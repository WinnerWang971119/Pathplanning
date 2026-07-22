"""Phase 2 crossing-traffic substrate for the Arena harness.

This module provides `DynamicObstacleState` (a frozen per-tick snapshot record)
and `TrafficSpawner` (the live spawner/advancer/wall-bouncer). Obstacles reflect
off the arena walls rather than exiting; the despawn/refill path is a safety net.
It is designed to be deterministic under a fixed `traffic_rng` seed: same seed +
same static world + same step count must produce the same sequence of
`state_sha256()` digests.

Determinism verification: TC20 in `arena/arena.py --check` runs two
`Arena(seed=3, traffic=True)` instances over 200 zero-action ticks and asserts
byte-identical `dynamic_obstacles_sha256` sequences (including the reset-time
hash). TC24 goes further: two same-seed `--traffic` runs through the episode
runner produce byte-identical trace JSONL, so the full per-tick trace
(including `lidar_sha256` under moving obstacles) is pinned, not just the
spawner-side hash. The digest hashes physical state only (x, y, vx, vy,
radius) ordered by id — the irsim object id is excluded so the sequence is
reproducible across repeated `reset()` of one Arena, not only across fresh
constructions.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from arena._errors import ArenaRuntimeError


TARGET_POPULATION = 20
OBSTACLE_RADIUS = 0.3
SPEED_MIN_FACTOR = 0.3
SPEED_MAX_FACTOR = 1.5
SPAWN_OVERLAP_BUFFER = 1.0
DESPAWN_BUFFER = 0.5
# Finite-difference step for the static-obstacle surface normal (used when bouncing
# an obstacle off an interior wall/pillar). Small enough to sample the local normal,
# large enough to stay clear of float noise.
STATIC_NORMAL_EPS = 1e-4
# Generous attempt budget: a spawn almost always succeeds in 1-2 tries, so a high
# cap makes a silent short-population (the TC18 invariant) effectively impossible
# without changing RNG consumption on the common path.
SPAWN_MAX_ATTEMPTS = 100
DYNAMIC_OBSTACLE_NAME_FMT = "traffic_{idx}"


@dataclass(frozen=True)
class DynamicObstacleState:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float


@dataclass
class _LiveObstacle:
    """Internal bookkeeping record. Velocity AND position are owned by the
    spawner: irsim omni objects do not expose vx/vy directly, and we
    deliberately do NOT read x/y back from the irsim handle either — keeping
    a spawner-side cache rules out determinism coupling to irsim's float
    round-tripping through its state-storage path (AC4)."""

    handle: Any
    x: float
    y: float
    vx: float
    vy: float
    radius: float


class TrafficSpawner:
    """Spawns and advances circular dynamic obstacles around a square arena.
    Deterministic under fixed RNG state.

    Obstacles spawn on the perimeter with an inward heading, then travel in
    straight lines and BOUNCE (elastic reflection) off both the arena walls (the
    [0,W]x[0,H] boundary) AND the interior static obstacles (walls/pillars), so
    the population stays inside, never exits, and never passes through a static
    obstacle. The despawn/refill machinery is retained only as a safety net: with
    reflection keeping every center in-bounds it is inert in normal operation, but
    a fresh obstacle would still refill if one ever escaped, keeping the
    population at the target.

    `live_ids` and `_next_idx` are intentionally distinct: `live_ids`
    reflects which irsim object ids exist *right now*; `_next_idx` is a
    monotonically increasing counter so obstacle names never collide across
    an Arena's lifetime (even after delete + respawn).
    """

    def __init__(
        self,
        env: Any,
        robot: Any,
        traffic_rng: np.random.Generator,
        motion_rng: np.random.Generator,
        dt: float,
        arena_w: float,
        arena_h: float,
        static_obstacles: Sequence[Any],
        *,
        speed_min_factor: float = SPEED_MIN_FACTOR,
        speed_max_factor: float = SPEED_MAX_FACTOR,
    ) -> None:
        if not (0.0 < speed_min_factor <= speed_max_factor):
            raise ValueError(
                "TrafficSpawner speed factors must satisfy 0 < min <= max, got "
                f"speed_min_factor={speed_min_factor}, "
                f"speed_max_factor={speed_max_factor}"
            )
        self._env = env
        self._robot = robot
        self._traffic_rng = traffic_rng
        self._motion_rng = motion_rng  # plumbed for forward-compat; unused in Phase 2
        self._dt = float(dt)
        self._arena_w = float(arena_w)
        self._arena_h = float(arena_h)
        self._static_obstacles = list(static_obstacles)
        self._speed_min_factor = float(speed_min_factor)
        self._speed_max_factor = float(speed_max_factor)

        robot_state = self._robot.state
        self._robot_start_xy = np.array(
            [float(robot_state[0, 0]), float(robot_state[1, 0])], dtype=np.float64
        )

        self._live: dict[int, _LiveObstacle] = {}
        self._next_idx = 0
        self._closed = False

        # Cache the point-to-obstacle distance callable once per spawner lifetime.
        # Lazy import to avoid a hard dependency cycle: arena.arena -> arena.dynamic
        # at module import time. Mirrors the TC10 sys.path pattern in arena/arena.py.
        import sys as _sys
        from pathlib import Path as _Path

        _repo_root = str(_Path(__file__).resolve().parent.parent)
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        from manual_astar import (  # type: ignore[import-not-found]
            MAX_LINEAR_SPEED,
            point_to_obstacle_distance,
        )

        self._point_to_obstacle_distance = point_to_obstacle_distance
        # Source the obstacle speed band from the robot's actual top speed instead of
        # a hand-copied constant, so the two cannot silently drift apart.
        self._robot_top_speed = float(MAX_LINEAR_SPEED)

    @property
    def live_ids(self) -> frozenset[int]:
        return frozenset(self._live.keys())

    def reset(
        self,
        traffic_rng: np.random.Generator,
        motion_rng: np.random.Generator,
    ) -> None:
        """Clear all live obstacles and rebind the RNGs for a new episode WITHOUT
        rebuilding the spawner: the cached point-distance callable, static-obstacle
        list, and cached robot start stay put (env.reset() restores the same robot
        pose). _next_idx is intentionally NOT reset, so obstacle names never collide
        across the Arena's lifetime even after delete + respawn."""
        if self._closed:
            raise ArenaRuntimeError("TrafficSpawner.reset() called after close()")
        for obs_id in list(self._live.keys()):
            try:
                self._env.delete_object(obs_id)
            except (KeyError, ValueError, AttributeError):
                pass
        self._live = {}
        self._traffic_rng = traffic_rng
        self._motion_rng = motion_rng

    def initialize(self) -> tuple[DynamicObstacleState, ...]:
        self._refill()
        return self.snapshot()

    def step(self) -> tuple[DynamicObstacleState, ...]:
        self._advance()
        self._despawn()
        self._refill()
        return self.snapshot()

    def snapshot(self) -> tuple[DynamicObstacleState, ...]:
        out: list[DynamicObstacleState] = []
        for obs_id in sorted(self._live.keys()):
            live = self._live[obs_id]
            out.append(
                DynamicObstacleState(
                    id=obs_id,
                    x=live.x,
                    y=live.y,
                    vx=live.vx,
                    vy=live.vy,
                    radius=live.radius,
                )
            )
        return tuple(out)

    def state_sha256(
        self, snap: tuple[DynamicObstacleState, ...] | None = None
    ) -> str:
        # Accept an already-built snapshot so callers don't rebuild it (step() and
        # initialize() already produced one). Hash physical state only: obstacle id is
        # an irsim handle that climbs across reset() (id_iter resets per env.make(),
        # not per env.reset()), so hashing it would make the digest differ across
        # repeated reset() of one Arena. Row order stays by id via snapshot(), so the
        # ordering is still deterministic.
        if snap is None:
            snap = self.snapshot()
        if not snap:
            arr = np.empty((0, 5), dtype=np.float64)
        else:
            arr = np.array(
                [[s.x, s.y, s.vx, s.vy, s.radius] for s in snap],
                dtype=np.float64,
            )
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def close(self) -> None:
        if self._closed:
            return
        for obs_id in list(self._live.keys()):
            try:
                self._env.delete_object(obs_id)
            except (KeyError, ValueError, AttributeError):
                # id-not-found is expected during torn-down env; other errors are programmer bugs we want to surface — but close() must not raise per Arena.close() contract.
                pass
        self._live = {}
        self._closed = True

    def _inject_for_test(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        radius: float = OBSTACLE_RADIUS,
    ) -> DynamicObstacleState:
        """Spawn an obstacle at an explicit state without drawing from traffic_rng.

        Mirrors the create_obstacle + add_object + record flow of a normal spawn,
        but does NOT consume any RNG draws so subsequent normal spawns see the
        same RNG state they would have seen without the injection.
        """
        if not all(math.isfinite(v) for v in (x, y, vx, vy, radius)):
            raise ValueError(
                f"_inject_for_test got non-finite values: x={x}, y={y}, vx={vx}, vy={vy}, radius={radius}"
            )
        if radius <= 0:
            raise ValueError(f"_inject_for_test requires radius > 0, got {radius}")
        handle = self._create_and_attach(x, y, radius)
        self._live[handle.id] = _LiveObstacle(
            handle=handle,
            x=float(x),
            y=float(y),
            vx=float(vx),
            vy=float(vy),
            radius=float(radius),
        )
        return DynamicObstacleState(
            id=handle.id, x=float(x), y=float(y), vx=float(vx), vy=float(vy), radius=float(radius)
        )

    def _advance(self) -> None:
        for live in self._live.values():
            # Integrate, then bounce off (1) interior static walls/pillars and (2) the
            # arena rectangle [0,W]x[0,H], so obstacles stay inside and never pass through
            # a static obstacle. The spawner-side cache is the source of truth for
            # determinism; push to the irsim handle once for lidar/collision consumers.
            # Both reflections are pure geometry (no RNG draw), so every spawn-draw
            # determinism guard (TC50's 3-draws-per-spawn) is untouched.
            x = live.x + live.vx * self._dt
            y = live.y + live.vy * self._dt
            vx, vy = live.vx, live.vy
            # Statics first, then the arena walls LAST so the wall reflection is the final
            # in-bounds clamp (a static push-out toward the boundary can never leave the
            # obstacle outside the arena). One arena reflection per axis suffices: the max
            # per-tick step |v|*dt = 2.0*0.1 = 0.2 m is far smaller than the 50 m span.
            x, y, vx, vy = self._reflect_off_statics(x, y, vx, vy, live.radius)
            x, vx = self._reflect(x, vx, self._arena_w)
            y, vy = self._reflect(y, vy, self._arena_h)
            live.x, live.y, live.vx, live.vy = x, y, vx, vy
            self._write_xy(live.handle, live.x, live.y)

    @staticmethod
    def _reflect(pos: float, vel: float, upper: float) -> tuple[float, float]:
        """Reflect a 1-D position/velocity across the walls at 0 and ``upper``.

        Mirrors the position back inside and flips the velocity sign (speed is
        conserved — an elastic bounce). A center that lands exactly on a wall or
        stays inside is returned unchanged.
        """
        if pos < 0.0:
            return -pos, -vel
        if pos > upper:
            return 2.0 * upper - pos, -vel
        return pos, vel

    def _reflect_off_statics(
        self, x: float, y: float, vx: float, vy: float, radius: float
    ) -> tuple[float, float, float, float]:
        """Bounce a moving obstacle off any interior static obstacle it is touching.

        Contact is when the obstacle CENTER is within ``radius`` of a static surface
        (`point_to_obstacle_distance < radius`). On contact the center is pushed back
        out to the surface (so it never sinks in or tunnels through) and the velocity
        is reflected across the surface normal, `v' = v - 2 (v·n) n`, but only when it
        is moving INTO the surface (`v·n < 0`) so a grazing obstacle is not flipped.

        The normal is the gradient of the analytic distance field (finite differences),
        which is valid for every obstacle kind — circle, rectangle, polygon, linestring.
        Detecting at `d < radius` (center still ``radius`` outside the boundary) keeps
        the gradient well-defined and, since the per-tick step (<=0.2 m) is smaller than
        the ``radius`` (0.3 m) contact band, prevents tunnelling through a thin wall.
        """
        if not self._static_obstacles:
            return x, y, vx, vy
        pos = np.array([x, y], dtype=np.float64)
        vel = np.array([vx, vy], dtype=np.float64)
        for static_obs in self._static_obstacles:
            d = self._point_to_obstacle_distance(pos, static_obs)
            if d >= radius:
                continue
            normal = self._surface_normal(pos, static_obs)
            if normal is None:
                continue
            pos = pos + (radius - d) * normal
            v_dot_n = float(vel @ normal)
            if v_dot_n < 0.0:
                vel = vel - 2.0 * v_dot_n * normal
        return float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1])

    def _surface_normal(self, pos: np.ndarray, static_obs: Any) -> np.ndarray | None:
        """Outward unit normal of a static obstacle at ``pos`` via central differences
        of the distance field. Returns None when the gradient is degenerate (e.g. the
        obstacle center coincides with a pillar center, or ``pos`` is inside a polygon
        where the clamped-to-0 distance has no slope)."""
        eps = STATIC_NORMAL_EPS
        dx = np.array([eps, 0.0])
        dy = np.array([0.0, eps])
        grad = np.array(
            [
                self._point_to_obstacle_distance(pos + dx, static_obs)
                - self._point_to_obstacle_distance(pos - dx, static_obs),
                self._point_to_obstacle_distance(pos + dy, static_obs)
                - self._point_to_obstacle_distance(pos - dy, static_obs),
            ],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(grad))
        if norm < 1e-9:
            return None
        return grad / norm

    def _despawn(self) -> None:
        # Safety net only. Obstacles now reflect off the arena walls in _advance, so a
        # center never leaves [0,W]x[0,H] and this buffer (outside [-DESPAWN_BUFFER,
        # W+DESPAWN_BUFFER]) is never crossed in normal operation. Kept so a numerically
        # escaped obstacle would still be removed and refilled rather than lingering.
        lo_x = -DESPAWN_BUFFER
        hi_x = self._arena_w + DESPAWN_BUFFER
        lo_y = -DESPAWN_BUFFER
        hi_y = self._arena_h + DESPAWN_BUFFER

        to_remove: list[int] = []
        for obs_id, live in self._live.items():
            if live.x < lo_x or live.x > hi_x or live.y < lo_y or live.y > hi_y:
                to_remove.append(obs_id)

        for obs_id in to_remove:
            # Reconcile our own tracking FIRST so a delete_object failure cannot leave
            # a phantom id in _live (which snapshot()/state_sha256() would then
            # over-report relative to what irsim's lidar/collision tree actually sees).
            del self._live[obs_id]
            try:
                self._env.delete_object(obs_id)
            except Exception as exc:
                raise ArenaRuntimeError(
                    f"env.delete_object failed for tracked id {obs_id}: {exc}"
                ) from exc

    def _refill(self) -> None:
        while len(self._live) < TARGET_POPULATION:
            spawned = self._try_one_spawn()
            if not spawned:
                # Exhausted SPAWN_MAX_ATTEMPTS for this slot. Surface it instead of
                # silently shipping a short population (the harness/TC18 treat 20 as an
                # invariant) so an unlucky seed is debuggable. Non-fatal: next tick retries.
                warnings.warn(
                    f"TrafficSpawner: refill gave up at {len(self._live)}/"
                    f"{TARGET_POPULATION} live after {SPAWN_MAX_ATTEMPTS} attempts; "
                    "population is below target this tick.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return

    def _try_one_spawn(self) -> bool:
        for _ in range(SPAWN_MAX_ATTEMPTS):
            # Three RNG draws per attempt, always in this order: position, heading, speed.
            t = float(
                self._traffic_rng.uniform(0.0, 2.0 * (self._arena_w + self._arena_h))
            )
            x, y, heading_lo, heading_hi = self._perimeter_sample(t)
            heading = float(self._traffic_rng.uniform(heading_lo, heading_hi))
            speed = float(
                self._traffic_rng.uniform(
                    self._speed_min_factor, self._speed_max_factor
                )
                * self._robot_top_speed
            )

            if self._overlaps_robot_start(x, y):
                continue
            if self._overlaps_static(x, y):
                continue

            vx = speed * math.cos(heading)
            vy = speed * math.sin(heading)

            handle = self._create_and_attach(x, y, OBSTACLE_RADIUS)
            self._live[handle.id] = _LiveObstacle(
                handle=handle,
                x=float(x),
                y=float(y),
                vx=vx,
                vy=vy,
                radius=OBSTACLE_RADIUS,
            )
            return True
        return False

    def _perimeter_sample(self, t: float) -> tuple[float, float, float, float]:
        """Map t in [0, 2*(W+H)) onto the arena perimeter and return the inward
        heading half-cone for that edge. Each edge is mapped by its own length
        (south=W, east=H, north=W, west=H), so this works for any rectangle, not
        only the square arenas shipped today."""
        W = self._arena_w
        H = self._arena_h
        if t < W:
            return (t, 0.0, 0.0, math.pi)
        if t < W + H:
            return (W, t - W, math.pi / 2.0, 3.0 * math.pi / 2.0)
        if t < 2.0 * W + H:
            return (2.0 * W + H - t, H, math.pi, 2.0 * math.pi)
        return (0.0, 2.0 * W + 2.0 * H - t, -math.pi / 2.0, math.pi / 2.0)

    def _overlaps_robot_start(self, x: float, y: float) -> bool:
        dx = x - self._robot_start_xy[0]
        dy = y - self._robot_start_xy[1]
        return math.hypot(dx, dy) < OBSTACLE_RADIUS + SPAWN_OVERLAP_BUFFER

    def _overlaps_static(self, x: float, y: float) -> bool:
        if not self._static_obstacles:
            return False
        point = np.array([x, y], dtype=np.float64)
        threshold = OBSTACLE_RADIUS + SPAWN_OVERLAP_BUFFER
        for static_obs in self._static_obstacles:
            if self._point_to_obstacle_distance(point, static_obs) < threshold:
                return True
        return False

    def _create_and_attach(self, x: float, y: float, radius: float) -> Any:
        name = DYNAMIC_OBSTACLE_NAME_FMT.format(idx=self._next_idx)
        self._next_idx += 1
        try:
            obs = self._env.create_obstacle(
                kinematics={"name": "omni"},
                shape={"name": "circle", "radius": radius},
                state=[float(x), float(y), 0.0],
                name=name,
            )
            self._env.add_object(obs)
        except ValueError as exc:
            raise ArenaRuntimeError(
                f"env.add_object rejected obstacle name {name!r}: {exc}"
            ) from exc
        return obs

    @staticmethod
    def _write_xy(handle: Any, x: float, y: float) -> None:
        # ObjectBase.state is read-only; the public mutation API is set_state(state, init=False).
        # Passing init=False keeps env.reset()'s spawn-pose restore behavior untouched.
        handle.set_state([float(x), float(y), 0.0], init=False)
