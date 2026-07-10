from __future__ import annotations

import argparse
import dataclasses
import filecmp
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import irsim
import numpy as np
import yaml


DEFAULT_TIMEOUT_S = 120.0
LIDAR_BEAM_COUNT = 360
ACTION_SHAPE = (2, 1)
RENDER_PAUSE_SECONDS = 0.05

# Render-only prediction-overlay styling (T16). Consumed solely by
# Arena.draw_prediction, which is a hard no-op unless render=True — so none of
# these touch the headless step/trace pipeline (determinism is unaffected).
PREDICTION_CELL_COLOR = "tab:orange"   # translucent predicted-footprint cells
PREDICTION_CELL_ALPHA = 0.25
PREDICTION_ARROW_COLOR = "red"         # per-track velocity arrows
PREDICTION_CELL_ZORDER = 5             # above irsim's static/obstacle patches
PREDICTION_ARROW_ZORDER = 6            # arrows above the cell overlay
# Visual multiplier (seconds-equivalent) applied to each track's (vx, vy) so a
# slow 0.3 m/s obstacle's arrow is still legible in a 50 m arena. Debug-only;
# not load-bearing on any metric or trace.
VELOCITY_ARROW_SCALE = 3.0


# Bootstrap repo root on sys.path so the `from arena.* import ...` below resolve
# whether this file is run as `python arena/arena.py` (script-mode puts arena/ on
# sys.path, not the repo root) or as `python -m arena.arena` / via the runner
# (repo root already on sys.path). Mirrors runners/run_episode.py:39-43. This MUST
# precede the arena._errors / arena.dynamic imports: in script-mode, without the
# repo root inserted in front of sys.path, `import arena` would resolve to THIS
# file rather than the package and the imports below would raise ModuleNotFoundError.
import sys as _sys
from pathlib import Path as _Path
_repo_root = str(_Path(__file__).resolve().parent.parent)
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

# Re-exported from the leaf `arena._errors` module so callers can keep doing
# `from arena.arena import ArenaConfigError, ArenaRuntimeError` unchanged. The
# former arena.arena <-> arena.dynamic cycle is broken: ArenaRuntimeError now
# lives in arena._errors, which arena.dynamic imports instead of arena.arena.
from arena._errors import ArenaConfigError, ArenaRuntimeError  # noqa: E402
from arena.dynamic import DynamicObstacleState, TrafficSpawner  # noqa: E402


@dataclass(frozen=True)
class EpisodeInfo:
    sim_time: float
    step_idx: int
    crashed: bool
    timed_out: bool
    reached_goal: bool
    distance_to_goal: float
    wallclock_per_step: float
    dynamic_obstacle_count: int
    lidar_status: str
    dynamic_obstacles_sha256: str | None
    # Live post-_advance() snapshot for THIS tick (same tick as dynamic_obstacles_sha256,
    # both sourced from self._last_snapshot). () when traffic is off or pre-reset. Distinct
    # from Arena.initial_dynamic_snapshot, which is the frozen t=0 view from _initial_snapshot.
    dynamic_obstacles: tuple[DynamicObstacleState, ...]


class Arena:
    """50x50 arena wrapping irsim. Static-only by default; pass traffic=True for Phase 2 crossing traffic."""

    def __init__(
        self,
        yaml_path: str | Path,
        seed: int,
        render: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        traffic: bool = False,
        *,
        speed_min_factor: float | None = None,
        speed_max_factor: float | None = None,
    ) -> None:
        self._yaml_path = Path(yaml_path)
        self._render = bool(render)
        self._timeout_s = float(timeout_s)
        self._master_seed = int(seed)
        self._traffic = bool(traffic)

        # Optional dynamic-obstacle speed band (factors of robot top speed). Both None
        # ⇒ the TrafficSpawner keeps its own SPEED_MIN_FACTOR/SPEED_MAX_FACTOR defaults
        # (the default path stays byte-identical: the kwargs are OMITTED from the
        # TrafficSpawner(...) call below, never passed as None — passing None would trip
        # the spawner's 0 < None validation). Both set ⇒ forwarded so the spawner
        # validates 0 < min <= max. Exactly one set is a programmer error (a one-sided
        # band), rejected here before any irsim/spawner construction.
        if (speed_min_factor is None) != (speed_max_factor is None):
            raise ValueError(
                "speed_min_factor and speed_max_factor must both be set or both be "
                f"None, got speed_min_factor={speed_min_factor}, "
                f"speed_max_factor={speed_max_factor}"
            )
        self._speed_min_factor = speed_min_factor
        self._speed_max_factor = speed_max_factor

        # With traffic on, every dynamic obstacle (omni, no behavior) makes irsim log
        # a per-tick WARNING ("Behavior not defined ..."), ~20 lines/tick that would
        # flood the runner output. Raise the irsim log level to ERROR for traffic runs
        # (collision/arrival are read from flags, not these logs); Phase 0/1 runs keep
        # the default level so their logging is unchanged.
        log_level = "ERROR" if self._traffic else "INFO"
        self._env = irsim.make(
            str(self._yaml_path), display=self._render, log_level=log_level
        )
        self._robot = self._env.robot_list[0]
        self._dt = float(self._env.step_time)
        goal = self._robot.goal
        if goal is None:
            raise ArenaConfigError("YAML robot has no goal pose")
        self._goal_xy = goal[:2, 0].astype(np.float64)

        scan = self._robot.get_lidar_scan()
        if not scan:
            raise ArenaConfigError("YAML robot has no working lidar2d sensor")
        if "ranges" not in scan:
            raise ArenaConfigError(
                f"YAML lidar2d scan dict missing 'ranges' key: keys={list(scan.keys())}"
            )
        beam_count = len(scan["ranges"])
        if beam_count != LIDAR_BEAM_COUNT:
            raise ArenaConfigError(
                f"lidar scan returned {beam_count} beams, expected {LIDAR_BEAM_COUNT}"
            )

        # traffic first, motion second — Phase 2 spawner consumes in this order
        ss = np.random.SeedSequence(self._master_seed)
        traffic_seed, motion_seed = ss.spawn(2)
        self._traffic_rng = np.random.default_rng(traffic_seed)
        self._motion_rng = np.random.default_rng(motion_seed)

        # Cache the WorldModel ONCE for the spawner construction (and for reuse on reset()).
        # Lazy-import manual_astar (mirrors TC10 pattern) — keeps arena import-time cheap
        # and avoids cycles if manual_astar grows imports.
        self._world_model: Any | None = None
        if self._traffic:
            import sys as _sys
            _repo_root = str(Path(__file__).resolve().parent.parent)
            if _repo_root not in _sys.path:
                _sys.path.insert(0, _repo_root)
            from manual_astar import load_world  # type: ignore[import-not-found]

            self._world_model = load_world(str(self._yaml_path))
            # Splat the optional speed band: empty dict (both None) reproduces the prior
            # call verbatim so Arena(traffic=True) stays byte-identical and TC17-TC24 are
            # unchanged; the two factors (both set) are forwarded to the spawner.
            speed_kwargs: dict[str, float] = (
                {}
                if self._speed_min_factor is None
                else {
                    "speed_min_factor": self._speed_min_factor,
                    "speed_max_factor": self._speed_max_factor,
                }
            )
            self._spawner: TrafficSpawner | None = TrafficSpawner(
                env=self._env,
                robot=self._robot,
                traffic_rng=self._traffic_rng,
                motion_rng=self._motion_rng,
                dt=self._dt,
                arena_w=float(self._world_model.width),
                arena_h=float(self._world_model.height),
                static_obstacles=self._world_model.obstacles,
                **speed_kwargs,
            )
        else:
            self._spawner = None

        # Pre-reset snapshot caches: per AC13, initial_dynamic_snapshot must return ()
        # and EpisodeInfo.dynamic_obstacles_sha256 must be None until reset() runs.
        # _initial_snapshot is captured ONCE at reset-time (the t=0 view planners get
        # via the public property); _last_snapshot tracks the per-tick state for
        # EpisodeInfo + sha256 bookkeeping inside step().
        self._initial_snapshot: tuple[DynamicObstacleState, ...] = ()
        self._last_snapshot: tuple[DynamicObstacleState, ...] = ()
        self._last_sha256: str | None = None

        self._step_idx = 0
        self._done = False
        self._closed = False
        self._reset_called = False

        # Render-only prediction overlay (T16). `_prediction_artists` holds the
        # matplotlib artists drawn last tick so draw_prediction can remove them
        # before drawing the next tick (no frame-to-frame accumulation).
        # `_prediction_geometry` caches the (resolution, offset_x, offset_y)
        # cell->world conversion so the YAML is parsed at most once. Both stay
        # untouched when render=False — the draw path is never entered.
        self._prediction_artists: list[Any] = []
        self._prediction_geometry: tuple[float, float, float] | None = None

    def reset(self) -> tuple[np.ndarray, np.ndarray, EpisodeInfo]:
        if self._closed:
            raise RuntimeError("Arena is closed")

        # Step 1: irsim reset (resets all current objects to _init_state, runs warm-up step)
        self._env.reset()

        # Step 2: re-derive RNGs deterministically from master seed (mirrors __init__
        # exactly so reset() is byte-equivalent to fresh construct + reset).
        ss = np.random.SeedSequence(self._master_seed)
        traffic_seed, motion_seed = ss.spawn(2)
        self._traffic_rng = np.random.default_rng(traffic_seed)
        self._motion_rng = np.random.default_rng(motion_seed)

        # Step 3: clear the PRIOR episode's dynamic obstacles. env.reset() resets their
        # POSE but does not remove them; spawning again without clearing would DOUBLE
        # the population. The spawner owns its own teardown (delete-all + RNG rebind) —
        # the cached point-distance callable and static-obstacle list are preserved,
        # and its _next_idx keeps climbing so obstacle names never collide.
        if self._spawner is not None:
            self._spawner.reset(self._traffic_rng, self._motion_rng)

        # Step 4: spawn fresh population (if traffic enabled). The initial snapshot
        # is pinned here and exposed via the initial_dynamic_snapshot property for the
        # full episode — _last_snapshot is overwritten on every step() but the t=0
        # view planners depend on must never drift.
        if self._spawner is not None:
            self._initial_snapshot = self._spawner.initialize()
            # env.reset()'s warm-up sensed the lidar BEFORE these obstacles existed, so
            # re-sense now: lidar0 must be consistent with the snapshot/sha the planner
            # receives for the same t=0 (reactive planners consume lidar0).
            self._robot.sensor_step()
            self._last_snapshot = self._initial_snapshot
            self._last_sha256 = self._spawner.state_sha256(self._last_snapshot)
        else:
            self._initial_snapshot = ()
            self._last_snapshot = ()
            self._last_sha256 = None

        # Step 5: defensive flag re-clear — irsim's reset() warm-up step may set these
        # against the just-reset pose if it overlaps an obstacle (Phase 0 T0 note).
        self._robot.arrive_flag = False
        self._robot.collision_flag = False

        # Step 6: counter reset
        self._step_idx = 0
        self._done = False
        self._reset_called = True

        # Step 7: build initial state + lidar + EpisodeInfo
        state = self._robot.state[:, 0].astype(np.float64)
        lidar, lidar_status = self._extract_lidar()

        info = EpisodeInfo(
            sim_time=0.0,
            step_idx=0,
            crashed=False,
            timed_out=False,
            reached_goal=False,
            distance_to_goal=float(np.linalg.norm(state[:2] - self._goal_xy)),
            wallclock_per_step=0.0,
            dynamic_obstacle_count=len(self._last_snapshot),
            lidar_status=lidar_status,
            dynamic_obstacles_sha256=self._last_sha256,
            dynamic_obstacles=self._last_snapshot,
        )
        return state, lidar, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, EpisodeInfo]:
        if self._done:
            raise RuntimeError("Episode is done; call reset() first.")
        if self._closed:
            raise RuntimeError("Arena is closed")
        if not self._reset_called:
            raise RuntimeError("reset() must be called before step()")

        if not isinstance(action, np.ndarray):
            raise ValueError(
                f"action must be np.ndarray, got {type(action).__name__}"
            )
        if action.shape != ACTION_SHAPE:
            raise ValueError(
                f"action shape must be {ACTION_SHAPE}, got {action.shape}"
            )
        if not np.issubdtype(action.dtype, np.floating):
            raise ValueError(f"action dtype must be float, got {action.dtype}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action contains NaN or Inf")

        # Advance dynamic obstacles BEFORE env.step() so the lidar inside env.step()
        # samples post-move obstacle positions on the same tick.
        if self._spawner is not None:
            self._last_snapshot = self._spawner.step()
            self._last_sha256 = self._spawner.state_sha256(self._last_snapshot)

        # Snapshot flags BEFORE step: irsim's check_*_status overwrite them per tick
        # (see object_base.py:531-532), so harness-injected flags would be lost otherwise.
        pre_crashed = bool(getattr(self._robot, "collision_flag", False))
        pre_reached = bool(getattr(self._robot, "arrive_flag", False))

        start = time.perf_counter()
        self._env.step([action])
        wallclock = time.perf_counter() - start

        # Drive irsim's repaint loop when render mode is on. Without this, the window
        # never updates between steps and the user only sees the final frame.
        # Excluded from wallclock_per_step on purpose.
        if self._render:
            self._env.render(RENDER_PAUSE_SECONDS)

        self._step_idx += 1

        state = self._robot.state[:, 0].astype(np.float64)
        lidar, lidar_status = self._extract_lidar()

        crashed = pre_crashed or bool(getattr(self._robot, "collision_flag", False))
        reached_goal = pre_reached or bool(getattr(self._robot, "arrive_flag", False))
        sim_time = self._step_idx * self._dt
        timed_out = sim_time >= self._timeout_s
        distance_to_goal = float(np.linalg.norm(state[:2] - self._goal_xy))

        info = EpisodeInfo(
            sim_time=sim_time,
            step_idx=self._step_idx,
            crashed=crashed,
            timed_out=timed_out,
            reached_goal=reached_goal,
            distance_to_goal=distance_to_goal,
            wallclock_per_step=wallclock,
            dynamic_obstacle_count=len(self._last_snapshot),
            lidar_status=lidar_status,
            dynamic_obstacles_sha256=self._last_sha256,
            dynamic_obstacles=self._last_snapshot,
        )

        done = crashed or timed_out or reached_goal
        self._done = bool(done)
        return state, lidar, self._done, info

    def _extract_lidar(self) -> tuple[np.ndarray, str]:
        scan = self._robot.get_lidar_scan()
        if not scan:
            return (
                np.full((LIDAR_BEAM_COUNT,), np.nan, dtype=np.float64),
                "missing",
            )
        if "ranges" not in scan:
            raise ArenaRuntimeError(
                f"irsim lidar returned a non-falsy scan without 'ranges' key: "
                f"keys={list(scan.keys())}"
            )
        ranges = np.asarray(scan["ranges"], dtype=np.float64)
        if ranges.shape != (LIDAR_BEAM_COUNT,):
            raise ArenaRuntimeError(
                f"irsim lidar returned ranges of shape {ranges.shape}, "
                f"expected ({LIDAR_BEAM_COUNT},)"
            )
        range_max = float(scan.get("range_max", np.inf))
        ranges = np.where(
            np.isfinite(ranges) & (ranges < range_max), ranges, np.nan
        )
        return ranges, "ok"

    @property
    def initial_dynamic_snapshot(self) -> tuple[DynamicObstacleState, ...]:
        """Snapshot of dynamic obstacles at t=0 of the current episode.

        Phase 0/1: always (). Phase 2 with traffic=True: tuple of 20 DynamicObstacleState
        entries captured by TrafficSpawner.initialize() at reset-time. This is the
        t=0 view Mission.md guarantees to planners — it does NOT update on subsequent
        step() calls. Mid-episode dynamic state is not exposed in Phase 2; Phase 6
        replanners that need the live set will query the spawner separately.
        """
        return self._initial_snapshot

    # ------------------------------------------------------------------ #
    # Render-only prediction overlay (T16)                               #
    # ------------------------------------------------------------------ #

    def draw_prediction(
        self,
        cells: list[tuple[int, int]],
        tracks: list[Any],
    ) -> None:
        """Paint the predictive controller's debug overlay on irsim's axes.

        Strictly render-only and read-only w.r.t. the step/trace pipeline. When
        ``render=False`` this returns IMMEDIATELY — the draw path is never
        entered, so headless runs stay byte-identical (the AC11 determinism
        guard, covered by TC58). It never raises: any failure to locate irsim's
        matplotlib axes, derive grid geometry, or draw degrades to a no-op so a
        render run can never crash on the overlay.

        Parameters
        ----------
        cells:
            ``(row, col)`` grid cells of the predicted footprint (the
            controller's ``last_predicted_cells``). Each is converted to a
            translucent world-frame square of side ``GRID_RESOLUTION``. May be
            empty (pre-first-act, horizon 0, or no threats) — drawn as nothing.
        tracks:
            Tracked obstacles (the controller's ``last_tracks``); each is read
            for ``.x, .y, .vx, .vy`` and drawn as a velocity arrow from its
            position along its velocity. May be empty.

        The previous tick's artists are removed FIRST so the overlay refreshes
        in place and never accumulates frame-to-frame.
        """
        # AC11 hard guard: never enter the draw path when headless.
        if not self._render:
            return

        # Always clear the previous tick's overlay first, even when this tick
        # draws nothing, so a transient empty `cells`/`tracks` does not leave a
        # stale overlay on screen.
        self._clear_prediction_artists()

        ax = self._discover_render_axes()
        if ax is None:
            return  # irsim axes API differs — degrade to a no-op (never raise).

        try:
            resolution, offset_x, offset_y = self._prediction_grid_geometry()
        except Exception:
            return  # cannot place cells without grid geometry — no-op.

        new_artists: list[Any] = []
        try:
            # Lazy, defensive imports: matplotlib is already loaded by irsim in
            # render mode, but keep these inside the method so module import (and
            # the `python arena/arena.py` script-mode import order) is untouched.
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Rectangle

            # Translucent predicted-footprint cells, as one PatchCollection so a
            # single artist covers the whole stamp (cheap to add and remove).
            patches: list[Rectangle] = []
            for cell in cells or []:
                row, col = cell
                lower_left_x = offset_x + col * resolution
                lower_left_y = offset_y + row * resolution
                patches.append(
                    Rectangle((lower_left_x, lower_left_y), resolution, resolution)
                )
            if patches:
                collection = PatchCollection(
                    patches,
                    facecolor=PREDICTION_CELL_COLOR,
                    edgecolor="none",
                    alpha=PREDICTION_CELL_ALPHA,
                    zorder=PREDICTION_CELL_ZORDER,
                )
                ax.add_collection(collection)
                new_artists.append(collection)

            # Per-track velocity arrows, as one quiver (a single artist). The
            # (vx, vy) vector is scaled into data units so the arrow length is
            # proportional to speed and legible in the arena.
            origins_x: list[float] = []
            origins_y: list[float] = []
            vectors_x: list[float] = []
            vectors_y: list[float] = []
            for track in tracks or []:
                origins_x.append(float(track.x))
                origins_y.append(float(track.y))
                vectors_x.append(float(track.vx) * VELOCITY_ARROW_SCALE)
                vectors_y.append(float(track.vy) * VELOCITY_ARROW_SCALE)
            if origins_x:
                quiver = ax.quiver(
                    origins_x,
                    origins_y,
                    vectors_x,
                    vectors_y,
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    color=PREDICTION_ARROW_COLOR,
                    width=0.004,
                    zorder=PREDICTION_ARROW_ZORDER,
                )
                new_artists.append(quiver)
        except Exception:
            # A drawing failure must not crash a render run. Remove anything we
            # managed to add this tick so nothing leaks, then degrade to a no-op.
            for artist in new_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            self._prediction_artists = []
            return

        self._prediction_artists = new_artists

    def _clear_prediction_artists(self) -> None:
        """Remove the artists draw_prediction added last tick (best-effort)."""
        for artist in self._prediction_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._prediction_artists = []

    def _discover_render_axes(self) -> Any | None:
        """Best-effort discovery of irsim's matplotlib Axes; None if not found.

        irsim renders through its own ``EnvPlot`` figure/axes. The known path is
        ``self._env._env_plot.ax``; alternates are tried with ``getattr`` so a
        future irsim version that moves the attribute degrades to a no-op overlay
        instead of crashing the render run.
        """
        env = getattr(self, "_env", None)
        if env is None:
            return None
        # Primary: the EnvPlot instance's axes.
        env_plot = getattr(env, "_env_plot", None)
        ax = getattr(env_plot, "ax", None) if env_plot is not None else None
        if ax is not None:
            return ax
        # Alternate: a direct .ax on the env.
        ax = getattr(env, "ax", None)
        if ax is not None:
            return ax
        # Last resort: the current pyplot axes, only if a figure already exists
        # (never create a fresh blank figure as a side effect of discovery).
        try:
            import matplotlib.pyplot as plt

            if plt.get_fignums():
                return plt.gca()
        except Exception:
            return None
        return None

    def _prediction_grid_geometry(self) -> tuple[float, float, float]:
        """``(resolution, offset_x, offset_y)`` for the cell->world conversion.

        Mirrors how the planner sizes its occupancy grid: ``GRID_RESOLUTION``
        metres per cell and the world ``offset`` from the YAML (default
        ``[0, 0]``). Cached after the first call so the YAML is parsed at most
        once. Reuses the already-loaded ``self._world_model`` when traffic
        provided it; otherwise lazy-loads it via ``manual_astar`` behind the
        repo-root ``sys.path`` shim (mirrors the other manual_astar imports in
        this module so script-mode import order is preserved).
        """
        if self._prediction_geometry is not None:
            return self._prediction_geometry

        world_model = self._world_model
        if world_model is None:
            import sys as _sys

            _repo_root = str(Path(__file__).resolve().parent.parent)
            if _repo_root not in _sys.path:
                _sys.path.insert(0, _repo_root)
            from manual_astar import load_world  # type: ignore[import-not-found]

            world_model = load_world(str(self._yaml_path))

        from manual_astar import GRID_RESOLUTION  # type: ignore[import-not-found]

        offset = world_model.offset
        geometry = (float(GRID_RESOLUTION), float(offset[0]), float(offset[1]))
        self._prediction_geometry = geometry
        return geometry

    def close(self) -> None:
        if self._closed:
            return
        if self._spawner is not None:
            self._spawner.close()
        self._env.end()
        self._closed = True


# ---------------------------------------------------------------------------
# TC1..TC12 — executable verification suite (run via `--check` from __main__).
# Each TC builds its own Arena, runs its assertions, and always calls close()
# in a finally block. Raise AssertionError on failure with a clear message.
# ---------------------------------------------------------------------------


EXPECTED_EPISODE_INFO_FIELDS = (
    "sim_time",
    "step_idx",
    "crashed",
    "timed_out",
    "reached_goal",
    "distance_to_goal",
    "wallclock_per_step",
    "dynamic_obstacle_count",
    "lidar_status",
    "dynamic_obstacles_sha256",
    "dynamic_obstacles",
)


def tc1(yaml_path: str, seed: int) -> None:
    """Construct an Arena and close it cleanly."""
    arena = Arena(yaml_path, seed)
    arena.close()


def tc2(yaml_path: str, seed: int) -> None:
    """Reset returns correctly shaped state/lidar and a fully populated EpisodeInfo."""
    arena = Arena(yaml_path, seed)
    try:
        state, lidar, info = arena.reset()

        assert isinstance(state, np.ndarray), f"state must be np.ndarray, got {type(state).__name__}"
        assert state.shape == (3,), f"state.shape must be (3,), got {state.shape}"

        assert isinstance(lidar, np.ndarray), f"lidar must be np.ndarray, got {type(lidar).__name__}"
        assert lidar.shape == (LIDAR_BEAM_COUNT,), (
            f"lidar.shape must be ({LIDAR_BEAM_COUNT},), got {lidar.shape}"
        )
        assert lidar.dtype == np.float64, f"lidar.dtype must be float64, got {lidar.dtype}"

        assert isinstance(info, EpisodeInfo), (
            f"info must be an EpisodeInfo, got {type(info).__name__}"
        )
        field_names = tuple(f.name for f in dataclasses.fields(info))
        assert field_names == EXPECTED_EPISODE_INFO_FIELDS, (
            f"EpisodeInfo fields mismatch: got {field_names}, "
            f"expected {EXPECTED_EPISODE_INFO_FIELDS}"
        )
        assert info.lidar_status == "ok", (
            f"info.lidar_status must be 'ok' on a healthy reset, got {info.lidar_status!r}"
        )
    finally:
        arena.close()


def tc2b(yaml_path: str, seed: int) -> None:
    """Missing lidar tick: monkeypatch get_lidar_scan to return None and step once."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()
        original_scan = arena._robot.get_lidar_scan
        arena._robot.get_lidar_scan = lambda: None
        try:
            _, lidar, _, info = arena.step(np.array([[0.0], [0.0]], dtype=float))
            assert lidar.shape == (LIDAR_BEAM_COUNT,), (
                f"lidar.shape must be ({LIDAR_BEAM_COUNT},), got {lidar.shape}"
            )
            assert np.all(np.isnan(lidar)), "lidar must be all NaN when scan is missing"
            assert info.lidar_status == "missing", (
                f"info.lidar_status must be 'missing', got {info.lidar_status!r}"
            )
        finally:
            arena._robot.get_lidar_scan = original_scan
    finally:
        arena.close()


def tc3(yaml_path: str, seed: int) -> None:
    """One zero-action step: not done, step_idx advances, sim_time increments by dt."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()
        _, _, done, info = arena.step(np.array([[0.0], [0.0]], dtype=float))
        assert done is False, f"done must be False after one zero-action step, got {done}"
        assert info.step_idx == 1, f"info.step_idx must be 1, got {info.step_idx}"
        assert abs(info.sim_time - arena._dt) < 1e-9, (
            f"info.sim_time must equal dt={arena._dt}, got {info.sim_time}"
        )
    finally:
        arena.close()


def tc4(yaml_path: str, seed: int) -> None:
    """Deliberate crash: v=1.0, w=0.3 curves into pillar (5, 5) within 200 steps."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()
        action = np.array([[1.0], [0.3]], dtype=float)
        max_steps = 200
        done = False
        info: EpisodeInfo | None = None
        for _ in range(max_steps):
            _, _, done, info = arena.step(action)
            if done:
                break
        assert done, (
            f"Episode did not terminate within {max_steps} steps; "
            f"final info={info}"
        )
        assert info is not None and info.crashed, (
            f"Expected info.crashed == True after curved drive, got info={info}"
        )
    finally:
        arena.close()


def tc5(yaml_path: str, seed: int) -> None:
    """Standing still must trigger timeout once sim_time >= timeout_s."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()
        action = np.array([[0.0], [0.0]], dtype=float)
        max_iters = int(DEFAULT_TIMEOUT_S / arena._dt) + 5
        done = False
        info: EpisodeInfo | None = None
        for _ in range(max_iters):
            _, _, done, info = arena.step(action)
            if done:
                break
        assert done, (
            f"Episode did not terminate within {max_iters} zero-action steps; "
            f"final info={info}"
        )
        assert info is not None and info.timed_out, (
            f"Expected info.timed_out == True after standing still, got info={info}"
        )
    finally:
        arena.close()


def tc6(yaml_path: str, seed: int) -> None:
    """Calling step() after done == True must raise RuntimeError."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()
        arena._done = True
        try:
            arena.step(np.array([[0.0], [0.0]], dtype=float))
        except RuntimeError:
            return
        raise AssertionError("step() after done must raise RuntimeError, but it did not")
    finally:
        arena.close()


def tc7(yaml_path: str, seed: int) -> None:
    """reset() after a finished episode clears sticky state and zeroes counters."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()
        crash_action = np.array([[1.0], [0.3]], dtype=float)
        done = False
        for _ in range(200):
            _, _, done, _ = arena.step(crash_action)
            if done:
                break
        assert done, "Setup for TC7 failed: episode did not terminate via crash drive"

        _, _, info = arena.reset()
        assert info.sim_time == 0.0, f"info.sim_time must be 0.0 after reset, got {info.sim_time}"
        assert info.step_idx == 0, f"info.step_idx must be 0 after reset, got {info.step_idx}"
        assert info.crashed is False, f"info.crashed must be False after reset, got {info.crashed}"
        assert info.timed_out is False, (
            f"info.timed_out must be False after reset, got {info.timed_out}"
        )
        assert info.reached_goal is False, (
            f"info.reached_goal must be False after reset, got {info.reached_goal}"
        )
        assert arena._done is False, (
            f"Arena._done must be cleared after reset, got {arena._done}"
        )
    finally:
        arena.close()


def tc8(yaml_path: str, seed: int) -> None:
    """Injecting robot.arrive_flag=True before a zero step must mark reached_goal/done."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()
        arena._robot.arrive_flag = True
        _, _, done, info = arena.step(np.array([[0.0], [0.0]], dtype=float))
        assert done is True, f"done must be True after arrive_flag injection, got {done}"
        assert info.reached_goal is True, (
            f"info.reached_goal must be True after arrive_flag injection, got {info.reached_goal}"
        )
    finally:
        arena.close()


def tc9(yaml_path: str, seed: int) -> None:
    """All malformed actions must raise ValueError."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()
        bad_actions: list[tuple[str, Any]] = [
            ("list-not-ndarray", [0.0, 0.0]),
            ("wrong-shape-(1,2)", np.array([[0.0, 0.0]], dtype=float)),
            ("int-dtype", np.array([[0], [0]], dtype=int)),
            ("contains-NaN", np.array([[np.nan], [0.0]], dtype=float)),
            ("contains-Inf", np.array([[np.inf], [0.0]], dtype=float)),
        ]
        failures: list[str] = []
        for label, bad in bad_actions:
            try:
                arena.step(bad)
            except ValueError:
                continue
            except Exception as exc:
                failures.append(
                    f"{label}: expected ValueError, got {type(exc).__name__}: {exc}"
                )
                continue
            failures.append(f"{label}: expected ValueError, but step() returned normally")
        if failures:
            raise AssertionError("; ".join(failures))
    finally:
        arena.close()


def tc10(yaml_path: str, seed: int) -> None:  # noqa: ARG001 (seed unused — planner is deterministic in yaml)
    """manual_astar.py must accept the world: load, inflate, validate start/goal unblocked."""
    # Local import keeps Arena import-time cheap and avoids cycles if manual_astar grows imports.
    # manual_astar.py lives at the repo root; ensure it's importable when arena.py is invoked
    # from any cwd (e.g. `python arena/arena.py ...` puts `arena/` on sys.path, not the root).
    import sys
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from manual_astar import (  # type: ignore[import-not-found]
        GRID_RESOLUTION,
        SAFETY_MARGIN,
        build_occupancy_grid,
        load_world,
        validate_start_and_goal,
    )

    world = load_world(yaml_path)
    grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN)
    validate_start_and_goal(world, grid)


def tc11(yaml_path: str, seed: int) -> None:  # noqa: ARG001
    """YAML schema sanity: world size, start/goal poses, and obstacle composition."""
    with open(yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    assert data["world"]["width"] == 50, (
        f"world.width must be 50, got {data['world']['width']}"
    )
    assert data["world"]["height"] == 50, (
        f"world.height must be 50, got {data['world']['height']}"
    )
    assert data["robot"]["state"] == [2, 2, 0], (
        f"robot.state must be [2, 2, 0], got {data['robot']['state']}"
    )
    assert data["robot"]["goal"] == [48, 48, 0], (
        f"robot.goal must be [48, 48, 0], got {data['robot']['goal']}"
    )

    obstacles = data["obstacle"]
    assert len(obstacles) == 14, f"expected 14 obstacles, got {len(obstacles)}"

    rect_count = sum(1 for o in obstacles if o["shape"]["name"] == "rectangle")
    circle_count = sum(1 for o in obstacles if o["shape"]["name"] == "circle")
    assert rect_count == 2, f"expected exactly 2 rectangle obstacles, got {rect_count}"
    assert circle_count == 12, f"expected exactly 12 circle obstacles, got {circle_count}"


def tc12(yaml_path: str, seed: int) -> None:  # noqa: ARG001
    """A YAML whose lidar2d.number != 360 must trigger ArenaConfigError at construction."""
    with open(yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    # Mutate beam count so Arena.__init__'s validation rejects it.
    data["robot"]["sensors"][0]["number"] = 180

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    try:
        yaml.safe_dump(data, tmp)
        tmp.close()
        tmp_path = tmp.name
        try:
            Arena(tmp_path, seed=0)
        except ArenaConfigError:
            return
        except Exception as exc:
            raise AssertionError(
                f"expected ArenaConfigError, got {type(exc).__name__}: {exc}"
            )
        raise AssertionError(
            "expected ArenaConfigError when lidar2d.number != 360, but construction succeeded"
        )
    finally:
        try:
            tmp.close()
        except Exception:
            pass
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


TC13_MAX_STEPS = 100


def tc13(yaml_path: str, seed: int) -> None:
    """Teleport robot under Wall B, drive forward, and assert crash within budget."""
    arena = Arena(yaml_path, seed)
    try:
        arena.reset()

        # ObjectBase.state is read-only; set_state also refreshes geometry for collision checks.
        arena._robot.set_state([20.0, 19.0, np.pi / 2], init=False)

        arena._robot.collision_flag = False
        arena._robot.arrive_flag = False

        action = np.array([[1.0], [0.0]], dtype=float)
        done = False
        info: EpisodeInfo | None = None
        for _ in range(TC13_MAX_STEPS):
            _, _, done, info = arena.step(action)
            if done and info.crashed:
                break

        assert done and info is not None and info.crashed, (
            f"TC13 did not crash within {TC13_MAX_STEPS} steps; final info={info}"
        )
    finally:
        arena.close()


# ---------------------------------------------------------------------------
# TC14..TC16 — runner integration checks. These subprocess-invoke
# `python -m runners.run_episode`, then validate the resulting metrics JSON
# and trace JSONL artifacts under a tempdir. Subprocess cwd is the repo root
# (parent of arena/) so `runners.run_episode` resolves correctly and relative
# --world paths like `arena/arena_v1.yaml` resolve from the same anchor.
# ---------------------------------------------------------------------------


TC14_TRACE_REQUIRED_KEYS = frozenset(
    {"action", "crashed", "done", "lidar_sha256", "reached_goal", "state", "step"}
)
TC14_METRICS_REQUIRED_KEYS = frozenset(
    {
        "time_to_goal",
        "crashed",
        "timed_out",
        "path_length",
        "mean_speed",
        "wallclock_per_step",
        "planner_error",
    }
)
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def tc14(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses fixed internal seed for determinism
    """Full a_star_once drive through run_episode + trace-line schema audit."""
    repo_root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as td:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "runners.run_episode",
                "--algorithm",
                "a_star_once",
                "--seed",
                "42",
                "--world",
                yaml_path,
                "--results-dir",
                td,
                "--no-traffic",
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"TC14 runner exit code {result.returncode}; "
            f"stderr={result.stderr[-500:]}"
        )

        json_path = Path(td) / "arena_v1" / "a_star_once" / "42.json"
        jsonl_path = Path(td) / "arena_v1" / "a_star_once" / "42.trace.jsonl"
        assert json_path.exists(), f"TC14: metrics JSON missing at {json_path}"
        assert jsonl_path.exists(), f"TC14: trace JSONL missing at {jsonl_path}"

        metrics = json.loads(json_path.read_text(encoding="utf-8"))
        # Lazy-import speed bounds from manual_astar (mirrors tc10's sys.path pattern).
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        from manual_astar import (  # type: ignore[import-not-found]
            MAX_LINEAR_SPEED,
            MIN_LINEAR_SPEED,
        )

        assert set(metrics) == TC14_METRICS_REQUIRED_KEYS, (
            f"TC14 metrics keys mismatch: got {set(metrics)}, "
            f"expected {set(TC14_METRICS_REQUIRED_KEYS)}"
        )
        assert metrics["planner_error"] is None, f"TC14 planner_error not None: {metrics}"
        assert metrics["crashed"] is False, f"TC14 crashed: {metrics}"
        assert metrics["timed_out"] is False, f"TC14 timed_out: {metrics}"
        assert metrics["time_to_goal"] is not None, f"TC14 time_to_goal is None: {metrics}"
        assert 50.0 < metrics["time_to_goal"] < 120.0, (
            f"TC14 time_to_goal out of range (50, 120): {metrics}"
        )
        assert metrics["path_length"] > 64.0, f"TC14 path_length too short: {metrics}"
        assert MIN_LINEAR_SPEED <= metrics["mean_speed"] <= MAX_LINEAR_SPEED, (
            f"TC14 mean_speed out of [{MIN_LINEAR_SPEED}, {MAX_LINEAR_SPEED}]: {metrics}"
        )
        assert metrics["mean_speed"] > 0.5, f"TC14 mean_speed too low: {metrics}"

        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) > 1, f"TC14 trace too short: {len(lines)} lines"
        for idx, line in enumerate(lines):
            row = json.loads(line)
            assert set(row) == TC14_TRACE_REQUIRED_KEYS, (
                f"TC14 trace line {idx} keys mismatch: got {set(row)}, "
                f"expected {set(TC14_TRACE_REQUIRED_KEYS)}"
            )
            assert isinstance(row["step"], int), (
                f"TC14 line {idx} step type: {type(row['step']).__name__}"
            )
            assert isinstance(row["state"], list) and len(row["state"]) == 3, (
                f"TC14 line {idx} state shape: {row['state']!r}"
            )
            assert all(isinstance(x, (int, float)) for x in row["state"]), (
                f"TC14 line {idx} state element types: {row['state']!r}"
            )
            assert isinstance(row["action"], list) and len(row["action"]) == 2, (
                f"TC14 line {idx} action shape: {row['action']!r}"
            )
            assert all(isinstance(x, (int, float)) for x in row["action"]), (
                f"TC14 line {idx} action element types: {row['action']!r}"
            )
            assert isinstance(row["lidar_sha256"], str) and _HEX64_RE.match(
                row["lidar_sha256"]
            ), f"TC14 line {idx} lidar_sha256: {row['lidar_sha256']!r}"
            assert isinstance(row["crashed"], bool), (
                f"TC14 line {idx} crashed type: {type(row['crashed']).__name__}"
            )
            assert isinstance(row["reached_goal"], bool), (
                f"TC14 line {idx} reached_goal type: {type(row['reached_goal']).__name__}"
            )
            assert isinstance(row["done"], bool), (
                f"TC14 line {idx} done type: {type(row['done']).__name__}"
            )

        first = json.loads(lines[0])
        assert first["step"] == 0, f"TC14 first line step != 0: {first}"
        assert first["state"] == [2.0, 2.0, 0.0], (
            f"TC14 first line state != [2.0, 2.0, 0.0]: {first}"
        )
        assert first["action"] == [0.0, 0.0], (
            f"TC14 first line action != [0.0, 0.0]: {first}"
        )
        assert first["done"] is False and first["reached_goal"] is False, (
            f"TC14 first line done/reached_goal flags: {first}"
        )

        last = json.loads(lines[-1])
        assert last["done"] is True, f"TC14 last line done != True: {last}"
        assert last["reached_goal"] is True, (
            f"TC14 last line reached_goal != True: {last}"
        )


def tc15(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """Determinism: two same-seed subprocess runs produce byte-identical trace JSONL."""
    repo_root = str(Path(__file__).resolve().parent.parent)
    runner_args = [
        sys.executable,
        "-m",
        "runners.run_episode",
        "--algorithm",
        "a_star_once",
        "--seed",
        "42",
        "--world",
        yaml_path,
        "--no-traffic",
    ]
    with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
        for td in (td_a, td_b):
            r = subprocess.run(
                [*runner_args, "--results-dir", td],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert r.returncode == 0, (
                f"TC15 runner exit {r.returncode}; stderr={r.stderr[-400:]}"
            )

        jsonl_a = Path(td_a) / "arena_v1" / "a_star_once" / "42.trace.jsonl"
        jsonl_b = Path(td_b) / "arena_v1" / "a_star_once" / "42.trace.jsonl"
        assert jsonl_a.exists() and jsonl_b.exists(), (
            f"TC15 trace JSONLs missing: a_exists={jsonl_a.exists()}, "
            f"b_exists={jsonl_b.exists()}"
        )
        assert filecmp.cmp(str(jsonl_a), str(jsonl_b), shallow=False), (
            "TC15 trace JSONLs differ — same-seed determinism broken (issue lives in "
            "runners/run_episode.py, not arena.py)"
        )

        json_a = json.loads(
            (Path(td_a) / "arena_v1" / "a_star_once" / "42.json").read_text(encoding="utf-8")
        )
        json_b = json.loads(
            (Path(td_b) / "arena_v1" / "a_star_once" / "42.json").read_text(encoding="utf-8")
        )
        json_a.pop("wallclock_per_step", None)
        json_b.pop("wallclock_per_step", None)
        assert json_a == json_b, (
            f"TC15 metrics differ (excluding wallclock_per_step): "
            f"a={json_a} b={json_b}"
        )


def tc16(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal world
    """Planner failure path: arena_no_path.yaml yields planner_error and no trace JSONL."""
    repo_root = Path(__file__).resolve().parent.parent
    no_path_yaml = str(repo_root / "arena" / "arena_no_path.yaml")
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "runners.run_episode",
                "--algorithm",
                "a_star_once",
                "--seed",
                "0",
                "--world",
                no_path_yaml,
                "--results-dir",
                td,
                "--no-traffic",
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r.returncode == 0, (
            f"TC16 runner exit {r.returncode}; stderr={r.stderr[-400:]}"
        )

        json_path = Path(td) / "arena_no_path" / "a_star_once" / "0.json"
        jsonl_path = Path(td) / "arena_no_path" / "a_star_once" / "0.trace.jsonl"
        assert json_path.exists(), f"TC16 metrics JSON missing at {json_path}"
        assert not jsonl_path.exists(), (
            f"TC16 trace JSONL must NOT exist on planner failure; found {jsonl_path}"
        )

        metrics = json.loads(json_path.read_text(encoding="utf-8"))
        assert metrics["planner_error"] is not None, (
            f"TC16 planner_error must not be None: {metrics}"
        )
        assert "could not find a path" in metrics["planner_error"], (
            f"TC16 planner_error must contain 'could not find a path': {metrics}"
        )
        assert metrics["time_to_goal"] is None, (
            f"TC16 time_to_goal must be None on planner failure: {metrics}"
        )
        assert metrics["crashed"] is False, f"TC16 crashed flag: {metrics}"
        assert metrics["timed_out"] is False, f"TC16 timed_out flag: {metrics}"


# ---------------------------------------------------------------------------
# TC17..TC23 — Phase 2 traffic checks (TC17..TC21) + path partitioning (TC22)
# + import-cycle guard (TC23). All use arena/arena_v1.yaml unless noted.
# ---------------------------------------------------------------------------


def tc17(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own seed (0) for determinism
    """Init population: 20 obstacles on perimeter edges with inward headings."""
    from arena.dynamic import DynamicObstacleState, TARGET_POPULATION, OBSTACLE_RADIUS

    arena = Arena(yaml_path, seed=0, traffic=True)
    try:
        _, _, info = arena.reset()
        assert info.dynamic_obstacle_count == TARGET_POPULATION, (
            f"TC17: dynamic_obstacle_count must be {TARGET_POPULATION}, got {info.dynamic_obstacle_count}"
        )
        snapshot = arena.initial_dynamic_snapshot
        assert len(snapshot) == TARGET_POPULATION, (
            f"TC17: snapshot length must be {TARGET_POPULATION}, got {len(snapshot)}"
        )
        # Per-obstacle invariants
        # Read the world dims from the YAML so the perimeter check tracks any future arena size change.
        import yaml as _yaml
        with open(yaml_path, "r", encoding="utf-8") as fh:
            world_data = _yaml.safe_load(fh)
        W = float(world_data["world"]["width"])
        H = float(world_data["world"]["height"])

        tol = 1e-6
        for i, obs in enumerate(snapshot):
            assert isinstance(obs, DynamicObstacleState), (
                f"TC17: snapshot[{i}] is {type(obs).__name__}, expected DynamicObstacleState"
            )
            assert obs.radius == OBSTACLE_RADIUS, (
                f"TC17: snapshot[{i}].radius must be {OBSTACLE_RADIUS}, got {obs.radius}"
            )
            # Perimeter check: must lie on one of the four edges within tol.
            on_south = abs(obs.y - 0.0) < tol
            on_north = abs(obs.y - H) < tol
            on_west  = abs(obs.x - 0.0) < tol
            on_east  = abs(obs.x - W) < tol
            assert on_south or on_north or on_west or on_east, (
                f"TC17: snapshot[{i}] at ({obs.x}, {obs.y}) is not on a perimeter edge "
                f"(W={W}, H={H}, tol={tol})"
            )
            # Inward-heading check: the velocity must have a non-negative inward
            # component for AT LEAST ONE edge the obstacle lies on. The spawner draws
            # heading from a half-open cone, so the inward component can be exactly 0 at
            # a cone endpoint (non-strict), and a corner spawn lies on two edges while
            # only the edge it was drawn from is guaranteed inward — so require ANY
            # satisfying edge rather than asserting every edge it touches.
            inward = (
                (on_south and obs.vy >= 0.0)
                or (on_north and obs.vy <= 0.0)
                or (on_west and obs.vx >= 0.0)
                or (on_east and obs.vx <= 0.0)
            )
            assert inward, (
                f"TC17: snapshot[{i}] at ({obs.x}, {obs.y}) vel ({obs.vx}, {obs.vy}) "
                f"is not inward for any edge it lies on "
                f"(S={on_south}, N={on_north}, W={on_west}, E={on_east})"
            )
            # Speed in [0.3, 1.5] m/s (factors of MAX_LINEAR_SPEED=1.0).
            speed = (obs.vx**2 + obs.vy**2) ** 0.5
            assert 0.3 - tol <= speed <= 1.5 + tol, (
                f"TC17: snapshot[{i}] speed must be in [0.3, 1.5], got {speed}"
            )
    finally:
        arena.close()


def tc18(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own seed
    """Refill maintains population at 20 across a full-traversal window, with at least one despawn."""
    from arena.dynamic import TARGET_POPULATION

    arena = Arena(yaml_path, seed=1, traffic=True)
    try:
        _, _, _ = arena.reset()
        initial_live_ids = set(arena.initial_dynamic_snapshot[i].id for i in range(TARGET_POPULATION))

        # Run enough ticks for the slowest obstacle (0.3 m/s) to traverse 50 m at dt=0.1:
        # 50 / 0.3 ≈ 167 ticks, plus 50 margin.
        max_ticks = int(50.0 / 0.3 / arena._dt) + 50
        zero = np.array([[0.0], [0.0]], dtype=float)
        for _ in range(max_ticks):
            _, _, _, info = arena.step(zero)
            assert info.dynamic_obstacle_count == TARGET_POPULATION, (
                f"TC18: dynamic_obstacle_count fell to {info.dynamic_obstacle_count} at step {info.step_idx}; "
                f"refill broken"
            )
            if info.crashed or info.timed_out or info.reached_goal:
                # Done early — should not happen with a stationary robot in arena_v1's
                # safe (2,2) start, but break cleanly if it does.
                break
        # initial_dynamic_snapshot is frozen at t=0, so read the final live set
        # straight from the spawner to detect despawn churn.
        assert arena._spawner is not None
        final_live_ids = set(arena._spawner.live_ids)
        churned = initial_live_ids.symmetric_difference(final_live_ids)
        assert len(churned) > 0, (
            f"TC18: expected at least one despawn over {max_ticks} ticks, but the live-id set "
            f"is unchanged ({len(initial_live_ids)} ids). Despawn path may be broken."
        )
    finally:
        arena.close()


def tc19(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own seed
    """Robot-vs-dynamic-obstacle collision fires info.crashed via _inject_for_test."""
    arena = Arena(yaml_path, seed=2, traffic=True)
    try:
        _, _, _ = arena.reset()
        assert arena._spawner is not None, "TC19: spawner must be live with traffic=True"
        # Inject an obstacle 1.0 m east of (2,2), moving west at 1.0 m/s.
        # Collision contact distance = robot_radius (0.2) + obstacle_radius (0.3) = 0.5 m.
        # Obstacle reaches contact distance after moving 0.5 m → 5 ticks at dt=0.1.
        arena._spawner._inject_for_test(x=3.0, y=2.0, vx=-1.0, vy=0.0)
        zero = np.array([[0.0], [0.0]], dtype=float)
        crashed = False
        for _ in range(20):
            _, _, _, info = arena.step(zero)
            if info.crashed:
                crashed = True
                break
        assert crashed, (
            "TC19: robot did not crash within 20 ticks of an obstacle traveling toward it at 1 m/s "
            "from 1 m east — irsim collision detection on dynamic obstacles may be broken"
        )
    finally:
        arena.close()


def tc20(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own seed
    """Traffic determinism: two same-seed arenas produce identical dynamic_obstacles_sha256 sequences."""
    seed_value = 3
    n_ticks = 200
    zero = np.array([[0.0], [0.0]], dtype=float)

    def collect_hashes() -> list[str]:
        arena = Arena(yaml_path, seed=seed_value, traffic=True)
        try:
            _, _, info0 = arena.reset()
            hashes: list[str] = []
            assert info0.dynamic_obstacles_sha256 is not None, (
                "TC20: reset() must produce a non-None dynamic_obstacles_sha256 when traffic=True"
            )
            hashes.append(info0.dynamic_obstacles_sha256)
            for _ in range(n_ticks):
                _, _, _, info = arena.step(zero)
                assert info.dynamic_obstacles_sha256 is not None, (
                    f"TC20: step {info.step_idx} sha256 is None with traffic=True"
                )
                hashes.append(info.dynamic_obstacles_sha256)
                if info.crashed or info.timed_out or info.reached_goal:
                    break
            return hashes
        finally:
            arena.close()

    hashes_a = collect_hashes()
    hashes_b = collect_hashes()
    assert hashes_a == hashes_b, (
        f"TC20: dynamic_obstacles_sha256 sequences differ between two same-seed runs. "
        f"len_a={len(hashes_a)}, len_b={len(hashes_b)}. First mismatch at tick "
        f"{next((i for i, (a, b) in enumerate(zip(hashes_a, hashes_b)) if a != b), 'n/a')}"
    )


def tc21(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own seed
    """Snapshot shape, type, and immutability."""
    import dataclasses as _dc
    from arena.dynamic import DynamicObstacleState, OBSTACLE_RADIUS, TARGET_POPULATION

    # traffic=False, pre-reset: ()
    arena_off = Arena(yaml_path, seed=5, traffic=False)
    try:
        assert arena_off.initial_dynamic_snapshot == (), (
            f"TC21: traffic=False, pre-reset snapshot must be (), got {arena_off.initial_dynamic_snapshot}"
        )
        # traffic=False, post-reset: still ()
        arena_off.reset()
        assert arena_off.initial_dynamic_snapshot == (), (
            f"TC21: traffic=False, post-reset snapshot must be (), got {arena_off.initial_dynamic_snapshot}"
        )
    finally:
        arena_off.close()

    # traffic=True
    arena_on = Arena(yaml_path, seed=5, traffic=True)
    try:
        # Pre-reset: ()
        assert arena_on.initial_dynamic_snapshot == (), (
            f"TC21: traffic=True, pre-reset snapshot must be (), got len={len(arena_on.initial_dynamic_snapshot)}"
        )
        # Post-reset: 20 frozen entries
        arena_on.reset()
        snap = arena_on.initial_dynamic_snapshot
        assert isinstance(snap, tuple), f"TC21: snapshot must be tuple, got {type(snap).__name__}"
        assert len(snap) == TARGET_POPULATION, f"TC21: snapshot len must be {TARGET_POPULATION}, got {len(snap)}"
        first = snap[0]
        assert _dc.is_dataclass(first), f"TC21: snapshot[0] must be a dataclass, got {type(first).__name__}"
        assert first.radius == OBSTACLE_RADIUS, (
            f"TC21: snapshot[0].radius must be {OBSTACLE_RADIUS}, got {first.radius}"
        )
        # Frozen: attempting to mutate must raise FrozenInstanceError
        try:
            first.x = 999.0  # type: ignore[misc]
        except _dc.FrozenInstanceError:
            pass
        else:
            raise AssertionError("TC21: DynamicObstacleState must be frozen; field assignment did not raise")
    finally:
        arena_on.close()


def tc22(yaml_path: str, seed: int) -> None:  # noqa: ARG001
    """World-stem partitioning: same seed, two different worlds, two distinct result files."""
    repo_root = Path(__file__).resolve().parent.parent
    v1_yaml = str(repo_root / "arena" / "arena_v1.yaml")
    v2_yaml = str(repo_root / "arena" / "arena_v2_hard.yaml")
    common = [
        sys.executable, "-m", "runners.run_episode",
        "--algorithm", "a_star_once",
        "--seed", "42",
        "--no-traffic",  # so A* succeeds on both worlds
    ]
    with tempfile.TemporaryDirectory() as td:
        for world in (v1_yaml, v2_yaml):
            r = subprocess.run(
                [*common, "--world", world, "--results-dir", td],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert r.returncode == 0, (
                f"TC22 runner failed on {world}: exit={r.returncode}; stderr={r.stderr[-400:]}"
            )

        json_v1 = Path(td) / "arena_v1" / "a_star_once" / "42.json"
        json_v2 = Path(td) / "arena_v2_hard" / "a_star_once" / "42.json"
        jsonl_v1 = Path(td) / "arena_v1" / "a_star_once" / "42.trace.jsonl"
        jsonl_v2 = Path(td) / "arena_v2_hard" / "a_star_once" / "42.trace.jsonl"

        for p in (json_v1, json_v2, jsonl_v1, jsonl_v2):
            assert p.exists(), f"TC22: expected output missing at {p}"

        data_v1 = json.loads(json_v1.read_text(encoding="utf-8"))
        data_v2 = json.loads(json_v2.read_text(encoding="utf-8"))
        # Different worlds at the same seed must produce different runs.
        # The simplest non-trivial check: at least one of (path_length, time_to_goal) differs.
        differs = (
            data_v1.get("path_length") != data_v2.get("path_length")
            or data_v1.get("time_to_goal") != data_v2.get("time_to_goal")
        )
        assert differs, (
            f"TC22: arena_v1 and arena_v2_hard at seed=42 produced identical metrics; "
            f"world-stem partitioning is silently clobbering. v1={data_v1}, v2={data_v2}"
        )


def tc23(yaml_path: str, seed: int) -> None:  # noqa: ARG001
    """Import-cycle guard: both import orders succeed in a clean subprocess."""
    repo_root = str(Path(__file__).resolve().parent.parent)
    for order in ("import planners; import arena.arena",
                  "import arena.arena; import planners"):
        r = subprocess.run(
            [sys.executable, "-c", order],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert r.returncode == 0, (
            f"TC23: import order failed (`{order}`): exit={r.returncode}; "
            f"stderr={r.stderr[-400:]}"
        )


def tc24(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own seed (7)
    """Traffic-ON runner end-to-end: 8-key trace + byte-identical across two seeded runs.

    The runner's shipped default is traffic=True, but TC14/TC15/TC16/TC22 all force
    --no-traffic, so without this case the default code path is untested: the 8th
    trace key wiring (step-0 reset sha + per-step post-step sha) and trace-level
    determinism through the runner under traffic.
    """
    repo_root = Path(__file__).resolve().parent.parent
    world_stem = Path(yaml_path).stem
    cmd = [
        sys.executable, "-m", "runners.run_episode",
        "--algorithm", "a_star_once",
        "--seed", "7",
        "--world", yaml_path,
        "--traffic",
    ]
    with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
        for td in (td_a, td_b):
            r = subprocess.run(
                [*cmd, "--results-dir", td],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert r.returncode == 0, (
                f"TC24 runner exit {r.returncode}; stderr={r.stderr[-400:]}"
            )

        jsonl_a = Path(td_a) / world_stem / "a_star_once" / "7.trace.jsonl"
        jsonl_b = Path(td_b) / world_stem / "a_star_once" / "7.trace.jsonl"
        assert jsonl_a.exists() and jsonl_b.exists(), (
            f"TC24 trace JSONLs missing: a={jsonl_a.exists()}, b={jsonl_b.exists()}"
        )

        lines_a = jsonl_a.read_text(encoding="utf-8").splitlines()
        assert lines_a, "TC24: traffic trace JSONL is empty"
        for ln, raw in enumerate(lines_a):
            rec = json.loads(raw)
            assert "dynamic_obstacles_sha256" in rec, (
                f"TC24: trace line {ln} missing dynamic_obstacles_sha256 with traffic on; "
                f"keys={sorted(rec)}"
            )
            assert len(rec) == 8, (
                f"TC24: trace line {ln} must have 8 keys with traffic on, got {len(rec)}: {sorted(rec)}"
            )

        assert filecmp.cmp(jsonl_a, jsonl_b, shallow=False), (
            "TC24: two same-seed traffic runs produced differing trace JSONL; "
            "traffic determinism through the runner is broken"
        )


def tc25(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure computation, no world used
    """Phase 3 seed derivation: determinism, uniqueness, prefix property, master-sensitivity."""
    from runners.run_experiment import derive_episode_seeds

    fifty = derive_episode_seeds(7, 50)
    assert len(fifty) == 50, f"TC25: expected 50 seeds, got {len(fifty)}"
    assert len(set(fifty)) == 50, "TC25: derived seeds are not unique"
    assert all(isinstance(s, int) and s >= 0 for s in fifty), (
        "TC25: seeds must be non-negative ints"
    )
    assert derive_episode_seeds(7, 50) == fifty, "TC25: derivation is not deterministic"
    assert derive_episode_seeds(7, 3) == fifty[:3], (
        "TC25: prefix property broken (spawn(3) != spawn(50)[:3])"
    )
    assert derive_episode_seeds(8, 3) != fifty[:3], (
        "TC25: a different master seed produced an identical prefix"
    )


def tc26(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses arena_no_path fixture
    """Phase 3 batch determinism + parallel-ordering.

    Runs the batch runner on the boxed-in-start world (A* fails fast, so each episode
    terminates in seconds with no driving loop). A and B at --jobs 1 must be byte-identical;
    C at --jobs 3 must keep the manifest in derivation order (completion order must not leak).
    """
    repo_root = Path(__file__).resolve().parent.parent
    world = str(repo_root / "arena" / "arena_no_path.yaml")
    # Master seed 1 yields a DESCENDING 3-seed prefix, so derivation order differs from
    # sort-by-seed order — this is what gives the ordering assertion below real teeth.
    base = [
        sys.executable, "-m", "runners.run_experiment",
        "--algorithm", "a_star_once",
        "--world", world,
        "--master-seed", "1",
        "--num-seeds", "3",
        "--no-traffic",
    ]

    def _run(td: str, extra: list[str]) -> Path:
        r = subprocess.run(
            [*base, "--results-dir", td, *extra],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, (
            f"TC26 batch failed (extra={extra}): exit={r.returncode}; stderr={r.stderr[-400:]}"
        )
        return Path(td) / "arena_no_path" / "a_star_once"

    def _manifest_no_git(out_dir: Path) -> dict:
        m = json.loads((out_dir / "_manifest.json").read_text(encoding="utf-8"))
        m.pop("git_sha", None)  # robust to dirty tree / absent git
        return m

    # ignore_cleanup_errors: child subprocesses wrote into these dirs; on Windows a lingering
    # handle or an AV/indexer lock can make rmtree raise PermissionError at block exit, which
    # would fail --check for a reason unrelated to the assertions.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_a, \
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_b, \
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_c:
        dir_a = _run(td_a, ["--jobs", "1"])
        dir_b = _run(td_b, ["--jobs", "1"])
        dir_c = _run(td_c, ["--jobs", "3"])

        seeds = sorted(int(p.stem) for p in dir_a.glob("[0-9]*.json"))
        assert len(seeds) == 3, f"TC26: expected 3 episode JSONs, got {len(seeds)}"

        for s in seeds:
            assert filecmp.cmp(dir_a / f"{s}.json", dir_b / f"{s}.json", shallow=False), (
                f"TC26: per-seed metrics JSON differ across two same-master-seed runs (seed={s})"
            )
            assert not (dir_a / f"{s}.trace.jsonl").exists(), (
                f"TC26: planner-failure world wrote a trace for seed={s}"
            )

        man_a = _manifest_no_git(dir_a)
        assert man_a == _manifest_no_git(dir_b), (
            "TC26: manifests differ across two same-master-seed --jobs 1 runs"
        )

        # Assert the manifest order against the KNOWN derivation order, not against itself.
        # With a descending prefix this catches a sorted-by-seed build AND a completion-order
        # leak in the --jobs 3 path; the old order_a == order_c check compared two outputs of the
        # same code against ascending default-master seeds and so could distinguish neither.
        from runners.run_experiment import derive_episode_seeds

        derived = list(derive_episode_seeds(1, 3))
        assert derived != sorted(derived), "TC26: chosen master must give a non-monotonic prefix"
        man_c = _manifest_no_git(dir_c)
        order_a = [e["seed"] for e in man_a["episodes"]]
        order_c = [e["seed"] for e in man_c["episodes"]]
        assert order_a == derived, (
            f"TC26: --jobs 1 manifest not in derivation order: {order_a} != {derived}"
        )
        assert order_c == derived, (
            f"TC26: --jobs 3 reordered the manifest episodes (completion order leaked in): "
            f"{order_c} != {derived}"
        )
        assert man_a["derived_seeds"] == derived, (
            "TC26: manifest derived_seeds not in derivation order"
        )
        assert man_a["derived_seeds"] == man_c["derived_seeds"], (
            "TC26: derived_seeds differ between --jobs 1 and --jobs 3"
        )


def tc27(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — writes its own malformed world
    """Phase 3 failure accounting: a malformed (but existing) world makes every child exit
    non-zero; the batch continues, reports the failures, and itself exits non-zero."""
    repo_root = Path(__file__).resolve().parent.parent
    # ignore_cleanup_errors: a child subprocess wrote into this dir; on Windows a lingering
    # handle / AV / indexer lock can make rmtree raise PermissionError at block exit.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        bad_yaml = Path(td) / "bad.yaml"
        bad_yaml.write_text("not: [valid: arena", encoding="utf-8")  # irsim/yaml rejects this
        r = subprocess.run(
            [
                sys.executable, "-m", "runners.run_experiment",
                "--algorithm", "a_star_once",
                "--world", str(bad_yaml),
                "--num-seeds", "2",
                "--no-traffic",
                "--results-dir", td,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r.returncode != 0, (
            f"TC27: batch should exit non-zero when all seeds fail; got {r.returncode}"
        )
        # The manifest checks below are the authoritative assertions: they prove both children
        # were recorded as runner failures with non-zero exit codes. They also guard the fixture
        # itself — if this "malformed" world ever parsed and failed only at the planner stage, the
        # child would exit 0, flipping those checks to status "ok"/exit_code 0 (a loud failure)
        # instead of a silent pass. We keep one light console check (the failure-detail section
        # prints) but deliberately do NOT assert the exact "<n> runner-failed" wording, which would
        # couple the test to console phrasing for no added coverage.
        assert "runner failures:" in r.stdout, (
            "TC27: summary omitted the per-seed failure detail section"
        )

        manifest = json.loads(
            (Path(td) / "bad" / "a_star_once" / "_manifest.json").read_text(encoding="utf-8")
        )
        statuses = [e["status"] for e in manifest["episodes"]]
        assert statuses == ["runner_error", "runner_error"], (
            f"TC27: manifest episodes should both be runner_error, got {statuses}"
        )
        assert all(e["exit_code"] != 0 for e in manifest["episodes"]), (
            "TC27: failed episodes must record a non-zero exit_code"
        )


# ---------------------------------------------------------------------------
# TC28..TC34 — Group A: the lidar-folding replanning family (a_star_replan /
# dijkstra_once / dijkstra_replan) and the planner registry. Pure-unit cases
# (TC28/TC31/TC32/TC33) build controllers/grids in-process; subprocess cases
# (TC29/TC30/TC34) shell out to `python -m runners.run_episode` exactly like
# TC14/TC15/TC22. Repo root must be importable for the in-process imports of
# `planners` / `planners._grid` / `manual_astar` (mirrors tc10's sys.path bump).
# ---------------------------------------------------------------------------


def _ensure_repo_root_on_path() -> Path:
    """Put the repo root on sys.path (idempotent) and return it.

    The in-process Group-A cases import `planners`, `planners._grid`, and
    `manual_astar`, all of which live at the repo root. `python arena/arena.py`
    only puts `arena/` on sys.path, so bump the root the same way tc10 does.
    """
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def tc28(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure unit; synthesizes its own pose/lidar
    """Lidar→grid fold geometry: one finite beam marks its world-hit cell, memorylessly."""
    _ensure_repo_root_on_path()
    from planners._grid import lidar_to_occupancy, load_lidar_geometry  # type: ignore[import-not-found]
    from manual_astar import (  # type: ignore[import-not-found]
        GRID_RESOLUTION,
        SAFETY_MARGIN,
        build_occupancy_grid,
        load_world,
        world_to_grid,
    )

    world = load_world(yaml_path)
    grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN)
    static_cells = grid.cells
    geom = load_lidar_geometry(yaml_path)
    inflation = world.robot_radius + SAFETY_MARGIN

    bearings = np.linspace(geom.angle_min, geom.angle_max, geom.number)

    # Open-space pose near the (2,2) start; beam 180 (bearing ~0) points ~+x into
    # the clear region, so the hit lands on an unblocked static cell.
    state = np.array([5.0, 5.0, 0.0], dtype=np.float64)
    beam_index = 180
    beam_range = 2.0

    lidar = np.full((geom.number,), np.nan, dtype=np.float64)
    lidar[beam_index] = beam_range

    world_angle = float(state[2]) + float(bearings[beam_index])
    hit = state[:2] + beam_range * np.array(
        [np.cos(world_angle), np.sin(world_angle)], dtype=np.float64
    )
    hit_cell = world_to_grid(hit, grid)
    assert not bool(static_cells[hit_cell]), (
        f"TC28 setup: chosen hit cell {hit_cell} is already blocked statically; "
        f"pick a clearer beam/pose"
    )

    static_sum_before = int(static_cells.sum())
    folded = lidar_to_occupancy(static_cells, grid, state, lidar, geom, inflation)

    # 1) The hit's cell is now blocked in the fold.
    assert bool(folded[hit_cell]), (
        f"TC28: folded hit cell {hit_cell} must be blocked after folding a finite "
        f"return at beam {beam_index}"
    )
    # 2) A far-away open cell stays free (the fold is local to the hit disk).
    far_cell = world_to_grid(np.array([45.0, 5.0], dtype=np.float64), grid)
    assert not bool(static_cells[far_cell]), (
        f"TC28 setup: far cell {far_cell} must be statically open"
    )
    assert not bool(folded[far_cell]), (
        f"TC28: a far-away open cell {far_cell} must stay free after folding one beam"
    )
    # 3) The fold returns a NEW array and never mutates the static cells.
    assert folded is not static_cells, "TC28: fold must return a new array, not the static one"
    assert int(static_cells.sum()) == static_sum_before, (
        f"TC28: static_cells was mutated by the fold "
        f"(sum {static_sum_before} -> {int(static_cells.sum())})"
    )
    # 4) Folding an all-NaN scan equals the static grid exactly (no returns => no marks).
    all_nan = np.full((geom.number,), np.nan, dtype=np.float64)
    empty_fold = lidar_to_occupancy(static_cells, grid, state, all_nan, geom, inflation)
    assert np.array_equal(empty_fold, static_cells), (
        "TC28: folding an all-NaN lidar must reproduce the static grid"
    )
    # 5) Pose-dependence: the SAME single-beam lidar folded at a DIFFERENT pose
    #    marks a different cell (the fold reads the live robot pose).
    state2 = np.array([10.0, 10.0, 0.0], dtype=np.float64)
    world_angle2 = float(state2[2]) + float(bearings[beam_index])
    hit2 = state2[:2] + beam_range * np.array(
        [np.cos(world_angle2), np.sin(world_angle2)], dtype=np.float64
    )
    hit_cell2 = world_to_grid(hit2, grid)
    assert hit_cell2 != hit_cell, (
        f"TC28 setup: the two poses must map to distinct hit cells "
        f"({hit_cell} vs {hit_cell2})"
    )
    folded2 = lidar_to_occupancy(static_cells, grid, state2, lidar, geom, inflation)
    assert bool(folded2[hit_cell2]), (
        f"TC28: pose-2 fold must block its own hit cell {hit_cell2}"
    )
    assert not bool(folded2[hit_cell]) or bool(static_cells[hit_cell]), (
        f"TC28: pose-2 fold must NOT block pose-1's hit cell {hit_cell} "
        f"(the fold is pose-dependent)"
    )


def tc29(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """Dijkstra == A* optimal (equal octile cost) + dijkstra_once reaches the goal."""
    repo_root = _ensure_repo_root_on_path()
    from manual_astar import (  # type: ignore[import-not-found]
        GRID_RESOLUTION,
        SAFETY_MARGIN,
        astar_search,
        build_occupancy_grid,
        load_world,
        validate_start_and_goal,
    )

    world = load_world(yaml_path)
    grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN)
    start_cell, goal_cell = validate_start_and_goal(world, grid)

    astar_path = astar_search(grid, start_cell, goal_cell)
    dijkstra_path = astar_search(grid, start_cell, goal_cell, lambda *_: 0.0)

    def _octile_cost(path: list[tuple[int, int]]) -> float:
        total = 0.0
        for (r0, c0), (r1, c1) in zip(path, path[1:]):
            total += float(np.hypot(r1 - r0, c1 - c0))
        return total

    cost_astar = _octile_cost(astar_path)
    cost_dijkstra = _octile_cost(dijkstra_path)
    assert abs(cost_astar - cost_dijkstra) < 1e-9, (
        f"TC29: Dijkstra path cost {cost_dijkstra} != A* path cost {cost_astar}; "
        f"Dijkstra must recover the same optimal cost"
    )

    # Subprocess part: dijkstra_once must actually reach the goal through the runner.
    seed_value = "29"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        r = subprocess.run(
            [
                sys.executable, "-m", "runners.run_episode",
                "--algorithm", "dijkstra_once",
                "--seed", seed_value,
                "--world", yaml_path,
                "--no-traffic",
                "--results-dir", td,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, (
            f"TC29 dijkstra_once runner exit {r.returncode}; stderr={r.stderr[-400:]}"
        )
        world_stem = Path(yaml_path).stem
        json_path = Path(td) / world_stem / "dijkstra_once" / f"{seed_value}.json"
        assert json_path.exists(), f"TC29: metrics JSON missing at {json_path}"
        metrics = json.loads(json_path.read_text(encoding="utf-8"))
        assert metrics["planner_error"] is None, f"TC29 planner_error not None: {metrics}"
        assert metrics["time_to_goal"] is not None, (
            f"TC29 dijkstra_once did not reach the goal (time_to_goal is None): {metrics}"
        )


def tc30(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """a_star_replan end-to-end through the runner: labeled dir, 8-key trace, runs to completion."""
    repo_root = _ensure_repo_root_on_path()
    seed_value = "30"
    replan_k = "5"
    world_stem = Path(yaml_path).stem
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        r = subprocess.run(
            [
                sys.executable, "-m", "runners.run_episode",
                "--algorithm", "a_star_replan",
                "--replan-k", replan_k,
                "--seed", seed_value,
                "--world", yaml_path,
                "--traffic",  # default; stated explicitly
                "--results-dir", td,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, (
            f"TC30 a_star_replan runner exit {r.returncode}; stderr={r.stderr[-400:]}"
        )

        out_dir = Path(td) / world_stem / "a_star_replan_k5"
        json_path = out_dir / f"{seed_value}.json"
        jsonl_path = out_dir / f"{seed_value}.trace.jsonl"
        assert json_path.exists(), (
            f"TC30: metrics JSON missing at {json_path} — label must be 'a_star_replan_k5'"
        )
        assert jsonl_path.exists(), f"TC30: trace JSONL missing at {jsonl_path}"

        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert lines, "TC30: a_star_replan trace JSONL is empty"
        for idx, raw in enumerate(lines):
            rec = json.loads(raw)
            assert isinstance(rec, dict), f"TC30: trace line {idx} is not an object"
            assert "dynamic_obstacles_sha256" in rec, (
                f"TC30: trace line {idx} missing dynamic_obstacles_sha256 with traffic on; "
                f"keys={sorted(rec)}"
            )
            assert len(rec) == 8, (
                f"TC30: trace line {idx} must have 8 keys with traffic on, got {len(rec)}: "
                f"{sorted(rec)}"
            )
        # The episode may crash or time out — that is fine. We assert only that it RAN
        # to completion (metrics written, no runner fault), not that it reached the goal.
        metrics = json.loads(json_path.read_text(encoding="utf-8"))
        assert metrics["planner_error"] is None, (
            f"TC30: a_star_replan must plan successfully at t=0; planner_error={metrics['planner_error']}"
        )


def tc31(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure unit; synthesizes its own pose/lidar
    """Replan cadence (every K-th act) + memoryless fold (no occupancy accumulation)."""
    _ensure_repo_root_on_path()
    import planners._grid as grid_module  # type: ignore[import-not-found]
    from planners import build_controller  # type: ignore[import-not-found]
    from planners._grid import (  # type: ignore[import-not-found]
        lidar_to_occupancy as real_lidar_to_occupancy,
        load_lidar_geometry,
    )
    from manual_astar import (  # type: ignore[import-not-found]
        GRID_RESOLUTION,
        SAFETY_MARGIN,
        build_occupancy_grid,
        load_world,
    )

    replan_k = 3
    controller = build_controller("a_star_replan", replan_k)

    # Synthesize a valid post-reset state0/lidar0 at the (2,2) start with an all-NaN
    # scan — equivalent to a throwaway Arena's reset() for this lidar-only family.
    state0 = np.array([2.0, 2.0, 0.0], dtype=np.float64)
    nan_lidar = np.full((360,), np.nan, dtype=np.float64)
    controller.reset(yaml_path, (), nan_lidar, state0)

    # Count compute_path invocations via the instance method (the cadence gate).
    call_indices: list[int] = []
    original_compute_path = controller.compute_path

    def counting_compute_path(state: np.ndarray, lidar: np.ndarray) -> Any:
        call_indices.append(len(recorded_folds))  # marker; index filled by the fold spy
        return original_compute_path(state, lidar)

    # Record the occupancy each replan actually folds. compute_path reads the
    # MODULE-level lidar_to_occupancy, so patch it there to capture the result.
    recorded_folds: list[np.ndarray] = []

    def spying_fold(static_cells, grid, state, lidar, geom, inflation):  # type: ignore[no-untyped-def]
        folded = real_lidar_to_occupancy(static_cells, grid, state, lidar, geom, inflation)
        recorded_folds.append(folded.copy())
        return folded

    # Two distinct lidar frames across the cadence window: frame_a carries an extra
    # return (an obstacle present ONLY in the first replan), frame_b is empty.
    world = load_world(yaml_path)
    static_grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN)
    geom = load_lidar_geometry(yaml_path)
    inflation = world.robot_radius + SAFETY_MARGIN

    frame_a = np.full((360,), np.nan, dtype=np.float64)
    frame_a[180] = 2.0  # beam ~+x: a finite return that adds cells beyond static
    frame_b = np.full((360,), np.nan, dtype=np.float64)  # empty: no extra returns

    controller.compute_path = counting_compute_path  # type: ignore[assignment]
    grid_module.lidar_to_occupancy = spying_fold
    try:
        # 9 acts at K=3: replans fire on acts 3, 6, 9 only.
        acts_per_frame = [frame_a, frame_a, frame_a, frame_b, frame_b, frame_b,
                          frame_a, frame_a, frame_a]
        fired_on: list[int] = []
        for act_number, frame in enumerate(acts_per_frame, start=1):
            before = len(recorded_folds)
            controller.act(state0, frame)
            if len(recorded_folds) > before:
                fired_on.append(act_number)
    finally:
        grid_module.lidar_to_occupancy = real_lidar_to_occupancy
        controller.compute_path = original_compute_path  # type: ignore[assignment]

    assert fired_on == [3, 6, 9], (
        f"TC31: compute_path must fire on acts 3, 6, 9 only (every K-th act), fired on {fired_on}"
    )
    assert len(recorded_folds) == 3, (
        f"TC31: expected 3 recorded folds (one per replan), got {len(recorded_folds)}"
    )

    # Memoryless: each recorded fold equals static ∪ that-call's frame, with NO
    # carry-over. The replan at act 3 folded frame_a (extra obstacle); the replan
    # at act 6 folded frame_b (empty) and must equal the static grid exactly — the
    # frame_a obstacle must NOT persist into it.
    expected_a = real_lidar_to_occupancy(
        static_grid.cells, static_grid, state0, frame_a, geom, inflation
    )
    expected_b = real_lidar_to_occupancy(
        static_grid.cells, static_grid, state0, frame_b, geom, inflation
    )
    assert np.array_equal(recorded_folds[0], expected_a), (
        "TC31: act-3 replan fold != static ∪ frame_a"
    )
    assert np.array_equal(recorded_folds[1], expected_b), (
        "TC31: act-6 replan fold != static ∪ frame_b (frame_a obstacle leaked across replans)"
    )
    assert np.array_equal(recorded_folds[2], expected_a), (
        "TC31: act-9 replan fold != static ∪ frame_a"
    )
    # The frame_a obstacle genuinely adds cells, so the memoryless check has teeth.
    assert int(recorded_folds[0].sum()) > int(static_grid.cells.sum()), (
        "TC31 setup: frame_a must add occupied cells beyond static"
    )
    assert np.array_equal(recorded_folds[1], static_grid.cells), (
        "TC31: act-6 replan fold must equal the bare static grid (empty frame, no accumulation)"
    )


def tc32(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure unit; synthesizes its own pose/lidar
    """Commitment-horizon swap semantics for a_star_replan (k=1): the three branches.

    A replan firing on a K-th act does NOT unconditionally rebuild the follower —
    that was the pre-fix behavior the commitment horizon corrects. The follower is
    swapped ONLY when the held commitment is exhausted or its immediate segment is
    no longer clear; otherwise the fresh plan is stored but the committed follower
    is KEPT. This case exercises all three branches at replan_k=1:
      (a) replan FAILS               -> KEEP the follower (error swallowed).
      (b) replan SUCCEEDS, segment CLEAR   -> KEEP the follower (committed).
      (c) replan SUCCEEDS, segment BLOCKED -> SWAP the follower (recommit).
    """
    _ensure_repo_root_on_path()
    from planners import build_controller  # type: ignore[import-not-found]
    from planners._grid import load_lidar_geometry  # type: ignore[import-not-found]

    state0 = np.array([2.0, 2.0, 0.0], dtype=np.float64)
    nan_lidar = np.full((360,), np.nan, dtype=np.float64)

    # --- (a) failure -> KEEP -------------------------------------------------
    controller = build_controller("a_star_replan", 1)  # replan on every act
    controller.reset(yaml_path, (), nan_lidar, state0)

    good_follower = controller._follower
    assert good_follower is not None, "TC32(a) setup: reset() must build a follower"

    def raising_compute_path(state: np.ndarray, lidar: np.ndarray) -> Any:
        raise RuntimeError("TC32 injected replan failure")

    controller.compute_path = raising_compute_path  # type: ignore[assignment]
    try:
        action = controller.act(state0, nan_lidar)
    except Exception as exc:  # noqa: BLE001 — the whole point is that nothing escapes
        raise AssertionError(
            f"TC32(a): a failed replan must not propagate out of act(); got "
            f"{type(exc).__name__}: {exc}"
        )

    assert isinstance(action, np.ndarray), (
        f"TC32(a): act() must return an ndarray after a failed replan, got {type(action).__name__}"
    )
    assert action.shape == (2, 1), f"TC32(a): action shape must be (2, 1), got {action.shape}"
    assert np.issubdtype(action.dtype, np.floating), (
        f"TC32(a): action dtype must be float, got {action.dtype}"
    )
    assert np.all(np.isfinite(action)), "TC32(a): action must be finite after a failed replan"
    assert controller._follower is good_follower, (
        "TC32(a): a failed replan must KEEP the existing follower object, not rebuild it"
    )

    # --- (b) success + clear segment -> KEEP ---------------------------------
    # Restore the real compute_path. An all-NaN frame folds to the bare static
    # grid, so the immediate segment (2,2)->current target waypoint is clear and
    # the follower is not finished: the successful replan must KEEP the follower.
    del controller.compute_path  # restore the bound base-class method
    controller.act(state0, nan_lidar)
    assert controller._follower is good_follower, (
        "TC32(b): a successful replan whose immediate segment stays clear must KEEP "
        "the committed follower (the commitment horizon must not rebuild it)"
    )

    # --- (c) success + blocked segment -> SWAP -------------------------------
    # Fresh controller at the start; place a single finite lidar return ON the
    # bearing of the current target waypoint so the resulting fold marks the
    # immediate segment blocked. A* still routes around the single disk, so the
    # replan succeeds and the follower must be SWAPPED.
    controller_c = build_controller("a_star_replan", 1)
    controller_c.reset(yaml_path, (), nan_lidar, state0)
    follower_c = controller_c._follower
    assert follower_c is not None, "TC32(c) setup: reset() must build a follower"

    position = np.array([2.0, 2.0], dtype=np.float64)
    target = follower_c.current_waypoint(position)
    delta = np.asarray(target, dtype=np.float64) - position
    delta_norm = float(np.linalg.norm(delta))
    assert delta_norm > 1e-6, (
        "TC32(c) setup: the current target waypoint must not coincide with the start"
    )
    desired_bearing = float(np.arctan2(delta[1], delta[0]))  # robot theta = 0

    geom = load_lidar_geometry(yaml_path)
    bearings = np.linspace(geom.angle_min, geom.angle_max, geom.number)
    beam = int(np.argmin(np.abs(bearings - desired_bearing)))

    blocking_lidar = np.full((geom.number,), np.nan, dtype=np.float64)
    # Hit at half the segment length lands ON the segment, inside the inflation
    # band, so segment_is_clear_grid reports the immediate segment blocked.
    blocking_lidar[beam] = 0.5 * delta_norm

    controller_c.act(state0, blocking_lidar)
    assert controller_c._follower is not follower_c, (
        "TC32(c): a successful replan whose immediate segment is BLOCKED must SWAP "
        "the follower (recommit to the fresh plan)"
    )


def tc33(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure unit; uses the registry only
    """--replan-k validation + name==key invariant + algorithm_label + ALGORITHMS membership."""
    _ensure_repo_root_on_path()
    from planners import ALGORITHMS, algorithm_label, build_controller  # type: ignore[import-not-found]

    # Invalid (algorithm, cadence) pairs must all raise ValueError.
    invalid_pairs: list[tuple[str, Any]] = [
        ("a_star_replan", None),     # replan family without a cadence
        ("a_star_once", 5),          # once family with a forbidden cadence
        ("dijkstra_replan", None),   # replan family without a cadence
        ("d_star_lite", 5),          # non-replan family with a forbidden cadence
    ]
    for name, k in invalid_pairs:
        try:
            build_controller(name, k)
        except ValueError:
            continue
        raise AssertionError(
            f"TC33: build_controller({name!r}, {k!r}) must raise ValueError but did not"
        )

    # Valid combos construct, and the constructed controller's .name == its key (AC15).
    valid_pairs: list[tuple[str, Any]] = [
        ("a_star_once", None),
        ("a_star_replan", 5),
        ("dijkstra_once", None),
        ("dijkstra_replan", 5),
        ("d_star_lite", None),
    ]
    for name, k in valid_pairs:
        controller = build_controller(name, k)
        assert controller.name == name, (
            f"TC33: build_controller({name!r}, {k!r}).name == {controller.name!r}, expected {name!r}"
        )
        assert name in ALGORITHMS, f"TC33: {name!r} must be a key in ALGORITHMS"

    # Labels: replan families fold the cadence in, the rest use the bare key (AC6).
    assert algorithm_label("a_star_replan", 5) == "a_star_replan_k5", (
        f"TC33: algorithm_label('a_star_replan', 5) == "
        f"{algorithm_label('a_star_replan', 5)!r}, expected 'a_star_replan_k5'"
    )
    assert algorithm_label("a_star_once", None) == "a_star_once", (
        f"TC33: algorithm_label('a_star_once', None) == "
        f"{algorithm_label('a_star_once', None)!r}, expected 'a_star_once'"
    )


def tc34(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """a_star_once parity through the redesigned loop: reaches goal + byte-identical traces."""
    repo_root = _ensure_repo_root_on_path()
    seed_value = "34"
    world_stem = Path(yaml_path).stem
    runner_args = [
        sys.executable, "-m", "runners.run_episode",
        "--algorithm", "a_star_once",
        "--seed", seed_value,
        "--world", yaml_path,
        "--no-traffic",
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_a, \
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_b:
        for td in (td_a, td_b):
            r = subprocess.run(
                [*runner_args, "--results-dir", td],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert r.returncode == 0, (
                f"TC34 a_star_once runner exit {r.returncode}; stderr={r.stderr[-400:]}"
            )

        json_a = Path(td_a) / world_stem / "a_star_once" / f"{seed_value}.json"
        json_b = Path(td_b) / world_stem / "a_star_once" / f"{seed_value}.json"
        jsonl_a = Path(td_a) / world_stem / "a_star_once" / f"{seed_value}.trace.jsonl"
        jsonl_b = Path(td_b) / world_stem / "a_star_once" / f"{seed_value}.trace.jsonl"
        for p in (json_a, json_b, jsonl_a, jsonl_b):
            assert p.exists(), f"TC34: expected output missing at {p}"

        for json_path in (json_a, json_b):
            metrics = json.loads(json_path.read_text(encoding="utf-8"))
            assert metrics["planner_error"] is None, (
                f"TC34 planner_error not None at {json_path}: {metrics}"
            )
            assert metrics["time_to_goal"] is not None, (
                f"TC34 a_star_once did not reach the goal at {json_path}: {metrics}"
            )

        assert filecmp.cmp(str(jsonl_a), str(jsonl_b), shallow=False), (
            "TC34: two same-seed a_star_once --no-traffic runs produced differing trace JSONL; "
            "the runner redesign regressed the shipped a_star_once determinism path"
        )


# ---------------------------------------------------------------------------
# TC35..TC37 — Group B: the incremental D* Lite family (d_star_lite). TC35/TC36
# are in-process unit cases over the search core (TC35 also shells out for the
# static-map drive); TC36 is the BINDING incremental==from-scratch proof; TC37
# mixes a pure-registry check with two subprocess drives (forbidden --replan-k +
# the slow traffic-ON end-to-end). All in-process imports need the repo root on
# sys.path, so reuse the tc28-tc34 helper.
# ---------------------------------------------------------------------------


def _octile_path_cost(path: list[tuple[int, int]]) -> float:
    """Octile cost of a cell path: Σ hypot(Δrow, Δcol) over consecutive cells.

    Identical metric to TC29's `_octile_cost` — both `astar_search` and
    `DStarLiteSearch` charge `np.hypot(dr, dc)` per step (1.0 orthogonal,
    sqrt(2) diagonal), so this is the common cost model all Group-B cost
    comparisons reduce to.
    """
    total = 0.0
    for (row0, col0), (row1, col1) in zip(path, path[1:]):
        total += float(np.hypot(row1 - row0, col1 - col0))
    return total


def tc35(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """D* Lite optimal static path (== A* cost, collision-free) + reaches goal via runner."""
    repo_root = _ensure_repo_root_on_path()
    from manual_astar import (  # type: ignore[import-not-found]
        GRID_RESOLUTION,
        SAFETY_MARGIN,
        astar_search,
        build_occupancy_grid,
        load_world,
        validate_start_and_goal,
    )
    from planners.d_star_lite import DStarLiteSearch  # type: ignore[import-not-found]

    # --- Unit part: D* Lite over the arena_v1 STATIC grid (the controller's t=0
    # substrate when traffic is off) must produce a path of the SAME optimal
    # octile cost A* does, since both share the cost model. ---
    world = load_world(yaml_path)
    grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN)
    start_cell, goal_cell = validate_start_and_goal(world, grid)

    astar_path = astar_search(grid, start_cell, goal_cell)
    astar_cost = _octile_path_cost(astar_path)

    search = DStarLiteSearch(grid.cells, start_cell, goal_cell)
    search.compute_shortest_path()
    dstar_path = search.extract_path()
    dstar_cost = _octile_path_cost(dstar_path)

    assert abs(astar_cost - dstar_cost) < 1e-9, (
        f"TC35: D* Lite static cost {dstar_cost} != A* cost {astar_cost}; "
        f"D* Lite must recover the same optimal cost"
    )
    assert dstar_path[0] == start_cell and dstar_path[-1] == goal_cell, (
        f"TC35: D* Lite path must run {start_cell} -> {goal_cell}, "
        f"got {dstar_path[0]} -> {dstar_path[-1]}"
    )
    # Clearance: every cell on the extracted grid path is unoccupied (the path is
    # collision-free on the static grid).
    for cell in dstar_path:
        assert not bool(grid.cells[cell]), (
            f"TC35: D* Lite path traverses an occupied cell {cell}"
        )

    # --- Subprocess part: d_star_lite must reach the goal on the static map. ---
    # D* Lite runs its full incremental search every tick, so even the no-traffic
    # drive is far more CPU-heavy per step than the A* _once runners — under the
    # contention of a full --check pass an 812-step traversal can blow a 300 s
    # budget. Give it the same 600 s timeout TC37's traffic drive uses.
    seed_value = "35"
    world_stem = Path(yaml_path).stem
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        r = subprocess.run(
            [
                sys.executable, "-m", "runners.run_episode",
                "--algorithm", "d_star_lite",
                "--seed", seed_value,
                "--world", yaml_path,
                "--no-traffic",
                "--results-dir", td,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert r.returncode == 0, (
            f"TC35 d_star_lite runner exit {r.returncode}; stderr={r.stderr[-400:]}"
        )
        json_path = Path(td) / world_stem / "d_star_lite" / f"{seed_value}.json"
        assert json_path.exists(), f"TC35: metrics JSON missing at {json_path}"
        metrics = json.loads(json_path.read_text(encoding="utf-8"))
        assert metrics["planner_error"] is None, (
            f"TC35 planner_error not None: {metrics}"
        )
        assert metrics["time_to_goal"] is not None, (
            f"TC35 d_star_lite did not reach the goal on the static map "
            f"(time_to_goal is None): {metrics}"
        )


def tc36(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure unit; hand-built grid
    """D* Lite incremental update == from-scratch A* (BINDING): blocking the path lengthens it.

    Hand-built 9x9 grid: a vertical wall at col 4 spanning rows 0..6 leaves a single
    passage at cell C=(7, 4). The unique optimal (0,0)->(8,8) path threads C. Blocking
    C forces a strictly costlier detour around the wall's bottom, so the incremental
    update MUST bind: a no-op/ignored update would leave the cheaper pre-block cost in
    place and fail the strict-increase assertion. We compare octile COST only (not the
    exact cell set), per AC10's equal-cost tie-break allowance.
    """
    _ensure_repo_root_on_path()
    from manual_astar import OccupancyGrid, astar_search  # type: ignore[import-not-found]
    from planners.d_star_lite import DStarLiteSearch  # type: ignore[import-not-found]

    rows, cols = 9, 9
    start_cell = (0, 0)
    goal_cell = (8, 8)
    block_cell = (7, 4)

    def build_grid() -> np.ndarray:
        cells = np.zeros((rows, cols), dtype=np.bool_)
        for row in range(0, 7):
            cells[row, 4] = True  # vertical wall, gap at row 7 (=> passage at C)
        return cells

    # (a) Pre-block: compute the optimal path and assert it traverses C.
    cells = build_grid()
    assert not bool(cells[block_cell]), "TC36 setup: C must start free"
    search = DStarLiteSearch(cells, start_cell, goal_cell)
    search.compute_shortest_path()
    pre_path = search.extract_path()
    pre_cost = _octile_path_cost(pre_path)
    assert block_cell in pre_path, (
        f"TC36 precondition: optimal pre-block path must traverse C={block_cell}; "
        f"got {pre_path}"
    )

    # (b)/(c) Block C in the SAME array the search references (it holds a reference,
    # not a copy), report the flip, and re-solve incrementally.
    cells[block_cell] = True
    search.update_cells([block_cell])
    search.compute_shortest_path()
    post_path = search.extract_path()
    post_cost = _octile_path_cost(post_path)

    # Oracle: a FRESH A* on the updated grid, built from the same astar_search so the
    # cost model matches exactly.
    oracle_grid = OccupancyGrid(
        cells=cells.copy(),
        resolution=1.0,
        offset=np.array([0.0, 0.0], dtype=float),
    )
    oracle_path = astar_search(oracle_grid, start_cell, goal_cell)
    oracle_cost = _octile_path_cost(oracle_path)

    assert abs(post_cost - oracle_cost) < 1e-9, (
        f"TC36: incremental post-update cost {post_cost} != fresh-A* oracle cost "
        f"{oracle_cost}; the incremental repair diverged from from-scratch"
    )
    # The block must BIND: an ignored/no-op update would leave the cheaper pre-block
    # cost in place, so the strict increase is the load-bearing assertion.
    assert post_cost > pre_cost + 1e-9, (
        f"TC36: blocking C did not lengthen the optimum (post {post_cost} <= pre "
        f"{pre_cost}); the update was a no-op — incremental edge repair is broken"
    )
    assert post_path[0] == start_cell and post_path[-1] == goal_cell, (
        f"TC36: post-update path must still run {start_cell} -> {goal_cell}, "
        f"got {post_path[0]} -> {post_path[-1]}"
    )
    assert block_cell not in post_path, (
        f"TC36: post-update path must route AROUND the now-blocked C={block_cell}; "
        f"got {post_path}"
    )


def tc37(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """d_star_lite registered + rejects --replan-k + traffic-ON end-to-end (8-key trace).

    NOTE: the traffic drive is the SLOWEST single TC (~1-3 min): d_star_lite dodges and
    reaches the goal under traffic, replanning every step over ~800 steps. The generous
    timeout below is intentional; mirror TC30's subprocess pattern.
    """
    repo_root = _ensure_repo_root_on_path()
    from planners import ALGORITHMS, build_controller  # type: ignore[import-not-found]

    # --- Registration: the controller module registered itself at import. ---
    assert "d_star_lite" in ALGORITHMS, "TC37: 'd_star_lite' must be a key in ALGORITHMS"
    controller = build_controller("d_star_lite", None)
    assert controller.name == "d_star_lite", (
        f"TC37: build_controller('d_star_lite', None).name == {controller.name!r}, "
        f"expected 'd_star_lite'"
    )

    # --- d_star_lite is NOT a REPLAN family: a --replan-k must be rejected. ---
    try:
        build_controller("d_star_lite", 5)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "TC37: build_controller('d_star_lite', 5) must raise ValueError "
            "(d_star_lite is not a REPLAN family)"
        )

    seed_value = "37"
    world_stem = Path(yaml_path).stem

    # A forbidden --replan-k through the runner must be a config error (exit 2).
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        r_bad = subprocess.run(
            [
                sys.executable, "-m", "runners.run_episode",
                "--algorithm", "d_star_lite",
                "--replan-k", "5",
                "--seed", seed_value,
                "--world", yaml_path,
                "--no-traffic",
                "--results-dir", td,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r_bad.returncode == 2, (
            f"TC37: forbidden --replan-k must exit 2, got {r_bad.returncode}; "
            f"stderr={r_bad.stderr[-400:]}"
        )

    # --- Traffic e2e: d_star_lite dodges and reaches the goal under traffic; every
    # trace line must carry the 8th dynamic_obstacles_sha256 key. This is the slowest
    # single TC — replans every step over a full ~800-step traversal. ---
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        r = subprocess.run(
            [
                sys.executable, "-m", "runners.run_episode",
                "--algorithm", "d_star_lite",
                "--seed", seed_value,
                "--world", yaml_path,
                "--traffic",  # default; stated explicitly
                "--results-dir", td,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert r.returncode == 0, (
            f"TC37 d_star_lite traffic runner exit {r.returncode}; stderr={r.stderr[-400:]}"
        )

        out_dir = Path(td) / world_stem / "d_star_lite"
        json_path = out_dir / f"{seed_value}.json"
        jsonl_path = out_dir / f"{seed_value}.trace.jsonl"
        assert json_path.exists(), f"TC37: metrics JSON missing at {json_path}"
        assert jsonl_path.exists(), f"TC37: trace JSONL missing at {jsonl_path}"

        # The episode RAN to completion (no runner fault): t=0 planning succeeded.
        metrics = json.loads(json_path.read_text(encoding="utf-8"))
        assert metrics["planner_error"] is None, (
            f"TC37: d_star_lite must plan successfully at t=0; "
            f"planner_error={metrics['planner_error']}"
        )

        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert lines, "TC37: d_star_lite traffic trace JSONL is empty"
        for idx, raw in enumerate(lines):
            rec = json.loads(raw)
            assert isinstance(rec, dict), f"TC37: trace line {idx} is not an object"
            assert "dynamic_obstacles_sha256" in rec, (
                f"TC37: trace line {idx} missing dynamic_obstacles_sha256 with traffic on; "
                f"keys={sorted(rec)}"
            )
            assert len(rec) == 8, (
                f"TC37: trace line {idx} must have 8 keys with traffic on, got {len(rec)}: "
                f"{sorted(rec)}"
            )


def tc46(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure in-process; drives the controller directly
    """D* Lite deferred settle: no per-tick settle, on-demand settle == fresh A*.

    Drives `DStarLiteController` directly against the real world YAML (no irsim, no
    subprocess) with a counting spy wrapped around `compute_shortest_path`:

    - Phase A: committed clear ticks (all-NaN lidar) must NOT settle (spy == 0).
    - Phase B: lidar changes that land BEHIND the robot run the per-tick
      bookkeeping (self._cells diverges from the static grid) yet still must NOT
      settle (the deferral is real, not just "nothing changed").
    - Phase C: a return ON the robot->target-waypoint segment forces a settle
      (spy >= 1).
    - Oracle: after the forced settle the incrementally repaired path matches a
      fresh A* on the same folded grid (the binding batched-update correctness
      proof).

    Reaching into controller._search/._cells/._follower privates matches the
    established TC31/TC32 pattern.
    """
    _ensure_repo_root_on_path()
    from manual_astar import (  # type: ignore[import-not-found]
        OccupancyGrid,
        astar_search,
        world_to_grid,
    )
    from manual_astar import load_world as _load_world  # type: ignore[import-not-found]
    from planners._grid import (  # type: ignore[import-not-found]
        lidar_to_occupancy,
        load_lidar_geometry,
    )
    from planners.d_star_lite import DStarLiteController  # type: ignore[import-not-found]

    # --- Build the controller and plan at t=0 from an all-NaN fold (==static). ---
    world = _load_world(yaml_path)
    start_x, start_y = float(world.start[0]), float(world.start[1])
    raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    theta = float(raw["robot"]["state"][2]) if len(raw["robot"]["state"]) > 2 else 0.0

    geom = load_lidar_geometry(yaml_path)
    range_max = float(raw["robot"]["sensors"][0].get("range_max", 5.0))
    bearings = np.linspace(geom.angle_min, geom.angle_max, geom.number)

    state0 = np.array([start_x, start_y, theta], dtype=np.float64)
    nan_lidar = np.full((geom.number,), np.nan, dtype=np.float64)

    controller = DStarLiteController()
    controller.reset(yaml_path, (), nan_lidar, state0)

    # --- Counting spy over compute_shortest_path (closure over the bound method). ---
    real_csp = controller._search.compute_shortest_path
    spy = {"count": 0}

    def _counting_csp() -> None:
        spy["count"] += 1
        real_csp()

    controller._search.compute_shortest_path = _counting_csp  # type: ignore[method-assign]

    static_cells = controller._grid.cells

    def _make_directed_lidar(world_angle: float, beam_range: float) -> np.ndarray:
        """A lidar scan with the 3 beams closest to `world_angle` set to `beam_range`."""
        target_bearing = world_angle - theta
        # Wrap to [-pi, pi) so the nearest-beam search is correct.
        target_bearing = (target_bearing + np.pi) % (2.0 * np.pi) - np.pi
        wrapped = (bearings + np.pi) % (2.0 * np.pi) - np.pi
        diffs = np.abs((wrapped - target_bearing + np.pi) % (2.0 * np.pi) - np.pi)
        center = int(np.argmin(diffs))
        scan = np.full((geom.number,), np.nan, dtype=np.float64)
        for offset in (-1, 0, 1):
            scan[(center + offset) % geom.number] = beam_range
        return scan

    # --- Phase A: committed clear ticks must NOT settle. ---
    for _ in range(5):
        action = controller.act(state0, nan_lidar)
        assert action.shape == (2, 1), (
            f"TC46: act() must return a (2,1) action, got shape {action.shape}"
        )
    assert spy["count"] == 0, (
        f"TC46 Phase A: clear committed ticks must NOT settle; "
        f"compute_shortest_path fired {spy['count']} times"
    )

    # --- Phase B: changes BEHIND the robot — bookkeeping runs, settle does not. ---
    # The current target waypoint sits ahead (toward the goal); fire beams in the
    # opposite (backward) world direction so the fold flips cells off the robot's
    # committed segment. The backward range is clamped so the hit (plus its
    # inflation disk) stays inside the grid even from this corner start, since a
    # too-long beam would land off-grid and mark nothing.
    target = controller._follower.current_waypoint(state0[:2])
    forward_angle = float(np.arctan2(target[1] - start_y, target[0] - start_x))
    backward_angle = forward_angle + np.pi
    grid = controller._grid
    offset_x, offset_y = float(grid.offset[0]), float(grid.offset[1])
    world_w = grid.shape[1] * grid.resolution
    world_h = grid.shape[0] * grid.resolution
    margin = controller._inflation + 2.0 * grid.resolution
    cos_b, sin_b = float(np.cos(backward_angle)), float(np.sin(backward_angle))
    # Largest range r such that (start + r*dir) stays `margin` inside every wall.
    max_back = range_max - 0.5
    for component, lo, hi, origin in (
        (cos_b, offset_x + margin, offset_x + world_w - margin, start_x),
        (sin_b, offset_y + margin, offset_y + world_h - margin, start_y),
    ):
        if component > 1e-9:
            max_back = min(max_back, (hi - origin) / component)
        elif component < -1e-9:
            max_back = min(max_back, (lo - origin) / component)
    back_range = float(np.clip(min(2.5, max_back), 0.5, range_max - 0.5))
    back_lidar = _make_directed_lidar(backward_angle, back_range)

    # Sanity: the chosen behind-the-robot scan must actually flip cells (else the
    # "deferral is real" assertion below would pass vacuously).
    back_fold = lidar_to_occupancy(
        static_cells, grid, state0, back_lidar, geom, controller._inflation
    )
    assert not np.array_equal(back_fold, static_cells), (
        f"TC46 setup: the backward scan at range {back_range} marked no cells; "
        f"adjust the geometry"
    )

    for _ in range(2):
        controller.act(state0, back_lidar)
    assert spy["count"] == 0, (
        f"TC46 Phase B: a behind-the-robot change must NOT settle; "
        f"compute_shortest_path fired {spy['count']} times"
    )
    assert not np.array_equal(controller._cells, static_cells), (
        "TC46 Phase B: the per-tick bookkeeping must have flipped cells in "
        "controller._cells (deferral is real, not 'nothing changed')"
    )

    # --- Phase C: a return ON the committed segment forces the settle. ---
    # Hit between the robot's own inflation boundary and the current target
    # waypoint, along the robot->target direction. The inflation disk
    # (robot_radius + SAFETY_MARGIN) is wide enough that the marked cells straddle
    # the segment's line-of-sight check, yet placing the hit beyond the inflation
    # radius keeps the robot's own cell free (so a finite detour to the goal still
    # exists — the oracle below requires it). Deriving the range from the live
    # segment length keeps the hit on-segment regardless of the first leg length.
    seg_len = float(np.linalg.norm(target - state0[:2]))
    inflation = controller._inflation
    assert seg_len > inflation, (
        f"TC46 setup: committed segment {seg_len:.3f} m is shorter than the "
        f"inflation radius {inflation:.3f} m; cannot block it without sealing "
        f"the robot cell"
    )
    fwd_range = 0.5 * (inflation + seg_len)
    fwd_lidar = _make_directed_lidar(forward_angle, fwd_range)
    controller.act(state0, fwd_lidar)
    assert spy["count"] >= 1, (
        f"TC46 Phase C: a return on the committed segment must force a settle; "
        f"compute_shortest_path fired {spy['count']} times (expected >= 1)"
    )

    # --- Oracle: the deferred-batch settle reaches the same optimum a fresh A* does. ---
    inc_path = controller._search.extract_path()
    inc_cost = _octile_path_cost(inc_path)

    oracle_grid = OccupancyGrid(
        cells=controller._cells.copy(),
        resolution=controller._grid.resolution,
        offset=controller._grid.offset,
    )
    cur_cell = world_to_grid(state0[:2], controller._grid)
    goal_cell = world_to_grid(controller._goal_xy, controller._grid)
    oracle_path = astar_search(oracle_grid, cur_cell, goal_cell)
    oracle_cost = _octile_path_cost(oracle_path)

    assert abs(inc_cost - oracle_cost) < 1e-9, (
        f"TC46: deferred-batch incremental cost {inc_cost} != fresh-A* oracle cost "
        f"{oracle_cost}; a batch of update_cells + one settle diverged from "
        f"from-scratch"
    )


def tc47(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure in-process; fixed internal RNG
    """rrt-local LOS helper == segment_is_clear_grid (stratified equivalence fuzz).

    Proves the allocation-free scalar twin `planners.rrt._segment_clear_fast`
    returns the BIT-IDENTICAL bool as the frozen reference
    `planners._grid.segment_is_clear_grid` over a large stratified set of
    segments. This is the safety net that lets the RRT hot loop swap the slow
    numpy collision check for the fast scalar one without breaking the
    byte-identical-trace guarantee; if a future edit to either function diverges
    them on even one input, this case fails loud with the offending segment.

    In-process only (NO irsim, NO subprocess). A FIXED internal RNG makes the
    case deterministic regardless of the `seed` arg, which is ignored.

    Several random boolean occupancy grids are built, each with occupied cells
    pressed against ALL FOUR edges (a clip-to-edge-cell OOB endpoint only
    distinguishes the two implementations when the edge cell is sometimes
    occupied — without edge occupancy stratum 1 is toothless), and with varying
    resolution and offset (so the clamp arithmetic is exercised under different
    scales and origins). >= 100,000 (p0, p1) segments are apportioned across
    four strata, each well-represented:

    1. Out-of-bounds endpoints — one or both endpoints with negative coords
       and/or coords past width/height (exercises the clip-to-edge-cell-then-read
       path, the named LOS-equivalence trap).
    2. Degenerate near-zero segments — endpoints within sub-1e-9 of each other,
       plus exactly-equal endpoints (exercises the `length < 1e-9` branch).
    3. Length spread — segment lengths spanning multiple sample_count regimes:
       short (single-sample), mid, and long multi-sample. This stratum is what
       makes a future `math.hypot` swap FAIL: a 1-ULP length difference flips the
       sample count on ~17% of inputs.
    4. In-bounds ordinary segments — the baseline stratum.
    """
    import math

    _ensure_repo_root_on_path()
    from manual_astar import OccupancyGrid  # type: ignore[import-not-found]
    from planners._grid import segment_is_clear_grid  # type: ignore[import-not-found]
    from planners.rrt import _segment_clear_fast  # type: ignore[import-not-found]

    rng = np.random.default_rng(4_700_000_047)  # fixed: TC47 is self-deterministic

    def build_grid(rows: int, cols: int, resolution: float, offset: np.ndarray) -> OccupancyGrid:
        """A random boolean grid with ~18% interior fill AND occupied edge rings.

        The four edges each get a random subset of occupied cells (so a clipped
        OOB endpoint sometimes reads an occupied edge cell — the only way
        stratum 1 can tell the two implementations apart).
        """
        cells = rng.random((rows, cols)) < 0.18
        # Press occupancy against every edge: ~40% of each border cell occupied.
        cells[0, :] |= rng.random(cols) < 0.40
        cells[rows - 1, :] |= rng.random(cols) < 0.40
        cells[:, 0] |= rng.random(rows) < 0.40
        cells[:, cols - 1] |= rng.random(rows) < 0.40
        # Keep a free interior anchor so not every read is trivially blocked.
        cells[rows // 2, cols // 2] = False
        return OccupancyGrid(
            cells=np.ascontiguousarray(cells, dtype=np.bool_),
            resolution=float(resolution),
            offset=np.asarray(offset, dtype=float),
        )

    # A handful of grids spanning resolution and offset so the clamp arithmetic
    # (floor((coord - offset) / resolution), clamped into [0, n-1]) is exercised
    # under different scales and origins.
    grids = [
        build_grid(rows=50, cols=50, resolution=0.10, offset=np.array([0.0, 0.0])),
        build_grid(rows=40, cols=60, resolution=0.25, offset=np.array([-5.0, 3.0])),
        build_grid(rows=64, cols=48, resolution=0.05, offset=np.array([2.5, -1.25])),
        build_grid(rows=33, cols=33, resolution=0.50, offset=np.array([-10.0, -10.0])),
    ]

    # Per-stratum budget: 4 strata x ~26,000 = ~104,000 total segments (>= 1e5).
    per_stratum = 26_000
    total = 0
    mismatches = 0
    sample_factor = 0.5  # planners._grid.SEGMENT_SAMPLE_FACTOR; sample_step = res*this

    def world_extent(grid: OccupancyGrid) -> tuple[float, float, float, float]:
        rows, cols = grid.shape
        ox, oy = float(grid.offset[0]), float(grid.offset[1])
        return ox, oy, ox + cols * grid.resolution, oy + rows * grid.resolution

    def in_bounds_point(grid: OccupancyGrid) -> np.ndarray:
        ox, oy, hx, hy = world_extent(grid)
        return np.array(
            [ox + float(rng.random()) * (hx - ox), oy + float(rng.random()) * (hy - oy)],
            dtype=float,
        )

    def out_of_bounds_point(grid: OccupancyGrid) -> np.ndarray:
        """A point with at least one coord pushed outside the grid's world extent."""
        ox, oy, hx, hy = world_extent(grid)
        width, height = hx - ox, hy - oy
        # Each coord independently lands below-min, above-max, or in-range; force
        # at least one axis out so the endpoint is genuinely OOB.
        def coord(lo: float, span: float) -> tuple[float, bool]:
            choice = int(rng.integers(0, 3))
            if choice == 0:  # below the minimum (negative-relative)
                return lo - float(rng.random()) * (span + 1.0) - 0.01, True
            if choice == 1:  # past the maximum
                return lo + span + float(rng.random()) * (span + 1.0) + 0.01, True
            return lo + float(rng.random()) * span, False  # in-range

        px, x_out = coord(ox, width)
        py, y_out = coord(oy, height)
        if not (x_out or y_out):  # guarantee OOB: shove x past the max edge
            px = hx + float(rng.random()) * (width + 1.0) + 0.01
        return np.array([px, py], dtype=float)

    def check(grid: OccupancyGrid, p0: np.ndarray, p1: np.ndarray) -> None:
        nonlocal total, mismatches
        cells = grid.cells
        fast = _segment_clear_fast(cells, grid, p0, p1)
        slow = segment_is_clear_grid(cells, grid, p0, p1)
        total += 1
        if fast != slow:
            mismatches += 1
            raise AssertionError(
                f"TC47 LOS mismatch: p0={tuple(float(c) for c in p0)} "
                f"p1={tuple(float(c) for c in p1)}; grid resolution="
                f"{grid.resolution} offset={tuple(float(c) for c in grid.offset)} "
                f"shape={grid.shape}; _segment_clear_fast={fast} "
                f"segment_is_clear_grid={slow}"
            )

    # --- Stratum 1: out-of-bounds endpoints (one or both OOB). ---
    for _ in range(per_stratum):
        grid = grids[int(rng.integers(0, len(grids)))]
        if int(rng.integers(0, 2)) == 0:
            # One OOB, one in-bounds (random which end).
            a, b = out_of_bounds_point(grid), in_bounds_point(grid)
            if int(rng.integers(0, 2)) == 0:
                a, b = b, a
        else:
            # Both OOB.
            a, b = out_of_bounds_point(grid), out_of_bounds_point(grid)
        check(grid, a, b)

    # --- Stratum 2: degenerate near-zero + exactly-equal endpoints. ---
    for index in range(per_stratum):
        grid = grids[int(rng.integers(0, len(grids)))]
        base = in_bounds_point(grid)
        if index % 4 == 0:
            # Exactly-equal endpoints (the canonical length == 0 case).
            check(grid, base, base.copy())
        else:
            # Sub-1e-9 perturbation: still inside the `length < 1e-9` branch.
            jitter = (rng.random(2) - 0.5) * 2.0e-10
            check(grid, base, base + jitter)

    # --- Stratum 3: length spread across sample_count regimes. ---
    # sample_step = resolution * 0.5; lengths chosen to land in the single-sample
    # (length < sample_step => count clamped to 2), mid (a few samples), and long
    # (many samples) regimes so a future hypot-vs-sqrt last-bit drift in the
    # sample count would show up here.
    for index in range(per_stratum):
        grid = grids[int(rng.integers(0, len(grids)))]
        sample_step = grid.resolution * sample_factor
        ox, oy, hx, hy = world_extent(grid)
        regime = index % 3
        if regime == 0:  # short: below one sample_step (count clamps to 2)
            target_len = float(rng.random()) * sample_step * 0.9
        elif regime == 1:  # mid: a handful of samples
            target_len = sample_step * (1.0 + float(rng.random()) * 6.0)
        else:  # long: many samples
            target_len = sample_step * (10.0 + float(rng.random()) * 60.0)
        angle = float(rng.random()) * 2.0 * math.pi
        dx = target_len * math.cos(angle)
        dy = target_len * math.sin(angle)
        # Anchor so both endpoints stay in-bounds (keep this stratum about the
        # length->sample_count behaviour, not OOB clamping).
        margin = abs(target_len) + grid.resolution
        ax = ox + margin + float(rng.random()) * max(hx - ox - 2.0 * margin, 0.0)
        ay = oy + margin + float(rng.random()) * max(hy - oy - 2.0 * margin, 0.0)
        p0 = np.array([ax, ay], dtype=float)
        p1 = np.array([ax + dx, ay + dy], dtype=float)
        check(grid, p0, p1)

    # --- Stratum 4: in-bounds ordinary segments (the baseline). ---
    for _ in range(per_stratum):
        grid = grids[int(rng.integers(0, len(grids)))]
        check(grid, in_bounds_point(grid), in_bounds_point(grid))

    assert total >= 100_000, (
        f"TC47: stratified coverage too small ({total} segments); the plan "
        f"requires >= 100,000"
    )

    # --- Length-formula guard: catches a future `math.hypot` swap in the helper. ---
    # The helper MUST compute length as math.sqrt(dx*dx + dy*dy), which is
    # bit-identical to float(np.linalg.norm(end - start)). math.hypot uses an
    # extended-precision intermediate that flips the last bit on ~17% of inputs
    # (e.g. dx=1.0, dy=2.4000000000000004 yields a different value), which would
    # propagate into the sample count and flip the returned bool. Assert the sqrt
    # form is bit-for-bit equal to numpy's norm over a large random sample, and
    # pin the documented counterexample so the trap is explicit in the test.
    dx_trap, dy_trap = 1.0, 2.4000000000000004
    sqrt_trap = math.sqrt(dx_trap * dx_trap + dy_trap * dy_trap)
    hypot_trap = math.hypot(dx_trap, dy_trap)
    assert sqrt_trap != hypot_trap, (
        "TC47: the documented hypot/sqrt counterexample no longer diverges on "
        "this platform; pick a fresh dx/dy pair to keep the guard meaningful"
    )
    for _ in range(20_000):
        dx_g = float(rng.standard_normal()) * 10.0
        dy_g = float(rng.standard_normal()) * 10.0
        sqrt_len = math.sqrt(dx_g * dx_g + dy_g * dy_g)
        norm_len = float(np.linalg.norm(np.asarray([dx_g, dy_g], dtype=float)))
        assert sqrt_len == norm_len, (
            f"TC47 length-formula guard: math.sqrt(dx*dx+dy*dy)={sqrt_len!r} != "
            f"float(np.linalg.norm)={norm_len!r} for dx={dx_g!r} dy={dy_g!r}; "
            f"the helper must use math.sqrt (math.hypot would diverge here)"
        )

    print(f"TC47: {total} segments, {mismatches} mismatches")


# ---------------------------------------------------------------------------
# TC38..TC45 — the reactive (DWA / APF) + sampling (RRT / RRT*) families plus
# the commitment-horizon fix proof. TC38/TC39 are traffic-ON reactive drives;
# TC40/TC41 are the --no-traffic sampling drives (TC40 also proves trace
# determinism, TC41 also prints the RRT*-vs-RRT planned-cost observation);
# TC42 is the sealed-start planner-failure audit for both samplers; TC43 is the
# pure-registry validation across all six new keys; TC44 is the two sampling
# replan families end-to-end with --replan-k; TC45 is the BINDING gate that the
# commitment horizon lets the grid replanners reach the goal. All in-process
# imports reuse the tc28-tc37 repo-root helper.
# ---------------------------------------------------------------------------


def tc38(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """dwa traffic-ON drive via runner: exit 0, runs to completion, 8-key trace per line.

    DWA reacts to the live lidar, so under traffic it may crash or time out — that
    is fine. We assert only that the episode RAN (reset never raises, so the trace
    is always written) and that every trace line carries the 8th traffic key.
    """
    repo_root = _ensure_repo_root_on_path()
    seed_value = "38"
    world_stem = Path(yaml_path).stem
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        r = subprocess.run(
            [
                sys.executable, "-m", "runners.run_episode",
                "--algorithm", "dwa",
                "--seed", seed_value,
                "--world", yaml_path,
                "--traffic",  # default; stated explicitly
                "--results-dir", td,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert r.returncode == 0, (
            f"TC38 dwa runner exit {r.returncode}; stderr={r.stderr[-400:]}"
        )

        out_dir = Path(td) / world_stem / "dwa"
        json_path = out_dir / f"{seed_value}.json"
        jsonl_path = out_dir / f"{seed_value}.trace.jsonl"
        assert json_path.exists(), f"TC38: metrics JSON missing at {json_path}"
        assert jsonl_path.exists(), f"TC38: trace JSONL missing at {jsonl_path}"

        metrics = json.loads(json_path.read_text(encoding="utf-8"))
        assert metrics["planner_error"] is None, (
            f"TC38: dwa reset must not raise; planner_error={metrics['planner_error']}"
        )

        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert lines, "TC38: dwa traffic trace JSONL is empty"
        for idx, raw in enumerate(lines):
            rec = json.loads(raw)
            assert isinstance(rec, dict), f"TC38: trace line {idx} is not an object"
            assert "dynamic_obstacles_sha256" in rec, (
                f"TC38: trace line {idx} missing dynamic_obstacles_sha256 with traffic on; "
                f"keys={sorted(rec)}"
            )
            assert len(rec) == 8, (
                f"TC38: trace line {idx} must have 8 keys with traffic on, got {len(rec)}: "
                f"{sorted(rec)}"
            )


def tc39(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """apf traffic-ON drive via runner: exit 0, runs to completion, 8-key trace per line.

    Like TC38: APF reacts to the live lidar and may crash or time out under
    traffic; we assert only that it RAN and that every trace line is the 8-key
    traffic schema.
    """
    repo_root = _ensure_repo_root_on_path()
    seed_value = "39"
    world_stem = Path(yaml_path).stem
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        r = subprocess.run(
            [
                sys.executable, "-m", "runners.run_episode",
                "--algorithm", "apf",
                "--seed", seed_value,
                "--world", yaml_path,
                "--traffic",  # default; stated explicitly
                "--results-dir", td,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert r.returncode == 0, (
            f"TC39 apf runner exit {r.returncode}; stderr={r.stderr[-400:]}"
        )

        out_dir = Path(td) / world_stem / "apf"
        json_path = out_dir / f"{seed_value}.json"
        jsonl_path = out_dir / f"{seed_value}.trace.jsonl"
        assert json_path.exists(), f"TC39: metrics JSON missing at {json_path}"
        assert jsonl_path.exists(), f"TC39: trace JSONL missing at {jsonl_path}"

        metrics = json.loads(json_path.read_text(encoding="utf-8"))
        assert metrics["planner_error"] is None, (
            f"TC39: apf reset must not raise; planner_error={metrics['planner_error']}"
        )

        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert lines, "TC39: apf traffic trace JSONL is empty"
        for idx, raw in enumerate(lines):
            rec = json.loads(raw)
            assert isinstance(rec, dict), f"TC39: trace line {idx} is not an object"
            assert "dynamic_obstacles_sha256" in rec, (
                f"TC39: trace line {idx} missing dynamic_obstacles_sha256 with traffic on; "
                f"keys={sorted(rec)}"
            )
            assert len(rec) == 8, (
                f"TC39: trace line {idx} must have 8 keys with traffic on, got {len(rec)}: "
                f"{sorted(rec)}"
            )


def tc40(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """rrt_once --no-traffic reaches the goal + two same-seed runs are byte-identical.

    AC5: rrt_once drives arena_v1 to the goal (~73 s measured) well under the cap.
    AC4: the deterministic single-plan RNG makes two same-seed runs produce
    byte-identical trace JSONL.
    """
    repo_root = _ensure_repo_root_on_path()
    seed_value = "40"
    world_stem = Path(yaml_path).stem
    runner_args = [
        sys.executable, "-m", "runners.run_episode",
        "--algorithm", "rrt_once",
        "--seed", seed_value,
        "--world", yaml_path,
        "--no-traffic",
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_a, \
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_b:
        for td in (td_a, td_b):
            r = subprocess.run(
                [*runner_args, "--results-dir", td],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=600,
            )
            assert r.returncode == 0, (
                f"TC40 rrt_once runner exit {r.returncode}; stderr={r.stderr[-400:]}"
            )

        json_a = Path(td_a) / world_stem / "rrt_once" / f"{seed_value}.json"
        json_b = Path(td_b) / world_stem / "rrt_once" / f"{seed_value}.json"
        jsonl_a = Path(td_a) / world_stem / "rrt_once" / f"{seed_value}.trace.jsonl"
        jsonl_b = Path(td_b) / world_stem / "rrt_once" / f"{seed_value}.trace.jsonl"
        for p in (json_a, json_b, jsonl_a, jsonl_b):
            assert p.exists(), f"TC40: expected output missing at {p}"

        for json_path in (json_a, json_b):
            metrics = json.loads(json_path.read_text(encoding="utf-8"))
            assert metrics["planner_error"] is None, (
                f"TC40 planner_error not None at {json_path}: {metrics}"
            )
            assert metrics["time_to_goal"] is not None, (
                f"TC40 rrt_once did not reach the goal at {json_path}: {metrics}"
            )
            assert metrics["time_to_goal"] <= 110.0, (
                f"TC40 rrt_once time_to_goal {metrics['time_to_goal']} exceeds the 110 s "
                f"margin (measured ~73 s) at {json_path}"
            )

        assert filecmp.cmp(str(jsonl_a), str(jsonl_b), shallow=False), (
            "TC40: two same-seed rrt_once --no-traffic runs produced differing trace JSONL; "
            "the deterministic single-plan RNG (AC4) regressed"
        )


def tc41(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """rrt_star_once --no-traffic reaches the goal (BLOCKING) + planned-cost observation (AC7-obs).

    Subprocess part: rrt_star_once drives arena_v1 to the goal (~71 s measured)
    under the 110 s margin. In-process part (NON-blocking): plan RRT and RRT* on
    the SAME static grid from the SAME seed and print both planned costs as an
    observation — RRT* is expected to be <= RRT, but the costs are NOT asserted.
    """
    repo_root = _ensure_repo_root_on_path()
    seed_value = "41"
    world_stem = Path(yaml_path).stem
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        r = subprocess.run(
            [
                sys.executable, "-m", "runners.run_episode",
                "--algorithm", "rrt_star_once",
                "--seed", seed_value,
                "--world", yaml_path,
                "--no-traffic",
                "--results-dir", td,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert r.returncode == 0, (
            f"TC41 rrt_star_once runner exit {r.returncode}; stderr={r.stderr[-400:]}"
        )
        json_path = Path(td) / world_stem / "rrt_star_once" / f"{seed_value}.json"
        assert json_path.exists(), f"TC41: metrics JSON missing at {json_path}"
        metrics = json.loads(json_path.read_text(encoding="utf-8"))
        assert metrics["planner_error"] is None, (
            f"TC41 planner_error not None: {metrics}"
        )
        assert metrics["time_to_goal"] is not None, (
            f"TC41 rrt_star_once did not reach the goal (time_to_goal is None): {metrics}"
        )
        assert metrics["time_to_goal"] <= 110.0, (
            f"TC41 rrt_star_once time_to_goal {metrics['time_to_goal']} exceeds the 110 s "
            f"margin (measured ~71 s)"
        )

    # --- AC7-obs: print the RRT vs RRT* planned costs on the same static grid. ---
    import planners.rrt as rrt  # type: ignore[import-not-found]
    import planners.rrt_star as rrt_star  # type: ignore[import-not-found]
    from planners.rrt import RRT_SEED  # type: ignore[import-not-found]
    from manual_astar import (  # type: ignore[import-not-found]
        GRID_RESOLUTION,
        SAFETY_MARGIN,
        build_occupancy_grid,
        load_world,
    )

    world = load_world(yaml_path)
    grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN)
    start_xy = np.asarray(world.start, dtype=float)[:2]
    goal_xy = np.asarray(world.goal, dtype=float)[:2]

    rrt_points = rrt.rrt_plan(
        grid.cells, grid, start_xy, goal_xy, np.random.default_rng(RRT_SEED)
    )
    rrt_star_points = rrt_star.rrt_star_plan(
        grid.cells, grid, start_xy, goal_xy, np.random.default_rng(RRT_SEED)
    )
    rrt_cost = rrt.rrt_planned_cost(rrt_points)
    rrt_star_cost = rrt.rrt_planned_cost(rrt_star_points)
    print(
        f"TC41 AC7-obs: rrt planned cost = {rrt_cost:.3f} m, "
        f"rrt_star planned cost = {rrt_star_cost:.3f} m"
    )


def tc42(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal world
    """Sealed-start planner failure: rrt_once and rrt_star_once both raise, no trace written.

    On arena_no_path.yaml the start is walled in, so both samplers exhaust their
    iteration budget and raise — the runner must record planner_error and write
    NO trace JSONL (mirrors TC16's audit).
    """
    repo_root = _ensure_repo_root_on_path()
    no_path_yaml = str(repo_root / "arena" / "arena_no_path.yaml")
    for algorithm, seed_value in (("rrt_once", "42"), ("rrt_star_once", "42")):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            r = subprocess.run(
                [
                    sys.executable, "-m", "runners.run_episode",
                    "--algorithm", algorithm,
                    "--seed", seed_value,
                    "--world", no_path_yaml,
                    "--no-traffic",
                    "--results-dir", td,
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert r.returncode == 0, (
                f"TC42 {algorithm} runner exit {r.returncode}; stderr={r.stderr[-400:]}"
            )

            json_path = Path(td) / "arena_no_path" / algorithm / f"{seed_value}.json"
            jsonl_path = Path(td) / "arena_no_path" / algorithm / f"{seed_value}.trace.jsonl"
            assert json_path.exists(), f"TC42 {algorithm}: metrics JSON missing at {json_path}"
            assert not jsonl_path.exists(), (
                f"TC42 {algorithm}: trace JSONL must NOT exist on planner failure; "
                f"found {jsonl_path}"
            )

            metrics = json.loads(json_path.read_text(encoding="utf-8"))
            assert metrics["planner_error"] is not None, (
                f"TC42 {algorithm} planner_error must not be None: {metrics}"
            )
            assert metrics["time_to_goal"] is None, (
                f"TC42 {algorithm} time_to_goal must be None on planner failure: {metrics}"
            )


def tc43(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure unit; uses the registry only
    """Registration + --replan-k validation for all six new keys (reactive + sampling).

    Mirrors TC33 across dwa / apf / rrt_once / rrt_star_once (reject --replan-k)
    and rrt_replan / rrt_star_replan (require it), checking the name==key invariant
    (AC15), the _k<K> label folding (AC6), and ALGORITHMS membership for all six.
    """
    _ensure_repo_root_on_path()
    from planners import ALGORITHMS, algorithm_label, build_controller  # type: ignore[import-not-found]

    once_like = ("dwa", "apf", "rrt_once", "rrt_star_once")
    replan_like = ("rrt_replan", "rrt_star_replan")

    # The non-replan keys must REJECT a --replan-k.
    for name in once_like:
        try:
            build_controller(name, 5)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"TC43: build_controller({name!r}, 5) must raise ValueError "
                f"({name} is not a REPLAN family)"
            )

    # The replan keys must REQUIRE a --replan-k.
    for name in replan_like:
        try:
            build_controller(name, None)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"TC43: build_controller({name!r}, None) must raise ValueError "
                f"({name} requires --replan-k)"
            )

    # Valid combos construct, name==key (AC15), membership, and label folding (AC6).
    valid_pairs: list[tuple[str, Any]] = [
        ("dwa", None),
        ("apf", None),
        ("rrt_once", None),
        ("rrt_star_once", None),
        ("rrt_replan", 5),
        ("rrt_star_replan", 5),
    ]
    for name, k in valid_pairs:
        controller = build_controller(name, k)
        assert controller.name == name, (
            f"TC43: build_controller({name!r}, {k!r}).name == {controller.name!r}, "
            f"expected {name!r}"
        )
        assert name in ALGORITHMS, f"TC43: {name!r} must be a key in ALGORITHMS"

    # Labels: the two replan families fold _k5, the others use the bare key.
    assert algorithm_label("rrt_replan", 5) == "rrt_replan_k5", (
        f"TC43: algorithm_label('rrt_replan', 5) == {algorithm_label('rrt_replan', 5)!r}, "
        f"expected 'rrt_replan_k5'"
    )
    assert algorithm_label("rrt_star_replan", 5) == "rrt_star_replan_k5", (
        f"TC43: algorithm_label('rrt_star_replan', 5) == "
        f"{algorithm_label('rrt_star_replan', 5)!r}, expected 'rrt_star_replan_k5'"
    )
    for name in once_like:
        assert algorithm_label(name, 5) == name, (
            f"TC43: algorithm_label({name!r}, 5) == {algorithm_label(name, 5)!r}, "
            f"expected the bare key {name!r}"
        )


def tc44(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """rrt_replan and rrt_star_replan traffic-ON via runner (--replan-k 5): labeled dir, 8-key trace.

    Mirrors TC30: each sampling replan family must plan at t=0 (no planner_error),
    write to its _k5 labeled dir, and emit the 8-key traffic trace per line. The
    episode may crash or time out — only completion + schema are asserted.
    """
    repo_root = _ensure_repo_root_on_path()
    seed_value = "44"
    replan_k = "5"
    world_stem = Path(yaml_path).stem
    for algorithm, label in (("rrt_replan", "rrt_replan_k5"),
                             ("rrt_star_replan", "rrt_star_replan_k5")):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            r = subprocess.run(
                [
                    sys.executable, "-m", "runners.run_episode",
                    "--algorithm", algorithm,
                    "--replan-k", replan_k,
                    "--seed", seed_value,
                    "--world", yaml_path,
                    "--traffic",  # default; stated explicitly
                    "--results-dir", td,
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=600,
            )
            assert r.returncode == 0, (
                f"TC44 {algorithm} runner exit {r.returncode}; stderr={r.stderr[-400:]}"
            )

            out_dir = Path(td) / world_stem / label
            json_path = out_dir / f"{seed_value}.json"
            jsonl_path = out_dir / f"{seed_value}.trace.jsonl"
            assert json_path.exists(), (
                f"TC44 {algorithm}: metrics JSON missing at {json_path} — "
                f"label must be {label!r}"
            )
            assert jsonl_path.exists(), f"TC44 {algorithm}: trace JSONL missing at {jsonl_path}"

            metrics = json.loads(json_path.read_text(encoding="utf-8"))
            assert metrics["planner_error"] is None, (
                f"TC44 {algorithm} must plan successfully at t=0; "
                f"planner_error={metrics['planner_error']}"
            )

            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
            assert lines, f"TC44 {algorithm}: trace JSONL is empty"
            for idx, raw in enumerate(lines):
                rec = json.loads(raw)
                assert isinstance(rec, dict), f"TC44 {algorithm}: trace line {idx} is not an object"
                assert "dynamic_obstacles_sha256" in rec, (
                    f"TC44 {algorithm}: trace line {idx} missing dynamic_obstacles_sha256 "
                    f"with traffic on; keys={sorted(rec)}"
                )
                assert len(rec) == 8, (
                    f"TC44 {algorithm}: trace line {idx} must have 8 keys with traffic on, "
                    f"got {len(rec)}: {sorted(rec)}"
                )


def tc45(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """Commitment-horizon fix proof (BINDING): grid replanners reach the goal + commit.

    Part 1 (subprocess): a_star_replan and dijkstra_replan at --replan-k 5
    --no-traffic each reach the goal (~86 s measured) with no crash and no
    timeout. These previously timed out / drove into a wall — the fix is what
    makes them traverse.

    Part 2 (in-process follower-identity proof): build a_star_replan k=5, reset at
    (2,2) with an all-NaN lidar, then act() five times. On a clear fold the
    immediate segment stays clear, so the K-th (5th) replan must KEEP the existing
    follower — the commitment held, the follower was not rebuilt.
    """
    repo_root = _ensure_repo_root_on_path()
    world_stem = Path(yaml_path).stem

    # --- Part 1: both grid replan families reach the goal --no-traffic. ---
    for algorithm, label, seed_value in (
        ("a_star_replan", "a_star_replan_k5", "45"),
        ("dijkstra_replan", "dijkstra_replan_k5", "45"),
    ):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            r = subprocess.run(
                [
                    sys.executable, "-m", "runners.run_episode",
                    "--algorithm", algorithm,
                    "--replan-k", "5",
                    "--seed", seed_value,
                    "--world", yaml_path,
                    "--no-traffic",
                    "--results-dir", td,
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=600,
            )
            assert r.returncode == 0, (
                f"TC45 {algorithm} runner exit {r.returncode}; stderr={r.stderr[-400:]}"
            )
            json_path = Path(td) / world_stem / label / f"{seed_value}.json"
            assert json_path.exists(), f"TC45 {algorithm}: metrics JSON missing at {json_path}"
            metrics = json.loads(json_path.read_text(encoding="utf-8"))
            assert metrics["planner_error"] is None, (
                f"TC45 {algorithm} planner_error not None: {metrics}"
            )
            assert metrics["time_to_goal"] is not None, (
                f"TC45 {algorithm} did not reach the goal (time_to_goal is None) — the "
                f"commitment-horizon fix regressed: {metrics}"
            )
            assert metrics["time_to_goal"] <= 110.0, (
                f"TC45 {algorithm} time_to_goal {metrics['time_to_goal']} exceeds the 110 s "
                f"margin (measured ~86 s): {metrics}"
            )
            assert metrics["crashed"] is False, (
                f"TC45 {algorithm} must not crash on the static map: {metrics}"
            )
            assert metrics["timed_out"] is False, (
                f"TC45 {algorithm} must not time out on the static map: {metrics}"
            )

    # --- Part 2: follower-identity proof (commitment actually held). ---
    from planners import build_controller  # type: ignore[import-not-found]

    controller = build_controller("a_star_replan", 5)
    state0 = np.array([2.0, 2.0, 0.0], dtype=np.float64)
    nan_lidar = np.full((360,), np.nan, dtype=np.float64)
    controller.reset(yaml_path, (), nan_lidar, state0)

    good = controller._follower
    assert good is not None, "TC45 setup: reset() must build a follower"

    # Five acts: the 5th is the replan tick. On the bare static fold the immediate
    # segment stays clear and the follower is not finished, so the follower must be
    # KEPT (committed), not rebuilt.
    for _ in range(5):
        controller.act(state0, nan_lidar)

    assert controller._follower is good, (
        "TC45: on a clear fold the K-th replan must KEEP the committed follower "
        "(the commitment horizon must not rebuild it every K acts)"
    )


# ---------------------------------------------------------------------------
# TC48..TC52 + TC-CLI/TC-FWD — the obstacle-speed-cap sweep (issue #11).
# TC48 is the pure regime table/resolver; TC49 is the spawner/Arena bound
# validation; TC50 is THE binding baseline-determinism + draw-order guard (a
# reordered/added traffic_rng draw breaks the byte-identical baseline here);
# TC51 proves the band is wired at the t=0 snapshot only (positions/heading
# identical across regimes, speeds scaled); TC52 is non-baseline determinism
# across a despawn/refill cycle; TC-CLI subprocess-asserts the runner rejects
# bad/conflicting speed flags with exit 2; TC-FWD proves run_experiment's pure
# command-builder forwards the flags and the manifest records the band.
# ---------------------------------------------------------------------------


def tc48(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure unit; no world/seed used
    """Regime table + resolver: exact 4 bands, cross-module agreement, resolver paths."""
    from arena.speed_regimes import (
        SPEED_REGIMES,
        SPEED_REGIME_CAP,
        resolve_speed_factors,
    )
    from arena.dynamic import SPEED_MIN_FACTOR, SPEED_MAX_FACTOR

    assert SPEED_REGIMES == {
        "slow": (0.3, 0.7),
        "matched": (0.3, 1.0),
        "current": (0.3, 1.5),
        "fast": (0.5, 2.0),
    }, f"TC48: SPEED_REGIMES table mismatch, got {SPEED_REGIMES}"

    # Cross-module agreement: the "current" regime must reproduce the spawner's
    # own default constants exactly (the Mission baseline) — if these drift the
    # baseline-determinism guard (TC50) and the live spawner would disagree.
    assert SPEED_REGIMES["current"] == (SPEED_MIN_FACTOR, SPEED_MAX_FACTOR), (
        "TC48: SPEED_REGIMES['current'] must equal "
        f"(SPEED_MIN_FACTOR, SPEED_MAX_FACTOR)=({SPEED_MIN_FACTOR}, {SPEED_MAX_FACTOR}), "
        f"got {SPEED_REGIMES['current']}"
    )

    assert SPEED_REGIME_CAP == {
        "slow": 0.7,
        "matched": 1.0,
        "current": 1.5,
        "fast": 2.0,
    }, f"TC48: SPEED_REGIME_CAP mismatch, got {SPEED_REGIME_CAP}"

    # Resolver: regime lookup, both-override passthrough, unknown-key ValueError.
    assert resolve_speed_factors("current", None, None) == (0.3, 1.5), (
        "TC48: resolve_speed_factors('current', None, None) must be (0.3, 1.5)"
    )
    assert resolve_speed_factors(None, 0.5, 2.0) == (0.5, 2.0), (
        "TC48: resolve_speed_factors(None, 0.5, 2.0) must passthrough (0.5, 2.0)"
    )
    try:
        resolve_speed_factors("bogus", None, None)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "TC48: resolve_speed_factors('bogus', None, None) must raise ValueError"
        )


def tc49(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own seed
    """Speed-band bound validation: one-sided / bad bounds raise ValueError; min==max OK."""
    seed_value = 0

    def _expect_value_error(label: str, **speed_kwargs: float | None) -> None:
        try:
            arena = Arena(yaml_path, seed=seed_value, traffic=True, **speed_kwargs)
        except ValueError:
            return
        except Exception as exc:  # wrong exception type — surface it
            try:
                arena.close()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            raise AssertionError(
                f"TC49 ({label}): expected ValueError, got {type(exc).__name__}: {exc}"
            )
        # Construction unexpectedly succeeded — clean up and fail.
        try:
            arena.close()
        except Exception:
            pass
        raise AssertionError(
            f"TC49 ({label}): expected ValueError, but construction succeeded"
        )

    # One-sided band (caught in Arena.__init__ BEFORE irsim.make).
    _expect_value_error(
        "one-sided", speed_min_factor=0.5, speed_max_factor=None
    )
    # Non-positive lower bound (from the spawner's 0 < min validation).
    _expect_value_error(
        "min<=0", speed_min_factor=-0.1, speed_max_factor=1.0
    )
    # max < min (from the spawner's min <= max validation).
    _expect_value_error(
        "max<min", speed_min_factor=1.5, speed_max_factor=0.5
    )

    # min == max is a degenerate-but-legal band: construction must succeed.
    arena = Arena(yaml_path, seed=seed_value, traffic=True,
                  speed_min_factor=0.7, speed_max_factor=0.7)
    try:
        pass  # constructed OK — that is the assertion
    finally:
        arena.close()


def tc50(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own seed
    """Baseline determinism + draw-order guard (THE binding gate).

    The explicit baseline band (0.3, 1.5) must reproduce the default
    Arena(traffic=True) dynamic_obstacles_sha256 sequence byte-for-byte: the
    speed draw stays the same Generator.uniform(lo, hi) call with the same
    bounds, so it consumes the same RNG bits. THIS byte-identity is the
    draw-order/count guard — any added or reordered traffic_rng draw (or a
    branch on the band) changes the consumed bits and breaks the match HERE,
    even before the hash content would differ. A non-baseline band (0.5, 2.0)
    must then DIFFER, proving the band actually reaches the speed draw.
    """
    seed_value = 3
    n_ticks = 50  # modest tick budget to bound runtime; enough to diverge the fast band
    zero = np.array([[0.0], [0.0]], dtype=float)

    def collect_hashes(**speed_kwargs: float) -> list[str]:
        arena = Arena(yaml_path, seed=seed_value, traffic=True, **speed_kwargs)
        try:
            _, _, info0 = arena.reset()
            assert info0.dynamic_obstacles_sha256 is not None, (
                "TC50: reset() must produce a non-None sha256 when traffic=True"
            )
            hashes = [info0.dynamic_obstacles_sha256]
            for _ in range(n_ticks):
                _, _, _, info = arena.step(zero)
                assert info.dynamic_obstacles_sha256 is not None, (
                    f"TC50: step {info.step_idx} sha256 is None with traffic on"
                )
                hashes.append(info.dynamic_obstacles_sha256)
                if info.crashed or info.timed_out or info.reached_goal:
                    break
            return hashes
        finally:
            arena.close()

    default_hashes = collect_hashes()
    baseline_hashes = collect_hashes(speed_min_factor=0.3, speed_max_factor=1.5)
    assert default_hashes == baseline_hashes, (
        "TC50: explicit baseline band (0.3, 1.5) is NOT byte-identical to the "
        "default Arena(traffic=True); a traffic_rng draw was added/reordered or "
        "the draw branched on the band. First mismatch at tick "
        f"{next((i for i, (a, b) in enumerate(zip(default_hashes, baseline_hashes)) if a != b), 'n/a')}"
    )

    fast_hashes = collect_hashes(speed_min_factor=0.5, speed_max_factor=2.0)
    assert fast_hashes != baseline_hashes, (
        "TC50: the fast band (0.5, 2.0) produced an identical sha256 sequence to "
        "the baseline — the speed band is not reaching the speed draw"
    )


def tc51(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own seed
    """Band wired at the INITIAL snapshot only: same positions/heading, scaled speeds.

    Overlap rejection is position-only and speed-independent, so at a fixed seed
    the two regimes' t=0 spawn population shares identical (x, y) and identical
    velocity DIRECTION; only the speed magnitude scales with the band. This holds
    ONLY at the initial snapshot — once a (faster) obstacle despawns and refills,
    the regimes draw a differing count of traffic_rng values and diverge — so we
    assert on the snapshot only, NEVER after stepping.
    """
    import math

    seed_value = 5

    def initial_snapshot(
        min_f: float, max_f: float
    ) -> tuple[DynamicObstacleState, ...]:
        arena = Arena(
            yaml_path,
            seed=seed_value,
            traffic=True,
            speed_min_factor=min_f,
            speed_max_factor=max_f,
        )
        try:
            arena.reset()  # populate; do NOT step
            return arena.initial_dynamic_snapshot
        finally:
            arena.close()

    slow = initial_snapshot(0.3, 0.7)
    fast = initial_snapshot(0.5, 2.0)

    assert len(slow) == len(fast) and len(slow) > 0, (
        f"TC51: snapshots must be same non-zero length, got {len(slow)} vs {len(fast)}"
    )

    slow_speeds: list[float] = []
    fast_speeds: list[float] = []
    # Snapshots are id-sorted with the same population, so pair by index.
    for i, (s, f) in enumerate(zip(slow, fast)):
        assert (s.x, s.y) == (f.x, f.y), (
            f"TC51: obstacle {i} spawn position differs across regimes: "
            f"slow=({s.x}, {s.y}) fast=({f.x}, {f.y}); overlap rejection must be "
            "speed-independent at t=0"
        )
        slow_dir = math.atan2(s.vy, s.vx)
        fast_dir = math.atan2(f.vy, f.vx)
        assert math.isclose(slow_dir, fast_dir, abs_tol=1e-9), (
            f"TC51: obstacle {i} velocity direction differs: slow={slow_dir} "
            f"fast={fast_dir} (heading must be identical across regimes at t=0)"
        )
        s_speed = math.hypot(s.vx, s.vy)
        f_speed = math.hypot(f.vx, f.vy)
        assert f_speed > s_speed, (
            f"TC51: obstacle {i} fast speed {f_speed} must exceed slow speed "
            f"{s_speed} (the band must scale the magnitude)"
        )
        slow_speeds.append(s_speed)
        fast_speeds.append(f_speed)

    assert max(fast_speeds) > max(slow_speeds), (
        f"TC51: max fast speed {max(fast_speeds)} must exceed max slow speed "
        f"{max(slow_speeds)}"
    )


def tc52(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own seed
    """Non-baseline determinism across a despawn/refill cycle.

    Two same-seed Arenas at the fast band (0.5, 2.0) must produce byte-identical
    dynamic_obstacles_sha256 sequences over enough ticks to force at least one
    despawn+refill. Fast obstacles clear the arena quickly, so ~180 ticks
    guarantees the refill RNG path is exercised — proving determinism holds at a
    non-baseline cap, not just at the baseline TC50 covers.
    """
    seed_value = 11
    n_ticks = 180
    zero = np.array([[0.0], [0.0]], dtype=float)

    def collect_hashes() -> tuple[list[str], frozenset[int], frozenset[int]]:
        arena = Arena(
            yaml_path,
            seed=seed_value,
            traffic=True,
            speed_min_factor=0.5,
            speed_max_factor=2.0,
        )
        try:
            _, _, info0 = arena.reset()
            assert info0.dynamic_obstacles_sha256 is not None, (
                "TC52: reset() must produce a non-None sha256 when traffic=True"
            )
            assert arena._spawner is not None, "TC52: spawner must be live with traffic=True"
            initial_ids = frozenset(obs.id for obs in arena.initial_dynamic_snapshot)
            hashes = [info0.dynamic_obstacles_sha256]
            for _ in range(n_ticks):
                _, _, _, info = arena.step(zero)
                assert info.dynamic_obstacles_sha256 is not None, (
                    f"TC52: step {info.step_idx} sha256 is None with traffic on"
                )
                hashes.append(info.dynamic_obstacles_sha256)
                if info.crashed or info.timed_out or info.reached_goal:
                    break
            # Read the final live id-set straight from the spawner (the t=0
            # initial_dynamic_snapshot is frozen) so the despawn/refill turnover
            # is observable.
            final_ids = frozenset(arena._spawner.live_ids)
            return hashes, initial_ids, final_ids
        finally:
            arena.close()

    hashes_a, initial_ids, final_ids = collect_hashes()
    hashes_b, _, _ = collect_hashes()
    assert hashes_a == hashes_b, (
        "TC52: two same-seed fast-band runs produced differing sha256 sequences; "
        "non-baseline determinism is broken. First mismatch at tick "
        f"{next((i for i, (a, b) in enumerate(zip(hashes_a, hashes_b)) if a != b), 'n/a')}"
    )
    # The label "across a despawn/refill cycle" must hold: the population has to
    # actually turn over (at least one id despawned AND a new id appeared), else
    # the fast-band refill RNG path was never exercised.
    despawned = initial_ids - final_ids
    appeared = final_ids - initial_ids
    assert despawned and appeared, (
        f"TC52: expected a despawn/refill turnover over {n_ticks} ticks at the fast band, "
        f"but the live-id set did not turn over (despawned={sorted(despawned)}, "
        f"appeared={sorted(appeared)}). The refill path was not exercised, so the "
        "non-baseline determinism claim is not actually testing a refill cycle."
    )


def tc_cli(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — subprocess; own world
    """Speed-flag CLI rejection: each bad/conflicting flag exits 2, writes no JSON."""
    repo_root = Path(__file__).resolve().parent.parent
    base = [
        sys.executable, "-m", "runners.run_episode",
        "--algorithm", "a_star_once",
        "--seed", "42",
        "--world", yaml_path,
        "--no-traffic",
    ]
    bad_flag_sets: list[tuple[str, list[str]]] = [
        ("unknown regime", ["--speed-regime", "bogus"]),
        ("lone min", ["--speed-min-factor", "0.5"]),
        ("regime+override", ["--speed-regime", "current",
                             "--speed-min-factor", "0.4", "--speed-max-factor", "1.0"]),
        ("max<min", ["--speed-min-factor", "1.5", "--speed-max-factor", "0.5"]),
    ]
    for label, extra in bad_flag_sets:
        # ignore_cleanup_errors: an irsim grandchild can briefly outlive its parent
        # subprocess and still hold a trace-file handle at teardown, which raises
        # PermissionError [WinError 32] on Windows. Make that race a no-op.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            r = subprocess.run(
                [*base, "--results-dir", td, *extra],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert r.returncode == 2, (
                f"TC-CLI ({label}): expected exit 2, got {r.returncode}; "
                f"stderr={r.stderr[-400:]}"
            )
            stray = list(Path(td).rglob("42.json"))
            assert not stray, (
                f"TC-CLI ({label}): a rejected run must write no <seed>.json, "
                f"found {stray}"
            )


def tc_fwd(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure + one subprocess
    """run_experiment flag forwarding (pure builder) + manifest provenance (e2e)."""
    from runners.run_experiment import build_episode_cmd

    def _cmd(**kw: Any) -> list[str]:
        return build_episode_cmd(
            seed=1,
            algorithm="a_star_once",
            replan_k=None,
            world_abs="w",
            results_dir="r",
            traffic=False,
            **kw,
        )

    # (a) PURE: a named regime forwards --speed-regime <name>.
    fast_cmd = _cmd(speed_regime="fast", speed_min_override=None, speed_max_override=None)
    assert "--speed-regime" in fast_cmd and fast_cmd[fast_cmd.index("--speed-regime") + 1] == "fast", (
        f"TC-FWD: fast-regime call must forward --speed-regime fast, got {fast_cmd}"
    )

    # An override pair forwards both float flags, not --speed-regime.
    over_cmd = _cmd(speed_regime=None, speed_min_override=0.4, speed_max_override=1.0)
    assert "--speed-min-factor" in over_cmd and "--speed-max-factor" in over_cmd, (
        f"TC-FWD: override call must forward both float flags, got {over_cmd}"
    )
    min_token = over_cmd[over_cmd.index("--speed-min-factor") + 1]
    max_token = over_cmd[over_cmd.index("--speed-max-factor") + 1]
    assert float(min_token) == 0.4, (
        f"TC-FWD: override call must forward --speed-min-factor 0.4, got {min_token!r} in {over_cmd}"
    )
    assert float(max_token) == 1.0, (
        f"TC-FWD: override call must forward --speed-max-factor 1.0, got {max_token!r} in {over_cmd}"
    )
    assert "--speed-regime" not in over_cmd, (
        f"TC-FWD: override call must NOT forward --speed-regime, got {over_cmd}"
    )

    # A default call (no regime, no overrides) forwards NEITHER (byte-identity).
    default_cmd = _cmd(speed_regime=None, speed_min_override=None, speed_max_override=None)
    assert "--speed-regime" not in default_cmd, (
        f"TC-FWD: default call must forward no --speed-regime, got {default_cmd}"
    )
    assert "--speed-min-factor" not in default_cmd and "--speed-max-factor" not in default_cmd, (
        f"TC-FWD: default call must forward no float flags, got {default_cmd}"
    )

    # (b) INTEGRATION: a 1-seed --no-traffic matched run records the band in the manifest.
    repo_root = Path(__file__).resolve().parent.parent
    world_stem = Path(yaml_path).stem
    # ignore_cleanup_errors: an irsim grandchild can briefly outlive its parent
    # subprocess and still hold a trace-file handle at teardown, which raises
    # PermissionError [WinError 32] on Windows. Make that race a no-op.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        r = subprocess.run(
            [
                sys.executable, "-m", "runners.run_experiment",
                "--algorithm", "a_star_once",
                "--world", yaml_path,
                "--num-seeds", "1",
                "--no-traffic",
                "--speed-regime", "matched",
                "--results-dir", td,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, (
            f"TC-FWD: run_experiment exit {r.returncode}; stderr={r.stderr[-400:]}"
        )
        manifest_path = Path(td) / world_stem / "a_star_once" / "_manifest.json"
        assert manifest_path.exists(), f"TC-FWD: manifest missing at {manifest_path}"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest.get("speed_regime") == "matched", (
            f"TC-FWD: manifest speed_regime must be 'matched', got {manifest.get('speed_regime')!r}"
        )
        assert manifest.get("speed_min_factor") == 0.3, (
            f"TC-FWD: manifest speed_min_factor must be 0.3, got {manifest.get('speed_min_factor')!r}"
        )
        assert manifest.get("speed_max_factor") == 1.0, (
            f"TC-FWD: manifest speed_max_factor must be 1.0, got {manifest.get('speed_max_factor')!r}"
        )


# ---------------------------------------------------------------------------
# TC53..TC62 — Predictive (motion-aware) D* Lite checks. The pure predictor
# (TC53..TC56b, TC60, TC61) is in-process (no irsim, no subprocess), mirroring
# TC46/TC47; the oracle e2e/validation cases (TC57..TC59) and the sweep-plotter
# selfcheck (TC62) shell `python -m runners.run_episode` / the plotter, mirroring
# TC15/TC24/TC37/TC-CLI. Every `from planners...` / `from runners...` import sits
# INSIDE the function body (the script-mode import-order gotcha — see module top).
# ---------------------------------------------------------------------------


def tc53(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure unit; synthesizes its own track/grid
    """predict_blocked_cells capsule geometry: constant-radius disk train, sorted/deduped, deterministic.

    Builds a tiny OccupancyGrid and one synthetic +x-moving Track, then asserts the
    capsule stamp is a CONSTANT-radius disk train (the per-step disk cell count does
    not grow with the lookahead step k), the returned cells are sorted row-major and
    deduped, and two calls on identical inputs return byte-identical output (AC4).
    """
    _ensure_repo_root_on_path()
    from manual_astar import OccupancyGrid, world_to_grid  # type: ignore[import-not-found]
    from planners._geometry import iter_disk_cells  # type: ignore[import-not-found]
    from planners._predict import (  # type: ignore[import-not-found]
        PREDICT_DT,
        Track,
        predict_blocked_cells,
    )

    # A 5 m x 5 m grid at the harness resolution, origin-aligned (offset 0).
    grid = OccupancyGrid(
        cells=np.zeros((80, 80), dtype=bool),
        resolution=0.1,
        offset=np.array([0.0, 0.0], dtype=float),
    )
    inflation = 0.25
    horizon_steps = 10

    # One obstacle moving along +x straight down the planned-path corridor.
    track = Track(id=3, x=1.0, y=4.0, vx=1.0, vy=0.0, radius=0.3)
    planned_path = [np.array([0.5, 4.0]), np.array([7.5, 4.0])]
    robot_xy = np.array([0.3, 4.0])

    groups = predict_blocked_cells(
        [track], planned_path, robot_xy, grid, inflation, horizon_steps, PREDICT_DT,
        geometry="capsule", exclusion_radius=inflation, corridor_half_width=inflation,
    )
    assert len(groups) == 1, (
        f"TC53: the single corridor-crossing track must produce one stamp group, "
        f"got {len(groups)}"
    )
    key, cells = groups[0]
    assert key.track_id == track.id, (
        f"TC53: stamp group key must carry the track id {track.id}, got {key.track_id}"
    )
    assert cells, "TC53: the capsule stamp must be non-empty"

    # Sorted row-major + deduped: the returned list equals sorted(set(...)).
    assert cells == sorted(set(cells)), (
        "TC53: capsule cells must be sorted row-major and deduplicated"
    )

    # Constant-radius disk train: recompute each lookahead step's disk via the same
    # shared planners._geometry.iter_disk_cells scan the predictor now uses (capsule
    # => r_k is the constant body-aware band radius). The per-step disk cell COUNT
    # must NOT grow with k (it would for a cone). Distinct k's give distinct centers,
    # so the union is genuinely a TRAIN of equal-radius disks, not a single disk.
    base_radius = track.radius + inflation
    per_step_counts: list[int] = []
    centers: list[tuple[int, int]] = []
    for k in range(1, horizon_steps + 1):
        center_x = track.x + track.vx * k * PREDICT_DT
        center_y = track.y + track.vy * k * PREDICT_DT
        disk = set(iter_disk_cells(grid, center_x, center_y, base_radius))
        per_step_counts.append(len(disk))
        centers.append(world_to_grid(np.array([center_x, center_y], dtype=float), grid))
    assert len(set(per_step_counts)) == 1, (
        f"TC53: capsule per-step disk cell count must be CONSTANT along v (constant "
        f"radius), got varying counts {per_step_counts}"
    )
    assert len(set(centers)) > 1, (
        f"TC53 setup: the +x track must sweep through distinct disk centers (a train), "
        f"got centers {set(centers)}"
    )

    # Determinism (AC4): two calls on identical inputs return byte-identical output.
    groups_again = predict_blocked_cells(
        [track], planned_path, robot_xy, grid, inflation, horizon_steps, PREDICT_DT,
        geometry="capsule", exclusion_radius=inflation, corridor_half_width=inflation,
    )
    assert groups_again == groups, (
        "TC53: predict_blocked_cells must be deterministic — two identical calls "
        "produced differing output"
    )


def tc54(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure unit; synthesizes its own track/grid
    """predict_blocked_cells cone widening + exclusion zone + non-intersecting gate.

    Same synthetic setup as TC53 with geometry="cone": the per-step disk radius GROWS
    with the lookahead step (the cone stamp strictly supersets the capsule stamp for the
    same track/horizon), NO stamped cell lies within exclusion_radius of the robot (AC5),
    and a track that does NOT cross the planned-path corridor is dropped by the gate.
    """
    _ensure_repo_root_on_path()
    from manual_astar import OccupancyGrid  # type: ignore[import-not-found]
    from planners._predict import (  # type: ignore[import-not-found]
        CONE_GROWTH_PER_STEP,
        PREDICT_DT,
        Track,
        predict_blocked_cells,
    )

    grid = OccupancyGrid(
        cells=np.zeros((100, 100), dtype=bool),
        resolution=0.1,
        offset=np.array([0.0, 0.0], dtype=float),
    )
    inflation = 0.25
    horizon_steps = 10
    assert CONE_GROWTH_PER_STEP > 0.0, (
        "TC54 setup: CONE_GROWTH_PER_STEP must be positive for the cone to widen"
    )

    track = Track(id=3, x=1.0, y=5.0, vx=1.0, vy=0.0, radius=0.3)
    planned_path = [np.array([0.5, 5.0]), np.array([9.5, 5.0])]
    robot_xy = np.array([0.3, 5.0])

    cap_groups = predict_blocked_cells(
        [track], planned_path, robot_xy, grid, inflation, horizon_steps, PREDICT_DT,
        geometry="capsule", exclusion_radius=inflation, corridor_half_width=inflation,
    )
    cone_groups = predict_blocked_cells(
        [track], planned_path, robot_xy, grid, inflation, horizon_steps, PREDICT_DT,
        geometry="cone", exclusion_radius=inflation, corridor_half_width=inflation,
    )
    assert len(cap_groups) == 1 and len(cone_groups) == 1, (
        f"TC54: both geometries must stamp the corridor-crossing track "
        f"(capsule={len(cap_groups)}, cone={len(cone_groups)})"
    )
    cap_cells = set(cap_groups[0][1])
    cone_cells = set(cone_groups[0][1])

    # Cone widens with k => its stamp strictly supersets the capsule's for the same
    # track/horizon (the radius grows, never shrinks).
    assert cap_cells <= cone_cells, (
        "TC54: the cone stamp must SUPERSET the capsule stamp (radius grows with step)"
    )
    assert len(cone_cells) > len(cap_cells), (
        f"TC54: the cone stamp must be STRICTLY larger than the capsule stamp "
        f"(cone={len(cone_cells)}, capsule={len(cap_cells)})"
    )

    # Robot exclusion zone (AC5): no stamped cell's center lies within
    # exclusion_radius of the robot, for either geometry.
    exclusion_sq = inflation * inflation
    resolution = grid.resolution
    offset_x, offset_y = float(grid.offset[0]), float(grid.offset[1])
    for label, cells in (("capsule", cap_cells), ("cone", cone_cells)):
        for row, col in cells:
            cell_x = offset_x + (col + 0.5) * resolution
            cell_y = offset_y + (row + 0.5) * resolution
            dx = cell_x - float(robot_xy[0])
            dy = cell_y - float(robot_xy[1])
            assert dx * dx + dy * dy > exclusion_sq, (
                f"TC54: {label} stamped cell ({row},{col}) lies within the robot "
                f"exclusion radius {inflation} of {tuple(robot_xy)} (AC5 violated)"
            )

    # Gate: a track that never crosses the corridor is dropped. Place it far above
    # the path and moving further away (no [0, T] intersection with the corridor).
    away = Track(id=4, x=5.0, y=12.0, vx=0.0, vy=2.0, radius=0.3)
    away_groups = predict_blocked_cells(
        [away], planned_path, robot_xy, grid, inflation, horizon_steps, PREDICT_DT,
        geometry="cone", exclusion_radius=inflation, corridor_half_width=inflation,
    )
    assert away_groups == [], (
        f"TC54: a track that never intersects the planned-path corridor must be "
        f"dropped by the gate, got {len(away_groups)} group(s)"
    )


def tc55(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure unit; synthesizes its own tracks/grid
    """Predicted-conflict gate: divergent-now-collide-later crosser stamped; receding dropped.

    The gate is geometric over [0, T], NOT an instantaneous closing-course test. Proof:
    a shallow-angle FAST crosser that is RECEDING from the robot at t=0 (an instantaneous
    closing-rate gate would drop it) yet whose capsule crosses the planned-path corridor
    within the horizon IS stamped (finite time-to-conflict). A track moving directly away
    from the corridor (genuinely receding) is NOT stamped.
    """
    _ensure_repo_root_on_path()
    from manual_astar import OccupancyGrid  # type: ignore[import-not-found]
    from planners._predict import (  # type: ignore[import-not-found]
        PREDICT_DT,
        Track,
        predict_blocked_cells,
    )

    grid = OccupancyGrid(
        cells=np.zeros((200, 200), dtype=bool),
        resolution=0.1,
        offset=np.array([0.0, 0.0], dtype=float),
    )
    inflation = 0.25
    horizon_steps = 20
    # Robot at (2,5); a long corridor along +x toward (16,5).
    planned_path = [np.array([2.0, 5.0]), np.array([16.0, 5.0])]
    robot_xy = np.array([2.0, 5.0])

    # Divergent-now-collide-later: far ahead at (11, 1.5), moving up-and-forward. The
    # closing rate to the ROBOT is strictly positive (it is moving AWAY from the robot
    # now — an instantaneous-heading/closing-course gate would drop it), yet its capsule
    # sweeps up into the corridor band within the horizon (a finite TTC).
    crosser = Track(id=1, x=11.0, y=1.5, vx=3.0, vy=2.5, radius=0.3)
    los_x = crosser.x - float(robot_xy[0])
    los_y = crosser.y - float(robot_xy[1])
    closing_rate = (los_x * crosser.vx + los_y * crosser.vy) / float(
        np.hypot(los_x, los_y)
    )
    assert closing_rate > 0.0, (
        f"TC55 setup: the crosser must be RECEDING from the robot at t=0 (closing "
        f"rate must be > 0 so an instantaneous-course gate would drop it), got "
        f"{closing_rate:.4f}"
    )
    crosser_groups = predict_blocked_cells(
        [crosser], planned_path, robot_xy, grid, inflation, horizon_steps, PREDICT_DT,
        geometry="capsule", exclusion_radius=inflation, corridor_half_width=inflation,
    )
    assert len(crosser_groups) == 1, (
        f"TC55: the divergent-now-collide-later crosser must be stamped by the "
        f"geometric gate, got {len(crosser_groups)} group(s)"
    )
    key, cells = crosser_groups[0]
    assert isinstance(key.ttc_steps, int) and 1 <= key.ttc_steps <= horizon_steps, (
        f"TC55: the crosser must carry a finite time-to-conflict in [1, {horizon_steps}], "
        f"got {key.ttc_steps}"
    )
    assert cells, "TC55: a gated crosser must stamp at least one cell"

    # Receding: directly away from the corridor (above the path, moving further up).
    # Its footprint never reaches the corridor, so the gate drops it.
    receding = Track(id=2, x=9.0, y=12.0, vx=0.0, vy=3.0, radius=0.3)
    receding_groups = predict_blocked_cells(
        [receding], planned_path, robot_xy, grid, inflation, horizon_steps, PREDICT_DT,
        geometry="capsule", exclusion_radius=inflation, corridor_half_width=inflation,
    )
    assert receding_groups == [], (
        f"TC55: a clearly receding track must NOT be stamped, got "
        f"{len(receding_groups)} group(s)"
    )


def tc56(yaml_path: str, seed: int) -> None:
    """Settle-time bounded peel: a map-sealing stamp is peeled farthest-future-first.

    Drives the oracle controller's settle-time fail-open peel directly (after an
    in-process reset() against the real world YAML — no irsim, no subprocess; mirrors
    TC46). The re-architected peel lives in ``_settle_and_extract``, not a per-tick
    ``_peel``: the FULL predicted stamp is committed into ``self._cells`` first
    (mimicking act()'s fold -> diff -> update_cells commit), then ``_settle_and_extract``
    settles with that full stamp, finds the grid sealed (the base settle returns None
    because ``g(start)`` is infinite), and peels the farthest-future (least-imminent)
    group — un-stamping its cells and re-settling — until a path re-exists. The peel
    receives two threat-ordered groups: a tiny, harmless MOST-IMMINENT group (smallest
    TTC) and a farthest-future group that is a full grid-spanning wall whose stamp SEALS
    the robot from the goal. The wall must be peeled (un-stamped) and the most-imminent
    group retained, with ``self._cells`` mutated in place and never rebound.

    The imminent group also SHARES one wall-column cell with the wall group, so the
    peel must keep that shared cell stamped (a retained group still needs it) while
    un-stamping the rest of the wall -- exercising ``_unstamp_group``'s subtlest
    property (a cell present in both a dropped and a kept group survives the peel).
    """
    _ensure_repo_root_on_path()
    from manual_astar import world_to_grid  # type: ignore[import-not-found]
    from planners import build_controller  # type: ignore[import-not-found]
    from planners._predict import ThreatKey  # type: ignore[import-not-found]
    from planners.d_star_lite import DStarLiteController  # type: ignore[import-not-found]

    controller = build_controller("d_star_lite_oracle", None, predict_horizon=10)
    raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    start = raw["robot"]["state"]
    state0 = np.array(
        [float(start[0]), float(start[1]), float(start[2]) if len(start) > 2 else 0.0],
        dtype=np.float64,
    )
    nan_lidar = np.full((LIDAR_BEAM_COUNT,), np.nan, dtype=np.float64)
    controller.reset(yaml_path, (), nan_lidar, state0)

    grid = controller._grid
    robot_cell = world_to_grid(state0[:2], grid)
    goal_cell = world_to_grid(controller._goal_xy, grid)
    rows = controller._cells.shape[0]

    # Most-imminent group (ttc=1): a single harmless cell off the robot's route.
    imminent_cell = (robot_cell[0] + 5, robot_cell[1] + 3)
    # Farthest-future group (ttc=9): a full vertical wall between start and goal. With
    # no corner-cutting (the search forbids diagonal squeezes), a complete column seals
    # the grid.
    wall_col = (robot_cell[1] + goal_cell[1]) // 2
    wall_cells = [(r, wall_col) for r in range(rows)]

    # Shared-cell-across-groups case: pick a stamp-only cell (currently False in the
    # static grid) in the wall column and place it in BOTH the imminent (kept) group
    # and the wall (dropped) group. The peel must NOT un-stamp this cell, because a
    # retained group still needs it -- this exercises _unstamp_group's subtlest
    # property (a cell present in both a dropped and a kept group survives the peel).
    shared_row = next(r for r in range(rows) if not controller._cells[r, wall_col])
    shared_cell = (shared_row, wall_col)

    # Threat order: most-imminent first (ttc=1), farthest-future last (ttc=9). The peel
    # drops from the END (least-imminent first). The imminent group also carries the
    # shared wall cell so it must stay stamped after the wall group is dropped.
    groups = [
        (ThreatKey(1, 1), [imminent_cell, shared_cell]),
        (ThreatKey(9, 2), wall_cells),
    ]

    # Mimic act()'s per-tick commit: store the un-stamped fold and the threat-ordered
    # groups, then OR the FULL stamp (both groups) into self._cells IN PLACE, reporting
    # the newly-flipped cells through move_start + update_cells so the search sees the
    # seal. self._cells is mutated, never rebound (grid-ownership invariant).
    controller._last_fold = controller._cells.copy()
    controller._pending_groups = groups
    cells_obj = controller._cells  # capture identity to prove no rebind below.
    changed: list[tuple[int, int]] = []
    for _key, group_cells in groups:
        for cell in group_cells:
            row, col = cell
            if not controller._cells[row, col]:
                controller._cells[row, col] = True
                changed.append(cell)
    controller._search.move_start(robot_cell)
    controller._search.update_cells(sorted(changed))

    # The full stamp must seal the grid: the BASE settle (bypassing the peel override)
    # returns None exactly when g(start) is infinite. If this passes vacuously the wall
    # did not actually disconnect start from goal and the peel test would be meaningless.
    assert DStarLiteController._settle_and_extract(controller, state0[:2]) is None, (
        "TC56 setup: the full stamp must seal the grid (the base settle must find no "
        "path before the peel runs)"
    )

    # The override peel: drop the farthest-future group until a path re-exists.
    result = controller._settle_and_extract(state0[:2])
    assert result is not None, (
        "TC56: the settle-time peel must restore a path by dropping the sealing wall"
    )
    # The farthest-future sealing wall is peeled away: every stamp-only wall cell (one
    # that was False in the un-stamped fold) is restored to False -- EXCEPT the cell
    # shared with the retained imminent group, which must survive (asserted below).
    assert all(
        not controller._cells[r, wall_col]
        for r in range(rows)
        if not controller._last_fold[r, wall_col] and r != shared_row
    ), (
        "TC56: the farthest-future sealing wall must be peeled (un-stamped) so a path "
        "to the goal re-exists"
    )
    # The most-imminent group is retained (still stamped after the peel).
    assert bool(controller._cells[imminent_cell]), (
        "TC56: the most-imminent group must be retained after the peel"
    )
    # Shared-cell-across-groups: the wall cell that ALSO belongs to the retained
    # imminent group must NOT be un-stamped -- a dropped group may never erase a cell a
    # kept group still needs. (The rest of the wall column IS peeled, asserted above.)
    assert bool(controller._cells[shared_cell]) is True, (
        "TC56: a cell shared between the dropped wall group and the retained imminent "
        "group must survive the peel (not be un-stamped)"
    )
    # Grid-ownership invariant: self._cells is mutated in place, never rebound.
    assert controller._cells is cells_obj, (
        "TC56: the peel must mutate self._cells in place, never rebind it"
    )


def tc56b(yaml_path: str, seed: int) -> None:
    """Genuine dead-end: settle-time peel-to-zero stays None; act() does NOT raise (AC6).

    Two complementary checks against the oracle controller (in-process reset() on the
    real world YAML; no irsim, no subprocess):

    (a) A genuine static dead-end — a full vertical wall that is part of the un-stamped
        FOLD (not a stamp). ``_unstamp_group`` refuses to erase a real fold obstacle, so
        even peeling every predicted group leaves the grid sealed and
        ``_settle_and_extract`` returns None: the controller keeps its last valid
        follower instead of raising. Built from a crafted MID-episode dead-end fold,
        avoiding arena_no_path.yaml whose sealed START would raise at reset().

    (b) Full-pipeline non-raise through act() under a dense stationary prediction. Because
        act() re-folds memorylessly, the pure static fold stays solvable, so a prediction
        seal is always peelable — this exercises the peel-SUCCESS path and proves act()
        returns a valid finite (2,1) action with a follower. The genuine-dead-end (None)
        assertion lives in (a); (b) does NOT assert follower identity (a successful peel
        may legitimately rebuild it). A FRESH controller is used for each check so (a)'s
        mutated grid never leaks into (b).
    """
    _ensure_repo_root_on_path()
    from arena.dynamic import DynamicObstacleState  # type: ignore[import-not-found]
    from manual_astar import world_to_grid  # type: ignore[import-not-found]
    from planners import build_controller  # type: ignore[import-not-found]
    from planners._predict import ThreatKey  # type: ignore[import-not-found]

    raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    start = raw["robot"]["state"]
    state0 = np.array(
        [float(start[0]), float(start[1]), float(start[2]) if len(start) > 2 else 0.0],
        dtype=np.float64,
    )
    nan_lidar = np.full((LIDAR_BEAM_COUNT,), np.nan, dtype=np.float64)

    # --- (a) Direct dead-end: the wall is a REAL fold obstacle, so the peel cannot
    #         erase it and the grid stays sealed even after peeling to zero. ---
    controller = build_controller("d_star_lite_oracle", None, predict_horizon=10)
    controller.reset(yaml_path, (), nan_lidar, state0)

    grid = controller._grid
    robot_cell = world_to_grid(state0[:2], grid)
    goal_cell = world_to_grid(controller._goal_xy, grid)
    rows = controller._cells.shape[0]
    wall_col = (robot_cell[1] + goal_cell[1]) // 2

    # Seal the grid with a full column that is part of the FOLD: mutate self._cells and
    # snapshot it into _last_fold so the wall is NOT stamp-only — _unstamp_group refuses
    # to erase a real fold obstacle (self._last_fold[row, col] is True there).
    for r in range(rows):
        controller._cells[r, wall_col] = True
    controller._last_fold = controller._cells.copy()
    # A single harmless off-route pending group: the peel will try to drop it, but it is
    # not the wall, so the wall survives and the grid stays sealed at zero stamp.
    controller._pending_groups = [
        (ThreatKey(1, 1), [(robot_cell[0] + 5, robot_cell[1] + 3)])
    ]
    controller._search.move_start(robot_cell)
    controller._search.update_cells([(r, wall_col) for r in range(rows)])

    assert controller._settle_and_extract(state0[:2]) is None, (
        "TC56b(a): a genuine fold dead-end (a real-obstacle wall) must leave "
        "_settle_and_extract returning None even after peeling every predicted group"
    )

    # --- (b) Full pipeline: act() under a dense stationary prediction must not raise.
    #         A FRESH controller, so (a)'s sealed grid never leaks in. ---
    controller_b = build_controller("d_star_lite_oracle", None, predict_horizon=10)
    controller_b.reset(yaml_path, (), nan_lidar, state0)
    assert controller_b._follower is not None, (
        "TC56b(b) setup: reset() must build a follower"
    )

    dense = tuple(
        DynamicObstacleState(id=i, x=24.0, y=float(2 + i), vx=0.0, vy=0.0, radius=0.3)
        for i in range(20)
    )
    controller_b.observe_truth(dense)
    action = controller_b.act(state0, nan_lidar)
    assert action.shape == (2, 1), (
        f"TC56b(b): act() must return a (2,1) action under a dense prediction, got "
        f"shape {action.shape}"
    )
    assert bool(np.all(np.isfinite(action))), (
        "TC56b(b): act() must return a finite action under a dense prediction"
    )
    assert controller_b._follower is not None, (
        "TC56b(b): act() must keep a valid follower (the peelable seal re-solves)"
    )


def tc57(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """d_star_lite_oracle_h0 trace == plain d_star_lite trace (byte-identical, AC2).

    Runs both algorithms on the SAME seed/world through the runner subprocess, in BOTH
    --no-traffic and traffic-on regimes (4 runs total), into a temp results dir. The
    zero-horizon oracle stamps nothing, so its trace must be byte-identical to plain
    d_star_lite in each regime — proving --predict-horizon 0 is a true no-op baseline.
    """
    repo_root = _ensure_repo_root_on_path()
    seed_value = "57"
    world_stem = Path(yaml_path).stem

    def _run(algorithm: str, td: str, traffic_flag: str, extra: list[str]) -> None:
        r = subprocess.run(
            [
                sys.executable, "-m", "runners.run_episode",
                "--algorithm", algorithm,
                "--seed", seed_value,
                "--world", yaml_path,
                traffic_flag,
                "--results-dir", td,
                *extra,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert r.returncode == 0, (
            f"TC57 {algorithm} ({traffic_flag}) runner exit {r.returncode}; "
            f"stderr={r.stderr[-400:]}"
        )

    for traffic_flag in ("--no-traffic", "--traffic"):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            _run("d_star_lite", td, traffic_flag, [])
            _run("d_star_lite_oracle", td, traffic_flag, ["--predict-horizon", "0"])

            plain_jsonl = (
                Path(td) / world_stem / "d_star_lite" / f"{seed_value}.trace.jsonl"
            )
            oracle_jsonl = (
                Path(td) / world_stem / "d_star_lite_oracle_h0" / f"{seed_value}.trace.jsonl"
            )
            assert plain_jsonl.exists(), (
                f"TC57 ({traffic_flag}): plain d_star_lite trace missing at {plain_jsonl}"
            )
            assert oracle_jsonl.exists(), (
                f"TC57 ({traffic_flag}): oracle_h0 trace missing at {oracle_jsonl} "
                f"(label must be 'd_star_lite_oracle_h0')"
            )
            assert filecmp.cmp(str(plain_jsonl), str(oracle_jsonl), shallow=False), (
                f"TC57 ({traffic_flag}): d_star_lite_oracle_h0 trace differs from plain "
                f"d_star_lite — zero-horizon stamping is not a true no-op (AC2 broken)"
            )


def tc58(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """d_star_lite_oracle traffic-on e2e + determinism (AC3).

    Runs d_star_lite_oracle --predict-horizon 10 traffic-on to a terminal state, asserts
    every trace line carries the 8-key schema (incl. dynamic_obstacles_sha256), and that
    two same-seed runs produce byte-identical trace JSONL. NOTE: the oracle peel makes
    this the slowest predictive TC; kept to one horizon (10) and one seed pair.

    PERFORMANCE GATE: this case is also a de-facto guard on the oracle's per-tick
    cost. The shipped PredictiveDStarLiteController runs NO per-tick reachability
    probe: the per-tick hook only stamps the predicted footprint onto the fold, and
    reachability is answered at settle-time by _settle_and_extract reusing the D*
    Lite search's own g(start) values (the settle-time fail-open peel), which fires
    only when the follower finishes or its committed segment is blocked. That keeps
    per-tick cost near baseline (~16 ms/tick), so a full non-zero-horizon traffic
    episode terminates well within the 600 s per-run wall. The byte-for-byte
    zero-horizon no-op (AC2) is independently proven fast by TC57.
    """
    repo_root = _ensure_repo_root_on_path()
    seed_value = "58"
    horizon = "10"
    world_stem = Path(yaml_path).stem
    cmd = [
        sys.executable, "-m", "runners.run_episode",
        "--algorithm", "d_star_lite_oracle",
        "--predict-horizon", horizon,
        "--seed", seed_value,
        "--world", yaml_path,
        "--traffic",  # default; stated explicitly
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_a, \
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_b:
        for td in (td_a, td_b):
            r = subprocess.run(
                [*cmd, "--results-dir", td],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=600,
            )
            assert r.returncode == 0, (
                f"TC58 oracle traffic runner exit {r.returncode}; stderr={r.stderr[-400:]}"
            )

        out_a = Path(td_a) / world_stem / f"d_star_lite_oracle_h{horizon}"
        out_b = Path(td_b) / world_stem / f"d_star_lite_oracle_h{horizon}"
        json_a = out_a / f"{seed_value}.json"
        jsonl_a = out_a / f"{seed_value}.trace.jsonl"
        jsonl_b = out_b / f"{seed_value}.trace.jsonl"
        assert json_a.exists(), (
            f"TC58: metrics JSON missing at {json_a} (label must be "
            f"'d_star_lite_oracle_h{horizon}')"
        )
        assert jsonl_a.exists() and jsonl_b.exists(), (
            f"TC58: trace JSONLs missing: a={jsonl_a.exists()}, b={jsonl_b.exists()}"
        )

        # The episode RAN to completion (no runner fault): t=0 planning succeeded.
        metrics = json.loads(json_a.read_text(encoding="utf-8"))
        assert metrics["planner_error"] is None, (
            f"TC58: d_star_lite_oracle must plan successfully at t=0; "
            f"planner_error={metrics['planner_error']}"
        )

        lines = jsonl_a.read_text(encoding="utf-8").splitlines()
        assert lines, "TC58: oracle traffic trace JSONL is empty"
        for idx, raw in enumerate(lines):
            rec = json.loads(raw)
            assert isinstance(rec, dict), f"TC58: trace line {idx} is not an object"
            assert "dynamic_obstacles_sha256" in rec, (
                f"TC58: trace line {idx} missing dynamic_obstacles_sha256 with traffic "
                f"on; keys={sorted(rec)}"
            )
            assert len(rec) == 8, (
                f"TC58: trace line {idx} must have 8 keys with traffic on, got "
                f"{len(rec)}: {sorted(rec)}"
            )

        assert filecmp.cmp(str(jsonl_a), str(jsonl_b), shallow=False), (
            "TC58: two same-seed oracle traffic runs produced differing trace JSONL; "
            "predictive determinism through the runner is broken"
        )


def tc59(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — subprocess exit codes
    """--predict-horizon validation: required for the predict family, rejected elsewhere.

    Asserts (via subprocess exit codes, mirroring TC-CLI/TC37) that --predict-horizon is
    REQUIRED for d_star_lite_oracle (omitting it -> exit 2), REJECTED for a non-predict
    family (a_star_once --predict-horizon 5 -> exit 2), and that --replan-k is REJECTED
    for d_star_lite_oracle (-> exit 2); each rejected run writes NO <seed>.json. A valid
    oracle run (horizon 0, the fast no-op path) writes to the d_star_lite_oracle_h0
    label dir, and algorithm_label folds a non-zero horizon into `_h<steps>` (pure).
    """
    repo_root = _ensure_repo_root_on_path()
    seed_value = "59"
    world_stem = Path(yaml_path).stem

    def _run(extra: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            r = subprocess.run(
                [
                    sys.executable, "-m", "runners.run_episode",
                    "--algorithm", extra[0],
                    "--seed", seed_value,
                    "--world", yaml_path,
                    "--no-traffic",
                    "--results-dir", td,
                    *extra[1:],
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            # Capture whether any <seed>.json leaked, while td is still alive.
            r.stray_json = list(Path(td).rglob(f"{seed_value}.json"))  # type: ignore[attr-defined]
            return r

    # (a) predict family WITHOUT --predict-horizon -> exit 2, no JSON.
    r_missing = _run(["d_star_lite_oracle"], timeout=120)
    assert r_missing.returncode == 2, (
        f"TC59: d_star_lite_oracle without --predict-horizon must exit 2, got "
        f"{r_missing.returncode}; stderr={r_missing.stderr[-400:]}"
    )
    assert not r_missing.stray_json, (  # type: ignore[attr-defined]
        f"TC59: a rejected run must write no <seed>.json, found {r_missing.stray_json}"  # type: ignore[attr-defined]
    )

    # (b) non-predict family WITH --predict-horizon -> exit 2, no JSON.
    r_forbidden = _run(["a_star_once", "--predict-horizon", "5"], timeout=120)
    assert r_forbidden.returncode == 2, (
        f"TC59: a_star_once with --predict-horizon must exit 2, got "
        f"{r_forbidden.returncode}; stderr={r_forbidden.stderr[-400:]}"
    )
    assert not r_forbidden.stray_json, (  # type: ignore[attr-defined]
        f"TC59: a rejected run must write no <seed>.json, found {r_forbidden.stray_json}"  # type: ignore[attr-defined]
    )

    # (c) predict family WITH --replan-k -> exit 2, no JSON.
    r_replan = _run(
        ["d_star_lite_oracle", "--predict-horizon", "10", "--replan-k", "5"],
        timeout=120,
    )
    assert r_replan.returncode == 2, (
        f"TC59: d_star_lite_oracle with --replan-k must exit 2, got "
        f"{r_replan.returncode}; stderr={r_replan.stderr[-400:]}"
    )
    assert not r_replan.stray_json, (  # type: ignore[attr-defined]
        f"TC59: a rejected run must write no <seed>.json, found {r_replan.stray_json}"  # type: ignore[attr-defined]
    )

    # (d) a valid oracle run writes to the d_star_lite_oracle_h<steps> label dir.
    # Use --predict-horizon 0 here: it is the true no-op fast path (the predictive
    # hook returns [] before any tracker/peel, so this run is as fast as plain
    # d_star_lite, ~80 s) yet still lands in the `_h0` label dir, proving the
    # `_h<steps>` naming end-to-end through the runner. A NON-ZERO horizon run is
    # avoided here on purpose — the oracle's per-tick reachability peel makes a full
    # episode drive far too slow for a `--check` case; the non-zero label folding is
    # covered purely below and the non-zero traffic e2e is TC58's job.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        r_ok = subprocess.run(
            [
                sys.executable, "-m", "runners.run_episode",
                "--algorithm", "d_star_lite_oracle",
                "--predict-horizon", "0",
                "--seed", seed_value,
                "--world", yaml_path,
                "--no-traffic",
                "--results-dir", td,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert r_ok.returncode == 0, (
            f"TC59: a valid oracle run must exit 0, got {r_ok.returncode}; "
            f"stderr={r_ok.stderr[-400:]}"
        )
        json_path = (
            Path(td) / world_stem / "d_star_lite_oracle_h0" / f"{seed_value}.json"
        )
        assert json_path.exists(), (
            f"TC59: a valid oracle run must write to the d_star_lite_oracle_h0 label "
            f"dir; missing {json_path}"
        )

    # (e) PURE: a NON-ZERO horizon folds into the `_h<steps>` label (no episode).
    # algorithm_label owns the label folding; assert it directly so the non-zero
    # naming is covered without a slow drive.
    from planners import algorithm_label  # type: ignore[import-not-found]
    assert algorithm_label("d_star_lite_oracle", None, 10) == "d_star_lite_oracle_h10", (
        "TC59: algorithm_label must fold a non-zero --predict-horizon into "
        "'d_star_lite_oracle_h10'"
    )
    assert algorithm_label("d_star_lite_oracle", None, 0) == "d_star_lite_oracle_h0", (
        "TC59: algorithm_label must fold --predict-horizon 0 into "
        "'d_star_lite_oracle_h0'"
    )


def tc60(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — drives a short in-process Arena
    """Truth seam + tick alignment: EpisodeInfo.dynamic_obstacles + observe_truth (AC10).

    1) EpisodeInfo.dynamic_obstacles is () when traffic is off / pre-reset, and a length-20
       tuple of DynamicObstacleState when traffic is on post-reset.
    2) Tick alignment: driving an Arena (traffic on) a few steps as the runner does, the
       snapshot the oracle observes before each act(state, lidar) equals the
       info.dynamic_obstacles from the SAME reset()/step() call that produced that
       state/lidar (no off-by-one).
    3) A non-oracle controller (plain d_star_lite) never has observe_truth called: its
       wants_truth flag is falsey.
    """
    _ensure_repo_root_on_path()
    from arena.dynamic import (  # type: ignore[import-not-found]
        DynamicObstacleState,
        TARGET_POPULATION,
    )
    from planners import build_controller  # type: ignore[import-not-found]

    # --- (1a) traffic OFF, pre-reset and post-reset: dynamic_obstacles is (). ---
    arena_off = Arena(yaml_path, seed=3, traffic=False)
    try:
        _, _, info_off = arena_off.reset()
        assert info_off.dynamic_obstacles == (), (
            f"TC60: traffic-off post-reset dynamic_obstacles must be (), got "
            f"{info_off.dynamic_obstacles!r}"
        )
    finally:
        arena_off.close()

    # --- (3) wants_truth flags: oracle opts in, plain D* Lite does not. ---
    oracle = build_controller("d_star_lite_oracle", None, predict_horizon=5)
    plain = build_controller("d_star_lite", None)
    assert getattr(oracle, "wants_truth", False), (
        "TC60: d_star_lite_oracle must set wants_truth=True (opts into observe_truth)"
    )
    assert not getattr(plain, "wants_truth", False), (
        "TC60: plain d_star_lite must have a falsey wants_truth (observe_truth is "
        "never called for it)"
    )

    # --- (1b) traffic ON: post-reset dynamic_obstacles is a length-20 tuple of state. ---
    arena_on = Arena(yaml_path, seed=3, traffic=True)
    try:
        state0, lidar0, info0 = arena_on.reset()
        assert isinstance(info0.dynamic_obstacles, tuple), (
            "TC60: traffic-on dynamic_obstacles must be a tuple"
        )
        assert len(info0.dynamic_obstacles) == TARGET_POPULATION, (
            f"TC60: traffic-on dynamic_obstacles must have {TARGET_POPULATION} entries, "
            f"got {len(info0.dynamic_obstacles)}"
        )
        for entry in info0.dynamic_obstacles:
            assert isinstance(entry, DynamicObstacleState), (
                f"TC60: dynamic_obstacles entries must be DynamicObstacleState, got "
                f"{type(entry).__name__}"
            )

        # --- (2) tick alignment, mimicking the runner loop. ---
        oracle.reset(yaml_path, arena_on.initial_dynamic_snapshot, lidar0, state0)
        observed: list[tuple] = []
        original_observe = oracle.observe_truth

        def _spy(snapshot: tuple) -> None:
            observed.append(snapshot)
            original_observe(snapshot)

        oracle.observe_truth = _spy  # type: ignore[method-assign]

        state, lidar, current_info = state0, lidar0, info0
        steps_driven = 0
        for _ in range(4):
            # `current_info` is the EpisodeInfo from the SAME call that produced
            # `state`/`lidar` (reset() at t=0, the same step() thereafter).
            info_for_this_state = current_info
            oracle.observe_truth(current_info.dynamic_obstacles)
            assert observed[-1] == info_for_this_state.dynamic_obstacles, (
                f"TC60: tick misalignment at step {steps_driven} — the snapshot the "
                f"oracle observed is not the one tick-aligned with its state/lidar"
            )
            action = oracle.act(state, lidar)
            state, lidar, done, current_info = arena_on.step(action)
            steps_driven += 1
            if done:
                break
        assert steps_driven >= 2, (
            f"TC60: the alignment drive must cover at least 2 steps, got {steps_driven}"
        )
    finally:
        arena_on.close()


def tc61(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure import + set algebra
    """run_all tolerates experimental keys (AC8).

    Importing planners + runners.run_all must not raise, and run_all's canonical-set
    assertion must hold as set(_CANONICAL_ORDER) == set(ALGORITHMS) - EXPERIMENTAL_KEYS
    (the experimental d_star_lite_oracle key is carved out of the canonical study set;
    d_star_lite_predictive is now canonical).
    """
    _ensure_repo_root_on_path()
    import planners  # type: ignore[import-not-found]
    from planners._grid import EXPERIMENTAL_KEYS  # type: ignore[import-not-found]
    import runners.run_all as run_all  # type: ignore[import-not-found]

    expected = set(planners.ALGORITHMS) - set(EXPERIMENTAL_KEYS)
    canonical = set(run_all._CANONICAL_ORDER)
    assert canonical == expected, (
        f"TC61: run_all._CANONICAL_ORDER must equal ALGORITHMS minus EXPERIMENTAL_KEYS; "
        f"missing={expected - canonical}, extra={canonical - expected}"
    )
    # The experimental keys must be excluded from the canonical set (carve-out holds).
    assert not (canonical & set(EXPERIMENTAL_KEYS)), (
        f"TC61: experimental keys must not appear in _CANONICAL_ORDER, found "
        f"{canonical & set(EXPERIMENTAL_KEYS)}"
    )


def tc62(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — subprocess selfcheck
    """plot_horizon_sweep --selfcheck passes with no irsim (AC9).

    Runs `python -m runners.plot_horizon_sweep --selfcheck` as a subprocess (its
    synthetic-fixture suite builds everything in a TemporaryDirectory — no irsim, no
    real episodes) and asserts exit code 0.
    """
    repo_root = _ensure_repo_root_on_path()
    r = subprocess.run(
        [sys.executable, "-m", "runners.plot_horizon_sweep", "--selfcheck"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, (
        f"TC62: plot_horizon_sweep --selfcheck must exit 0, got {r.returncode}; "
        f"stdout={r.stdout[-400:]} stderr={r.stderr[-400:]}"
    )


def tc63(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """d_star_lite_predictive (lidar) traffic-on e2e + determinism.

    The lidar-fed, Mission-faithful predictive variant: it estimates obstacle
    velocities from frame-to-frame lidar clustering (no truth seam — wants_truth=False)
    and stamps a WIDENING cone (geometry="cone") to absorb that estimation noise.
    Runs d_star_lite_predictive --predict-horizon 10 traffic-on to a terminal state,
    asserts every trace line carries the 8-key schema (incl. dynamic_obstacles_sha256),
    and that two same-seed runs produce byte-identical trace JSONL. NOTE: kept to one
    horizon (10) and one seed pair (mirrors TC58's oracle e2e).

    PERFORMANCE GATE: like the oracle (TC58), the lidar variant shares the settle-time
    fail-open peel — no per-tick reachability probe. The per-tick hook only runs the
    LidarTracker + cone stamp; reachability is answered at settle-time by
    _settle_and_extract reusing the D* Lite search's own g(start) values, which fires
    only when the follower finishes or its committed segment is blocked. That keeps
    per-tick cost near baseline, so a full non-zero-horizon traffic episode terminates
    well within the 600 s per-run wall.
    """
    repo_root = _ensure_repo_root_on_path()
    seed_value = "63"
    horizon = "10"
    world_stem = Path(yaml_path).stem
    cmd = [
        sys.executable, "-m", "runners.run_episode",
        "--algorithm", "d_star_lite_predictive",
        "--predict-horizon", horizon,
        "--seed", seed_value,
        "--world", yaml_path,
        "--traffic",  # default; stated explicitly
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_a, \
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_b:
        for td in (td_a, td_b):
            r = subprocess.run(
                [*cmd, "--results-dir", td],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=600,
            )
            assert r.returncode == 0, (
                f"TC63 predictive traffic runner exit {r.returncode}; stderr={r.stderr[-400:]}"
            )

        out_a = Path(td_a) / world_stem / f"d_star_lite_predictive_h{horizon}"
        out_b = Path(td_b) / world_stem / f"d_star_lite_predictive_h{horizon}"
        json_a = out_a / f"{seed_value}.json"
        jsonl_a = out_a / f"{seed_value}.trace.jsonl"
        jsonl_b = out_b / f"{seed_value}.trace.jsonl"
        assert json_a.exists(), (
            f"TC63: metrics JSON missing at {json_a} (label must be "
            f"'d_star_lite_predictive_h{horizon}')"
        )
        assert jsonl_a.exists() and jsonl_b.exists(), (
            f"TC63: trace JSONLs missing: a={jsonl_a.exists()}, b={jsonl_b.exists()}"
        )

        # The episode RAN to completion (no runner fault): t=0 planning succeeded.
        metrics = json.loads(json_a.read_text(encoding="utf-8"))
        assert metrics["planner_error"] is None, (
            f"TC63: d_star_lite_predictive must plan successfully at t=0; "
            f"planner_error={metrics['planner_error']}"
        )

        lines = jsonl_a.read_text(encoding="utf-8").splitlines()
        assert lines, "TC63: predictive traffic trace JSONL is empty"
        for idx, raw in enumerate(lines):
            rec = json.loads(raw)
            assert isinstance(rec, dict), f"TC63: trace line {idx} is not an object"
            assert "dynamic_obstacles_sha256" in rec, (
                f"TC63: trace line {idx} missing dynamic_obstacles_sha256 with traffic "
                f"on; keys={sorted(rec)}"
            )
            assert len(rec) == 8, (
                f"TC63: trace line {idx} must have 8 keys with traffic on, got "
                f"{len(rec)}: {sorted(rec)}"
            )

        assert filecmp.cmp(str(jsonl_a), str(jsonl_b), shallow=False), (
            "TC63: two same-seed predictive traffic runs produced differing trace JSONL; "
            "predictive determinism through the runner is broken"
        )


def tc64(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure in-process; fixed synthetic fixture
    """LidarTracker determinism across a multi-frame cluster-count change.

    Drives a LidarTracker over a >=4-frame synthetic lidar fixture WHERE THE CLUSTER
    COUNT CHANGES between frames (one obstacle for 3 frames, a 2nd appears, the 1st
    leaves), twice on fresh instances. This exercises the association-stability hazard
    — the danger that a changing cluster population reorders or mis-associates tracks
    and silently desyncs the velocity estimate — not just a single 2-frame diff.

    In-process only (NO irsim, NO subprocess). The fixture is fully fixed (no RNG), so
    the case is self-deterministic regardless of the `seed` arg, which is ignored. The
    grid is all-free (80x80 cells at 0.5 m, offset (-10,-10)) so the static occupancy
    subtracts nothing — this case targets determinism + association, not static
    subtraction.

    Asserts:
      1. Determinism (binding): the two list[list[Track]] sequences are byte-identical
         via dataclasses.astuple comparison.
      2. The cluster count genuinely changed across frames (per-frame counts ==
         [1, 1, 1, 2, 1], an enter then a leave).
      3. The first frame has zero velocity (no prior) and a +x-moving obstacle yields a
         non-zero, correctly-signed velocity after the second frame (the estimator is
         actually frame-differencing, not returning zeros).
    """
    import math

    _ensure_repo_root_on_path()
    from manual_astar import OccupancyGrid  # type: ignore[import-not-found]
    from planners._predict import PREDICT_DT, LidarTracker, Track  # type: ignore[import-not-found]

    def make_grid() -> OccupancyGrid:
        """An all-free 40x40 m grid (offset (-10,-10)) so nothing is subtracted."""
        rows = cols = 80
        cells = np.zeros((rows, cols), dtype=bool)
        return OccupancyGrid(
            cells=cells, resolution=0.5, offset=np.array([-10.0, -10.0], dtype=float)
        )

    def ray_disk_range(
        bearing: float, center: tuple[float, float], radius: float
    ) -> float | None:
        """Nearest forward hit range of a beam from the origin against a disk."""
        dx, dy = math.cos(bearing), math.sin(bearing)
        cx, cy = center
        d_dot_c = dx * cx + dy * cy
        disc = d_dot_c * d_dot_c - (cx * cx + cy * cy - radius * radius)
        if disc < 0.0:
            return None
        t = d_dot_c - math.sqrt(disc)
        return t if t > 0.0 else None

    def synth_lidar(
        bearings: np.ndarray, disks: list[tuple[tuple[float, float], float]]
    ) -> np.ndarray:
        """Synthesize a scan (robot at origin, theta=0) hitting the given disks."""
        ranges = np.full(bearings.shape[0], np.nan, dtype=float)
        for i, bearing in enumerate(bearings):
            best: float | None = None
            for center, radius in disks:
                r = ray_disk_range(float(bearing), center, radius)
                if r is not None and (best is None or r < best):
                    best = r
            if best is not None:
                ranges[i] = best
        return ranges

    def run_sequence(
        bearings: np.ndarray,
        frames: list[list[tuple[tuple[float, float], float]]],
    ) -> list[list[Track]]:
        grid = make_grid()
        tracker = LidarTracker(grid, bearings)
        state = np.array([0.0, 0.0, 0.0], dtype=float)
        out: list[list[Track]] = []
        for disks in frames:
            lidar = synth_lidar(bearings, disks)
            tracks = tracker.update(snapshot=(), state=state, lidar=lidar, dt=PREDICT_DT)
            out.append(tracks)
        return out

    # Slightly under 2*pi (math.pi * 0.999) so WrapTo2Pi does not collapse the scan to
    # a single ray (gotcha-lidar-wraptopi-collapses-2pi).
    bearings = np.linspace(-math.pi, math.pi * 0.999, 180)
    radius = 0.4

    # Obstacle A is on-axis and moves +x in 0.15 m straight-line steps (=> ~1.5 m/s,
    # vy~0 by symmetry of the visible near arc). Obstacle B appears in frame 4; A
    # leaves in frame 5, so the cluster count walks 1 -> 1 -> 1 -> 2 -> 1.
    a0 = (5.0, 0.0)
    a1 = (5.15, 0.0)
    a2 = (5.30, 0.0)
    a3 = (5.45, 0.0)
    b = (3.0, -3.0)

    frames: list[list[tuple[tuple[float, float], float]]] = [
        [(a0, radius)],                 # frame 1: 1 cluster
        [(a1, radius)],                 # frame 2: 1 cluster
        [(a2, radius)],                 # frame 3: 1 cluster
        [(a3, radius), (b, radius)],    # frame 4: 2 clusters (B appears)
        [(b, radius)],                  # frame 5: 1 cluster (A leaves)
    ]

    seq1 = run_sequence(bearings, frames)
    seq2 = run_sequence(bearings, frames)

    # 1. Determinism (binding): two fresh-instance runs byte-identical.
    tup1 = [[dataclasses.astuple(t) for t in frame] for frame in seq1]
    tup2 = [[dataclasses.astuple(t) for t in frame] for frame in seq2]
    assert tup1 == tup2, (
        "TC64: two fresh LidarTracker runs over the same 5-frame fixture diverged; "
        "the estimator/association is non-deterministic"
    )

    # 2. The cluster count actually changed across frames (enter/leave exercised, not a
    #    static count) — fail loud with the observed counts if not.
    counts = [len(frame) for frame in seq1]
    assert counts == [1, 1, 1, 2, 1], (
        f"TC64: fixture did not exercise a cluster-count change; per-frame track counts "
        f"were {counts}, expected [1, 1, 1, 2, 1]"
    )

    # 3a. The first frame has no prior, so its velocity estimate must be exactly zero.
    frame1 = seq1[0]
    assert len(frame1) == 1, f"TC64: expected 1 track in frame 1, got {len(frame1)}"
    assert frame1[0].vx == 0.0 and frame1[0].vy == 0.0, (
        f"TC64: first-frame velocity must be (0, 0) with no prior, got "
        f"({frame1[0].vx}, {frame1[0].vy})"
    )

    # 3b. After the obstacle moves +x, the estimate must be non-zero and correctly
    #     signed — proving the estimator is differencing, not returning zeros.
    frame2 = seq1[1]
    assert len(frame2) == 1, f"TC64: expected 1 track in frame 2, got {len(frame2)}"
    track_a = frame2[0]
    assert track_a.vx > 0.5 and abs(track_a.vy) < 0.2, (
        f"TC64: a +x-moving obstacle must yield vx>0.5 and |vy|<0.2 (estimator is "
        f"frame-differencing), got vx={track_a.vx:.4f} vy={track_a.vy:.4f}"
    )


def tc65(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """Plain dwa unchanged; paper-only h0 == plain dwa; global h0 DIFFERS (AC1, AC2).

    dwa_predictive / dwa_predictive_oracle are now the paper+global (braking-
    inevitability + cost-to-go field) variants, so their h0 trace is NO LONGER a
    no-op against plain dwa — the global-guidance heading term is active even at
    horizon 0 (it stamps no tracks, but _heading_term still reads the cost-to-go
    field). That `!=` assertion moved to TCa. This case instead pins:

    1. AC1: plain `dwa` is byte-preserving under the T1 `_heading_term` /
       `state`-threading refactor. There is nothing to compare it against here
       other than reuse across regimes, so this is really an existence/shape
       smoke check; AC1's real guard is TCa/TCb/TCe not perturbing plain dwa's
       own trace file, which every OTHER dwa-vs-predictive TC in this family
       cross-checks by construction (they all run plain dwa fresh each time).
    2. AC2: the two PAPER-ONLY keys (`dwa_predictive_paper`,
       `dwa_predictive_paper_oracle`) at h0 delegate straight to vanilla DWA (no
       tracker update, no space-time layer, no global guidance), so their h0
       trace.jsonl MUST be byte-identical to plain dwa — in BOTH --no-traffic and
       traffic-on regimes.
    """
    repo_root = _ensure_repo_root_on_path()
    seed_value = "65"
    world_stem = Path(yaml_path).stem

    def _run(algorithm: str, td: str, traffic_flag: str, extra: list) -> None:
        r = subprocess.run(
            [
                sys.executable, "-m", "runners.run_episode",
                "--algorithm", algorithm, "--seed", seed_value, "--world", yaml_path,
                traffic_flag, "--results-dir", td, *extra,
            ],
            cwd=str(repo_root), capture_output=True, text=True, timeout=600,
        )
        assert r.returncode == 0, (
            f"TC65 {algorithm} ({traffic_flag}) exit {r.returncode}; stderr={r.stderr[-400:]}"
        )

    for traffic_flag in ("--no-traffic", "--traffic"):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            _run("dwa", td, traffic_flag, [])
            _run("dwa_predictive_paper_oracle", td, traffic_flag, ["--predict-horizon", "0"])
            _run("dwa_predictive_paper", td, traffic_flag, ["--predict-horizon", "0"])

            base = Path(td) / world_stem / "dwa" / f"{seed_value}.trace.jsonl"
            paper_oracle = (
                Path(td) / world_stem / "dwa_predictive_paper_oracle_h0"
                / f"{seed_value}.trace.jsonl"
            )
            paper_lidar = (
                Path(td) / world_stem / "dwa_predictive_paper_h0" / f"{seed_value}.trace.jsonl"
            )
            assert base.exists() and paper_oracle.exists() and paper_lidar.exists(), (
                f"TC65 ({traffic_flag}): a trace file is missing "
                f"(base={base.exists()} paper_oracle={paper_oracle.exists()} "
                f"paper_lidar={paper_lidar.exists()})"
            )
            assert filecmp.cmp(str(base), str(paper_oracle), shallow=False), (
                f"TC65 ({traffic_flag}): dwa_predictive_paper_oracle_h0 trace differs from "
                f"plain dwa — zero-horizon paper-only stamping is not a true no-op (AC2 broken)"
            )
            assert filecmp.cmp(str(base), str(paper_lidar), shallow=False), (
                f"TC65 ({traffic_flag}): dwa_predictive_paper_h0 trace differs from plain dwa "
                f"(AC2 broken)"
            )


def tc66(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — subprocess exit codes
    """--predict-horizon validation for the space-time DWA family (AC6).

    dwa_predictive / dwa_predictive_oracle REQUIRE --predict-horizon (omitting it ->
    exit 2); dwa REJECTS it (exit 2); --replan-k is REJECTED for dwa_predictive
    (exit 2). Each rejected run writes NO <seed>.json. The pure `_h<steps>` label
    fold is checked directly.
    """
    repo_root = _ensure_repo_root_on_path()
    seed_value = "66"

    def _run(extra: list) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            r = subprocess.run(
                [
                    sys.executable, "-m", "runners.run_episode",
                    "--algorithm", extra[0], "--seed", seed_value, "--world", yaml_path,
                    "--no-traffic", "--results-dir", td, *extra[1:],
                ],
                cwd=str(repo_root), capture_output=True, text=True, timeout=120,
            )
            r.stray_json = list(Path(td).rglob(f"{seed_value}.json"))  # type: ignore[attr-defined]
            return r

    checks = [
        (["dwa_predictive"], "dwa_predictive without --predict-horizon"),
        (["dwa_predictive_oracle"], "dwa_predictive_oracle without --predict-horizon"),
        (["dwa", "--predict-horizon", "10"], "dwa with --predict-horizon"),
        (["dwa_predictive", "--predict-horizon", "10", "--replan-k", "5"], "dwa_predictive with --replan-k"),
    ]
    for extra, why in checks:
        r = _run(extra)
        assert r.returncode == 2, f"TC66: {why} must exit 2, got {r.returncode}; stderr={r.stderr[-300:]}"
        assert not r.stray_json, f"TC66: {why} (rejected) must write no <seed>.json, found {r.stray_json}"  # type: ignore[attr-defined]

    from planners import algorithm_label  # type: ignore[import-not-found]
    assert algorithm_label("dwa_predictive", None, 10) == "dwa_predictive_h10", "TC66: label must fold _h10"
    assert algorithm_label("dwa_predictive_oracle", None, 0) == "dwa_predictive_oracle_h0", "TC66: label must fold _h0"


def tc67(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """dwa_predictive & dwa_predictive_oracle traffic-on e2e + determinism (AC3/AC5).

    Each variant at --predict-horizon 10, traffic on, runs to a terminal state with
    planner_error null (DWA reset never raises), every trace line carrying the 8-key
    schema (incl. dynamic_obstacles_sha256), and two same-seed runs byte-identical.
    Exercises the real space-time layer (the tracker + trajectory_conflict) through
    the runner, per variant.
    """
    repo_root = _ensure_repo_root_on_path()
    seed_value = "67"
    horizon = "10"
    world_stem = Path(yaml_path).stem

    for algorithm in ("dwa_predictive", "dwa_predictive_oracle"):
        label = f"{algorithm}_h{horizon}"
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_a, \
                tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_b:
            for td in (td_a, td_b):
                r = subprocess.run(
                    [
                        sys.executable, "-m", "runners.run_episode",
                        "--algorithm", algorithm, "--predict-horizon", horizon,
                        "--seed", seed_value, "--world", yaml_path, "--traffic",
                        "--results-dir", td,
                    ],
                    cwd=str(repo_root), capture_output=True, text=True, timeout=600,
                )
                assert r.returncode == 0, (
                    f"TC67 {algorithm} traffic runner exit {r.returncode}; stderr={r.stderr[-400:]}"
                )
            out_a = Path(td_a) / world_stem / label
            out_b = Path(td_b) / world_stem / label
            json_a = out_a / f"{seed_value}.json"
            jsonl_a = out_a / f"{seed_value}.trace.jsonl"
            jsonl_b = out_b / f"{seed_value}.trace.jsonl"
            assert json_a.exists() and jsonl_a.exists() and jsonl_b.exists(), (
                f"TC67 {algorithm}: output missing (label must be {label!r})"
            )
            metrics = json.loads(json_a.read_text(encoding="utf-8"))
            assert metrics["planner_error"] is None, (
                f"TC67 {algorithm}: DWA reset must not raise; planner_error={metrics['planner_error']}"
            )
            lines = jsonl_a.read_text(encoding="utf-8").splitlines()
            assert lines, f"TC67 {algorithm}: traffic trace JSONL is empty"
            for idx, raw in enumerate(lines):
                rec = json.loads(raw)
                assert "dynamic_obstacles_sha256" in rec and len(rec) == 8, (
                    f"TC67 {algorithm}: trace line {idx} must be 8-key with traffic on, got {sorted(rec)}"
                )
            assert filecmp.cmp(str(jsonl_a), str(jsonl_b), shallow=False), (
                f"TC67 {algorithm}: two same-seed traffic runs produced differing trace JSONL; "
                f"space-time DWA determinism through the runner is broken"
            )


def tc68(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure in-process, no irsim
    """trajectory_conflict pure space-time geometry + determinism (AC4).

    A head-on track within the horizon collides at the correct earliest step with a
    non-positive gap; a receding track does not; a horizon shorter than the meeting
    keeps a larger min gap than the full horizon; horizon 0 / empty tracks are a
    no-op (+inf gap); two calls on the same inputs are byte-identical. Pure (builds
    its own robot trajectory + tracks; no irsim, no subprocess) — mirrors TC53.
    """
    _ensure_repo_root_on_path()
    from planners._predict import Track, trajectory_conflict  # type: ignore[import-not-found]

    dt = 0.1
    # Robot marches +x at 1 m/s: step k (1-based) is at (0.1*k, 0), 12 steps.
    robot = np.array([[0.1 * k, 0.0] for k in range(1, 13)], dtype=float)
    head_on = [Track(id=1, x=1.5, y=0.0, vx=-1.0, vy=0.0, radius=0.3)]

    full = trajectory_conflict(robot, head_on, robot_radius=0.2, horizon_steps=10, dt=dt, margin=0.05)
    assert full.collides, "TC68: a head-on track within the horizon must collide"
    assert full.ttc_step is not None and 1 <= full.ttc_step <= 10, (
        f"TC68: ttc_step out of range: {full.ttc_step}"
    )
    assert full.min_gap <= 0.05, f"TC68: a colliding min_gap must be <= margin, got {full.min_gap}"

    # Horizon 3 is before the bodies meet (earliest collision is step 5): no collision.
    early = trajectory_conflict(robot, head_on, robot_radius=0.2, horizon_steps=3, dt=dt, margin=0.05)
    assert not early.collides, "TC68: horizon 3 is before the head-on meeting -> no collision yet"
    assert early.min_gap > full.min_gap, "TC68: the earlier horizon must keep a larger min gap"

    receding = [Track(id=2, x=3.0, y=3.0, vx=1.0, vy=1.0, radius=0.3)]
    away = trajectory_conflict(robot, receding, robot_radius=0.2, horizon_steps=10, dt=dt, margin=0.05)
    assert not away.collides and away.ttc_step is None, "TC68: a receding track must not collide"
    assert away.min_gap > 0.0, "TC68: a non-colliding track must have a positive gap"

    zero = trajectory_conflict(robot, head_on, robot_radius=0.2, horizon_steps=0, dt=dt, margin=0.05)
    assert not zero.collides and zero.min_gap == float("inf"), "TC68: horizon 0 must be a no-op"
    empty = trajectory_conflict(robot, [], robot_radius=0.2, horizon_steps=10, dt=dt, margin=0.05)
    assert not empty.collides and empty.min_gap == float("inf"), "TC68: empty tracks must be a no-op"

    again = trajectory_conflict(robot, head_on, robot_radius=0.2, horizon_steps=10, dt=dt, margin=0.05)
    assert (full.collides, full.ttc_step, full.min_gap) == (again.collides, again.ttc_step, again.min_gap), (
        "TC68: trajectory_conflict must be deterministic across identical calls"
    )


def tc69(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure in-process, no irsim
    """run_all canonical set = 13; the assertion tolerates the DWA oracle carve-out (AC7).

    Importing planners + run_all does not raise (the import-time
    set(_CANONICAL_ORDER) == set(ALGORITHMS) - EXPERIMENTAL_KEYS assertion holds),
    dwa_predictive is canonical, dwa_predictive_oracle is experimental (not
    canonical), and both DWA predict keys are in PREDICT_FAMILIES.
    """
    _ensure_repo_root_on_path()
    from planners import ALGORITHMS, EXPERIMENTAL_KEYS, PREDICT_FAMILIES  # type: ignore[import-not-found]
    from runners.run_all import _CANONICAL_ORDER, canonical_planner_set  # type: ignore[import-not-found]

    canonical = set(_CANONICAL_ORDER)
    assert canonical == set(ALGORITHMS) - EXPERIMENTAL_KEYS, (
        "TC69: _CANONICAL_ORDER must equal registry minus experimental keys"
    )
    assert len(canonical_planner_set()) == 13, (
        f"TC69: canonical set must be 13, got {len(canonical_planner_set())}"
    )
    assert "dwa_predictive" in canonical, "TC69: dwa_predictive must be canonical"
    assert "dwa_predictive_oracle" in EXPERIMENTAL_KEYS, "TC69: dwa_predictive_oracle must be experimental"
    assert "dwa_predictive_oracle" not in canonical, "TC69: the DWA oracle must not be canonical"
    assert {"dwa_predictive", "dwa_predictive_oracle"} <= PREDICT_FAMILIES, (
        "TC69: both DWA predict keys must be in PREDICT_FAMILIES"
    )


def tca(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """Paper+global h0 is deterministic AND != plain dwa (AC3).

    `dwa_predictive` and `dwa_predictive_oracle` (the paper+global keys) are now
    GLOBAL variants: their `_heading_term` override reads the static cost-to-go
    field even at horizon 0 (no tracks, but the field-guided heading term is
    still active), so their h0 trace must DIFFER from plain `dwa`. This is the
    `!=` half of the old TC65 assertion, moved here per the plan. For each of
    the two keys: two same-seed h0 runs are byte-identical to EACH OTHER
    (determinism), and NEITHER is byte-identical to plain dwa (field guidance
    active). Uses --no-traffic (cheaper, and traffic-on determinism is already
    covered at horizon 10 by TCb).
    """
    repo_root = _ensure_repo_root_on_path()
    seed_value = "165"
    world_stem = Path(yaml_path).stem

    def _run(algorithm: str, td: str, extra: list) -> None:
        r = subprocess.run(
            [
                sys.executable, "-m", "runners.run_episode",
                "--algorithm", algorithm, "--seed", seed_value, "--world", yaml_path,
                "--no-traffic", "--results-dir", td, *extra,
            ],
            cwd=str(repo_root), capture_output=True, text=True, timeout=600,
        )
        assert r.returncode == 0, (
            f"TCa {algorithm} exit {r.returncode}; stderr={r.stderr[-400:]}"
        )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        _run("dwa", td, [])
        base = Path(td) / world_stem / "dwa" / f"{seed_value}.trace.jsonl"
        assert base.exists(), f"TCa: plain dwa trace missing at {base}"

        for algorithm in ("dwa_predictive", "dwa_predictive_oracle"):
            label = f"{algorithm}_h0"
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_a, \
                    tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_b:
                for td_run in (td_a, td_b):
                    _run(algorithm, td_run, ["--predict-horizon", "0"])
                jsonl_a = Path(td_a) / world_stem / label / f"{seed_value}.trace.jsonl"
                jsonl_b = Path(td_b) / world_stem / label / f"{seed_value}.trace.jsonl"
                assert jsonl_a.exists() and jsonl_b.exists(), (
                    f"TCa {algorithm}: trace file missing (label must be {label!r})"
                )
                assert filecmp.cmp(str(jsonl_a), str(jsonl_b), shallow=False), (
                    f"TCa {algorithm}: two same-seed h0 runs diverged; global-guidance "
                    f"h0 must still be deterministic"
                )
                assert not filecmp.cmp(str(base), str(jsonl_a), shallow=False), (
                    f"TCa {algorithm}: h0 trace is byte-identical to plain dwa, but "
                    f"{algorithm} is the paper+global variant — the cost-to-go field "
                    f"guidance must be active even at horizon 0 (AC3 broken)"
                )


def tcb(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses its own internal seed
    """All four keys traffic-on e2e + determinism + 8-key schema at h10 (AC4).

    For dwa_predictive, dwa_predictive_oracle, dwa_predictive_paper, and
    dwa_predictive_paper_oracle at --predict-horizon 10, traffic on: each runs to
    a terminal state, two same-seed runs are byte-identical, and every trace
    line carries the 8-key schema (incl. dynamic_obstacles_sha256). Mirrors
    TC67 (which covers only the two global keys) extended to all four.
    """
    repo_root = _ensure_repo_root_on_path()
    seed_value = "166"
    horizon = "10"
    world_stem = Path(yaml_path).stem

    for algorithm in (
        "dwa_predictive",
        "dwa_predictive_oracle",
        "dwa_predictive_paper",
        "dwa_predictive_paper_oracle",
    ):
        label = f"{algorithm}_h{horizon}"
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_a, \
                tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td_b:
            for td in (td_a, td_b):
                r = subprocess.run(
                    [
                        sys.executable, "-m", "runners.run_episode",
                        "--algorithm", algorithm, "--predict-horizon", horizon,
                        "--seed", seed_value, "--world", yaml_path, "--traffic",
                        "--results-dir", td,
                    ],
                    cwd=str(repo_root), capture_output=True, text=True, timeout=600,
                )
                assert r.returncode == 0, (
                    f"TCb {algorithm} traffic runner exit {r.returncode}; "
                    f"stderr={r.stderr[-400:]}"
                )
            out_a = Path(td_a) / world_stem / label
            out_b = Path(td_b) / world_stem / label
            json_a = out_a / f"{seed_value}.json"
            jsonl_a = out_a / f"{seed_value}.trace.jsonl"
            jsonl_b = out_b / f"{seed_value}.trace.jsonl"
            assert json_a.exists() and jsonl_a.exists() and jsonl_b.exists(), (
                f"TCb {algorithm}: output missing (label must be {label!r})"
            )
            metrics = json.loads(json_a.read_text(encoding="utf-8"))
            assert metrics["planner_error"] is None, (
                f"TCb {algorithm}: DWA reset must not raise; "
                f"planner_error={metrics['planner_error']}"
            )
            lines = jsonl_a.read_text(encoding="utf-8").splitlines()
            assert lines, f"TCb {algorithm}: traffic trace JSONL is empty"
            for idx, raw in enumerate(lines):
                rec = json.loads(raw)
                assert isinstance(rec, dict), f"TCb {algorithm}: trace line {idx} is not an object"
                assert "dynamic_obstacles_sha256" in rec and len(rec) == 8, (
                    f"TCb {algorithm}: trace line {idx} must be 8-key with traffic on, "
                    f"got {sorted(rec)}"
                )
            assert filecmp.cmp(str(jsonl_a), str(jsonl_b), shallow=False), (
                f"TCb {algorithm}: two same-seed traffic runs produced differing trace "
                f"JSONL; determinism through the runner is broken"
            )


def tcc(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — subprocess exit codes
    """--predict-horizon required / --replan-k rejected for all four keys (AC5).

    Mirrors TC66 (which covers only the two global keys) extended to all four
    dwa_predictive* keys: omitting --predict-horizon exits 2 with no <seed>.json
    written; --replan-k is rejected (exit 2) for each. The pure `_h<steps>`
    label fold is checked directly for the two paper-only keys (the two global
    keys are already checked by TC66).
    """
    repo_root = _ensure_repo_root_on_path()
    seed_value = "167"

    def _run(extra: list) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            r = subprocess.run(
                [
                    sys.executable, "-m", "runners.run_episode",
                    "--algorithm", extra[0], "--seed", seed_value, "--world", yaml_path,
                    "--no-traffic", "--results-dir", td, *extra[1:],
                ],
                cwd=str(repo_root), capture_output=True, text=True, timeout=120,
            )
            r.stray_json = list(Path(td).rglob(f"{seed_value}.json"))  # type: ignore[attr-defined]
            return r

    predict_keys = (
        "dwa_predictive",
        "dwa_predictive_oracle",
        "dwa_predictive_paper",
        "dwa_predictive_paper_oracle",
    )
    for algorithm in predict_keys:
        r_missing = _run([algorithm])
        assert r_missing.returncode == 2, (
            f"TCc: {algorithm} without --predict-horizon must exit 2, got "
            f"{r_missing.returncode}; stderr={r_missing.stderr[-300:]}"
        )
        assert not r_missing.stray_json, (  # type: ignore[attr-defined]
            f"TCc: {algorithm} rejected run must write no <seed>.json, found "
            f"{r_missing.stray_json}"  # type: ignore[attr-defined]
        )

        r_replan = _run([algorithm, "--predict-horizon", "10", "--replan-k", "5"])
        assert r_replan.returncode == 2, (
            f"TCc: {algorithm} with --replan-k must exit 2, got "
            f"{r_replan.returncode}; stderr={r_replan.stderr[-300:]}"
        )
        assert not r_replan.stray_json, (  # type: ignore[attr-defined]
            f"TCc: {algorithm} --replan-k rejected run must write no <seed>.json, found "
            f"{r_replan.stray_json}"  # type: ignore[attr-defined]
        )

    from planners import algorithm_label  # type: ignore[import-not-found]
    for algorithm in predict_keys:
        assert algorithm_label(algorithm, None, 10) == f"{algorithm}_h10", (
            f"TCc: algorithm_label must fold --predict-horizon 10 into '{algorithm}_h10'"
        )


def tcd(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure in-process, no irsim
    """build_cost_to_go_field == A*-cost oracle on reachable cells; inf elsewhere (AC8).

    Builds arena_v1's static occupancy grid (no irsim, no subprocess) and the
    cost-to-go field rooted at the goal cell, then checks:
      1. On several reachable free cells, the field value equals the octile path
         cost a fresh `astar_search` FROM that cell TO the goal returns (the field
         is a Dijkstra-from-goal, so it must agree with A* run the other way).
      2. The field is `inf` at an occupied cell.
      3. The field is `inf` at a genuinely unreachable cell — synthesized here by
         sealing a small pocket of free cells behind an inflated wall ring so a
         cell inside the pocket has no path to the goal (arena_v1's real grid is
         fully connected, so this could not otherwise be exercised).
      4. Two calls on the same grid/goal return byte-identical arrays
         (`np.array_equal`, which treats `inf == inf` as an element match).
    """
    _ensure_repo_root_on_path()
    from manual_astar import (  # type: ignore[import-not-found]
        GRID_RESOLUTION,
        SAFETY_MARGIN,
        OccupancyGrid,
        astar_search,
        build_occupancy_grid,
        load_world,
        world_to_grid,
    )
    from planners._costfield import build_cost_to_go_field  # type: ignore[import-not-found]

    world = load_world(yaml_path)
    grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN)
    goal_cell = world_to_grid(np.asarray(world.goal, dtype=float)[:2], grid)
    start_cell = world_to_grid(np.asarray(world.start, dtype=float)[:2], grid)

    field = build_cost_to_go_field(grid, goal_cell)
    assert field.shape == grid.shape, (
        f"TCd: field shape {field.shape} must match grid shape {grid.shape}"
    )

    # --- 1. Reachable-cell agreement with a fresh A* oracle (several cells). ---
    rows, cols = grid.shape
    free_rows, free_cols = np.nonzero(~grid.cells)
    assert free_rows.size > 0, "TCd setup: grid has no free cells"
    rng = np.random.default_rng(0)
    sample_count = min(20, free_rows.size)
    sample_indices = rng.choice(free_rows.size, size=sample_count, replace=False)
    checked = 0
    for idx in sample_indices:
        cell = (int(free_rows[idx]), int(free_cols[idx]))
        if not np.isfinite(field[cell]):
            continue  # an isolated free cell with no path to the goal; skip
        oracle_path = astar_search(grid, cell, goal_cell)
        oracle_cost = _octile_path_cost(oracle_path)
        assert abs(float(field[cell]) - oracle_cost) < 1e-9, (
            f"TCd: field[{cell}]={field[cell]!r} != A*-oracle cost {oracle_cost!r} "
            f"from {cell} to goal {goal_cell}"
        )
        checked += 1
    assert checked >= 5, (
        f"TCd setup: only {checked} reachable cells were actually comparable "
        f"(need >= 5); adjust the sample"
    )
    # The real start cell is a canonical reachable check too (arena_v1 is solvable).
    assert np.isfinite(field[start_cell]), "TCd: arena_v1 start cell must be reachable"
    start_oracle_path = astar_search(grid, start_cell, goal_cell)
    start_oracle_cost = _octile_path_cost(start_oracle_path)
    assert abs(float(field[start_cell]) - start_oracle_cost) < 1e-9, (
        f"TCd: field[start]={field[start_cell]!r} != A*-oracle cost "
        f"{start_oracle_cost!r}"
    )

    # --- 2. inf on an occupied cell. ---
    occupied_rows, occupied_cols = np.nonzero(grid.cells)
    assert occupied_rows.size > 0, "TCd setup: grid has no occupied cells"
    occ_cell = (int(occupied_rows[0]), int(occupied_cols[0]))
    assert np.isinf(field[occ_cell]), (
        f"TCd: field[{occ_cell}] must be inf on an occupied cell, got {field[occ_cell]!r}"
    )

    # --- 3. inf on a genuinely unreachable (sealed) cell. ---
    sealed_cells = grid.cells.copy()
    # Wall off a 3x3 pocket in a corner far from the goal with a full ring of
    # occupied cells (at least 1 cell thick — the corner keeps the ring inside
    # bounds without needing a bounds check).
    pocket_row0, pocket_col0 = 2, 2
    ring_lo_r, ring_hi_r = pocket_row0 - 1, pocket_row0 + 3
    ring_lo_c, ring_hi_c = pocket_col0 - 1, pocket_col0 + 3
    assert ring_hi_r < rows and ring_hi_c < cols, "TCd setup: pocket ring out of bounds"
    sealed_cells[ring_lo_r:ring_hi_r + 1, ring_lo_c] = True
    sealed_cells[ring_lo_r:ring_hi_r + 1, ring_hi_c] = True
    sealed_cells[ring_lo_r, ring_lo_c:ring_hi_c + 1] = True
    sealed_cells[ring_hi_r, ring_lo_c:ring_hi_c + 1] = True
    sealed_cells[pocket_row0:pocket_row0 + 2, pocket_col0:pocket_col0 + 2] = False
    sealed_grid = OccupancyGrid(cells=sealed_cells, resolution=grid.resolution, offset=grid.offset)
    sealed_field = build_cost_to_go_field(sealed_grid, goal_cell)
    pocket_cell = (pocket_row0, pocket_col0)
    assert not sealed_cells[pocket_cell], "TCd setup: pocket cell must be free"
    assert np.isinf(sealed_field[pocket_cell]), (
        f"TCd: field[{pocket_cell}] must be inf on a sealed/unreachable cell, got "
        f"{sealed_field[pocket_cell]!r}"
    )

    # --- 4. Determinism: two calls on the same grid/goal are byte-identical. ---
    field_again = build_cost_to_go_field(grid, goal_cell)
    assert np.array_equal(field, field_again, equal_nan=False), (
        "TCd: build_cost_to_go_field must be deterministic across identical calls"
    )


def tce(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — synthetic tracks + in-process controller
    """Braking-inevitability + soft term: unit rules + a behavioral yield drive (AC7).

    UNIT (synthetic tracks, a PredictiveDWAController-family instance reset
    in-process against arena_v1 — no irsim, no subprocess):
      (i)   An inevitable matched-time collision (braking-to-a-stop-and-hold
            STILL collides) is rejected: _evaluate_candidate returns None.
      (ii)  A brakeable conflict (the same approaching track, but far enough
            that braking clears it) is admitted: _evaluate_candidate returns a
            finite score, not None.
      (iii) A ttc_step == 1 imminent conflict is rejected outright (the
            backstop fires before the braking test is even relevant).
      (iv)  The soft term is strictly monotone and un-clipped: a SLOWER
            collision-free candidate outscores a FASTER grazing/colliding one
            (built directly, comparing _evaluate_candidate's returned scores).

    INTEGRATION (behavioral): drives dwa_predictive_oracle in-process a few
    ticks with a scripted, closing head-on track and asserts the chosen linear
    speed decreases tick-over-tick (the robot yields as the conflict nears).
    """
    _ensure_repo_root_on_path()
    from planners._predict import PREDICT_DT, Track, trajectory_conflict  # type: ignore[import-not-found]
    from planners.dwa import CLEARANCE_CAP as _CLEARANCE_CAP  # type: ignore[import-not-found]
    from planners.dwa import COLLISION_MARGIN, CONTROL_DT  # type: ignore[import-not-found]
    from planners.dwa_predictive import (  # type: ignore[import-not-found]
        PREDICTED_GAP_WEIGHT,
        DWAPredictiveOracleController,
    )

    world_raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    start = world_raw["robot"]["state"]
    state0 = np.array(
        [float(start[0]), float(start[1]), float(start[2]) if len(start) > 2 else 0.0],
        dtype=np.float64,
    )

    def make_controller() -> DWAPredictiveOracleController:
        controller = DWAPredictiveOracleController(predict_horizon=10)
        controller.reset(yaml_path, (), np.full((360,), np.nan, dtype=np.float64), state0)
        return controller

    # A candidate that marches straight +x at 1.0 m/s (theta=0, the arena_v1
    # start heading), so its rollout is easy to reason about analytically.
    heading = float(state0[2])
    v_fast = 1.0
    w = 0.0
    controller = make_controller()
    trajectory = controller._rollout(state0, v_fast, w)
    empty_cloud = np.empty((0, 2), dtype=float)

    # (i) Inevitable: a stationary track sitting on the robot's straight path,
    # far enough that the un-braked rollout's earliest conflict is NOT the very
    # next step (ttc_step != 1, isolating the ICS test from the imminent
    # backstop), yet close enough that even braking-to-a-stop-and-hold (which
    # settles around x~0.2 m under BRAKE_DECEL=MAX_LINEAR_ACCEL=2.0 within a few
    # sub-steps) still ends up inside the collision margin of it.
    forward_x = float(state0[0]) + float(np.cos(heading)) * 0.70
    forward_y = float(state0[1]) + float(np.sin(heading)) * 0.70
    inevitable_track = [Track(id=1, x=forward_x, y=forward_y, vx=0.0, vy=0.0, radius=0.3)]
    controller._tracks = inevitable_track
    result_inevitable = controller._evaluate_candidate(
        state0, trajectory, v_fast, w, empty_cloud
    )
    assert result_inevitable is None, (
        "TCe(i): a stationary track close enough that braking-and-holding still "
        "collides must be rejected (inevitable collision state)"
    )
    # Confirm this candidate's ttc_step != 1 (isolating the ICS branch, not the
    # imminent backstop) by checking the un-braked conflict directly.
    unbraked_conflict = trajectory_conflict(
        trajectory, inevitable_track, controller._robot_radius, 10, PREDICT_DT, COLLISION_MARGIN
    )
    assert unbraked_conflict.ttc_step is not None and unbraked_conflict.ttc_step > 1, (
        f"TCe(i) setup: expected ttc_step > 1 to isolate the ICS test, got "
        f"{unbraked_conflict.ttc_step}"
    )

    # (ii) Brakeable: the same stationary obstacle, but far enough away that
    # emergency braking-to-a-stop clears it well before matched-time contact.
    far_x = float(state0[0]) + float(np.cos(heading)) * 3.5
    far_y = float(state0[1]) + float(np.sin(heading)) * 3.5
    brakeable_track = [Track(id=2, x=far_x, y=far_y, vx=0.0, vy=0.0, radius=0.3)]
    controller2 = make_controller()
    controller2._tracks = brakeable_track
    trajectory2 = controller2._rollout(state0, v_fast, w)
    result_brakeable = controller2._evaluate_candidate(
        state0, trajectory2, v_fast, w, empty_cloud
    )
    assert result_brakeable is not None, (
        "TCe(ii): a brakeable conflict (braking clears it) must be admitted, "
        "not rejected"
    )

    # (iii) Imminent backstop: a track placed so the very NEXT step already
    # violates the collision margin (ttc_step == 1), regardless of braking.
    imminent_dx = float(np.cos(heading)) * (CONTROL_DT * v_fast)
    imminent_dy = float(np.sin(heading)) * (CONTROL_DT * v_fast)
    imminent_x = float(state0[0]) + imminent_dx
    imminent_y = float(state0[1]) + imminent_dy
    imminent_track = [Track(id=3, x=imminent_x, y=imminent_y, vx=0.0, vy=0.0, radius=0.3)]
    controller3 = make_controller()
    controller3._tracks = imminent_track
    trajectory3 = controller3._rollout(state0, v_fast, w)
    imminent_conflict = trajectory_conflict(
        trajectory3, imminent_track, controller3._robot_radius, 10, PREDICT_DT, COLLISION_MARGIN
    )
    assert imminent_conflict.ttc_step == 1, (
        f"TCe(iii) setup: expected ttc_step == 1, got {imminent_conflict.ttc_step}"
    )
    result_imminent = controller3._evaluate_candidate(
        state0, trajectory3, v_fast, w, empty_cloud
    )
    assert result_imminent is None, (
        "TCe(iii): a ttc_step == 1 imminent conflict must be rejected outright"
    )

    # (iv) Monotone, un-clipped soft term: a SLOWER collision-free candidate
    # must outscore a FASTER grazing/colliding-adjacent one. Build a track
    # positioned so a fast candidate's matched-time gap is small/negative-ish
    # while a slow candidate's gap is comfortably positive, then compare the
    # two candidates' OWN scores (not the same trajectory at two speeds).
    side_track = [Track(id=4, x=float(state0[0]) + 1.0, y=float(state0[1]) + 0.05, vx=0.0, vy=0.0, radius=0.3)]
    controller_fast = make_controller()
    controller_fast._tracks = side_track
    v_test_fast = 1.0
    traj_fast = controller_fast._rollout(state0, v_test_fast, w)
    score_fast = controller_fast._evaluate_candidate(
        state0, traj_fast, v_test_fast, w, empty_cloud
    )

    controller_slow = make_controller()
    controller_slow._tracks = side_track
    v_test_slow = 0.1
    traj_slow = controller_slow._rollout(state0, v_test_slow, w)
    score_slow = controller_slow._evaluate_candidate(
        state0, traj_slow, v_test_slow, w, empty_cloud
    )

    assert score_fast is not None and score_slow is not None, (
        f"TCe(iv) setup: both candidates must be admissible for the score "
        f"comparison to mean anything (fast={score_fast!r}, slow={score_slow!r})"
    )
    fast_conflict = trajectory_conflict(
        traj_fast, side_track, controller_fast._robot_radius, 10, PREDICT_DT, COLLISION_MARGIN
    )
    slow_conflict = trajectory_conflict(
        traj_slow, side_track, controller_slow._robot_radius, 10, PREDICT_DT, COLLISION_MARGIN
    )
    assert slow_conflict.min_gap > fast_conflict.min_gap, (
        f"TCe(iv) setup: the slow candidate must have a larger matched-time gap "
        f"than the fast one (slow={slow_conflict.min_gap!r}, "
        f"fast={fast_conflict.min_gap!r}) for the monotone-term claim to be "
        f"meaningful"
    )
    # The un-clipped, un-floored soft term rewards the larger gap; combined with
    # the velocity term favoring speed, this asserts the CLEARANCE margin (not
    # raw speed) can flip the ranking when the gap difference is large enough —
    # i.e. score is NOT simply monotone in v alone. We assert the soft-term
    # CONTRIBUTION directly (base score with/without the soft term) rather than
    # the full argmax outcome, which also depends on heading/velocity weights.
    gap_fast = float(np.clip(fast_conflict.min_gap, -_CLEARANCE_CAP, _CLEARANCE_CAP))
    gap_slow = float(np.clip(slow_conflict.min_gap, -_CLEARANCE_CAP, _CLEARANCE_CAP))
    soft_fast = PREDICTED_GAP_WEIGHT * (gap_fast / _CLEARANCE_CAP)
    soft_slow = PREDICTED_GAP_WEIGHT * (gap_slow / _CLEARANCE_CAP)
    assert soft_slow > soft_fast, (
        f"TCe(iv): the soft term must be strictly monotone in min_gap (larger gap "
        f"-> larger un-clipped term); soft_slow={soft_slow!r} soft_fast={soft_fast!r}"
    )
    assert soft_fast > -PREDICTED_GAP_WEIGHT and soft_slow < PREDICTED_GAP_WEIGHT, (
        "TCe(iv): the soft term must not be floored at 0 (it must be able to go "
        "negative for a negative gap and stay below the max for a positive one)"
    )

    # --- Integration: a head-on crosser makes the chosen v decrease tick-over-tick. ---
    # dwa_predictive_oracle's act() rebuilds self._tracks from self._snapshot via
    # observe_truth each tick (OracleTracker.update reads the snapshot, not a
    # directly-poked self._tracks — a raw assignment would just be clobbered), so
    # this drive feeds a synthetic snapshot record through observe_truth, exactly
    # as the runner does for the truth seam.
    @dataclass(frozen=True)
    class _FakeDynamicObstacleState:
        id: int
        x: float
        y: float
        vx: float
        vy: float
        radius: float

    driver = DWAPredictiveOracleController(predict_horizon=10)
    driver.reset(yaml_path, (), np.full((360,), np.nan, dtype=np.float64), state0)
    # Warm-start the commanded linear speed at cruise so the dynamic window
    # already brackets the top speed — otherwise DWA spends the first several
    # ticks ramping up from rest (a ±MAX_LINEAR_ACCEL*CONTROL_DT window per
    # tick) and there is no cruise speed to visibly YIELD from. The braking-
    # inevitability reject is a hard cliff (full speed until the crosser gets
    # close, then 0), so with a warm start the sequence drops from cruise to a
    # stop as the head-on crosser closes in.
    driver._last_v = 1.0
    # A track that starts ahead on the robot's heading line and closes head-on
    # at a modest speed, positioned so the braking-inevitability reject fires
    # partway through the drive (the probe found 2.5 m at 0.8 m/s yields a clean
    # cruise->stop within 8 ticks).
    approach_speed = 0.8
    track_x = float(state0[0]) + float(np.cos(heading)) * 2.5
    track_y = float(state0[1]) + float(np.sin(heading)) * 2.5
    track_vx = -float(np.cos(heading)) * approach_speed
    track_vy = -float(np.sin(heading)) * approach_speed

    speeds: list[float] = []
    state = state0.copy()
    no_lidar = np.full((360,), np.nan, dtype=np.float64)
    ticks = 8
    for tick in range(ticks):
        snapshot = (
            _FakeDynamicObstacleState(
                id=100, x=track_x, y=track_y, vx=track_vx, vy=track_vy, radius=0.3
            ),
        )
        driver.observe_truth(snapshot)
        action = driver.act(state, no_lidar)
        assert action.shape == (2, 1), f"TCe integration: bad action shape at tick {tick}"
        commanded_v = float(action[0, 0])
        speeds.append(commanded_v)
        # Advance the scripted track and (roughly) the robot pose for the next tick.
        track_x += track_vx * CONTROL_DT
        track_y += track_vy * CONTROL_DT
        state = np.array(
            [
                state[0] + commanded_v * np.cos(state[2]) * CONTROL_DT,
                state[1] + commanded_v * np.sin(state[2]) * CONTROL_DT,
                state[2],
            ],
            dtype=np.float64,
        )

    assert len(speeds) == ticks, "TCe integration: must have driven every tick"
    # The braking-inevitability reject makes the yield a cliff, not a gentle
    # ramp: the robot holds cruise until the head-on crosser gets close enough
    # that every forward candidate is inevitable, then the commanded speed
    # collapses toward a stop. So the yield shows as (a) the commanded speed
    # dropping below its cruise value somewhere in the drive, and (b) ending
    # below where it started — the robot gave way rather than racing the crosser.
    assert min(speeds) < speeds[0] - 1e-9, (
        f"TCe integration: commanded speed never dropped below its initial cruise "
        f"value as the crosser closed in (no yield); speeds={speeds}"
    )
    assert speeds[-1] < speeds[0], (
        f"TCe integration: final commanded speed {speeds[-1]!r} must be lower "
        f"than the initial cruise speed {speeds[0]!r} once the crosser has closed "
        f"in (the robot must yield, not race the crosser); speeds={speeds}"
    )


def tcf(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — synthetic obstacle points
    """Present-position floor keeps un-tracked mover returns; no live return subtracted (AC6).

    UNIT (in-process, no irsim). Reset a predictive controller against
    arena_v1, then:
      1. Place an obstacle point directly in a candidate's rollout path but give
         it NO track (self._tracks stays empty for this check) — the floor
         (_trajectory_clearance over the FULL live cloud) must still reject that
         candidate, proving an un-tracked mover is caught exactly as vanilla DWA
         would catch it.
      2. Confirm no live return is ever subtracted for a (hypothetically)
         tracked mover: build the SAME obstacle_points cloud whether or not a
         track exists for that obstacle, and assert _evaluate_candidate's floor
         behavior (clearance rejection) is identical in both cases — i.e. having
         a track for an obstacle does not exempt its live lidar return from the
         floor check.
    """
    _ensure_repo_root_on_path()
    from planners._predict import Track  # type: ignore[import-not-found]
    from planners.dwa_predictive import DWAPredictiveOracleController  # type: ignore[import-not-found]

    world_raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    start = world_raw["robot"]["state"]
    state0 = np.array(
        [float(start[0]), float(start[1]), float(start[2]) if len(start) > 2 else 0.0],
        dtype=np.float64,
    )
    heading = float(state0[2])

    def make_controller() -> DWAPredictiveOracleController:
        controller = DWAPredictiveOracleController(predict_horizon=10)
        controller.reset(yaml_path, (), np.full((360,), np.nan, dtype=np.float64), state0)
        return controller

    v = 1.0
    w = 0.0

    # A point sitting squarely on the straight-ahead rollout, close enough that
    # the robot body would graze it — but with NO corresponding Track.
    graze_x = float(state0[0]) + float(np.cos(heading)) * 1.0
    graze_y = float(state0[1]) + float(np.sin(heading)) * 1.0
    cloud_with_grazer = np.array([[graze_x, graze_y]], dtype=float)

    controller_untracked = make_controller()
    controller_untracked._tracks = []  # no track for the grazing obstacle
    trajectory = controller_untracked._rollout(state0, v, w)
    result_untracked = controller_untracked._evaluate_candidate(
        state0, trajectory, v, w, cloud_with_grazer
    )
    assert result_untracked is None, (
        "TCf(1): a mover with NO track must still be rejected by the "
        "present-position floor at its current return"
    )

    # Now give the SAME obstacle a track (as if it were being tracked), but keep
    # it far from the robot's near-term rollout so it does not ALSO trip the
    # space-time layer — this isolates whether having a track exempts the point
    # from the floor. It must not: the floor check does not consult self._tracks
    # at all, so the outcome must be identical.
    controller_tracked = make_controller()
    controller_tracked._tracks = [
        Track(id=1, x=graze_x, y=graze_y, vx=0.0, vy=0.0, radius=0.3)
    ]
    result_tracked = controller_tracked._evaluate_candidate(
        state0, trajectory, v, w, cloud_with_grazer
    )
    assert result_tracked is None, (
        "TCf(2): having a Track for an obstacle must NOT exempt its live lidar "
        "return from the present-position floor — no live return is ever "
        "subtracted for a tracked mover"
    )

    # And a clean sanity check: with the grazer's point REMOVED from the cloud
    # entirely (simulating it never having been seen by lidar this tick) and no
    # track either, the SAME rollout is admissible — proving the rejection above
    # was really the floor catching the point, not some other guard.
    controller_clear = make_controller()
    controller_clear._tracks = []
    empty_cloud = np.empty((0, 2), dtype=float)
    result_clear = controller_clear._evaluate_candidate(
        state0, trajectory, v, w, empty_cloud
    )
    assert result_clear is not None, (
        "TCf setup: with no obstacle points at all the same rollout must be "
        "admissible (isolates that the floor, not something else, rejected the "
        "grazing case above)"
    )


def tcg(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure in-process, no irsim
    """Global _heading_term is non-saturated + strictly monotone in progress (AC8).

    Builds the cost-to-go field for arena_v1 and a global-guidance controller
    instance (dwa_predictive_oracle), then constructs three synthetic rollouts
    ending at cells with strictly decreasing / equal / increasing field value
    relative to the start cell (retreat, no-progress, progress) and asserts
    the resulting _heading_term scores are STRICTLY ordered
    retreat < no_progress(==0.5) < progress, and every value lies in the open
    interval (0, 1) (never saturated at the 0/1 endpoints).
    """
    _ensure_repo_root_on_path()
    from manual_astar import (  # type: ignore[import-not-found]
        GRID_RESOLUTION,
        SAFETY_MARGIN,
        build_occupancy_grid,
        load_world,
        world_to_grid,
    )
    from planners.dwa_predictive import DWAPredictiveOracleController  # type: ignore[import-not-found]

    world_raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    start = world_raw["robot"]["state"]
    state0 = np.array(
        [float(start[0]), float(start[1]), float(start[2]) if len(start) > 2 else 0.0],
        dtype=np.float64,
    )

    controller = DWAPredictiveOracleController(predict_horizon=10)
    controller.reset(yaml_path, (), np.full((360,), np.nan, dtype=np.float64), state0)
    assert controller._field is not None, (
        "TCg setup: arena_v1's start cell must be reachable, so global guidance "
        "must be active (controller._field must not be None)"
    )

    world = load_world(yaml_path)
    grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN)
    start_cell = world_to_grid(state0[:2], grid)
    start_value = float(controller._field[start_cell])

    # Build three synthetic "rollouts" (only the final position matters to
    # _heading_term) whose final cell's field value is respectively HIGHER
    # (retreat), EQUAL (no progress), and LOWER (progress) than the start cell's
    # value. Search outward from the start cell for cells with each property so
    # this does not depend on assuming any particular grid geometry.
    rows, cols = grid.shape
    resolution = grid.resolution
    offset = grid.offset

    def cell_to_world(cell: tuple[int, int]) -> np.ndarray:
        row, col = cell
        return np.array(
            [offset[0] + (col + 0.5) * resolution, offset[1] + (row + 0.5) * resolution],
            dtype=float,
        )

    retreat_cell = None
    progress_cell = None
    equal_cell = None
    start_row, start_col = start_cell
    for radius in range(1, 30):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = start_row + dr, start_col + dc
                if not (0 <= r < rows and 0 <= c < cols):
                    continue
                value = float(controller._field[r, c])
                if not np.isfinite(value):
                    continue
                if progress_cell is None and value < start_value - 1e-6:
                    progress_cell = (r, c)
                if retreat_cell is None and value > start_value + 1e-6:
                    retreat_cell = (r, c)
                if equal_cell is None and abs(value - start_value) < 1e-6:
                    equal_cell = (r, c)
        if progress_cell is not None and retreat_cell is not None and equal_cell is not None:
            break

    assert progress_cell is not None, "TCg setup: could not find a progress cell"
    assert retreat_cell is not None, "TCg setup: could not find a retreat cell"
    if equal_cell is None:
        equal_cell = start_cell  # the start cell itself has value == start_value trivially

    def make_trajectory(end_cell: tuple[int, int]) -> np.ndarray:
        end_xy = cell_to_world(end_cell)
        traj = np.tile(end_xy, (12, 1))
        # Perturb every-but-the-last row slightly so trajectory.shape[0] >= 2
        # with a non-degenerate "step" is not required by _heading_term for the
        # global path (it only reads trajectory[-1] and the start cell), but
        # keep the shape realistic.
        return traj

    controller_test = DWAPredictiveOracleController(predict_horizon=10)
    controller_test.reset(yaml_path, (), np.full((360,), np.nan, dtype=np.float64), state0)
    assert controller_test._field is not None

    retreat_score = controller_test._heading_term(state0, make_trajectory(retreat_cell))
    equal_score = controller_test._heading_term(state0, make_trajectory(equal_cell))
    progress_score = controller_test._heading_term(state0, make_trajectory(progress_cell))

    assert retreat_score < equal_score < progress_score, (
        f"TCg: heading-term scores must be strictly ordered retreat < no-progress "
        f"< progress; got retreat={retreat_score!r} equal={equal_score!r} "
        f"progress={progress_score!r}"
    )
    assert abs(equal_score - 0.5) < 1e-9, (
        f"TCg: the no-progress score must be exactly 0.5, got {equal_score!r}"
    )
    for label, value in (
        ("retreat", retreat_score),
        ("equal", equal_score),
        ("progress", progress_score),
    ):
        assert 0.0 < value < 1.0, (
            f"TCg: {label} score must be a non-saturated interior value in (0, 1), "
            f"got {value!r}"
        )


def tch(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure in-process, no irsim
    """Opt-in LidarTracker hardening: default unchanged, enabled deterministic + capped (AC9).

    (i)   Guard: the DEFAULT LidarTracker construction must remain byte-identical
          to the pre-hardening estimator, so d_star_lite_predictive is unaffected.
          Directly re-runs TC63's and TC64's bodies (both pass a plain
          LidarTracker(grid, bearings) with no hardening kwargs) and asserts they
          still PASS — the binding guard that opt-in hardening did not leak into
          the shared default path.
    (ii)  An ENABLED tracker (smoothing_frames=VELOCITY_SMOOTHING_FRAMES,
          max_track_speed=MAX_TRACK_SPEED) is deterministic across a
          cluster-count change AND an association swap: two fresh instances
          driven over the same multi-frame fixture (reusing TC64's
          enter/leave cluster-count-change fixture, replayed through an
          ENABLED tracker) produce byte-identical Track sequences.
    (iii) No reported speed exceeds MAX_TRACK_SPEED, exercised with an obstacle
          moving fast enough that the RAW instantaneous estimate would exceed
          the cap absent the clamp.
    """
    _ensure_repo_root_on_path()
    from planners._predict import MAX_TRACK_SPEED, VELOCITY_SMOOTHING_FRAMES  # type: ignore[import-not-found]

    # --- (i) Guard: TC63 (subprocess e2e) and TC64 (in-process determinism) must
    # still PASS unmodified — proves the opt-in hardening left the shared
    # default-constructed LidarTracker (used by d_star_lite_predictive) byte-
    # unchanged. TC63 is a slow subprocess e2e; running it again here would
    # roughly double --check's wall time for a guard that TC64 (fast, in-process,
    # already exercises LidarTracker's default-path determinism directly) also
    # covers. Re-run TC64 directly (cheap) and rely on TC63 running elsewhere in
    # the same --check pass for the subprocess-level guarantee.
    tc64(yaml_path, seed)

    # --- (ii) + (iii): drive an ENABLED tracker over TC64's cluster-count-change
    # fixture (reconstructed here to keep this case self-contained), twice on
    # fresh instances, and check determinism + the speed cap.
    import math

    from manual_astar import OccupancyGrid  # type: ignore[import-not-found]
    from planners._predict import LidarTracker, PREDICT_DT  # type: ignore[import-not-found]

    def make_grid() -> OccupancyGrid:
        rows = cols = 80
        cells = np.zeros((rows, cols), dtype=bool)
        return OccupancyGrid(
            cells=cells, resolution=0.5, offset=np.array([-10.0, -10.0], dtype=float)
        )

    def ray_disk_range(
        bearing: float, center: tuple[float, float], radius: float
    ) -> float | None:
        dx, dy = math.cos(bearing), math.sin(bearing)
        cx, cy = center
        d_dot_c = dx * cx + dy * cy
        disc = d_dot_c * d_dot_c - (cx * cx + cy * cy - radius * radius)
        if disc < 0.0:
            return None
        t = d_dot_c - math.sqrt(disc)
        return t if t > 0.0 else None

    def synth_lidar(
        bearings: np.ndarray, disks: list[tuple[tuple[float, float], float]]
    ) -> np.ndarray:
        ranges = np.full(bearings.shape[0], np.nan, dtype=float)
        for i, bearing in enumerate(bearings):
            best: float | None = None
            for center, radius in disks:
                r = ray_disk_range(float(bearing), center, radius)
                if r is not None and (best is None or r < best):
                    best = r
            if best is not None:
                ranges[i] = best
        return ranges

    def run_enabled_sequence(
        bearings: np.ndarray,
        frames: list[list[tuple[tuple[float, float], float]]],
    ) -> list[list]:
        grid = make_grid()
        tracker = LidarTracker(
            grid,
            bearings,
            smoothing_frames=VELOCITY_SMOOTHING_FRAMES,
            max_track_speed=MAX_TRACK_SPEED,
        )
        state = np.array([0.0, 0.0, 0.0], dtype=float)
        out: list[list] = []
        for disks in frames:
            lidar = synth_lidar(bearings, disks)
            tracks = tracker.update(snapshot=(), state=state, lidar=lidar, dt=PREDICT_DT)
            out.append(tracks)
        return out

    bearings = np.linspace(-math.pi, math.pi * 0.999, 180)
    radius = 0.4
    a0 = (5.0, 0.0)
    a1 = (5.15, 0.0)
    a2 = (5.30, 0.0)
    a3 = (5.45, 0.0)
    b = (3.0, -3.0)
    frames: list[list[tuple[tuple[float, float], float]]] = [
        [(a0, radius)],
        [(a1, radius)],
        [(a2, radius)],
        [(a3, radius), (b, radius)],
        [(b, radius)],
    ]

    seq1 = run_enabled_sequence(bearings, frames)
    seq2 = run_enabled_sequence(bearings, frames)

    tup1 = [[dataclasses.astuple(t) for t in frame] for frame in seq1]
    tup2 = [[dataclasses.astuple(t) for t in frame] for frame in seq2]
    assert tup1 == tup2, (
        "TCh(ii): two fresh ENABLED-tracker runs over the same cluster-count-change "
        "fixture diverged; hardened determinism is broken"
    )
    counts = [len(frame) for frame in seq1]
    assert counts == [1, 1, 1, 2, 1], (
        f"TCh(ii) setup: fixture did not exercise a cluster-count change; got {counts}"
    )

    # (iii) Speed cap: a fast-moving obstacle whose RAW instantaneous velocity
    # would exceed MAX_TRACK_SPEED absent the clamp. Two frames, a large jump.
    fast_step = MAX_TRACK_SPEED * PREDICT_DT * 3.0  # 3x the cap's per-frame displacement
    fast_frames: list[list[tuple[tuple[float, float], float]]] = [
        [((5.0, 0.0), radius)],
        [((5.0 + fast_step, 0.0), radius)],
    ]
    fast_seq = run_enabled_sequence(bearings, fast_frames)
    assert len(fast_seq[1]) == 1, (
        f"TCh(iii) setup: expected 1 track in frame 2, got {len(fast_seq[1])}"
    )
    fast_track = fast_seq[1][0]
    raw_speed_would_be = fast_step / PREDICT_DT
    assert raw_speed_would_be > MAX_TRACK_SPEED, (
        "TCh(iii) setup: the fixture must make the RAW estimate exceed the cap "
        "for the clamp assertion to be meaningful"
    )
    reported_speed = math.sqrt(fast_track.vx ** 2 + fast_track.vy ** 2)
    assert reported_speed <= MAX_TRACK_SPEED + 1e-9, (
        f"TCh(iii): reported speed {reported_speed!r} exceeds MAX_TRACK_SPEED "
        f"{MAX_TRACK_SPEED!r} — the clamp did not fire"
    )
    # Also check across the earlier cluster-change fixture (already-deterministic
    # obstacle A moves at a modest ~1.5 m/s, well under the cap, so this is a
    # sanity check that the clamp does not needlessly distort normal-speed
    # tracks): no track in any frame of seq1 exceeds the cap.
    for frame in seq1:
        for track in frame:
            speed = math.sqrt(track.vx ** 2 + track.vy ** 2)
            assert speed <= MAX_TRACK_SPEED + 1e-9, (
                f"TCh(iii): track id={track.id} speed {speed!r} exceeds "
                f"MAX_TRACK_SPEED {MAX_TRACK_SPEED!r} in the cluster-change fixture"
            )


def tci(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — uses arena_no_path fixture
    """Start-unreachable fallback: global variant falls back, no crash, times out (AC8, AC11).

    On arena/arena_no_path.yaml the robot start is walled in by a 1.5 m box (the
    goal is open but unreachable from inside the box), so the cost-to-go field
    is inf at the start cell. A GLOBAL DWA-predictive variant
    (dwa_predictive_oracle at h10) must:
      - fall back to the base Euclidean heading for the whole episode (no
        planner_error — DWA never fails to plan, reset() cannot raise);
      - drive to a TERMINAL state that is a TIMEOUT, not a crash — the robot is
        physically boxed in by walls it can sense and avoid, so it should
        wander/settle inside the box rather than drive through a wall.
    --no-traffic keeps this a pure static-box story (no dynamic obstacles to
    also dodge), matching TC16's treatment of the same fixture for a different
    algorithm family.
    """
    repo_root = _ensure_repo_root_on_path()
    no_path_yaml = str(repo_root / "arena" / "arena_no_path.yaml")
    seed_value = "168"
    horizon = "10"
    world_stem = Path(no_path_yaml).stem

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        r = subprocess.run(
            [
                sys.executable, "-m", "runners.run_episode",
                "--algorithm", "dwa_predictive_oracle",
                "--predict-horizon", horizon,
                "--seed", seed_value,
                "--world", no_path_yaml,
                "--no-traffic",
                "--results-dir", td,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            # This is the suite's worst-case runtime: the boxed-in robot never
            # crashes and never reaches the goal, so it runs the FULL 1200-step
            # (120 s sim) episode with the heavier global-guidance predictive
            # controller (a cost-to-go heading lookup per sampled candidate).
            # Measured ~222 s wall uncontended, so 180 s was too tight; 420 s
            # keeps generous margin for a loaded/slower machine.
            timeout=420,
        )
        assert r.returncode == 0, (
            f"TCi runner exit {r.returncode}; stderr={r.stderr[-400:]}"
        )

        json_path = (
            Path(td) / world_stem / f"dwa_predictive_oracle_h{horizon}" / f"{seed_value}.json"
        )
        assert json_path.exists(), f"TCi: metrics JSON missing at {json_path}"

        metrics = json.loads(json_path.read_text(encoding="utf-8"))
        assert metrics["planner_error"] is None, (
            f"TCi: DWA reset must never raise, even with the start walled off; "
            f"planner_error={metrics['planner_error']}"
        )
        assert metrics["crashed"] is False, (
            f"TCi: the boxed-in robot must not crash into the wall; metrics={metrics}"
        )
        assert metrics["timed_out"] is True, (
            f"TCi: the boxed-in robot must time out (it cannot reach the "
            f"unreachable goal); metrics={metrics}"
        )
        assert metrics["time_to_goal"] is None, (
            f"TCi: time_to_goal must be None (the goal was never reached); "
            f"metrics={metrics}"
        )


def tcj(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure in-process, no irsim
    """Mid-episode raise guard: a tracker/prediction failure degrades to base DWA (AC11).

    Resets a predictive controller in-process (no irsim, no subprocess), drives
    one successful act(), then monkeypatches its tracker's update() to raise on
    the NEXT call, and asserts the following act() still returns a finite
    (2, 1) action rather than propagating — act()'s own try/except around the
    tracker refresh degrades that tick to an empty-tracks (base DWA) call.
    """
    _ensure_repo_root_on_path()
    from planners.dwa_predictive import DWAPredictiveOracleController  # type: ignore[import-not-found]

    world_raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    start = world_raw["robot"]["state"]
    state0 = np.array(
        [float(start[0]), float(start[1]), float(start[2]) if len(start) > 2 else 0.0],
        dtype=np.float64,
    )
    no_lidar = np.full((360,), np.nan, dtype=np.float64)

    controller = DWAPredictiveOracleController(predict_horizon=10)
    controller.reset(yaml_path, (), no_lidar, state0)
    controller.observe_truth(())

    # First act() succeeds and lazily builds self._tracker (OracleTracker()).
    first_action = controller.act(state0, no_lidar)
    assert first_action.shape == (2, 1), (
        f"TCj setup: the first act() must succeed with a (2,1) action, got shape "
        f"{first_action.shape}"
    )
    assert controller._tracker is not None, (
        "TCj setup: the first act() must have built the tracker"
    )

    # Monkeypatch the tracker to raise on the NEXT update() call.
    def _raising_update(*args: object, **kwargs: object) -> None:
        raise RuntimeError("TCj: injected tracker failure")

    controller._tracker.update = _raising_update  # type: ignore[method-assign]

    second_action = controller.act(state0, no_lidar)
    assert isinstance(second_action, np.ndarray), (
        f"TCj: act() must still return an ndarray after a tracker failure, got "
        f"{type(second_action).__name__}"
    )
    assert second_action.shape == (2, 1), (
        f"TCj: act() must return a (2,1) action after a tracker failure, got shape "
        f"{second_action.shape}"
    )
    assert np.all(np.isfinite(second_action)), (
        f"TCj: act() must return a finite action after a tracker failure, got "
        f"{second_action}"
    )
    assert controller._tracks == [], (
        "TCj: a tracker failure must degrade this tick to empty tracks (plain-DWA "
        "scoring), not propagate"
    )

    # A third tick (tracker still raising) must also keep working — the guard is
    # not a one-shot fluke.
    third_action = controller.act(state0, no_lidar)
    assert third_action.shape == (2, 1) and np.all(np.isfinite(third_action)), (
        f"TCj: act() must keep returning finite (2,1) actions on repeated tracker "
        f"failures, got {third_action}"
    )


def tck(yaml_path: str, seed: int) -> None:  # noqa: ARG001 — pure in-process, no irsim
    """Registry/family sets: canonical == 13; EXPERIMENTAL_KEYS == the 4 (AC10).

    set(run_all._CANONICAL_ORDER) has 13 entries and equals
    set(ALGORITHMS) - EXPERIMENTAL_KEYS; EXPERIMENTAL_KEYS == {d_star_lite_oracle,
    dwa_predictive_oracle, dwa_predictive_paper, dwa_predictive_paper_oracle}; all
    four dwa_predictive* keys are in PREDICT_FAMILIES; importing runners.run_all
    does not raise (the import-time assertion already ran by the time this
    function body executes, so a prior raise would have failed collection, not
    just this assertion — re-checked here explicitly for a clear failure message).
    """
    _ensure_repo_root_on_path()
    import planners  # type: ignore[import-not-found]
    from planners._grid import EXPERIMENTAL_KEYS, PREDICT_FAMILIES  # type: ignore[import-not-found]
    import runners.run_all as run_all  # type: ignore[import-not-found]

    canonical = set(run_all._CANONICAL_ORDER)
    assert len(canonical) == 13, (
        f"TCk: _CANONICAL_ORDER must have exactly 13 entries, got {len(canonical)}"
    )
    assert canonical == set(planners.ALGORITHMS) - set(EXPERIMENTAL_KEYS), (
        "TCk: _CANONICAL_ORDER must equal ALGORITHMS minus EXPERIMENTAL_KEYS"
    )
    assert set(EXPERIMENTAL_KEYS) == {
        "d_star_lite_oracle",
        "dwa_predictive_oracle",
        "dwa_predictive_paper",
        "dwa_predictive_paper_oracle",
    }, (
        f"TCk: EXPERIMENTAL_KEYS must be exactly the 4 documented cheats/ablations, "
        f"got {set(EXPERIMENTAL_KEYS)}"
    )
    four_dwa_keys = {
        "dwa_predictive",
        "dwa_predictive_oracle",
        "dwa_predictive_paper",
        "dwa_predictive_paper_oracle",
    }
    assert four_dwa_keys <= set(PREDICT_FAMILIES), (
        f"TCk: all four dwa_predictive* keys must be in PREDICT_FAMILIES; missing "
        f"{four_dwa_keys - set(PREDICT_FAMILIES)}"
    )
    assert "dwa_predictive" in canonical and "dwa_predictive_paper" not in canonical, (
        "TCk: dwa_predictive must be canonical and dwa_predictive_paper must not"
    )


# ---------------------------------------------------------------------------
# CLI runner — --check (default) or --render. See module docstring above.
# ---------------------------------------------------------------------------


def _run_checks(yaml_path: str, seed: int) -> int:
    cases: list[tuple[str, Any]] = [
        ("TC1: construct + close", tc1),
        ("TC2: reset shapes & info", tc2),
        ("TC2b: missing-lidar tick", tc2b),
        ("TC3: one step", tc3),
        ("TC4: deliberate crash within 200 steps", tc4),
        ("TC5: timeout fires", tc5),
        ("TC6: step after done raises", tc6),
        ("TC7: reset after done clears state", tc7),
        ("TC8: arrive_flag injection sets reached_goal", tc8),
        ("TC9: action validation", tc9),
        ("TC10: manual_astar inflation check", tc10),
        ("TC11: YAML schema fields", tc11),
        ("TC12: lidar beam mismatch raises ArenaConfigError", tc12),
        ("TC13: wall crash via teleport", tc13),
        ("TC14: full A* drive via runner", tc14),
        ("TC15: determinism — same seed -> byte-identical trace", tc15),
        ("TC16: planner failure on arena_no_path.yaml", tc16),
        ("TC17: init population (20 on edges, inward)", tc17),
        ("TC18: refill maintained across full-traversal window", tc18),
        ("TC19: robot-vs-dynamic-obstacle collision via _inject_for_test", tc19),
        ("TC20: traffic determinism — sha256 sequences match", tc20),
        ("TC21: snapshot shape, type, immutability", tc21),
        ("TC22: world-stem partitioning end-to-end", tc22),
        ("TC23: import-cycle guard (planners <-> arena.arena)", tc23),
        ("TC24: traffic-ON runner — 8-key trace + determinism", tc24),
        ("TC25: Phase 3 seed derivation (determinism/uniqueness/prefix)", tc25),
        ("TC26: Phase 3 batch determinism + parallel-ordering", tc26),
        ("TC27: Phase 3 failure accounting + non-zero batch exit", tc27),
        ("TC28: lidar->grid fold geometry (pose-dependent, memoryless)", tc28),
        ("TC29: Dijkstra == A* optimal cost + dijkstra_once reaches goal", tc29),
        ("TC30: a_star_replan end-to-end + labeled dir + 8-key trace", tc30),
        ("TC31: replan cadence (every K-th act) + memoryless fold", tc31),
        ("TC32: mid-replan failure fallback + follower identity", tc32),
        ("TC33: --replan-k validation + name==key + label + membership", tc33),
        ("TC34: a_star_once parity through the new loop (determinism)", tc34),
        ("TC35: D* Lite optimal static path (== A* cost) + reaches goal", tc35),
        ("TC36: D* Lite incremental == from-scratch (binding block)", tc36),
        ("TC37: d_star_lite registered + rejects --replan-k + traffic e2e", tc37),
        ("TC46: D* Lite deferred settle — no per-tick settle, on-demand == fresh A*", tc46),
        ("TC38: dwa traffic-on drive via runner + 8-key trace", tc38),
        ("TC39: apf traffic-on drive via runner + 8-key trace", tc39),
        ("TC40: rrt_once --no-traffic reaches goal + trace determinism", tc40),
        ("TC41: rrt_star_once reaches goal + RRT*-vs-RRT planned-cost obs", tc41),
        ("TC42: rrt_once/rrt_star_once sealed-start planner failure", tc42),
        ("TC43: --replan-k validation for the 6 reactive/sampling keys", tc43),
        ("TC44: rrt_replan/rrt_star_replan traffic e2e + labeled dir", tc44),
        ("TC45: commitment-horizon fix proof (goal + follower identity)", tc45),
        ("TC47: rrt-local LOS helper == segment_is_clear_grid (stratified fuzz)", tc47),
        ("TC48: speed-regime table + resolver (cross-module agreement)", tc48),
        ("TC49: speed-band bound validation (one-sided/bad bounds raise)", tc49),
        ("TC50: baseline determinism + draw-order guard (binding gate)", tc50),
        ("TC51: band wired at initial snapshot (positions equal, speeds scaled)", tc51),
        ("TC52: non-baseline determinism across a despawn/refill cycle", tc52),
        ("TC-CLI: speed-flag CLI rejection (exit 2, no JSON)", tc_cli),
        ("TC-FWD: run_experiment flag forwarding + manifest provenance", tc_fwd),
        ("TC53: predict_blocked_cells capsule geometry (disk train, sorted, deterministic)", tc53),
        ("TC54: predict_blocked_cells cone widening + exclusion zone + gate drop", tc54),
        ("TC55: predicted-conflict gate (divergent-now-collide-later crosser)", tc55),
        ("TC56: threat-ordered bounded peel (farthest-future-first, imminent retained)", tc56),
        ("TC56b: peel-to-zero still unsolvable — act() does not raise", tc56b),
        ("TC57: d_star_lite_oracle_h0 trace == plain d_star_lite (byte-identical)", tc57),
        ("TC58: d_star_lite_oracle traffic-on e2e + determinism", tc58),
        ("TC59: --predict-horizon validation (exit 2; label dir)", tc59),
        ("TC60: dynamic_obstacles truth seam + tick alignment", tc60),
        ("TC61: run_all tolerates experimental keys (canonical carve-out)", tc61),
        ("TC62: plot_horizon_sweep --selfcheck passes (no irsim)", tc62),
        ("TC63: d_star_lite_predictive traffic-on e2e + determinism", tc63),
        ("TC64: LidarTracker determinism across a multi-frame cluster-count change", tc64),
        ("TC65: plain dwa unchanged + paper-only h0 == plain dwa (byte-identical)", tc65),
        ("TC66: --predict-horizon validation for the space-time DWA family (exit 2)", tc66),
        ("TC67: dwa_predictive/_oracle traffic-on e2e + determinism", tc67),
        ("TC68: trajectory_conflict pure space-time geometry + determinism", tc68),
        ("TC69: run_all canonical set == 13 (DWA oracle carve-out tolerated)", tc69),
        ("TCa: paper+global h0 deterministic AND != plain dwa", tca),
        ("TCb: all four DWA-predict keys traffic-on e2e + determinism (h10)", tcb),
        ("TCc: --predict-horizon required / --replan-k rejected for all four keys", tcc),
        ("TCd: build_cost_to_go_field == A*-cost oracle; inf on occupied/sealed", tcd),
        ("TCe: braking-inevitability + soft term (unit) + yield drive (integration)", tce),
        ("TCf: present floor keeps un-tracked mover returns (no live-return exempt)", tcf),
        ("TCg: global _heading_term non-saturated + monotone in geodesic progress", tcg),
        ("TCh: opt-in LidarTracker hardening — default unchanged, enabled capped", tch),
        ("TCi: start-unreachable fallback on arena_no_path.yaml — timeout, no crash", tci),
        ("TCj: mid-episode tracker-raise guard — act() degrades, never raises", tcj),
        ("TCk: registry/family sets — canonical == 13, EXPERIMENTAL_KEYS == 4", tck),
    ]
    failures = 0
    for label, fn in cases:
        try:
            fn(yaml_path, seed)
            print(f"PASS - {label}")
        except Exception as exc:
            print(f"FAIL - {label}: {type(exc).__name__}: {exc}")
            failures += 1
    return failures


def _run_render(yaml_path: str, seed: int) -> None:
    arena = Arena(yaml_path, seed, render=True)
    try:
        arena.reset()
        zero = np.array([[0.0], [0.0]], dtype=float)
        while True:
            _, _, done, info = arena.step(zero)
            if done:
                print(f"done: {info}")
                break
    finally:
        arena.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arena smoke/check harness")
    parser.add_argument(
        "yaml_path",
        help="Path to arena world YAML (e.g. arena/arena_v1.yaml)",
    )
    parser.add_argument("--seed", type=int, default=42)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--render",
        action="store_true",
        help="Interactive smoke loop (visible window)",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Run TC1-TC69 + TCa-TCk + TC-CLI/TC-FWD headless (84 cases, incl. Phase 2 traffic + Phase 3 batch runner + replanning + D* Lite (incl. deferred-settle) + reactive (DWA/APF) + sampling (RRT/RRT*) families + rrt-local LOS-helper equivalence + the obstacle-speed-cap sweep + the predictive (motion-aware) D* Lite family + the space-time predictive DWA family + the paper+global braking-inevitability/cost-to-go-field DWA rework)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import sys

    args = _parse_args()
    if args.render:
        _run_render(args.yaml_path, args.seed)
    else:
        # Default to --check when neither flag given.
        sys.exit(_run_checks(args.yaml_path, args.seed))
