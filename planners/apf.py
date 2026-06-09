"""APF controller: reactive Khatib (1986) Artificial Potential Fields.

`APFController` (registry key `apf`) is a purely reactive, velocity-output
planner with NO global plan. Each `act()` builds a net force from two sources:

- an ATTRACTIVE force pulling the robot toward the YAML goal (conic far away,
  quadratic near the goal — the standard saturated attractive potential), and
- a REPULSIVE force pushing away from every live lidar return inside an
  influence radius, with the classic Khatib magnitude
  `REPULSIVE_GAIN * (1/d - 1/d0) * (1/d^2)`.

The net force is converted into a clamped `(v, omega)` command via a heading
controller plus a heading-gated forward-speed schedule (the same spirit as
`manual_astar.compute_action_from_state`). This is a pure Khatib field — there
are deliberately no goal-biased escape heuristics or local-minimum-escape
tricks, so the robot is expected to stall in `arena_v1`'s corridors (that local
minimum is the experimental signal, not a bug).
"""
from __future__ import annotations

import numpy as np

from manual_astar import (
    MAX_ANGULAR_SPEED,
    MAX_LINEAR_SPEED,
    load_world,
    wrap_to_pi,
)
from planners._grid import LidarGeometry, load_lidar_geometry, register

# --- Tunables (no magic numbers in the function bodies) ----------------------

# Attractive potential. The force magnitude grows linearly with goal distance up
# to ATTRACTIVE_SATURATION_DISTANCE, beyond which it is held constant so a far
# goal cannot dominate the repulsive field and blow the command up.
ATTRACTIVE_GAIN = 1.0
ATTRACTIVE_SATURATION_DISTANCE = 2.0

# Repulsive potential (Khatib). Beams returning farther than the influence
# radius contribute nothing; nearer beams push the robot away with the standard
# `GAIN * (1/d - 1/d0) * (1/d^2)` magnitude.
REPULSIVE_GAIN = 0.6
REPULSIVE_INFLUENCE_RADIUS = 1.5

# Floor on the obstacle distance used in the repulsive denominator, so a grazing
# return (d -> 0) cannot produce an infinite / NaN force.
MIN_OBSTACLE_DISTANCE = 0.05

# Heading controller gain and the heading-gate thresholds. Forward speed is
# curtailed while the heading error is large so the robot turns toward the net
# force before driving along it (mirrors compute_action_from_state).
K_OMEGA = 2.0
HEADING_GATE_STOP = 0.9          # |heading error| above this -> creep speed only
HEADING_GATE_SLOW = 0.4          # |heading error| above this -> slow speed
SLOW_LINEAR_SPEED = 0.15         # forward speed in the mid heading-error band
CREEP_LINEAR_SPEED = 0.2         # forward speed in the large heading-error band

# Maps net-force magnitude to a desired forward speed before the heading gate and
# the [0, MAX_LINEAR_SPEED] clamp are applied.
FORCE_TO_SPEED_GAIN = 1.0

# Net forces below this magnitude carry no usable direction (e.g. a perfect
# attractive/repulsive cancellation at a local minimum), so the robot holds
# still rather than chasing numerical noise.
MIN_FORCE_MAGNITUDE = 1e-6


class APFController:
    """Reactive Khatib Artificial Potential Fields controller (key ``apf``).

    Holds no global plan: `reset()` only caches the goal xy and the lidar beam
    geometry (it never raises, so `apf` never yields a `planner_error`), and
    every `act()` recomputes the attractive + repulsive net force from the live
    lidar frame and converts it to a clamped `(v, omega)`.
    """

    name = "apf"

    def __init__(self, replan_k: int | None = None) -> None:
        # `build_controller` rejects a non-None `replan_k` for reactive families
        # (apf is not in REPLAN_FAMILIES) before construction; the kwarg is
        # accepted only to match the uniform `ALGORITHMS[name](replan_k=...)`
        # construction seam, then ignored.
        del replan_k
        self._goal_xy: np.ndarray | None = None
        self._geom: LidarGeometry | None = None
        self._bearings: np.ndarray | None = None

    def reset(
        self,
        world_yaml: str,
        initial_snapshot: tuple,
        lidar0: np.ndarray,
        state0: np.ndarray,
    ) -> None:
        """Cache the goal and lidar beam bearings; never raises (no global plan).

        The live snapshot, the t=0 lidar, and the start pose carry no information
        a reactive field needs at setup time — the force is recomputed from the
        live lidar every `act()`.
        """
        del initial_snapshot, lidar0, state0

        # Static substrate: the goal xy and the beam bearings (robot frame).
        self._goal_xy = np.asarray(load_world(world_yaml).goal, dtype=float)[:2]
        self._geom = load_lidar_geometry(world_yaml)
        # irsim lays beams with linspace over the inclusive [angle_min, angle_max]
        # endpoints (the same recovery lidar_to_occupancy uses).
        self._bearings = np.linspace(
            self._geom.angle_min, self._geom.angle_max, self._geom.number
        )

    def act(self, state: np.ndarray, lidar: np.ndarray) -> np.ndarray:
        if self._goal_xy is None or self._geom is None or self._bearings is None:
            raise RuntimeError("act() called before reset().")
        if state.shape != (3,):
            raise ValueError(
                f"Expected (3,) [x, y, theta] state, received shape {state.shape}."
            )
        if lidar.shape != (self._geom.number,):
            raise ValueError(
                f"Expected lidar of shape {(self._geom.number,)}, received {lidar.shape}."
            )

        position = state[:2].astype(float)
        theta = float(state[2])

        net_force = self._attractive_force(position) + self._repulsive_force(
            position, theta, lidar
        )
        return self._force_to_action(net_force, theta)

    def _attractive_force(self, position: np.ndarray) -> np.ndarray:
        """Saturated attractive force pulling the robot toward the goal xy.

        Conic far from the goal (constant magnitude beyond the saturation
        distance) and quadratic within it, so the pull does not grow without
        bound as the goal recedes.
        """
        assert self._goal_xy is not None
        to_goal = self._goal_xy - position
        distance = float(np.linalg.norm(to_goal))
        if distance < MIN_FORCE_MAGNITUDE:
            return np.zeros(2, dtype=float)

        direction = to_goal / distance
        magnitude = ATTRACTIVE_GAIN * min(distance, ATTRACTIVE_SATURATION_DISTANCE)
        return magnitude * direction

    def _repulsive_force(
        self, position: np.ndarray, theta: float, lidar: np.ndarray
    ) -> np.ndarray:
        """Sum of Khatib repulsive forces from finite lidar returns within d0.

        Each finite beam closer than REPULSIVE_INFLUENCE_RADIUS is projected to
        its world-frame hit point and contributes a force pointing from the
        obstacle back toward the robot, with magnitude
        `REPULSIVE_GAIN * (1/d - 1/d0) * (1/d^2)`. NaN beams and beams beyond d0
        are skipped; the obstacle distance is floored at MIN_OBSTACLE_DISTANCE.
        """
        assert self._bearings is not None
        influence_radius = REPULSIVE_INFLUENCE_RADIUS
        force = np.zeros(2, dtype=float)

        for beam_index in range(lidar.shape[0]):
            beam_range = float(lidar[beam_index])
            if not np.isfinite(beam_range) or beam_range >= influence_radius:
                continue

            world_angle = theta + float(self._bearings[beam_index])
            hit = position + beam_range * np.array(
                [np.cos(world_angle), np.sin(world_angle)], dtype=float
            )

            away = position - hit
            distance = float(np.linalg.norm(away))
            distance = max(distance, MIN_OBSTACLE_DISTANCE)

            # Khatib repulsive gradient magnitude (positive only inside d0).
            magnitude = REPULSIVE_GAIN * (
                (1.0 / distance) - (1.0 / influence_radius)
            ) * (1.0 / (distance * distance))
            force += magnitude * (away / distance)

        return force

    def _force_to_action(self, net_force: np.ndarray, theta: float) -> np.ndarray:
        """Convert a net force vector to a clamped `(2,1)` `[[v],[w]]` command.

        Steers toward the force direction with a proportional heading
        controller, and sets forward speed proportional to the force magnitude
        but gated by the heading error (creep while badly misaligned), all
        clamped to `[0, MAX_LINEAR_SPEED]` / `[-MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED]`.
        """
        force_magnitude = float(np.linalg.norm(net_force))

        # A near-zero net force has no usable direction (e.g. a local minimum
        # where attraction and repulsion cancel) -> hold still.
        if force_magnitude < MIN_FORCE_MAGNITUDE:
            return np.array([[0.0], [0.0]], dtype=float)

        desired_heading = float(np.arctan2(net_force[1], net_force[0]))
        heading_error = wrap_to_pi(desired_heading - theta)

        angular_velocity = float(
            np.clip(K_OMEGA * heading_error, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
        )

        # Heading-gated forward-speed schedule (mirrors compute_action_from_state).
        if abs(heading_error) > HEADING_GATE_STOP:
            linear_velocity = CREEP_LINEAR_SPEED
        elif abs(heading_error) > HEADING_GATE_SLOW:
            linear_velocity = SLOW_LINEAR_SPEED
        else:
            linear_velocity = FORCE_TO_SPEED_GAIN * force_magnitude

        linear_velocity = float(np.clip(linear_velocity, 0.0, MAX_LINEAR_SPEED))

        return np.array([[linear_velocity], [angular_velocity]], dtype=float)


register("apf", APFController)
