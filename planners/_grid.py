"""Shared grid-planning substrate for the lidar-folding replanning controllers.

This module is the keystone foundation for the full-search replanning family
(A* / Dijkstra in T4/T5, D* Lite in T11). It reuses the static occupancy and
A* machinery from `manual_astar` and adds the lidar-fold + replanning glue:

- `LidarGeometry` / `load_lidar_geometry` — parse the world's `lidar2d` sensor
  block into the bearing endpoints irsim itself uses.
- `lidar_to_occupancy` — fold a live lidar scan onto a copy of the static
  occupancy grid (returns a NEW array; never mutates the static cells).
- `segment_is_clear_grid` / `grid_path_to_waypoints` — turn a grid cell path
  into a sparse, inflation-aware waypoint tuple anchored at the robot's current
  pose and the goal.
- `PathFollowingController` — a concrete `Controller` that plans once at
  `reset()` and (optionally) replans every k-th `act()` against the folded grid.
- registry machinery (`ALGORITHMS`, `register`, `build_controller`,
  `algorithm_label`) — controller modules register themselves at import; this
  module registers nothing and imports no controller module (avoids the cycle).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path as _FilePath

import numpy as np
import yaml

from manual_astar import (
    GRID_RESOLUTION,
    OccupancyGrid,
    SAFETY_MARGIN,
    WAYPOINT_REACHED_DISTANCE,
    WAYPOINT_STRIDE,
    WaypointFollower,
    astar_search,
    build_occupancy_grid,
    compute_action_from_state,
    grid_to_world,
    is_cell_in_bounds,
    load_world,
    world_to_grid,
)
from planners._geometry import iter_disk_cells, lidar_to_world_points
from planners._types import Controller, Path

# Lidar sits at the robot origin (the canonical worlds give it no sensor offset),
# so a beam's hit point is taken directly from the robot pose.
LIDAR_SENSOR_NAME = "lidar2d"
TWO_PI = 2.0 * np.pi
# Sampling pitch for the line-of-sight check: half a cell guarantees no occupied
# cell can hide between two consecutive samples.
SEGMENT_SAMPLE_FACTOR = 0.5
MIN_REPLAN_K = 1


@dataclass(frozen=True)
class LidarGeometry:
    """The inclusive bearing endpoints (robot frame) and beam count of a lidar."""

    angle_min: float
    angle_max: float
    number: int
    range_max: float


def load_lidar_geometry(world_yaml: str) -> LidarGeometry:
    """Read the `lidar2d` sensor block and mirror irsim's WrapTo2Pi on its range.

    irsim wraps the configured `angle_range` with `value % (2*pi)` (so an exact
    2*pi collapses to 0, while a value just under 2*pi is unchanged) and then
    lays the beams symmetrically about the robot heading over
    [-wrapped/2, +wrapped/2]. We reproduce that here so `lidar_to_occupancy`
    can recover each beam's true world bearing.
    """
    raw_world = yaml.safe_load(_FilePath(world_yaml).read_text(encoding="utf-8"))
    robot_section = raw_world.get("robot", {})
    sensors = robot_section.get("sensors", []) or []

    for sensor in sensors:
        if sensor.get("name") == LIDAR_SENSOR_NAME:
            angle_range = float(sensor["angle_range"])
            number = int(sensor["number"])
            if number < 1:
                raise ValueError(
                    f"lidar2d 'number' must be at least 1, received {number!r}."
                )
            # range_max is the no-hit sentinel: a beam that reaches it without
            # hitting comes back AT range_max. Absent in the YAML -> inf, kept to
            # mirror the Arena's own scan.get("range_max", inf) default.
            # CAVEAT: with range_max == inf the LidarTracker's near-rim deadband
            # (a `< range_max - RANGE_MAX_DEADBAND` cut) becomes a no-op, so a world
            # that omits range_max re-admits the rim no-hit ring as phantom dynamic
            # obstacles for the lidar predictive variant. Every shipped world sets
            # `range_max: 5.0`, so this is latent; a new world feeding the lidar
            # variant should set range_max explicitly.
            range_max = float(sensor.get("range_max", float("inf")))
            wrapped = angle_range % TWO_PI
            half = wrapped / 2.0
            return LidarGeometry(
                angle_min=-half, angle_max=half, number=number, range_max=range_max
            )

    raise ValueError(
        f"World {world_yaml!r} has no 'lidar2d' sensor block in robot.sensors."
    )


def _mark_disk(
    cells: np.ndarray,
    grid: OccupancyGrid,
    center_x: float,
    center_y: float,
    inflation: float,
) -> None:
    """Mark every grid cell whose CENTER lies within `inflation` of (cx, cy).

    Fills the boolean array from the shared :func:`iter_disk_cells` bounding-box
    scan (the same row-major scan `_predict` collects its predicted footprint from),
    so the lidar fold stays byte-deterministic and can never drift from the
    predicted stamp. Marking is idempotent, so the scan order does not affect the
    result.
    """
    for row, col in iter_disk_cells(grid, center_x, center_y, inflation):
        cells[row, col] = True


def lidar_to_occupancy(
    static_cells: np.ndarray,
    grid: OccupancyGrid,
    state: np.ndarray,
    lidar: np.ndarray,
    geom: LidarGeometry,
    inflation: float,
) -> np.ndarray:
    """Fold a live lidar scan onto a COPY of the static occupancy cells.

    Each finite beam return is projected to its world-frame hit point from the
    robot pose, and every grid cell whose center is within `inflation` of that
    hit is marked occupied. NaN beams (no return) are skipped. The input
    `static_cells` is never mutated; a fresh boolean array is returned.
    """
    if static_cells.shape != grid.shape:
        raise ValueError(
            f"static_cells shape {static_cells.shape} does not match grid shape {grid.shape}."
        )
    if state.shape != (3,):
        raise ValueError(f"Expected (3,) [x, y, theta] state, received shape {state.shape}.")
    if lidar.shape != (geom.number,):
        raise ValueError(
            f"Expected lidar of shape {(geom.number,)}, received {lidar.shape}."
        )

    folded = static_cells.copy()

    # irsim lays beams with linspace over the inclusive [angle_min, angle_max]
    # endpoints (NOT angle_increment spacing). The shared projection drops NaN
    # (no-return) beams and folds every finite one (range_max defaults to inf, so
    # NO rim deadband here — the fold wants the full scan); the points come back in
    # beam order, and marking is order-independent anyway.
    bearings = np.linspace(geom.angle_min, geom.angle_max, geom.number)
    points = lidar_to_world_points(state, lidar, bearings)
    for index in range(points.shape[0]):
        _mark_disk(folded, grid, float(points[index, 0]), float(points[index, 1]), inflation)

    return folded


def segment_is_clear_grid(
    grid_cells: np.ndarray,
    grid: OccupancyGrid,
    p0: np.ndarray,
    p1: np.ndarray,
) -> bool:
    """Line-of-sight check against an occupancy array along p0 -> p1.

    Samples the segment at <= half-resolution spacing; a sample is unsafe if it
    maps to an out-of-bounds cell or an occupied (True) cell. Returns True only
    when every sample is in-bounds and free.
    """
    start = np.asarray(p0, dtype=float)
    end = np.asarray(p1, dtype=float)
    segment = end - start
    length = float(np.linalg.norm(segment))
    sample_step = grid.resolution * SEGMENT_SAMPLE_FACTOR

    if length < 1e-9:
        cell = world_to_grid(start, grid)
        return is_cell_in_bounds(cell, grid) and not bool(grid_cells[cell])

    sample_count = max(2, int(np.ceil(length / sample_step)))
    for sample_index in range(sample_count + 1):
        ratio = sample_index / sample_count
        point = start + ratio * segment
        cell = world_to_grid(point, grid)
        if not is_cell_in_bounds(cell, grid) or bool(grid_cells[cell]):
            return False

    return True


def _append_clear_waypoints(
    output: list[np.ndarray],
    points: list[np.ndarray],
    start_index: int,
    end_index: int,
    grid: OccupancyGrid,
    grid_cells: np.ndarray,
) -> None:
    """Recursively keep `points[end_index]`, bisecting unclear spans.

    Mirrors `manual_astar.append_safe_waypoints` but checks line-of-sight
    against the folded occupancy array instead of analytic obstacle distances.
    """
    start_point = points[start_index]
    end_point = points[end_index]

    if end_index <= start_index + 1 or segment_is_clear_grid(grid_cells, grid, start_point, end_point):
        if np.linalg.norm(output[-1] - end_point) > 1e-9:
            output.append(end_point)
        return

    middle_index = (start_index + end_index) // 2
    _append_clear_waypoints(output, points, start_index, middle_index, grid, grid_cells)
    _append_clear_waypoints(output, points, middle_index, end_index, grid, grid_cells)


def grid_path_to_waypoints(
    cells_path: list[tuple[int, int]],
    grid: OccupancyGrid,
    grid_cells: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    stride: int,
) -> Path:
    """Collapse a grid cell path into a sparse, line-of-sight-safe waypoint tuple.

    The FIRST waypoint is anchored at `start_xy` (the robot's current pose) and
    the LAST at `goal_xy` — NOT the world's static start/goal, because a replan
    runs from wherever the robot is now. Intermediate cells are downsampled at
    `stride` and any span whose straight line is not clear in `grid_cells` is
    recursively bisected to reinsert detail. The returned `Path` always ends at
    `goal_xy`.
    """
    if stride < 1:
        raise ValueError("Waypoint stride must be at least 1.")
    if not cells_path:
        raise ValueError("grid_path_to_waypoints requires a non-empty cell path.")

    start_point = np.asarray(start_xy, dtype=float)
    goal_point = np.asarray(goal_xy, dtype=float)

    # World-frame points for every cell on the path, with the endpoints pinned to
    # the true robot pose / goal rather than their (rounded) cell centers.
    points: list[np.ndarray] = [start_point]
    points.extend(grid_to_world(cell, grid) for cell in cells_path[1:-1])
    points.append(goal_point)

    # Candidate anchor indices: endpoints plus every stride-th interior index.
    candidate_indices = [0]
    candidate_indices.extend(index for index in range(1, len(points) - 1) if index % stride == 0)
    candidate_indices.append(len(points) - 1)

    waypoints: list[np.ndarray] = [points[0]]
    for previous_index, next_index in zip(candidate_indices, candidate_indices[1:]):
        _append_clear_waypoints(waypoints, points, previous_index, next_index, grid, grid_cells)

    # Guarantee the path terminates exactly at the goal point.
    if np.linalg.norm(waypoints[-1] - goal_point) > 1e-9:
        waypoints.append(goal_point)

    return tuple(waypoints)


class PathFollowingController:
    """Full-search replanning `Controller` over a lidar-folded occupancy grid.

    Plans once at `reset()`; `act()` replans every `replan_k`-th call (or never,
    when `replan_k is None`) by re-running the configured graph search from the
    current pose against the freshly folded grid, then drives the resulting
    waypoints with the shared `WaypointFollower`. A mid-episode replan that
    fails is swallowed — the last valid follower is kept, never rebuilt — so
    `act()` never raises (AC8). Subclasses select the search variant by setting
    the class attributes `name` and `heuristic_fn` only (AC2/AC15).
    """

    name: str = ""
    heuristic_fn = None  # None => A* with Euclidean heuristic; staticmethod(lambda *_: 0.0) => Dijkstra

    def __init__(self, replan_k: int | None = None) -> None:
        if replan_k is not None and replan_k < MIN_REPLAN_K:
            raise ValueError(f"replan_k must be >= {MIN_REPLAN_K} or None, received {replan_k!r}.")

        self._replan_k = replan_k
        self._k = 0
        self._follower: WaypointFollower | None = None

        # Commitment-horizon state. `_last_fold` is the occupancy array the most
        # recent plan was built on (reused by `_immediate_segment_blocked` so the
        # segment check costs no extra lidar fold — AC9). `_planned_path` is the
        # freshest plan from the last K-th act, retained even when the follower
        # has not yet adopted it.
        self._last_fold: np.ndarray | None = None
        self._planned_path: Path | None = None

        # Static substrate caches, populated by reset().
        self._world = None
        self._grid: OccupancyGrid | None = None
        self._static_cells: np.ndarray | None = None
        self._goal_cell: tuple[int, int] | None = None
        self._goal_xy: np.ndarray | None = None
        self._geom: LidarGeometry | None = None
        self._inflation: float | None = None

    def reset(
        self,
        world_yaml: str,
        initial_snapshot: tuple,
        lidar0: np.ndarray,
        state0: np.ndarray,
    ) -> None:
        # `initial_snapshot` is ignored by design: lidar0 already encodes those
        # obstacles, and this family is lidar-only.
        del initial_snapshot

        world = load_world(world_yaml)
        grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN)

        goal_xy = np.asarray(world.goal, dtype=float)
        goal_cell = world_to_grid(goal_xy, grid)
        if not is_cell_in_bounds(goal_cell, grid):
            raise ValueError("The goal position is outside the occupancy grid.")
        if bool(grid.cells[goal_cell]):
            raise ValueError("The goal position is blocked after obstacle inflation.")

        self._world = world
        self._grid = grid
        self._static_cells = grid.cells
        self._goal_xy = goal_xy
        self._goal_cell = goal_cell
        self._geom = load_lidar_geometry(world_yaml)
        self._inflation = world.robot_radius + SAFETY_MARGIN

        # Initial plan from the start pose; propagate planner failures so the
        # runner can record planner_error.
        initial_path = self.compute_path(state0, lidar0)
        if not initial_path:
            raise ValueError("The initial plan produced no waypoints.")

        self._follower = WaypointFollower(list(initial_path), WAYPOINT_REACHED_DISTANCE)
        self._k = 0

    def act(self, state: np.ndarray, lidar: np.ndarray) -> np.ndarray:
        if self._follower is None:
            raise RuntimeError("act() called before reset().")

        self._k += 1
        if self._replan_k is not None and self._k % self._replan_k == 0:
            # Refresh knowledge every K acts (the cadence the experiments count),
            # but commit to it only when the held segment is exhausted or blocked.
            try:
                new_path = self.compute_path(state, lidar)
            except (ValueError, RuntimeError):
                # Keep the last valid path/follower; never rebuild it (AC8/AC10).
                # `_planned_path` is left untouched on failure.
                new_path = None

            if new_path:
                self._planned_path = new_path
                # Commitment horizon: adopt the fresh plan ONLY when the current
                # commitment is exhausted or its immediate segment is no longer
                # clear. On a clear run the committed segment stays clear and the
                # follower never finishes mid-traverse, so no swap fires and the
                # robot follows its t=0 plan identically to the `_once` variant —
                # the lidar-fold jitter that starved forward motion only appeared
                # when a swap fired every K acts (AC8/AC10).
                if self._follower.is_finished or self._immediate_segment_blocked(state[:2]):
                    self._follower = WaypointFollower(
                        list(new_path), WAYPOINT_REACHED_DISTANCE
                    )

        return compute_action_from_state(state, self._follower)

    def compute_path(self, state: np.ndarray, lidar: np.ndarray) -> Path:
        """Full-search replan from the current pose against the folded grid.

        Always stores the folded occupancy in `self._last_fold` (including the
        initial plan called from `reset()`) so the subsequent commitment check in
        `_immediate_segment_blocked` can reuse it without a second lidar fold
        (AC9). The actual search is delegated to the overridable `_plan` hook.
        """
        if (
            self._grid is None
            or self._static_cells is None
            or self._goal_cell is None
            or self._goal_xy is None
            or self._geom is None
            or self._inflation is None
        ):
            raise RuntimeError("compute_path() called before reset().")

        folded = lidar_to_occupancy(
            self._static_cells, self._grid, state, lidar, self._geom, self._inflation
        )
        self._last_fold = folded
        # astar_search reads grid.cells, so the folded ndarray must be re-wrapped.
        folded_grid = OccupancyGrid(
            cells=folded, resolution=self._grid.resolution, offset=self._grid.offset
        )
        return self._plan(folded_grid, folded, state)

    def _plan(self, folded_grid: OccupancyGrid, folded: np.ndarray, state: np.ndarray) -> Path:
        """Overridable search hook: grid A*/Dijkstra by default (AC2/AC15).

        Subclasses (RRT / RRT* in T4/T5) override ONLY this method to substitute
        a sampling search; the signature `(folded_grid, folded, state)` and the
        contract — return a `Path` of world-frame waypoints from the robot's
        current pose to the goal — are fixed. The grid variant selected by the
        class is recovered from `type(self).heuristic_fn` (None => A*, zero
        heuristic => Dijkstra).
        """
        cur_cell = world_to_grid(state[:2], folded_grid)
        cells_path = astar_search(folded_grid, cur_cell, self._goal_cell, type(self).heuristic_fn)
        return grid_path_to_waypoints(
            cells_path, self._grid, folded, state[:2], self._goal_xy, WAYPOINT_STRIDE
        )

    def _immediate_segment_blocked(self, position: np.ndarray) -> bool:
        """Is the segment the robot is about to traverse no longer clear?

        The commitment horizon is exactly the robot pose -> its current target
        waypoint: the one piece of the held path the robot drives across next.
        This reuses the stored `self._last_fold` and calls ONLY
        `segment_is_clear_grid`, so it performs no additional lidar fold (AC9).
        `current_waypoint` advances the follower's index past reached waypoints,
        which is intended and mirrors `DStarLiteController`.
        """
        assert self._follower is not None and self._grid is not None
        if self._last_fold is None:
            return False
        target = self._follower.current_waypoint(position)
        return not segment_is_clear_grid(self._last_fold, self._grid, position, target)


# --- Registry machinery -----------------------------------------------------

# Populated by controller modules (T4/T5 grid keys, T11 d_star_lite) via
# register() at import time. Empty after T3 alone — that is expected.
ALGORITHMS: dict[str, type] = {}

# Families that take a --replan-k cadence and label their results with _k<K>.
# The RRT replan families (T4/T5) subclass PathFollowingController and override
# only `_plan`, so they share this same cadence + commitment-horizon machinery.
REPLAN_FAMILIES = frozenset(
    {"a_star_replan", "dijkstra_replan", "rrt_replan", "rrt_star_replan"}
)

# Predictive (motion-aware) families. They take a --predict-horizon (int steps)
# and label their results with _h<steps>. They are NOT in REPLAN_FAMILIES, so the
# existing "--replan-k not allowed" branch rejects --replan-k for them. This holds
# both the grid-STAMPING D* Lite family (d_star_lite_*) and the space-time DWA
# family (dwa_predictive*, incl. the paper-only ablation pair), which share the
# --predict-horizon / _h<steps> machinery but reason very differently (stamp a
# grid vs. check collisions in (x, y, t)). Six keys total.
PREDICT_FAMILIES = frozenset(
    {
        "d_star_lite_oracle",
        "d_star_lite_predictive",
        "dwa_predictive",
        "dwa_predictive_oracle",
        "dwa_predictive_paper",
        "dwa_predictive_paper_oracle",
    }
)

# EXPERIMENTAL_KEYS is a STRICT SUBSET of PREDICT_FAMILIES (6 predict keys, 4
# experimental): the lidar-fed, paper+global-guidance predictive variants
# (d_star_lite_predictive, dwa_predictive) are full canonical study planners (run
# at h10 by run_all and charted on the main plots). Carved OUT of the canonical
# set are the perfect-velocity oracle cheats (d_star_lite_oracle,
# dwa_predictive_oracle) AND the paper-only ablation pair (dwa_predictive_paper,
# dwa_predictive_paper_oracle) — the ablation that isolates the braking layer from
# the cost-to-go guidance, reached only through the runner. All six predict
# families still take --predict-horizon and label with _h<steps> (that is
# PREDICT_FAMILIES, unchanged); membership here is ONLY about the canonical-set
# carve-out, not the horizon/label machinery. run_all's canonical-set assertion
# tolerates these four via this set, so they never land on the canonical scatter;
# the canonical set stays 13.
EXPERIMENTAL_KEYS = frozenset(
    {
        "d_star_lite_oracle",
        "dwa_predictive_oracle",
        "dwa_predictive_paper",
        "dwa_predictive_paper_oracle",
    }
)


def register(name: str, cls: type) -> None:
    """Register a controller class under its algorithm key (its `name`)."""
    ALGORITHMS[name] = cls


def algorithm_label(
    name: str, replan_k: int | None, predict_horizon: int | None = None
) -> str:
    """Results label for an (algorithm, cadence, horizon) triple (AC6/AC7).

    Replanning families fold the cadence into the label (`a_star_replan_k5`);
    predictive families fold the horizon (`d_star_lite_oracle_h10`); every other
    algorithm uses its bare key. `predict_horizon` is defaulted to None so the
    ~40 existing two-arg callers keep working untouched.
    """
    if name in REPLAN_FAMILIES:
        return f"{name}_k{replan_k}"
    if name in PREDICT_FAMILIES:
        return f"{name}_h{predict_horizon}"
    return name


def build_controller(
    name: str, replan_k: int | None, predict_horizon: int | None = None
) -> Controller:
    """Validate the (algorithm, cadence, horizon) triple and construct it (AC6/AC7).

    Raises ValueError on an unknown algorithm, a missing/forbidden/out-of-range
    `--replan-k`, or a missing/forbidden/negative `--predict-horizon`. The
    returned instance's `.name` equals `name` (AC15). `predict_horizon` is
    defaulted to None so the existing two-arg call sites keep working untouched.
    """
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {name!r}.")

    if name in REPLAN_FAMILIES and replan_k is None:
        raise ValueError(f"{name} requires --replan-k.")

    # d_star_lite_oracle / d_star_lite_predictive are NOT in REPLAN_FAMILIES, so
    # this branch already rejects --replan-k for the predict family (AC7).
    if name not in REPLAN_FAMILIES and replan_k is not None:
        raise ValueError(f"--replan-k not allowed for {name}.")

    if replan_k is not None and replan_k < MIN_REPLAN_K:
        raise ValueError(f"--replan-k must be >= {MIN_REPLAN_K}, received {replan_k!r}.")

    # --predict-horizon: required for the predict family, forbidden elsewhere.
    if name in PREDICT_FAMILIES and predict_horizon is None:
        raise ValueError(f"{name} requires --predict-horizon.")
    if name not in PREDICT_FAMILIES and predict_horizon is not None:
        raise ValueError(f"--predict-horizon not allowed for {name}.")
    if predict_horizon is not None and predict_horizon < 0:
        raise ValueError(
            f"--predict-horizon must be >= 0, received {predict_horizon!r}."
        )

    if name in PREDICT_FAMILIES:
        return ALGORITHMS[name](replan_k=replan_k, predict_horizon=predict_horizon)
    return ALGORITHMS[name](replan_k=replan_k)
