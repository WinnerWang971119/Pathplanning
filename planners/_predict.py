"""planners/_predict.py — shared predictive substrate for motion-aware D* Lite.

The LOGIC here is pure: plain floats/ints + numpy in, deterministic output, no
irsim/RNG calls, no set-iteration leaking into output order. The shared grid
geometry it builds on lives in the pure ``planners._geometry`` module; the one
manual_astar helper it needs at runtime (``point_to_polyline_distance`` for the
conflict gate) is lazy-imported inside ``predict_blocked_cells``.

NOTE: "pure" describes the computation, NOT the import. Like every ``planners``
submodule, importing this one runs the package ``__init__`` (which eagerly imports
the controllers) and therefore pulls irsim + matplotlib — the documented
"importing planners pulls irsim + matplotlib" gotcha. A headless tool does NOT
become irsim-free by importing from here; it must still lazy-import ``planners``
symbols inside functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

from planners._geometry import iter_disk_cells, lidar_to_world_points

if TYPE_CHECKING:
    # Annotation-only (stringified by `from __future__ import annotations`), so it
    # never runs at import time. The one runtime manual_astar symbol — the conflict
    # gate's point_to_polyline_distance — is lazy-imported inside predict_blocked_cells.
    from manual_astar import OccupancyGrid

# Prediction timestep in seconds.  Matches irsim step_time and DWA CONTROL_DT.
PREDICT_DT: float = 0.1

# Per-step radial growth (metres) applied to the cone geometry's stamp radius.
# The cone widens with the lookahead step to represent estimator uncertainty
# (zero for the oracle, nonzero for the lidar variant).  Chosen as roughly half
# a grid cell per step (GRID_RESOLUTION is 0.1 m, so ~0.05 m), a small default
# tuned later by the lidar variant (T12).  The capsule geometry ignores it.
CONE_GROWTH_PER_STEP: float = 0.05

# --- LidarTracker tunables (frame-differencing velocity estimator) ----------

# Grid-bucket edge length (metres) for clustering surviving dynamic lidar
# returns into obstacle clusters.  Roughly one obstacle radius (the arena's
# dynamic obstacles are r=0.3 m circles), so a single obstacle's arc of returns
# falls into a few adjacent buckets that 8-connectivity then merges into one
# cluster, while two distinct obstacles a metre apart stay separate.
CLUSTER_RESOLUTION: float = 0.3

# Floor (metres) on a cluster's estimated radius.  A one-point cluster has zero
# extent, so without a floor its predicted disk would collapse to nothing; lidar
# only ever sees the near surface and under-estimates the true radius anyway
# (the cone geometry in T12 widens to cover that), so a small sane floor is all
# that is required.
MIN_TRACK_RADIUS: float = 0.15

# Maximum centroid displacement (metres) accepted when associating a current
# cluster to a prior-frame cluster.  The fastest obstacle moves
# 1.5 * MAX_LINEAR_SPEED(=1.0) * PREDICT_DT(=0.1) = 0.15 m per frame, so 1.0 m
# comfortably covers that plus clustering/centroid jitter while staying tight
# enough to reject a spurious cross-association to an unrelated obstacle.
MAX_ASSOCIATION_DISTANCE: float = 1.0

# Multiplier folding a cluster's representative (smallest) bucket cell
# ``(row, col)`` into a single integer track id: ``id = row * MULT + col``.
# Chosen larger than any column index a 50x50 world reaches at
# CLUSTER_RESOLUTION (~167 buckets), so distinct representative cells map to
# distinct ids and a stationary obstacle keeps a stable id across frames.
# Assumes non-negative world bucket coordinates (the arena offset is (0, 0)).
CLUSTER_ID_MULTIPLIER: int = 100000

# The near-rim no-hit deadband (RANGE_MAX_DEADBAND) the LidarTracker applies when
# projecting beams now lives with the shared projection in planners._geometry.


class ThreatKey(NamedTuple):
    """Sort key placing soonest-conflict tracks first, ties broken by id.

    Ascending order over (ttc_steps, track_id) is exactly threat order:
    the track whose predicted footprint first intersects the planned-path
    corridor comes first, and equal time-to-conflict ties resolve by stable
    obstacle id so the grouped output is deterministic.
    """

    ttc_steps: int
    track_id: int


@dataclass(frozen=True)
class Track:
    """A single tracked obstacle at the current tick."""

    id: int          # stable identity (oracle: obstacle id; lidar: synthesized cluster id)
    x: float
    y: float
    vx: float        # m/s, world frame
    vy: float
    radius: float


class Tracker(Protocol):
    """Interface for velocity-source adapters consumed by the predictive controller.

    OracleTracker.update reads *snapshot* (ignores state/lidar/dt) — velocities
    are exact/known from the live truth seam.

    The future LidarTracker (T11) reads *state* and *lidar* (ignores snapshot) —
    velocities are estimated from frame-differencing.

    Both return tracks sorted by id ascending to guarantee deterministic ordering.
    """

    def update(
        self,
        *,
        snapshot: object,
        state: object,
        lidar: object,
        dt: float,
    ) -> list[Track]:
        ...


class OracleTracker:
    """Trivial tracker that converts the live oracle snapshot to a Track list.

    Reads ``snapshot`` (a tuple of DynamicObstacleState records from
    EpisodeInfo.dynamic_obstacles).  ``state``, ``lidar``, and ``dt`` are
    accepted for protocol compatibility but are ignored — the velocities are
    exact and already present in the snapshot.

    The DynamicObstacleState type is NOT imported here; attributes are read
    via duck typing so this module stays decoupled from arena.
    """

    def update(
        self,
        *,
        snapshot: object,
        state: object,
        lidar: object,
        dt: float,
    ) -> list[Track]:
        """Return id-sorted Tracks from the oracle snapshot.

        Parameters
        ----------
        snapshot:
            A tuple of records each exposing .id, .x, .y, .vx, .vy, .radius.
            Pass ``()`` (or an empty tuple) when traffic is off.
        state, lidar, dt:
            Ignored by this implementation.
        """
        # Suppress "unused argument" intent — these are accepted for protocol
        # conformance only.
        del state, lidar, dt

        tracks = [
            Track(
                id=rec.id,
                x=rec.x,
                y=rec.y,
                vx=rec.vx,
                vy=rec.vy,
                radius=rec.radius,
            )
            for rec in snapshot  # type: ignore[union-attr]
        ]
        tracks.sort(key=lambda t: t.id)
        return tracks


@dataclass(frozen=True)
class _Cluster:
    """One connected-component cluster of dynamic lidar returns (internal).

    ``rep_cell`` is the smallest bucket key in the component (its deterministic
    representative); ``track_id`` is that cell folded into a single integer. It is
    stable WITHIN a frame (distinct components have distinct rep_cells), and stable
    across frames only while the obstacle is stationary — a moving obstacle's
    rep_cell (and thus id) shifts as its return arc moves. Nothing relies on
    cross-frame id stability: cross-frame state is carried positionally via
    ``_prev_centroids``, and each tick's prediction is computed fresh.
    """

    track_id: int
    rep_cell: tuple[int, int]
    centroid_x: float
    centroid_y: float
    radius: float


class LidarTracker:
    """Frame-differencing velocity estimator — the lidar-only ``Tracker``.

    Each tick it projects the live lidar frame to world points, drops returns
    that land on the STATIC inflated grid (walls / pillars and their inflation
    band), clusters the surviving dynamic returns by grid bucket, associates
    each cluster to the nearest cluster from the PRIOR frame, and estimates the
    velocity as ``(centroid_now - centroid_prev) / dt``.  The prior-frame
    centroids are the only mutable state.

    Determinism is the whole contract (the plan's "landmine"): the pipeline never
    iterates a ``set`` to build output or to order points, draws no RNG, and
    feeds every ``np.mean`` reduction a fixed, sorted-order array, so two
    ``update`` sequences on byte-identical inputs (same constructor args, same
    per-frame ``state`` / ``lidar``) return byte-identical ``Track`` lists — even
    across a frame where the cluster count changes (an obstacle enters / leaves).

    The first ``update`` has no prior frame, so it yields zero velocities; that
    is correct (one frame cannot reveal motion).
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        bearings: np.ndarray,
        range_max: float = float("inf"),
    ) -> None:
        """Store the static grid, the per-beam bearings, and the no-hit range.

        Parameters
        ----------
        grid:
            The STATIC inflated occupancy grid.  Read-only here: ``grid.cells``
            is consulted to subtract static returns and is never mutated.
        range_max:
            The lidar's max range (the no-hit sentinel).  Returns within
            ``RANGE_MAX_DEADBAND`` of it are dropped as no-hits before clustering
            (see ``_lidar_to_world_points``).  Defaults to ``inf``, which disables
            the cut so a tracker built without a range stays byte-compatible.
        bearings:
            The per-beam angles, already recovered by the caller as
            ``np.linspace(angle_min, angle_max, number)`` (the exact recovery the
            rest of the harness uses — NOT ``i * angle_increment``).  No YAML is
            loaded and no bearings are recomputed here.
        """
        if grid is None:
            raise ValueError("LidarTracker requires a non-None OccupancyGrid.")

        bearings_array = np.asarray(bearings, dtype=float)
        if bearings_array.ndim != 1:
            raise ValueError(
                "bearings must be a 1-D array of beam angles, received shape "
                f"{bearings_array.shape}."
            )

        self._grid = grid
        self._bearings = bearings_array
        self._range_max = float(range_max)
        # Prior-frame cluster centroids, stored in the current frame's
        # rep-cell-sorted order.  Empty until the first update(), so the first
        # update yields zero velocities.
        self._prev_centroids: list[tuple[float, float]] = []

    def update(
        self,
        *,
        snapshot: object,
        state: np.ndarray,
        lidar: np.ndarray,
        dt: float,
    ) -> list[Track]:
        """Estimate one Track per dynamic cluster, sorted by ``id`` ascending.

        Parameters
        ----------
        snapshot:
            Ignored — this tracker is lidar-only (accepted for protocol parity).
        state:
            ``(3,)`` ``[x, y, theta]`` robot pose in the world frame.
        lidar:
            ``(number,)`` range scan (NaN = no return), one entry per bearing.
        dt:
            Frame interval in seconds (the caller passes ``PREDICT_DT``); the
            velocity denominator.  Must be positive.
        """
        del snapshot  # lidar-only: the truth snapshot is ignored.

        state_array = np.asarray(state, dtype=float)
        if state_array.shape != (3,):
            raise ValueError(
                f"Expected (3,) [x, y, theta] state, received shape "
                f"{state_array.shape}."
            )

        lidar_array = np.asarray(lidar, dtype=float)
        expected_lidar_shape = (self._bearings.shape[0],)
        if lidar_array.shape != expected_lidar_shape:
            raise ValueError(
                f"Expected lidar of shape {expected_lidar_shape}, received "
                f"{lidar_array.shape}."
            )

        if not dt > 0.0:
            raise ValueError(f"dt must be positive, received {dt!r}.")

        world_points = self._lidar_to_world_points(state_array, lidar_array)
        dynamic_points = self._drop_static_returns(world_points)
        clusters = self._cluster(dynamic_points)
        return self._associate_and_build(clusters, float(dt))

    # --- Internal helpers ---------------------------------------------------

    def _lidar_to_world_points(
        self, state: np.ndarray, lidar: np.ndarray
    ) -> np.ndarray:
        """Project finite lidar returns to world-frame obstacle points.

        Delegates to the shared :func:`planners._geometry.lidar_to_world_points`
        (the one projection the family uses — formerly copy-pasted here, in DWA, and
        inline in the occupancy fold). Passing ``self._range_max`` enables the
        near-rim no-hit deadband: a beam coming back AT range_max with float jitter
        survives the Arena's ``>= range_max`` filter and would otherwise cluster into
        a phantom rim obstacle, so a RANGE_MAX_DEADBAND cut below range_max drops it
        (``range_max == inf`` keeps every finite beam). Returns an ``(N, 2)`` array,
        possibly empty, with the surviving points in beam order.
        """
        return lidar_to_world_points(
            state, lidar, self._bearings, range_max=self._range_max
        )

    def _drop_static_returns(self, points: np.ndarray) -> np.ndarray:
        """Drop every hit point whose grid cell is occupied in the static grid.

        Vectorized for speed but byte-equivalent to per-point
        ``world_to_grid`` + ``grid.cells[cell]`` (same ``np.floor`` / ``np.clip``
        clamp, same offset and resolution): a wall or pillar return lands on a
        ``True`` cell (the obstacle or its inflation band) and is removed,
        leaving only dynamic-obstacle returns.  Input beam ORDER is preserved.
        """
        if points.shape[0] == 0:
            return points

        cells = self._grid.cells
        rows, cols = cells.shape
        resolution = self._grid.resolution
        offset_x = float(self._grid.offset[0])
        offset_y = float(self._grid.offset[1])

        raw_col = (points[:, 0] - offset_x) / resolution
        raw_row = (points[:, 1] - offset_y) / resolution
        grid_col = np.clip(np.floor(raw_col), 0, cols - 1).astype(int)
        grid_row = np.clip(np.floor(raw_row), 0, rows - 1).astype(int)

        occupied = cells[grid_row, grid_col]
        return points[~occupied]

    def _cluster(self, points: np.ndarray) -> list[_Cluster]:
        """Deterministically cluster dynamic points into obstacle clusters.

        Buckets each point into an integer ``(row, col)`` cell at
        ``CLUSTER_RESOLUTION`` (row = y, col = x to match the grid convention),
        runs 8-connected connected components over the occupied buckets seeded in
        sorted bucket-key order, and reduces each component to a ``_Cluster``.
        Returns the clusters in component-seed order; the caller re-sorts by
        ``rep_cell`` before association.
        """
        if points.shape[0] == 0:
            return []

        # 1. Bucket points, preserving input (beam) order within each bucket.
        buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for index in range(points.shape[0]):
            point_x = float(points[index, 0])
            point_y = float(points[index, 1])
            bucket_key = (
                int(math.floor(point_y / CLUSTER_RESOLUTION)),
                int(math.floor(point_x / CLUSTER_RESOLUTION)),
            )
            buckets.setdefault(bucket_key, []).append((point_x, point_y))

        # 2. Connected components (8-connectivity) over the occupied buckets,
        #    seeding components in sorted bucket-key order so component output
        #    order is deterministic.  Flood-fill internal order is irrelevant
        #    because each component's members are re-sorted in _build_cluster.
        occupied_keys = set(buckets.keys())
        visited: set[tuple[int, int]] = set()
        clusters: list[_Cluster] = []
        for seed_key in sorted(occupied_keys):
            if seed_key in visited:
                continue
            component_keys: list[tuple[int, int]] = []
            stack = [seed_key]
            visited.add(seed_key)
            while stack:
                current = stack.pop()
                component_keys.append(current)
                current_row, current_col = current
                for delta_row in (-1, 0, 1):
                    for delta_col in (-1, 0, 1):
                        if delta_row == 0 and delta_col == 0:
                            continue
                        neighbor = (current_row + delta_row, current_col + delta_col)
                        if neighbor in occupied_keys and neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
            clusters.append(self._build_cluster(component_keys, buckets))

        return clusters

    def _build_cluster(
        self,
        component_keys: list[tuple[int, int]],
        buckets: dict[tuple[int, int], list[tuple[float, float]]],
    ) -> _Cluster:
        """Reduce one connected component to a centroid, radius, and stable id.

        Member points are gathered and sorted by ``(bucket_key, x, y)`` into a
        fixed order, so the centroid ``np.mean`` reduces an identically-ordered
        array every run.  The radius is the max centroid-to-member distance,
        floored at ``MIN_TRACK_RADIUS``.  The id is the smallest bucket key
        folded by ``CLUSTER_ID_MULTIPLIER``.
        """
        member_points: list[tuple[tuple[int, int], float, float]] = []
        for bucket_key in component_keys:
            for point_x, point_y in buckets[bucket_key]:
                member_points.append((bucket_key, point_x, point_y))
        # Lexicographic sort by (bucket_key, x, y) — a fixed, reproducible order.
        member_points.sort()

        points_array = np.array(
            [[point_x, point_y] for (_, point_x, point_y) in member_points],
            dtype=float,
        )
        centroid = np.mean(points_array, axis=0)
        centroid_x = float(centroid[0])
        centroid_y = float(centroid[1])

        deltas = points_array - centroid
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        radius = max(float(distances.max()), MIN_TRACK_RADIUS)

        rep_cell = min(component_keys)
        track_id = rep_cell[0] * CLUSTER_ID_MULTIPLIER + rep_cell[1]
        return _Cluster(
            track_id=track_id,
            rep_cell=rep_cell,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            radius=radius,
        )

    def _associate_and_build(
        self, clusters: list[_Cluster], dt: float
    ) -> list[Track]:
        """Associate current clusters to the prior frame and build Tracks.

        Greedy, sorted, first-match-wins: current clusters are processed in
        ``rep_cell`` ascending order, each claiming the nearest UNUSED prior
        centroid within ``MAX_ASSOCIATION_DISTANCE`` (distance ties broken by the
        lowest prior index).  A cluster with no prior in range gets zero velocity
        (a freshly-appeared obstacle).  The current centroids replace the prior
        state for the next tick.  Tracks are returned sorted by ``id``.
        """
        ordered = sorted(clusters, key=lambda cluster: cluster.rep_cell)
        prior = self._prev_centroids
        used = [False] * len(prior)

        tracks: list[Track] = []
        for cluster in ordered:
            best_index = -1
            best_distance: float | None = None
            for prior_index in range(len(prior)):
                if used[prior_index]:
                    continue
                prior_x, prior_y = prior[prior_index]
                delta_x = cluster.centroid_x - prior_x
                delta_y = cluster.centroid_y - prior_y
                distance = math.sqrt(delta_x * delta_x + delta_y * delta_y)
                # First-match-wins on equal distance: strict `<` keeps the lowest
                # prior index (we iterate prior_index ascending).
                if distance <= MAX_ASSOCIATION_DISTANCE and (
                    best_distance is None or distance < best_distance
                ):
                    best_distance = distance
                    best_index = prior_index

            if best_index >= 0:
                used[best_index] = True
                prior_x, prior_y = prior[best_index]
                velocity_x = (cluster.centroid_x - prior_x) / dt
                velocity_y = (cluster.centroid_y - prior_y) / dt
            else:
                velocity_x = 0.0
                velocity_y = 0.0

            tracks.append(
                Track(
                    id=cluster.track_id,
                    x=cluster.centroid_x,
                    y=cluster.centroid_y,
                    vx=velocity_x,
                    vy=velocity_y,
                    radius=cluster.radius,
                )
            )

        # Store this frame's centroids (in current sorted order) for next tick.
        # A frame with zero dynamic returns (all-static or all-NaN) leaves this
        # empty, so the NEXT frame restarts every track at zero velocity. That is
        # an inherent frame-differencing limitation, deterministic, and accepted
        # for v1 (the cone widening + gate + fail-open peel absorb a one-frame
        # zero-velocity blip).
        self._prev_centroids = [
            (cluster.centroid_x, cluster.centroid_y) for cluster in ordered
        ]

        tracks.sort(key=lambda track: track.id)
        return tracks


def predict_blocked_cells(
    tracks: list[Track],
    planned_path: list[np.ndarray],     # ordered (2,) world-frame waypoints (the robot's current committed path)
    robot_xy: np.ndarray,               # (2,) current robot position
    grid: OccupancyGrid,
    inflation: float,                   # = robot_radius + SAFETY_MARGIN (body-aware band, same as the static grid)
    horizon_steps: int,
    dt: float,
    *,
    geometry: str,                      # "capsule" | "cone"
    exclusion_radius: float,
    corridor_half_width: float,
) -> list[tuple[ThreatKey, list[tuple[int, int]]]]:
    """Predict the grid cells each tracked obstacle will threaten over the horizon.

    For every track, the future footprint is the union over lookahead steps
    ``k = 1..horizon_steps`` of the disk centered at
    ``(x + vx*k*dt, y + vy*k*dt)`` with radius ``r_k``:

    - ``geometry == "capsule"``: ``r_k = track.radius + inflation`` (constant — a
      straight disk train along the velocity vector).
    - ``geometry == "cone"``: ``r_k = track.radius + inflation +
      CONE_GROWTH_PER_STEP * k`` (radius grows linearly with the lookahead step,
      widening to represent estimator uncertainty).

    A track is kept only if its predicted disk geometrically intersects a
    corridor of half-width ``corridor_half_width`` around ``planned_path`` for
    SOME ``k`` (the predicted-conflict gate); its time-to-conflict is the
    smallest such ``k``.  Cells whose center lies within ``exclusion_radius`` of
    ``robot_xy`` are removed from every track's set (the robot exclusion zone).

    Returns one ``(ThreatKey(ttc_steps, track.id), sorted_unique_cells)`` group
    per gated track, the list sorted by ``ThreatKey`` ascending (soonest
    conflict first, then by id), each cell list sorted row-major ascending.

    The function is PURE and deterministic: plain floats/ints + numpy in,
    deterministically-sorted cells out.  No RNG, no irsim calls, no set-iteration
    leaking into the output order.  Two calls on identical inputs return
    byte-identical output.
    """
    if geometry not in {"capsule", "cone"}:
        raise ValueError(
            f"geometry must be 'capsule' or 'cone', received {geometry!r}."
        )

    # h0 / empty guards.  horizon_steps == 0 yields no centers and is the true
    # no-op baseline (AC2/TC57); an empty track list or an empty path likewise
    # produces nothing to stamp.
    if horizon_steps <= 0:
        return []
    if not tracks:
        return []
    if len(planned_path) < 1:
        return []

    # The conflict gate's polyline distance comes from manual_astar's canonical
    # helper, lazy-imported here (not at module top) so this module's top-level
    # imports stay free of the manual_astar coupling.
    from manual_astar import point_to_polyline_distance

    robot_x = float(robot_xy[0])
    robot_y = float(robot_xy[1])
    exclusion_radius_sq = exclusion_radius * exclusion_radius

    groups: list[tuple[ThreatKey, list[tuple[int, int]]]] = []

    for track in tracks:
        base_radius = track.radius + inflation
        ttc_steps: int | None = None
        cells: set[tuple[int, int]] = set()

        for k in range(1, horizon_steps + 1):
            center_x = track.x + track.vx * k * dt
            center_y = track.y + track.vy * k * dt

            if geometry == "capsule":
                r_k = base_radius
            else:  # "cone": radius grows linearly with the lookahead step.
                r_k = base_radius + CONE_GROWTH_PER_STEP * k

            # Predicted-conflict gate: does this disk reach the path corridor?
            # Reuses manual_astar's canonical point_to_polyline_distance (open
            # polyline) so the gate measures distance the SAME way the rest of the
            # harness does, with no drifting private copy.
            distance = point_to_polyline_distance(
                np.array([center_x, center_y], dtype=float), planned_path, closed=False
            )
            if distance <= r_k + corridor_half_width:
                if ttc_steps is None:
                    ttc_steps = k
                # Collect the full disk footprint via the shared _geometry scan —
                # the SAME bounding-box row-major scan _grid._mark_disk fills the
                # lidar fold from — so the predicted stamp can never drift from the
                # fold's geometry.
                cells.update(iter_disk_cells(grid, center_x, center_y, r_k))

        # Drop tracks whose footprint never reaches the corridor.
        if ttc_steps is None:
            continue

        # Robot exclusion zone: never stamp a cell whose center is within
        # exclusion_radius of the robot (AC5).
        if exclusion_radius > 0.0 and cells:
            resolution = grid.resolution
            offset_x = float(grid.offset[0])
            offset_y = float(grid.offset[1])
            kept: set[tuple[int, int]] = set()
            for (row, col) in cells:
                cell_center_x = offset_x + (col + 0.5) * resolution
                cell_center_y = offset_y + (row + 0.5) * resolution
                dx = cell_center_x - robot_x
                dy = cell_center_y - robot_y
                if dx * dx + dy * dy > exclusion_radius_sq:
                    kept.add((row, col))
            cells = kept

        # A track whose every cell fell inside the exclusion zone contributes
        # nothing; skip the empty group so the output carries only real stamps.
        if not cells:
            continue

        sorted_cells = sorted(cells)  # row-major ascending (row, then col)
        groups.append((ThreatKey(ttc_steps, track.id), sorted_cells))

    groups.sort(key=lambda group: group[0])
    return groups


@dataclass(frozen=True)
class TrajectoryConflict:
    """Result of a space-time robot-trajectory-vs-tracks conflict check.

    ``collides`` is True when the robot body overlaps some track's body at a
    MATCHED time step within the horizon; ``ttc_step`` is the earliest such step
    (1-based, None when no collision); ``min_gap`` is the minimum matched-time
    body gap (``center_dist - robot_radius - track_radius``) over every checked
    (step, track) pair (``+inf`` when there are no tracks / no steps to check).
    A surviving (non-colliding) candidate has ``min_gap > margin``, so the caller
    can turn ``min_gap`` into a "predicted clearance" score term.
    """

    collides: bool
    ttc_step: int | None
    min_gap: float


def trajectory_conflict(
    robot_positions: np.ndarray,   # (S, 2) robot world positions at steps k = 1..S (step k is dt*k ahead)
    tracks: list[Track],
    robot_radius: float,
    horizon_steps: int,
    dt: float,
    margin: float,
) -> TrajectoryConflict:
    """Space-time collision check: robot(k) vs each track's constant-velocity pose(k).

    For each lookahead step ``k = 1..min(horizon_steps, S)`` the robot is at
    ``robot_positions[k-1]`` and each track is at
    ``(track.x + track.vx*k*dt, track.y + track.vy*k*dt)`` — the SAME sim time, so
    this is genuine ``(x, y, t)`` reasoning, not a 2-D footprint stamp. A collision
    is registered when the matched-time body gap
    ``dist - robot_radius - track.radius`` drops to within ``margin``.
    ``ttc_step`` is the earliest colliding step (steps are scanned ascending);
    ``min_gap`` is the minimum gap over every checked pair (kept even past a
    collision so a rejected candidate still has a meaningful negative gap and a
    surviving one a positive clearance).

    PURE and deterministic: plain floats + numpy in, a plain dataclass out; the
    track list is iterated in caller order (no set-iteration), no RNG. Two calls on
    identical inputs return byte-identical results (AC4).
    """
    no_conflict = TrajectoryConflict(collides=False, ttc_step=None, min_gap=float("inf"))
    if horizon_steps <= 0 or not tracks or robot_positions.shape[0] == 0:
        return no_conflict

    steps = min(int(horizon_steps), int(robot_positions.shape[0]))
    min_gap = float("inf")
    ttc_step: int | None = None

    for k in range(1, steps + 1):
        robot_x = float(robot_positions[k - 1, 0])
        robot_y = float(robot_positions[k - 1, 1])
        for track in tracks:
            obstacle_x = track.x + track.vx * k * dt
            obstacle_y = track.y + track.vy * k * dt
            delta_x = robot_x - obstacle_x
            delta_y = robot_y - obstacle_y
            gap = math.sqrt(delta_x * delta_x + delta_y * delta_y) - robot_radius - track.radius
            if gap < min_gap:
                min_gap = gap
            if ttc_step is None and gap <= margin:
                ttc_step = k

    return TrajectoryConflict(
        collides=ttc_step is not None, ttc_step=ttc_step, min_gap=min_gap
    )
