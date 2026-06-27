"""planners/_predict.py — shared predictive substrate for motion-aware D* Lite.

This module is PURE: plain floats/ints in, deterministic output, no irsim,
no RNG, no set-iteration.  T4 will add predict_blocked_cells() here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

from manual_astar import OccupancyGrid, world_to_grid

# Prediction timestep in seconds.  Matches irsim step_time and DWA CONTROL_DT.
PREDICT_DT: float = 0.1

# Per-step radial growth (metres) applied to the cone geometry's stamp radius.
# The cone widens with the lookahead step to represent estimator uncertainty
# (zero for the oracle, nonzero for the lidar variant).  Chosen as roughly half
# a grid cell per step (GRID_RESOLUTION is 0.1 m, so ~0.05 m), a small default
# tuned later by the lidar variant (T12).  The capsule geometry ignores it.
CONE_GROWTH_PER_STEP: float = 0.05


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


def _point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Euclidean distance from point (px, py) to segment [(ax,ay), (bx,by)].

    Degenerate (zero-length) segments collapse to a point-to-point distance.
    Pure scalar arithmetic — no numpy boxing — so the gate stays deterministic.
    """
    seg_dx = bx - ax
    seg_dy = by - ay
    seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy

    if seg_len_sq < 1e-18:
        # Degenerate segment: distance to the shared endpoint.
        ex = px - ax
        ey = py - ay
        return float(np.sqrt(ex * ex + ey * ey))

    # Projection parameter of the point onto the (infinite) line, clamped to
    # the segment so we measure to the nearest point ON the segment.
    projection = ((px - ax) * seg_dx + (py - ay) * seg_dy) / seg_len_sq
    if projection < 0.0:
        projection = 0.0
    elif projection > 1.0:
        projection = 1.0

    closest_x = ax + projection * seg_dx
    closest_y = ay + projection * seg_dy
    dx = px - closest_x
    dy = py - closest_y
    return float(np.sqrt(dx * dx + dy * dy))


def _dist_point_to_polyline(
    px: float,
    py: float,
    polyline: list[np.ndarray],
) -> float:
    """Minimum distance from (px, py) to a polyline of (2,) world waypoints.

    A single-point polyline degrades to a point-to-point distance.  The caller
    guarantees ``len(polyline) >= 1``.
    """
    if len(polyline) == 1:
        only = polyline[0]
        dx = px - float(only[0])
        dy = py - float(only[1])
        return float(np.sqrt(dx * dx + dy * dy))

    min_distance = np.inf
    for index in range(len(polyline) - 1):
        start = polyline[index]
        end = polyline[index + 1]
        distance = _point_to_segment_distance(
            px,
            py,
            float(start[0]),
            float(start[1]),
            float(end[0]),
            float(end[1]),
        )
        if distance < min_distance:
            min_distance = distance
    return float(min_distance)


def _cell_center_within(
    grid: OccupancyGrid,
    cell: tuple[int, int],
    center_x: float,
    center_y: float,
    radius_sq: float,
) -> bool:
    """True iff `cell`'s center lies within sqrt(radius_sq) of (center_x, center_y).

    Uses the same cell-center formula as ``_mark_disk`` so membership matches the
    bounding-box scan exactly.
    """
    row, col = cell
    resolution = grid.resolution
    cell_center_x = float(grid.offset[0]) + (col + 0.5) * resolution
    cell_center_y = float(grid.offset[1]) + (row + 0.5) * resolution
    delta_x = cell_center_x - center_x
    delta_y = cell_center_y - center_y
    return delta_x * delta_x + delta_y * delta_y <= radius_sq


def _collect_disk_cells(
    grid: OccupancyGrid,
    center_x: float,
    center_y: float,
    radius: float,
    accumulator: set[tuple[int, int]],
    center_cell: tuple[int, int],
) -> None:
    """Append every grid cell whose CENTER lies within `radius` of (cx, cy).

    Mirrors ``_grid._mark_disk`` exactly — the same axis-aligned bounding box
    (clamped to the grid), the same row-major scan order, and the same cell
    center formula ``offset + (idx + 0.5) * resolution`` with the squared-radius
    membership test — but COLLECTS ``(row, col)`` tuples into `accumulator``
    instead of mutating a boolean array.  Keeping the scan discipline identical
    guarantees the collected set is independent of insertion order.

    ``center_cell`` is the caller's ``world_to_grid`` conversion of the disk
    center (the function-owns-the-conversion contract).  It is added only when it
    genuinely passes the disk membership test, so a future center that lies
    off-grid — where ``world_to_grid`` clamps to a non-covered border cell —
    never contributes a spurious cell.  This keeps the result a strict function
    of the disk geometry (a subset the bounding-box scan would also emit).
    """
    radius_sq = radius * radius

    if _cell_center_within(grid, center_cell, center_x, center_y, radius_sq):
        accumulator.add(center_cell)

    rows, cols = grid.shape
    resolution = grid.resolution
    offset_x = float(grid.offset[0])
    offset_y = float(grid.offset[1])

    # Bounding box of candidate cell indices (centers within `radius`).
    min_col = int(np.floor((center_x - radius - offset_x) / resolution))
    max_col = int(np.floor((center_x + radius - offset_x) / resolution))
    min_row = int(np.floor((center_y - radius - offset_y) / resolution))
    max_row = int(np.floor((center_y + radius - offset_y) / resolution))

    min_col = max(min_col, 0)
    max_col = min(max_col, cols - 1)
    min_row = max(min_row, 0)
    max_row = min(max_row, rows - 1)

    for row in range(min_row, max_row + 1):
        cell_center_y = offset_y + (row + 0.5) * resolution
        delta_y = cell_center_y - center_y
        for col in range(min_col, max_col + 1):
            cell_center_x = offset_x + (col + 0.5) * resolution
            delta_x = cell_center_x - center_x
            if delta_x * delta_x + delta_y * delta_y <= radius_sq:
                accumulator.add((row, col))


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
            distance = _dist_point_to_polyline(center_x, center_y, planned_path)
            if distance <= r_k + corridor_half_width:
                if ttc_steps is None:
                    ttc_steps = k
                # The function owns the world->grid conversion of each future
                # center via world_to_grid (the contract), then collects the full
                # disk footprint with the _mark_disk-mirroring bounding-box
                # row-major scan around that converted center.
                center_cell = world_to_grid(
                    np.array([center_x, center_y], dtype=float), grid
                )
                _collect_disk_cells(
                    grid, center_x, center_y, r_k, cells, center_cell
                )

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
