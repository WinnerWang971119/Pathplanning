"""Grid-geometry primitives shared across the planner family.

The CODE here is pure: numpy in, deterministic output, no irsim/matplotlib/RNG
calls and no set-iteration leaking into output order. It is the lowest layer of the
planner stack — ``_grid``, ``dwa`` and ``_predict`` all build on it.

NOTE on imports: this is a ``planners`` submodule, so a bare
``import planners._geometry`` still runs the package ``__init__``, which eagerly
imports the controllers and therefore pulls irsim + matplotlib (the documented
"importing planners pulls irsim + matplotlib" gotcha). "Pure" here means the logic
is deterministic and irsim-free, NOT that importing the module is side-effect-free.
Headless tools must still lazy-import ``planners`` symbols inside functions; they do
not gain an irsim-free import by reaching for this module.

It hosts the two geometry primitives that used to be copy-pasted across the family,
so a fix to either now lands in one place instead of three:

- :func:`iter_disk_cells` — the row-major bounding-box disk-cell scan. The single
  source of truth behind both ``_grid._mark_disk``'s boolean-array fill and
  ``_predict.predict_blocked_cells``'s predicted-footprint collection, so the
  predicted stamp can never drift from the cells the lidar fold marks.
- :func:`lidar_to_world_points` — the finite-beam lidar -> world projection. The
  single source behind ``_grid.lidar_to_occupancy``'s fold, ``DWA``'s obstacle
  points, and the ``LidarTracker``'s velocity estimation, with an optional near-rim
  no-hit deadband for the tracker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    # Annotation-only: `from __future__ import annotations` stringifies every hint,
    # so OccupancyGrid is never evaluated at runtime and need not be imported then.
    # Keeping it out of the runtime import path is what makes this module pure
    # (manual_astar pulls irsim).
    from manual_astar import OccupancyGrid

# Deadband (metres) below the lidar's range_max within which a return is treated
# as a NO-HIT, not an obstacle. irsim returns a no-hit beam AT range_max, but
# float jitter scatters a few just under it; the Arena's strict ``< range_max``
# filter only catches the >= side, so those survivors reach a consumer as a ring
# of points at the sensing rim. Without this cut they cluster into phantom
# "obstacles" with spurious estimated velocities (the rim moves with the
# robot). 0.05 m sits in the clean gap between real returns (<= ~4.9 m here) and
# the ~range_max no-hit ring, and costs only the outermost 5 cm of a 5 m sensor.
# Only consumers that pass a finite ``range_max`` to :func:`lidar_to_world_points`
# (the LidarTracker) apply it; the occupancy fold / DWA leave range_max at inf.
RANGE_MAX_DEADBAND: float = 0.05


def iter_disk_cells(
    grid: "OccupancyGrid",
    center_x: float,
    center_y: float,
    radius: float,
) -> Iterator[tuple[int, int]]:
    """Yield every grid cell whose CENTER lies within ``radius`` of (cx, cy).

    Iterates only the axis-aligned bounding box of candidate cells (clamped to the
    grid) in stable row-major order, applying the squared-radius cell-center
    membership test ``offset + (idx + 0.5) * resolution``. This is the ONE disk-cell
    scan the family shares: ``_grid._mark_disk`` fills a boolean array from it and
    ``_predict.predict_blocked_cells`` collects a cell set from it, so the predicted
    footprint and the lidar fold can never disagree on which cells a disk covers.

    The scan order is row-major and fully determined by the disk geometry, so the
    set a caller accumulates is independent of insertion order.
    """
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

    radius_sq = radius * radius
    for row in range(min_row, max_row + 1):
        cell_center_y = offset_y + (row + 0.5) * resolution
        delta_y = cell_center_y - center_y
        for col in range(min_col, max_col + 1):
            cell_center_x = offset_x + (col + 0.5) * resolution
            delta_x = cell_center_x - center_x
            if delta_x * delta_x + delta_y * delta_y <= radius_sq:
                yield (row, col)


def lidar_to_world_points(
    state: np.ndarray,
    lidar: np.ndarray,
    bearings: np.ndarray,
    range_max: float = float("inf"),
) -> np.ndarray:
    """Project finite lidar returns to ``(N, 2)`` world-frame points, in beam order.

    For beam ``i`` with an accepted range ``r`` the world bearing is
    ``theta + bearings[i]`` and the hit is ``(x + r*cos, y + r*sin)``. A beam is
    accepted iff its range is finite AND below ``range_max - RANGE_MAX_DEADBAND``.
    The default ``range_max == inf`` accepts every finite beam (``r < inf`` is always
    true), so the occupancy fold and DWA — which want the full scan — simply omit it;
    the LidarTracker passes its sensor max so the near-rim no-hit ring is dropped.

    Returns an empty ``(0, 2)`` array when no beam is accepted. Beam ORDER is
    preserved (the mask keeps the surviving beams in index order), which the
    occupancy fold relies on for a deterministic mark sequence.
    """
    ranges = np.asarray(lidar, dtype=float)
    bearings_array = np.asarray(bearings, dtype=float)
    # `ranges < range_max - DEADBAND`: with range_max == inf this is `ranges < inf`,
    # true for every finite range, so the deadband is a no-op for the inf default.
    valid_mask = np.isfinite(ranges) & (ranges < range_max - RANGE_MAX_DEADBAND)
    if not valid_mask.any():
        return np.empty((0, 2), dtype=float)

    finite_ranges = ranges[valid_mask]
    world_angles = float(state[2]) + bearings_array[valid_mask]
    hit_x = float(state[0]) + finite_ranges * np.cos(world_angles)
    hit_y = float(state[1]) + finite_ranges * np.sin(world_angles)
    return np.column_stack((hit_x, hit_y))
