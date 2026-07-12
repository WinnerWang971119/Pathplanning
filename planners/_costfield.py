"""planners/_costfield.py — Dijkstra-from-goal cost-to-go field builder.

This module builds a static, plan-once "cost-to-go" field over the occupancy
grid: for every reachable cell, the octile-distance cost of the cheapest path
from that cell to the goal. It is the guidance substrate the global-guidance
DWA-predictive variant (T4, ``planners/dwa_predictive.py``) uses to score
candidates by geodesic progress instead of straight-line heading, which is
immune to the local-minima pathology a Euclidean heading term hits behind a
wall segment.

Cost model — identical to ``manual_astar.astar_search``:
- 8-connected neighbors.
- Octile step cost via ``np.hypot(delta_row, delta_col)`` in CELL units
  (orthogonal step cost 1.0, diagonal step cost sqrt(2)).
- No corner-cutting: a diagonal move is blocked if EITHER of the two
  orthogonal cells it passes between is occupied, even if the diagonal
  target cell itself is free.

This is plain single-source Dijkstra run BACKWARD from the goal (so one
field serves every candidate rollout endpoint in a single planning pass,
rather than re-running A* per candidate). Ties in the priority queue break on
cell index, exactly like ``astar_search``'s ``(priority, cell)`` heap, so the
traversal order — and therefore the field — is fully deterministic.

PURE: this module imports only ``manual_astar.OccupancyGrid``, ``numpy``, and
the standard-library ``heapq``. It performs no irsim calls and holds no
mutable module state, so it is safe to import from a headless context that
must not pull irsim — though in practice its only consumer,
``planners/dwa_predictive.py``, already imports the full ``planners``
package (and therefore irsim) at import time; see
[[gotcha-planners-import-pulls-irsim]].
"""

from __future__ import annotations

import heapq

import numpy as np

from manual_astar import OccupancyGrid

# The 8-connected neighbor offsets, in (delta_row, delta_col) form — identical
# ordering to ``manual_astar.astar_search`` (not that traversal order matters
# for Dijkstra's final distances, but it keeps the two implementations easy to
# compare line-by-line).
_NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def build_cost_to_go_field(grid: OccupancyGrid, goal_cell: tuple[int, int]) -> np.ndarray:
    """Return the octile goal-distance (in CELL units) for every grid cell.

    Runs single-source Dijkstra rooted at ``goal_cell`` over ``grid.cells``
    (``True`` = occupied), using the SAME cost model as
    ``manual_astar.astar_search``: 8-connected neighbors, octile step cost
    (``np.hypot(delta_row, delta_col)``) in cell units, and no corner-cutting
    (a diagonal move is rejected if either orthogonal intermediate cell is
    occupied). Because edge costs and connectivity are symmetric, the
    resulting distance from any reachable cell to ``goal_cell`` equals the
    cost a fresh ``astar_search`` from that cell to the goal would return.

    Occupied cells and cells with no path to the goal (through the free-space
    connected component containing ``goal_cell``) are ``np.inf``. Matching
    ``astar_search``'s own convention, the seed cell's occupancy is never
    checked — only a NEIGHBOR being relaxed into is checked against
    ``grid.cells`` — so an occupied ``goal_cell`` is not special-cased: it
    still seeds normally, and its distance-0 entry still relaxes into any
    free neighbor. A goal cell with only occupied neighbors is therefore the
    one case where nothing beyond the goal itself is ever relaxed, leaving
    every other cell ``np.inf``; callers that care should guard against an
    occupied (or walled-in) goal before consuming the field.

    The open set is a ``heapq`` of ``(distance, (row, col))`` tuples, so ties
    at equal distance break on cell index in ascending row-major order —
    exactly like ``astar_search``'s ``(priority, cell)`` heap — making the
    field fully deterministic across repeated calls on the same grid.

    Args:
        grid: the static occupancy grid to search over.
        goal_cell: the ``(row, col)`` cell to root the field at (distances are
            "distance-to-here", i.e. this is a Dijkstra-from-goal, not a
            Dijkstra-from-start).

    Returns:
        A ``grid.shape`` (rows, cols) float64 array. Reachable cells hold
        their octile-distance cost to ``goal_cell``; occupied and unreachable
        cells hold ``np.inf``.
    """
    rows, cols = grid.shape
    dist = np.full((rows, cols), np.inf, dtype=np.float64)

    goal_row, goal_col = goal_cell
    dist[goal_row, goal_col] = 0.0

    open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, goal_cell)]

    while open_heap:
        current_dist, current = heapq.heappop(open_heap)
        current_row, current_col = current

        if current_dist > dist[current_row, current_col]:
            # Stale heap entry superseded by a cheaper relaxation already
            # popped for this cell — skip it.
            continue

        for delta_row, delta_col in _NEIGHBOR_OFFSETS:
            neighbor_row = current_row + delta_row
            neighbor_col = current_col + delta_col

            if not (0 <= neighbor_row < rows and 0 <= neighbor_col < cols):
                continue
            if grid.cells[neighbor_row, neighbor_col]:
                continue

            if delta_row != 0 and delta_col != 0:
                row_neighbor_occupied = grid.cells[current_row + delta_row, current_col]
                col_neighbor_occupied = grid.cells[current_row, current_col + delta_col]
                if row_neighbor_occupied or col_neighbor_occupied:
                    # No corner-cutting: both orthogonal cells the diagonal
                    # move passes between must be free.
                    continue

            step_cost = float(np.hypot(delta_row, delta_col))
            tentative_dist = current_dist + step_cost

            if tentative_dist >= dist[neighbor_row, neighbor_col]:
                continue

            dist[neighbor_row, neighbor_col] = tentative_dist
            heapq.heappush(open_heap, (tentative_dist, (neighbor_row, neighbor_col)))

    return dist
