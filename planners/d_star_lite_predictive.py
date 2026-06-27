"""Motion-aware (predictive) D* Lite controllers.

These controllers subclass :class:`DStarLiteController` and override ONLY the
:meth:`DStarLiteController._extra_blocked_cells` hook (plus ``__init__`` and
``observe_truth``). They do NOT reimplement ``act()`` / ``reset()`` / the search,
so D* Lite's incremental invariants, grid-ownership contract, commitment horizon
and deferred settle are inherited unchanged. The prediction enters purely as
extra changed-occupancy cells through the existing fold -> diff -> ``update_cells``
seam.

Per tick the hook:

1. asks its :class:`~planners._predict.Tracker` for the current obstacle tracks
   (the oracle reads the live truth snapshot; the lidar variant frame-differences);
2. predicts each track's future footprint over the horizon via the pure
   :func:`~planners._predict.predict_blocked_cells` (threat-ordered groups,
   robot-exclusion zone already removed, gated to the planned-path corridor);
3. applies a per-tick stamped-cell area cap (soonest-TTC tracks first);
4. runs a threat-ordered, fail-open reachability peel so a stamp can never leave
   the grid unsolvable (drop farthest-future groups until a path re-exists);
5. returns the surviving cells, which the base ORs into the fold before the diff.

Determinism: every step above is deterministic — the tracker returns id-sorted
tracks, ``predict_blocked_cells`` returns sorted/deduped cells in threat order,
the area cap and peel are pure list operations, and the survivors are returned as
a sorted, deduped ``list[(row, col)]``. No RNG, no set-iteration leaks into the
output order.

The horizon-0 fast path is a true no-op (no tracker side effect, no stamp), so
``d_star_lite_oracle_h0`` produces a byte-identical trace to plain
``d_star_lite`` (AC2/TC57).
"""

from __future__ import annotations

import numpy as np

from manual_astar import OccupancyGrid, astar_search, world_to_grid
from planners._predict import (
    OracleTracker,
    PREDICT_DT,
    Tracker,
    predict_blocked_cells,
)
from planners.d_star_lite import DStarLiteController

# Per-tick stamped-cell area cap. The predictive hook stamps at most this many
# cells per tick, allocating the budget to the soonest-time-to-conflict tracks
# first (the area cap loop below stops at the first group that would overflow).
# This hard-bounds both the per-tick stamping cost and the fail-open peel's
# reachability-search frequency. ~6000 cells is a small fraction of a 50x50 grid
# at GRID_RESOLUTION 0.1 m (250000 cells total); it is a safety cap, tuned by the
# T10 horizon sweep, not a tight functional limit.
MAX_STAMP_CELLS: int = 6000


class PredictiveDStarLiteController(DStarLiteController):
    """Base for the motion-aware D* Lite family. Subclasses pick tracker + geometry.

    Subclasses set :attr:`geometry` (``"capsule"`` / ``"cone"``) and
    :attr:`wants_truth`, and implement :meth:`_make_tracker`. They MUST NOT
    override ``act()`` / ``reset()`` — the only behavioural override is
    :meth:`_extra_blocked_cells`.
    """

    # Prediction geometry; subclasses set "capsule" (oracle) or "cone" (lidar).
    geometry: str = "capsule"
    # Opt-in live-truth flag; the oracle sets True so the runner feeds it the
    # dynamic-obstacle snapshot via observe_truth().
    wants_truth: bool = False

    def _make_tracker(self) -> Tracker:
        """Return the velocity-source adapter (subclass responsibility)."""
        raise NotImplementedError(
            "PredictiveDStarLiteController subclasses must implement _make_tracker()."
        )

    def __init__(
        self,
        replan_k: int | None = None,
        predict_horizon: int | None = None,
    ) -> None:
        # The base rejects --replan-k for the predict family in build_controller
        # (it is not in REPLAN_FAMILIES), so replan_k is None here; pass it on for
        # the uniform construction seam.
        super().__init__(replan_k)

        if predict_horizon is None or int(predict_horizon) < 0:
            raise ValueError(
                f"predict_horizon must be a non-negative int, received {predict_horizon!r}."
            )
        self._horizon_steps: int = int(predict_horizon)
        self._tracker: Tracker = self._make_tracker()
        self._snapshot: tuple = ()

        # Read-only debug attributes for the render overlay (T16). Initialised to
        # [] (never None) so the overlay tolerates the pre-first-act case;
        # refreshed every act() via _extra_blocked_cells.
        self.last_predicted_cells: list[tuple[int, int]] = []
        self.last_tracks: list = []

    def observe_truth(self, snapshot: tuple) -> None:
        """Store the live dynamic-obstacle snapshot for the upcoming act() tick.

        Tick alignment is the runner's responsibility: it calls observe_truth
        with the snapshot from the SAME source call that produced the state/lidar
        the next act() receives.
        """
        self._snapshot = snapshot

    # ------------------------------------------------------------------ #
    # The predictive stamp hook                                          #
    # ------------------------------------------------------------------ #

    def _extra_blocked_cells(
        self, state: np.ndarray, lidar: np.ndarray, folded_new_cells: np.ndarray
    ) -> list[tuple[int, int]]:
        """Return the predicted extra-blocked cells for this tick.

        AC6 is structural here: the prediction body (tracker update -> predict ->
        area cap -> peel) is wrapped in a ``try/except Exception`` so a failed
        prediction tick degrades to ``[]`` (the controller behaves like plain
        D* Lite for that tick) instead of propagating out of ``act()`` and
        crashing the episode. The horizon-0 fast path stays outside the guard
        because it is already a pure no-op.
        """
        # h0 fast path: a true no-op. No tracker update (no side effect on the
        # trajectory) and no stamp, so h0 is byte-identical to plain d_star_lite
        # (AC2/TC57). Kept outside the guard below: it is already a pure no-op.
        if self._horizon_steps == 0:
            self.last_tracks = []
            self.last_predicted_cells = []
            return []

        # AC6: any failure in the prediction body degrades to no extra cells for
        # this tick (plain D* Lite behaviour) rather than crashing the episode.
        # except Exception (not bare) so KeyboardInterrupt/SystemExit propagate.
        try:
            robot_xy = np.asarray(state[:2], dtype=float)

            tracks = self._tracker.update(
                snapshot=self._snapshot, state=state, lidar=lidar, dt=PREDICT_DT
            )
            self.last_tracks = tracks

            planned_path = self._current_planned_path(state)

            groups = predict_blocked_cells(
                tracks,
                planned_path,
                robot_xy,
                self._grid,
                self._inflation,
                self._horizon_steps,
                PREDICT_DT,
                geometry=self.geometry,
                # The robot's own vicinity is never stamped (AC5); the body-aware
                # inflation band is the natural radius — a stamp inside it would seal
                # the robot in place. The path corridor uses the same band so the gate
                # is body-aware (the lidar is center-to-surface).
                exclusion_radius=self._inflation,
                corridor_half_width=self._inflation,
            )
            groups = self._apply_area_cap(groups)
            survivors = self._peel(groups, folded_new_cells, robot_xy)
            self.last_predicted_cells = survivors
            return survivors
        except Exception:
            # Degrade to "no extra cells": clear the debug attrs and behave like
            # plain D* Lite for this tick.
            self.last_tracks = []
            self.last_predicted_cells = []
            return []

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _current_planned_path(self, state: np.ndarray) -> list[np.ndarray]:
        """The robot's current committed path as a corridor polyline.

        ``[state[:2]]`` followed by the follower's remaining waypoints (from its
        current index onward). When no follower exists yet (defensive — reset()
        always builds one), the path degrades to the robot position alone.
        Private follower access is acceptable: DStarLiteController already couples
        tightly to its WaypointFollower.
        """
        head = np.asarray(state[:2], dtype=float)
        follower = self._follower
        if follower is None:
            return [head]
        remaining = follower._waypoints[follower.index:]
        return [head, *(np.asarray(wp, dtype=float) for wp in remaining)]

    def _apply_area_cap(
        self, groups: list[tuple[object, list[tuple[int, int]]]]
    ) -> list[tuple[object, list[tuple[int, int]]]]:
        """Cap the per-tick stamped-cell area, soonest-TTC tracks first.

        ``groups`` arrive threat-ordered (soonest time-to-conflict first). Keep
        groups greedily while the cumulative cell count stays within
        :data:`MAX_STAMP_CELLS`; stop at the first group that would overflow and
        skip the rest (the budget goes to the most imminent threats).
        """
        kept: list[tuple[object, list[tuple[int, int]]]] = []
        total = 0
        for group in groups:
            group_size = len(group[1])
            if total + group_size > MAX_STAMP_CELLS:
                break
            kept.append(group)
            total += group_size
        return kept

    def _peel(
        self,
        groups: list[tuple[object, list[tuple[int, int]]]],
        folded_new_cells: np.ndarray,
        robot_xy: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Threat-ordered, bounded, fail-open peel; return surviving cells.

        1. If the UNSTAMPED fold is already unsolvable, return [] — stamping
           cannot help, and the base D* Lite swallow keeps the last valid
           follower so act() never raises (AC6/TC56b). Do NOT raise.
        2. Otherwise stamp all kept groups; if that seals the grid, drop the LAST
           (farthest-future, least-imminent) group and retry, so the most
           imminent protection is retained (AC5/TC56).
        3. Return the union of surviving cells, sorted and deduped.
        """
        # Step 1: a real dead-end (even zero prediction is unsolvable) -> stamp
        # nothing; the base keeps committing to its last valid follower.
        if not self._reachable(folded_new_cells, robot_xy):
            return []

        # Step 2: peel farthest-future first until the trial grid is solvable.
        kept = list(groups)
        while kept:
            trial = folded_new_cells.copy()
            for _key, cells in kept:
                for row, col in cells:
                    trial[row, col] = True
            if self._reachable(trial, robot_xy):
                break
            kept.pop()  # drop the last (least-imminent) group and retry

        # Step 3: union the survivors, sorted + deduped row-major.
        survivors: set[tuple[int, int]] = set()
        for _key, cells in kept:
            survivors.update(cells)
        return sorted(survivors)

    def _reachable(self, cells: np.ndarray, robot_xy: np.ndarray) -> bool:
        """Is the goal reachable from the robot's current cell on ``cells``?

        Uses the same traversability as the search (8-connected, octile cost, no
        corner cutting) by running ``manual_astar.astar_search`` over an
        OccupancyGrid wrapping ``cells`` (reusing this grid's resolution/offset).
        ``astar_search`` RAISES RuntimeError when no path exists, so a raise (or
        any geometry error) is treated as "unreachable".

        Note: this peel cost shows only in total episode wall time, never in the
        runner's wallclock_per_step metric (which times only irsim's env.step,
        not act()).
        """
        if self._grid is None or self._goal_xy is None:
            return False
        robot_cell = world_to_grid(robot_xy, self._grid)
        goal_cell = world_to_grid(self._goal_xy, self._grid)
        probe_grid = OccupancyGrid(
            cells=cells,
            resolution=self._grid.resolution,
            offset=self._grid.offset,
        )
        try:
            astar_search(probe_grid, robot_cell, goal_cell)
        except (RuntimeError, ValueError):
            return False
        return True


class DStarLiteOracleController(PredictiveDStarLiteController):
    """Oracle-fed predictive D* Lite: perfect velocities, capsule geometry.

    A deliberate cheat (perfect live velocities via the truth seam) that measures
    the achievable motion-aware ceiling. Capsule geometry: with exact velocities
    each obstacle's future footprint is a straight constant-radius disk train, so
    no widening is warranted.
    """

    name = "d_star_lite_oracle"
    geometry = "capsule"
    wants_truth = True

    def _make_tracker(self) -> Tracker:
        return OracleTracker()


# Self-register at import (mirrors d_star_lite.py). Imported after the
# module-level imports so the registry sees the fully-defined class.
from planners._grid import register  # noqa: E402

register("d_star_lite_oracle", DStarLiteOracleController)
