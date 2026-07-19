"""Motion-aware (predictive) D* Lite controllers.

These controllers subclass :class:`DStarLiteController` and override the
:meth:`DStarLiteController._extra_blocked_cells` fold hook AND the
:meth:`DStarLiteController._settle_and_extract` settle hook (plus ``__init__`` and
``observe_truth``). They do NOT reimplement ``act()`` / ``reset()`` / the search,
so D* Lite's incremental invariants, grid-ownership contract, commitment horizon
and deferred settle are inherited unchanged. The prediction enters purely as
extra changed-occupancy cells through the existing fold -> diff -> ``update_cells``
seam.

Per-tick (cheap) — ``_extra_blocked_cells``:

1. asks its :class:`~planners._predict.Tracker` for the current obstacle tracks
   (the oracle reads the live truth snapshot; the lidar variant estimates velocities
   via the Kalman multi-object tracker);
2. predicts each track's future footprint over the horizon via the pure
   :func:`~planners._predict.predict_blocked_cells` (threat-ordered groups,
   robot-exclusion zone already removed, gated to the planned-path corridor);
3. applies a per-tick stamped-cell area cap (soonest-TTC tracks first);
4. STORES the threat-ordered groups and the un-stamped fold for the settle, and
   returns the FULL stamp (the union of all groups' cells). The base ORs that
   union into the fold before the diff, so the whole predicted footprint enters
   ``self._cells`` this tick.

This per-tick hook does NO reachability search — the expensive part is deferred.

At settle-time (rare — only when the follower finishes or its committed segment is
blocked) — ``_settle_and_extract``:

1. first settles + extracts with the FULL stamp already committed. This REUSES
   the D* Lite search's own ``g``-values: ``compute_shortest_path`` +
   ``extract_path`` returns a path exactly when ``g(start)`` is finite, and raises
   (-> None) exactly when the stamp sealed the grid. No separate from-scratch A*
   reachability probe is run — the incremental search already knows reachability.
2. if the full stamp sealed the grid, peels predicted groups farthest-future
   first (drop the least-imminent group, un-stamp its cells from ``self._cells``
   via ``update_cells``, re-settle incrementally) until a path re-exists, so the
   most-imminent protection is retained (AC5).
3. if even zero predicted stamp leaves the grid unsolvable (a real static
   dead-end), returns None so the base keeps its last valid follower — ``act()``
   never raises (AC6).

Why this is correct and fast. The reachability question the old per-tick A* probe
answered ("can the robot still reach the goal with this stamp?") is exactly what
``g(start) < inf`` answers after a settle — and the settle is incremental
(``move_start`` + batched ``update_cells`` accumulate the edge changes; one
``compute_shortest_path`` folds them into the same optimum a from-scratch A* would
find, the TC46 property). Reusing the search drops the per-tick cost back to
near-baseline: on a clear run no settle fires, so no search runs at all, and a
settle that does fire is incremental (cheap) plus, only when the stamp seals the
map, a bounded re-settle per peeled group.

Determinism: every step is deterministic — the tracker returns id-sorted tracks,
``predict_blocked_cells`` returns sorted/deduped cells in threat order, the area
cap and peel are pure list operations over the threat-ordered groups, the peel
drops groups from a fixed (soonest-first) order, and ``update_cells`` is reported
with sorted cell lists. No RNG, no set-iteration leaks into the output or the
update order.

Grid ownership: the peel mutates ``self._cells`` IN PLACE at the un-stamped
positions and reports them via ``self._search.update_cells`` — it NEVER rebinds
``self._cells`` (rebinding would detach the search's occupancy mirror, the same
load-bearing invariant the base relies on).

The horizon-0 fast path is a true no-op (no tracker side effect, no stamp, no
pending groups), so ``d_star_lite_oracle_h0`` produces a byte-identical trace to
plain ``d_star_lite`` (AC2/TC57).
"""

from __future__ import annotations

import numpy as np

from planners._predict import (
    LidarTracker,
    OracleTracker,
    PREDICT_DT,
    Tracker,
    predict_blocked_cells,
)
from planners._types import Path
from planners.d_star_lite import DStarLiteController

# Per-tick stamped-cell area cap. The predictive hook stamps at most this many
# cells per tick, allocating the budget to the soonest-time-to-conflict tracks
# first (the area cap loop below stops at the first group that would overflow).
# This hard-bounds both the per-tick stamping cost and the number of groups the
# fail-open settle-peel may have to drop. ~6000 cells is a small fraction of a
# 50x50 grid at GRID_RESOLUTION 0.1 m (250000 cells total); it is a safety cap,
# tuned by the T10 horizon sweep, not a tight functional limit.
MAX_STAMP_CELLS: int = 6000


class PredictiveDStarLiteController(DStarLiteController):
    """Base for the motion-aware D* Lite family. Subclasses pick tracker + geometry.

    Subclasses set :attr:`geometry` (``"capsule"`` / ``"cone"``) and
    :attr:`wants_truth`, and implement :meth:`_make_tracker`. They MUST NOT
    override ``act()`` / ``reset()`` — the only behavioural overrides are
    :meth:`_extra_blocked_cells` (the per-tick fold stamp) and
    :meth:`_settle_and_extract` (the settle-time fail-open peel).
    """

    # Abstract base: it carries no registry key. The concrete subclasses
    # (oracle / lidar) set their own real `name`. Blanking it here prevents the
    # base from inheriting d_star_lite's name and masquerading as that key.
    name = ""

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
        # Tracker construction is DEFERRED to first use (first non-h0 act()).
        # The lidar variant's tracker needs self._grid / self._geom, which only
        # exist after reset(); the oracle's tracker is stateless, so lazy build
        # is behaviourally identical for it. Built lazily inside
        # _extra_blocked_cells' try block so a _make_tracker() failure degrades
        # the tick (AC6) instead of crashing act().
        self._tracker: Tracker | None = None
        self._snapshot: tuple = ()

        # Settle-peel state, refreshed every act() by _extra_blocked_cells and
        # consumed by _settle_and_extract. `_pending_groups` is the threat-ordered
        # (soonest-TTC-first) list of (ThreatKey, cells) stamped this tick;
        # `_last_fold` is the un-stamped fold (False where the stamp added a cell),
        # needed to restore a peeled cell to its true fold value. Initialised so a
        # defensive early settle is a no-op (empty groups -> no peel).
        self._pending_groups: list[tuple[object, list[tuple[int, int]]]] = []
        self._last_fold: np.ndarray | None = None

        # Read-only debug attributes for the render overlay (T16). Initialised to
        # [] (never None) so the overlay tolerates the pre-first-act case;
        # refreshed every act() via _extra_blocked_cells.
        self.last_predicted_cells: list[tuple[int, int]] = []
        self.last_tracks: list = []

    def reset(
        self,
        world_yaml: str,
        initial_snapshot: tuple,
        lidar0: np.ndarray,
        state0: np.ndarray,
    ) -> None:
        """Rebuild the static substrate AND clear this subclass's per-episode state.

        The base ``reset`` only re-initializes its own search / cells / follower; it
        knows nothing about the tracker, the stored snapshot, or the pending-peel
        state this subclass adds in ``__init__``. The ``Controller`` contract allows
        an instance to be reset and reused for a second episode, so without this
        override the lazy ``if self._tracker is None`` guard would be skipped on
        reuse and episode 2's first frame would be tracked against episode 1's
        final track state — bogus velocities, phantom stamps. Null every added member
        back to its constructed state (``_horizon_steps`` / ``geometry`` /
        ``wants_truth`` are construction-time config, not episode state, so they are
        left intact). The tracker rebuilds lazily on the next non-h0 ``act()``.
        """
        super().reset(world_yaml, initial_snapshot, lidar0, state0)
        self._tracker = None
        self._snapshot = ()
        self._pending_groups = []
        self._last_fold = None
        self.last_predicted_cells = []
        self.last_tracks = []

    def observe_truth(self, snapshot: tuple) -> None:
        """Store the live dynamic-obstacle snapshot for the upcoming act() tick.

        Tick alignment is the runner's responsibility: it calls observe_truth
        with the snapshot from the SAME source call that produced the state/lidar
        the next act() receives.
        """
        self._snapshot = snapshot

    # ------------------------------------------------------------------ #
    # The predictive stamp hook (per tick, cheap — no reachability probe) #
    # ------------------------------------------------------------------ #

    def _extra_blocked_cells(
        self, state: np.ndarray, lidar: np.ndarray, folded_new_cells: np.ndarray
    ) -> list[tuple[int, int]]:
        """Return the full predicted stamp for this tick (no peel here).

        Cheap by design: tracker update -> predict -> area cap, then STORE the
        threat-ordered groups + the un-stamped fold for the settle-time peel and
        RETURN the union of all groups' cells (the full stamp). The reachability
        peel that used to run a from-scratch A* every stamp-bearing tick now lives
        in :meth:`_settle_and_extract`, which reuses the D* Lite search's own
        g-values and only fires when a fresh path is actually needed.

        AC6 is structural here: the prediction body is wrapped in a
        ``try/except Exception`` so a failed prediction tick degrades to ``[]``
        (the controller behaves like plain D* Lite for that tick) instead of
        propagating out of ``act()``. The horizon-0 fast path stays outside the
        guard because it is already a pure no-op.
        """
        # h0 fast path: a true no-op. No tracker update (no side effect on the
        # trajectory), no stamp, no pending groups, so h0 is byte-identical to
        # plain d_star_lite (AC2/TC57). Kept outside the guard: it cannot fail.
        if self._horizon_steps == 0:
            self.last_tracks = []
            self.last_predicted_cells = []
            self._pending_groups = []
            return []

        # AC6: any failure in the prediction body degrades to no extra cells for
        # this tick (plain D* Lite behaviour) rather than crashing the episode.
        # except Exception (not bare) so KeyboardInterrupt/SystemExit propagate.
        try:
            robot_xy = np.asarray(state[:2], dtype=float)

            # Lazy tracker build (first non-h0 act() after reset()). Inside the
            # try so a _make_tracker() failure degrades to no extra cells (AC6),
            # matching the plain-D*-Lite fallback for any other prediction fault.
            if self._tracker is None:
                self._tracker = self._make_tracker()

            tracks = self._tracker.update(
                snapshot=self._snapshot, state=state, lidar=lidar, dt=PREDICT_DT
            )
            self.last_tracks = tracks

            # No tracks -> nothing to predict or stamp. Clear the pending groups
            # so a settle this tick does no peel work.
            if not tracks:
                self.last_predicted_cells = []
                self._pending_groups = []
                return []

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

            # No gated threats survived the cap -> no stamp, no pending groups.
            if not groups:
                self.last_predicted_cells = []
                self._pending_groups = []
                return []

            # Store the threat-ordered groups (soonest-TTC first) and the
            # un-stamped fold so the settle-time peel can drop the least-imminent
            # groups and restore their cells to the true fold value.
            self._pending_groups = groups
            self._last_fold = folded_new_cells.copy()

            # The full stamp: the union of ALL groups' cells, sorted + deduped
            # row-major. Enters the fold this tick; the settle-time peel removes
            # farthest-future groups only if this full stamp seals the grid.
            stamped: set[tuple[int, int]] = set()
            for _key, cells in groups:
                stamped.update(cells)
            survivors = sorted(stamped)
            self.last_predicted_cells = survivors
            return survivors
        except Exception:
            # Degrade to "no extra cells": clear the debug attrs + pending groups
            # and behave like plain D* Lite for this tick.
            self.last_tracks = []
            self.last_predicted_cells = []
            self._pending_groups = []
            return []

    # ------------------------------------------------------------------ #
    # The settle-time fail-open peel (rare — reuses the search's g-values) #
    # ------------------------------------------------------------------ #

    def _settle_and_extract(self, position: np.ndarray) -> Path | None:
        """Settle + extract, peeling the predicted stamp only if it sealed the grid.

        First settle/extract with the FULL stamp already committed into
        ``self._cells``. This reuses the search's reachability: the base
        :meth:`DStarLiteController._settle_and_extract` runs
        ``compute_shortest_path`` + ``extract_path`` and returns None exactly when
        ``g(start)`` is infinite (the stamp sealed the robot off from the goal).

        On a seal, peel farthest-future groups (drop the least-imminent first —
        ``_pending_groups`` is soonest-TTC-first, so pop from the END), un-stamping
        each dropped group's cells from ``self._cells`` and re-settling
        incrementally, until a path re-exists. If even zero predicted stamp leaves
        the grid unsolvable (a real static dead-end), return None so the base keeps
        the last valid follower — ``act()`` never raises (AC6).
        """
        # Common path: the full stamp did not seal the grid (or there is no stamp
        # at all). The base settle reuses the incremental g-values; no peel needed.
        result = super()._settle_and_extract(position)
        if result is not None:
            return result

        # S1: wrapping the peel makes AC6 structural even if the
        # _pending_groups/_last_fold co-invariant is ever broken (e.g. by a future
        # subclass) -- any peel exception falls back to the un-peeled base settle.
        # except Exception (not bare) so KeyboardInterrupt/SystemExit propagate.
        try:
            # Sealed by the stamp (g(start) is infinite): peel the least-imminent
            # group, restore its cells, and re-settle until a path re-exists.
            kept = list(self._pending_groups)
            while kept:
                dropped = kept.pop()  # farthest-future (least-imminent) group
                self._unstamp_group(dropped, kept)
                result = super()._settle_and_extract(position)
                if result is not None:
                    # Render overlay (T16): expose the POST-peel stamp so the debug
                    # view never paints cells the peel un-stamped (which the robot
                    # then routed through). Render-only — `self._cells` already
                    # reflects the peel; this just refreshes the read-only debug attr
                    # to the cells that are STILL stamped (the surviving groups).
                    self.last_predicted_cells = self._stamped_cells(kept)
                    return result

            # Even with zero predicted stamp the grid is unsolvable: a real static
            # dead-end. Every predicted group was peeled, so nothing is stamped now.
            self.last_predicted_cells = []
            return None
        except Exception:
            # Fall back to the un-peeled settle. The base returns None on failure,
            # so the base keeps the last valid follower and act() never raises.
            return super()._settle_and_extract(position)

    def _unstamp_group(
        self,
        dropped: tuple[object, list[tuple[int, int]]],
        kept: list[tuple[object, list[tuple[int, int]]]],
    ) -> None:
        """Remove a dropped group's stamp-only cells from the live search grid.

        A dropped cell is un-stamped ONLY when it is currently stamped
        (``True`` in ``self._cells``), is NOT retained by any KEPT group, and was
        ``False`` in the un-stamped fold (i.e. it was stamp-only, not a real fold
        obstacle). Such a cell is restored to its fold value (``False``) IN PLACE
        and reported through :meth:`DStarLiteSearch.update_cells`, so the search
        re-syncs its occupancy mirror and repairs those vertices for the next
        incremental re-settle. ``self._cells`` is NEVER rebound (grid-ownership
        invariant).
        """
        # Union of the kept groups' cells: these must stay stamped, so they are
        # never un-stamped even if `dropped` also lists them.
        kept_cells: set[tuple[int, int]] = set()
        for _key, cells in kept:
            kept_cells.update(cells)

        _dropped_key, dropped_cells = dropped
        changed: list[tuple[int, int]] = []
        for cell in dropped_cells:
            row, col = cell
            if not self._cells[row, col]:
                continue  # already cleared (a prior peel iteration, or never set)
            if cell in kept_cells:
                continue  # a kept group still needs this cell stamped
            if self._last_fold[row, col]:
                continue  # a real fold obstacle, not stamp-only — never erase it
            self._cells[row, col] = self._last_fold[row, col]  # restore to False
            changed.append(cell)

        if changed:
            self._search.update_cells(sorted(changed))

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
        # NOTE: `follower.index` is read here during the fold, BEFORE the follower
        # advances it later in this same tick (its current_waypoint() runs inside
        # the base act() AFTER this hook). On a tick where the robot has just reached
        # its target waypoint the corridor therefore leads with one already-passed
        # waypoint ([head, reached_wp, next_wp, ...]). The effect is bounded and
        # safe: it only OVER-includes corridor area near the robot's recent past
        # (nudging the ttc ordering at worst) — it can never drop a real threat,
        # since a slightly longer corridor only ever admits MORE stamps, never fewer.
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

        The single most-imminent group is kept UNCONDITIONALLY — even if it alone
        exceeds the budget. Dropping it would silently degrade the controller to
        plain D* Lite for exactly the threat the cap claims to prioritize, so the
        cap can never strip the highest-priority group; it only bounds the tail.
        """
        kept: list[tuple[object, list[tuple[int, int]]]] = []
        total = 0
        for group in groups:
            group_size = len(group[1])
            # `kept and ...`: the first (most-imminent) group is appended before
            # this guard can fire, so an oversized leading group is always kept;
            # every later group is capped normally.
            if kept and total + group_size > MAX_STAMP_CELLS:
                break
            kept.append(group)
            total += group_size
        return kept

    @staticmethod
    def _stamped_cells(
        groups: list[tuple[object, list[tuple[int, int]]]]
    ) -> list[tuple[int, int]]:
        """The sorted, deduped union of `groups`' cells.

        After a peel the still-stamped predicted cells are exactly the union of the
        surviving (kept) groups — ``_unstamp_group`` only ever clears a cell that no
        kept group still lists. Used to refresh the render overlay's debug attr to
        the post-peel footprint.
        """
        union: set[tuple[int, int]] = set()
        for _key, cells in groups:
            union.update(cells)
        return sorted(union)


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


class DStarLitePredictiveController(PredictiveDStarLiteController):
    """Lidar-fed predictive D* Lite: estimated velocities, widening cone.

    The Mission-faithful variant: velocities come from the Kalman multi-object
    tracker (LidarTracker) over the live lidar, not the truth seam
    (wants_truth=False). The cone geometry widens the predicted footprint with
    the lookahead step to absorb the estimator's residual velocity error, where
    the oracle's exact velocities warrant only a constant-radius capsule.
    """

    name = "d_star_lite_predictive"
    geometry = "cone"
    wants_truth = False

    def _make_tracker(self) -> Tracker:
        # Lazy: reset() has populated self._geom / self._grid by the time this
        # first fires (first non-h0 act()). Recover the beam bearings exactly as
        # the rest of the harness does (np.linspace over the inclusive endpoints,
        # NOT i*angle_increment), then hand them + the static grid to the tracker.
        bearings = np.linspace(
            self._geom.angle_min, self._geom.angle_max, self._geom.number
        )
        # range_max lets the tracker drop no-hit returns at the sensing rim (which
        # otherwise cluster into phantom obstacles); see LidarTracker.
        return LidarTracker(self._grid, bearings, range_max=self._geom.range_max)


# Self-register at import (mirrors d_star_lite.py). Imported after the
# module-level imports so the registry sees the fully-defined class.
from planners._grid import register  # noqa: E402

register("d_star_lite_oracle", DStarLiteOracleController)
register("d_star_lite_predictive", DStarLitePredictiveController)
