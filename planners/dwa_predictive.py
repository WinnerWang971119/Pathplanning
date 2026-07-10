"""Predictive (space-time) DWA controllers — real (x, y, t) collision avoidance.

Unlike the D* Lite predictive family, which STAMPS each obstacle's predicted
future footprint onto the occupancy grid (a 2-D projection over [0, T] that
collapses the time axis, so a cell an obstacle merely passes through is blocked as
if permanently occupied), these controllers reason in space-time: inside DWA's
forward-simulation rollout they advance each tracked obstacle at constant velocity
and check the robot against the obstacle AT THE MATCHED time step. A candidate that
passes *behind* a crosser is fine — the crosser has moved on by the time the robot
reaches that point; a candidate that would *meet* the crosser is rejected. This is
the mechanism of the two cited DWA-prediction papers (Missura & Bennewitz, ICRA
2019; MDPI Actuators 2025 14(5):207).

Two-layer collision model. ``PredictiveDWAController`` subclasses ``DWAController``
and overrides ONLY ``_evaluate_candidate`` (plus ``__init__`` / ``reset`` /
``observe_truth`` and an ``act`` horizon-0 shortcut); it does NOT reimplement the
dynamic window, the sampling loop, the rollout, or the fallback:

1. **Present-position safety FLOOR** — vanilla DWA's clearance check against the
   full live lidar cloud (walls AND movers at their *current* positions),
   hard-rejecting any candidate whose rollout grazes a currently-visible body. This
   keeps the controller safe even when the tracker misses an obstacle. The floor is
   UNCHANGED: no live return is ever subtracted for a tracked mover, so an obstacle
   with no track is still caught at its current position.
2. **Space-time predictive LAYER** (braking-inevitability + soft yield) — this is
   NOT a blanket space-time hard reject (that freezes the robot: any matched-time
   conflict anywhere on the rollout kills every forward candidate). Instead, per
   candidate, over the horizon via :func:`planners._predict.trajectory_conflict`:
   - **Imminent backstop** — a conflict at the very next step (``ttc_step == 1``)
     is rejected outright (too close to react).
   - **Braking-inevitability (ICS) test** — simulate an emergency-braking
     trajectory (decelerate to a stop at ``BRAKE_DECEL`` then hold, via
     :meth:`_braking_trajectory`) and reject the candidate ONLY when even that
     braking-and-holding path still collides in matched time. A conflict the robot
     could brake out of is admitted (it is not an inevitable collision state), so
     the layer cannot freeze the robot the way a blanket reject does.
   - **Soft yield term** — a SYMMETRICALLY-clipped, un-floored predicted-clearance
     score term (``PREDICTED_GAP_WEIGHT * clip(min_gap, -cap, cap)/cap``) added to
     the base score. Being monotone in the matched-time gap and penalizing negative
     gaps, it makes a slower collision-free candidate outscore a faster grazing one
     — the mechanism that makes the robot yield early rather than only at the last
     feasible instant.

Velocity source behind the shared ``Tracker`` seam (mirrors D* Lite):
- ``DWAPredictiveController`` (key ``"dwa_predictive"``): ``LidarTracker``
  frame-differencing estimator, ``wants_truth=False`` — the Mission-faithful,
  CANONICAL variant.
- ``DWAPredictiveOracleController`` (key ``"dwa_predictive_oracle"``):
  ``OracleTracker`` perfect live velocities via the truth seam,
  ``wants_truth=True`` — the EXPERIMENTAL ceiling. Its ONLY difference from the
  lidar variant is the velocity source (walls are checked identically, against the
  live lidar cloud), so it isolates velocity estimation as the single variable.

Horizon. ``--predict-horizon H`` sets the space-time depth. The rollout is
lengthened to ``max(ROLLOUT_STEPS, H)`` steps, but the base heading/clearance/speed
score still reads only the first ``ROLLOUT_STEPS`` (a forward prefix), so the extra
steps feed only the space-time check and the base score is unchanged. ``h0`` is a
true no-op: ``act`` returns ``super().act(...)`` with no tracks, so
``dwa_predictive[_oracle]_h0`` produces a byte-identical trace to plain ``dwa``
(TC65).

Determinism. DWA has no RNG; both trackers are deterministic (the ``LidarTracker``
by construction — TC64); ``trajectory_conflict`` is pure. Traces stay
byte-identical across same-seed runs. ``act`` never raises mid-episode: a tracker /
prediction failure degrades that tick to plain DWA (AC5); DWA ``reset`` never
raises, so ``planner_error`` is always null for this family.
"""
from __future__ import annotations

import numpy as np

from manual_astar import (
    GRID_RESOLUTION,
    MAX_LINEAR_SPEED,
    SAFETY_MARGIN,
    build_occupancy_grid,
    load_world,
    world_to_grid,
    wrap_to_pi,
)
from planners._costfield import build_cost_to_go_field
from planners._predict import (
    PREDICT_DT,
    LidarTracker,
    OracleTracker,
    Tracker,
    trajectory_conflict,
)
from planners.dwa import (
    CLEARANCE_CAP,
    COLLISION_MARGIN,
    CONTROL_DT,
    GOAL_REACHED_RADIUS,
    MAX_LINEAR_ACCEL,
    ROLLOUT_STEPS,
    DWAController,
)

# Braking-simulation deceleration (m/s^2) used by the inevitable-collision-state
# test. The braking rollout decelerates the candidate's linear speed at this rate
# to a stop, then holds — so it uses the dynamic window's OWN physics (the same
# acceleration bound that shapes the reachable window each tick), rather than an
# arbitrary braking constant.
BRAKE_DECEL = MAX_LINEAR_ACCEL

# Weight on the space-time predicted-clearance ("min_gap") soft-score term added
# to DWA's weighted-sum score. The term is the SYMMETRICALLY-clipped, normalized
# matched-time body gap: clip(min_gap, -CLEARANCE_CAP, CLEARANCE_CAP)/CLEARANCE_CAP
# in [-1, 1]. Being un-floored (negative gaps penalize) makes a slower
# collision-free candidate outscore a faster grazing/colliding one — the mechanism
# that makes the robot yield rather than race a crosser. Small so foresight biases
# the argmax without overwhelming the goal-heading term.
PREDICTED_GAP_WEIGHT = 0.3

# Normalizer for the global-guidance heading term's geodesic-progress score, in
# CELL units to match the cost-to-go field (which stores octile distances in cell
# units). It is the field-progress a full-speed straight rollout achieves over one
# rollout window: MAX_LINEAR_SPEED * ROLLOUT_STEPS * CONTROL_DT metres, converted to
# cells by dividing by GRID_RESOLUTION. Progress at or beyond this saturates the
# term; the ratio is symmetrically clipped so a retreat of the same magnitude
# saturates the low end (= 0.0). Evaluates to ~12.0 for the current constants (the
# expression, kept verbatim so it tracks the constants, carries the usual 0.1
# float error and lands at 12.000000000000002 rather than an exact 12.0).
MAX_PROGRESS_CELLS = MAX_LINEAR_SPEED * ROLLOUT_STEPS * CONTROL_DT / GRID_RESOLUTION


class PredictiveDWAController(DWAController):
    """Base for the space-time DWA family. Subclasses pick the velocity source.

    Subclasses set :attr:`name` and :attr:`wants_truth` and implement
    :meth:`_make_tracker`. They MUST NOT override ``act`` beyond what this base
    already does — the only behavioural override is :meth:`_evaluate_candidate`
    (the per-candidate space-time layer), plus the ``__init__`` / ``reset`` /
    ``observe_truth`` / tracker plumbing here.
    """

    # Abstract base: no registry key (blanked so it cannot shadow "dwa").
    name = ""
    # Opt-in live-truth flag; the oracle sets True so the runner feeds it the
    # dynamic-obstacle snapshot via observe_truth().
    wants_truth = False
    # Opt-in cost-to-go global guidance. When True, reset() builds a static
    # Dijkstra-from-goal field and the overridden _heading_term scores candidates by
    # geodesic progress toward the goal (immune to the Euclidean-heading local minima
    # a wall segment induces) instead of straight-line heading. The paper-only
    # ablation classes leave this False (base Euclidean heading); the two global
    # concrete classes flip it True. Default off so the base stays paper-behaviour.
    use_global_guidance = False

    def _make_tracker(self) -> Tracker:
        """Return the velocity-source adapter (subclass responsibility)."""
        raise NotImplementedError(
            "PredictiveDWAController subclasses must implement _make_tracker()."
        )

    def __init__(
        self,
        replan_k: int | None = None,
        predict_horizon: int | None = None,
    ) -> None:
        # build_controller rejects --replan-k for the predict family (not in
        # REPLAN_FAMILIES), so replan_k is None here; pass it on for the uniform
        # construction seam (DWAController ignores it).
        super().__init__(replan_k)

        if predict_horizon is None or int(predict_horizon) < 0:
            raise ValueError(
                f"predict_horizon must be a non-negative int, received {predict_horizon!r}."
            )
        self._horizon_steps: int = int(predict_horizon)
        # Lengthen the rollout to cover the prediction horizon; the base score
        # still reads only the first ROLLOUT_STEPS, so h0 keeps the vanilla length
        # and stays byte-identical to plain dwa.
        self._rollout_steps = max(ROLLOUT_STEPS, self._horizon_steps)

        # Deferred to first non-h0 act(): the LidarTracker needs the post-reset()
        # static grid + beam geometry; the OracleTracker is stateless.
        self._tracker: Tracker | None = None
        self._snapshot: tuple = ()
        # Static inflated occupancy grid (built in reset(); consumed by the
        # LidarTracker's static-return subtraction — unused by the oracle — and, when
        # global guidance is on, by the cost-to-go field build and per-candidate
        # world_to_grid lookups).
        self._grid = None
        # Static cost-to-go field (goal-distances in CELL units), or None when global
        # guidance is off OR the start cell is walled off from the goal (the
        # start-unreachable fallback: guidance disabled for the episode, base
        # Euclidean heading used, planner_error stays null). Built in reset().
        self._field: np.ndarray | None = None
        # Current-tick tracks, refreshed once per act() and read per candidate.
        self._tracks: list = []

        # Read-only debug attrs for the render overlay. DWA stamps no grid cells,
        # so last_predicted_cells stays []; last_tracks carries the velocity arrows.
        self.last_predicted_cells: list = []
        self.last_tracks: list = []

    def reset(
        self,
        world_yaml: str,
        initial_snapshot: tuple,
        lidar0: np.ndarray,
        state0: np.ndarray,
    ) -> None:
        """Cache DWA's substrate, build the static grid, and clear per-episode state.

        The base ``reset`` caches the goal + lidar beam geometry. On top we build
        the STATIC inflated occupancy grid the ``LidarTracker`` needs for
        static-return subtraction (harmless/unused for the oracle), and null every
        per-episode member so a reused instance (the ``Controller`` contract allows
        a second episode) does not difference episode 2's first frame against
        episode 1's final tracker state.
        """
        super().reset(world_yaml, initial_snapshot, lidar0, state0)

        world = load_world(world_yaml)
        self._grid = build_occupancy_grid(world, GRID_RESOLUTION, SAFETY_MARGIN)

        self._rollout_steps = max(ROLLOUT_STEPS, self._horizon_steps)
        self._tracker = None
        self._snapshot = ()
        self._tracks = []
        self.last_predicted_cells = []
        self.last_tracks = []

        # Global guidance: build the static cost-to-go field once. Clear any
        # prior-episode field first, so a reused instance whose second episode has
        # guidance disabled (or a walled-off start) does not carry episode 1's field.
        self._field = None
        if self.use_global_guidance:
            assert self._grid is not None and self._goal_xy is not None
            goal_cell = world_to_grid(self._goal_xy, self._grid)
            field = build_cost_to_go_field(self._grid, goal_cell)
            start_cell = world_to_grid(np.asarray(state0, dtype=float)[:2], self._grid)
            # Start-unreachable fallback: if the goal is walled off from the start
            # the field is inf at the start cell — leave self._field None so
            # _heading_term falls back to the base Euclidean heading for the whole
            # episode. DWA never fails to plan, so planner_error stays null.
            if not np.isinf(field[start_cell]):
                self._field = field

    def observe_truth(self, snapshot: tuple) -> None:
        """Store the live dynamic-obstacle snapshot for the upcoming act() tick.

        Tick alignment is the runner's responsibility (it calls this with the
        snapshot from the SAME source call that produced the state/lidar the next
        act() receives). Only the oracle sets wants_truth=True, so only it is fed.
        """
        self._snapshot = snapshot

    def act(self, state: np.ndarray, lidar: np.ndarray) -> np.ndarray:
        """Refresh tracks, then run DWA's dynamic-window search with the space-time layer.

        The tracker update is the only work here; the actual window build, sampling
        loop, per-candidate space-time scoring (via the overridden
        :meth:`_evaluate_candidate`), and fallback all run inside ``super().act``.
        """
        # h0 fast path: a true no-op. No tracker update (no side effect), no
        # space-time layer, base rollout length -> byte-identical to plain dwa
        # (TC65). _evaluate_candidate returns the bare base score because
        # self._tracks is empty.
        if self._horizon_steps == 0:
            self._tracks = []
            self.last_tracks = []
            self.last_predicted_cells = []
            return super().act(state, lidar)

        # Refresh the tracks ONCE per act (read per candidate in _evaluate_candidate).
        # AC5: any tracker / prediction failure degrades this tick to plain DWA (no
        # space-time layer) rather than propagating out of act(). except Exception
        # (not bare) so KeyboardInterrupt/SystemExit still propagate.
        try:
            if self._tracker is None:
                self._tracker = self._make_tracker()
            self._tracks = self._tracker.update(
                snapshot=self._snapshot, state=state, lidar=lidar, dt=PREDICT_DT
            )
        except Exception:
            self._tracks = []
        self.last_tracks = self._tracks
        self.last_predicted_cells = []  # DWA has no grid stamp to draw

        return super().act(state, lidar)

    def _evaluate_candidate(
        self,
        state: np.ndarray,
        trajectory: np.ndarray,
        v: float,
        w: float,
        obstacle_points: np.ndarray,
    ) -> float | None:
        """Present floor + braking-inevitability space-time layer for one candidate.

        1. Present-position FLOOR + base score on the first ``ROLLOUT_STEPS`` (a
           forward prefix of the possibly-longer rollout) — identical to vanilla
           DWA, so with no tracks this returns exactly the base score. The cloud
           (``obstacle_points``) is the FULL live lidar projection: no mover return
           is subtracted, so a tracker miss is still rejected here (AC6).
        2. Space-time LAYER over the horizon: reject an imminent conflict
           (``ttc_step == 1``); reject an INEVITABLE conflict (even the
           braking-and-holding trajectory still collides in matched time); and add
           the symmetric, un-floored predicted-clearance soft term so a slower
           collision-free candidate outscores a faster grazing one (yielding).

        The whole space-time block is wrapped so an unexpected raise degrades this
        candidate to base scoring rather than propagating out of ``act()`` (AC11).
        A deliberate ``return None`` reject inside the block is normal control flow,
        not caught by the guard — it returns from the method directly.
        """
        # Base heading/clearance/speed on the first ROLLOUT_STEPS (byte-identical
        # to plain DWA — the extra rollout steps feed only the space-time check).
        # ``state`` is used below (the base ``_score`` ignores it, but the braking
        # rollout needs the start pose), so it is NOT deleted here.
        base_trajectory = trajectory[:ROLLOUT_STEPS]
        clearance = self._trajectory_clearance(base_trajectory, obstacle_points)
        if clearance is None:
            return None
        base_score = self._score(state, base_trajectory, v, clearance)

        # No tracks -> no space-time layer; behaves as plain DWA (also the h0 path).
        if not self._tracks:
            return base_score

        assert self._robot_radius is not None  # narrowed by super().act()'s guard
        robot_radius = float(self._robot_radius)

        try:
            # Predicted matched-time conflict on the constant-(v, w) rollout. Pass
            # the FULL (possibly longer) trajectory; trajectory_conflict clamps
            # internally to min(horizon_steps, len).
            conflict = trajectory_conflict(
                trajectory,
                self._tracks,
                robot_radius,
                self._horizon_steps,
                PREDICT_DT,
                COLLISION_MARGIN,
            )

            # Imminent backstop: a conflict at the very next step is too close to
            # react to; reject outright.
            if conflict.ttc_step == 1:
                return None

            # Braking-inevitability (ICS) test: if even an emergency-braking-and-
            # holding trajectory still collides at a matched time, the collision is
            # inevitable — reject. A conflict the robot could brake out of is
            # admitted (the soft term below biases the argmax toward yielding).
            braking = self._braking_trajectory(state, v, w)
            braking_conflict = trajectory_conflict(
                braking,
                self._tracks,
                robot_radius,
                self._horizon_steps,
                PREDICT_DT,
                COLLISION_MARGIN,
            )
            if braking_conflict.collides:
                return None

            # Soft yield term: symmetric-clipped, un-floored matched-time gap in
            # [-1, 1]. Negative gaps penalize, so a slower collision-free candidate
            # outscores a faster grazing/colliding one.
            gap = float(np.clip(conflict.min_gap, -CLEARANCE_CAP, CLEARANCE_CAP))
            soft = PREDICTED_GAP_WEIGHT * (gap / CLEARANCE_CAP)
            return base_score + soft
        except Exception:
            # AC11: an unexpected raise in the space-time layer degrades this
            # candidate to base scoring; act() never raises mid-episode.
            return base_score

    def _heading_term(self, state: np.ndarray, trajectory: np.ndarray) -> float:
        """Geodesic-progress heading score when global guidance is active.

        When no cost-to-go field is available (guidance off, or the
        start-unreachable fallback disabled it for the episode) this delegates to
        the base Euclidean goal-heading term unchanged, so a paper-only variant and
        the global-with-walled-off-start case behave exactly as plain DWA's heading.

        With a field, the score is a NON-SATURATED, strictly-monotone function of
        the geodesic progress the rollout makes toward the goal (the field-value
        DROP from the start cell to the rollout's final cell, in CELL units):

            0.5 + 0.5 * clip(progress / MAX_PROGRESS_CELLS, -1.0, 1.0)

        so retreat (< 0.5) < no progress (0.5) < progress (> 0.5), and the interior
        is never exactly 0 or 1. This cures the local-minima pathology a Euclidean
        heading hits behind a wall segment (the straight-line bearing points into
        the wall; the geodesic field points around it).

        Guards, in order:
        - Goal-reached: within GOAL_REACHED_RADIUS of the goal the heading is
          ill-defined, so return 1.0 (same convention as the base term).
        - Start cell unreachable: defensive — the reset guard should have disabled
          guidance (self._field would be None), but if a start read still lands on
          an inf cell, fall back to the base heading rather than divide meaning into
          an inf progress.
        - Rollout ends in a wall (end cell inf): disfavour with 0.0.

        world_to_grid clips to the grid bounds, so both cell reads are always
        in-bounds and this method never raises (AC11).
        """
        assert self._goal_xy is not None  # narrowed by act()'s guard

        # Paper-only variants and the walled-off-start fallback use the base
        # Euclidean heading (self._field is None in both cases).
        if self._field is None:
            return super()._heading_term(state, trajectory)

        assert self._grid is not None  # a field implies a grid was built in reset()

        goal_distance = float(np.linalg.norm(self._goal_xy - trajectory[-1]))
        if goal_distance < GOAL_REACHED_RADIUS:
            return 1.0

        start_cell = world_to_grid(state[:2], self._grid)
        end_cell = world_to_grid(trajectory[-1], self._grid)

        if np.isinf(self._field[start_cell]):
            # Defensive: should not happen after the reset start-unreachable guard.
            return super()._heading_term(state, trajectory)
        if np.isinf(self._field[end_cell]):
            # The rollout ends inside a wall's inflation band — disfavour it.
            return 0.0

        # Positive = the rollout's final cell is closer to the goal than the start.
        progress = self._field[start_cell] - self._field[end_cell]
        return 0.5 + 0.5 * float(np.clip(progress / MAX_PROGRESS_CELLS, -1.0, 1.0))

    def _braking_trajectory(
        self, state: np.ndarray, v: float, w: float
    ) -> np.ndarray:
        """Forward-simulate an emergency-braking-then-hold rollout for the ICS test.

        Mirrors :meth:`DWAController._rollout` exactly (same heading-first unicycle
        update at CONTROL_DT), but the linear speed decelerates at ``BRAKE_DECEL``:
        at sub-step ``k`` (1-based) the speed is
        ``v_k = max(0, v - BRAKE_DECEL * k * CONTROL_DT)``. Once ``v_k`` hits 0 the
        position increment is 0, so the robot HOLDS its stopped pose for the
        remaining steps — a robot stopped in a crosser's lane is still an inevitable
        collision state, so the space-time check must see the full braking+held
        path. The angular velocity ``w`` is applied unchanged (heading keeps
        turning), matching the base rollout's constant-``w`` assumption.

        RNG-free and fixed step count (``self._rollout_steps``), so the extra
        rollout is deterministic. Returns a ``(self._rollout_steps, 2)`` float64
        array of predicted xy positions (excluding the start pose).
        """
        x = float(state[0])
        y = float(state[1])
        theta = float(state[2])

        positions = np.empty((self._rollout_steps, 2), dtype=float)
        for step_index in range(self._rollout_steps):
            theta = wrap_to_pi(theta + w * CONTROL_DT)
            v_k = max(0.0, v - BRAKE_DECEL * (step_index + 1) * CONTROL_DT)
            x += v_k * np.cos(theta) * CONTROL_DT
            y += v_k * np.sin(theta) * CONTROL_DT
            positions[step_index, 0] = x
            positions[step_index, 1] = y

        return positions


class DWAPredictiveController(PredictiveDWAController):
    """Lidar-fed space-time DWA: estimated velocities (the Mission-faithful, canonical key).

    Velocities come from frame-differencing the live lidar (``LidarTracker``), not
    the truth seam (``wants_truth=False``). Promoted to a canonical study planner.
    """

    name = "dwa_predictive"
    wants_truth = False
    use_global_guidance = True

    def _make_tracker(self) -> Tracker:
        # Lazy: reset() has populated self._grid / self._bearings / self._geom by
        # the time this first fires (first non-h0 act()). The bearings are the exact
        # linspace recovery the rest of the harness uses (NOT i*angle_increment).
        assert self._grid is not None and self._bearings is not None and self._geom is not None
        return LidarTracker(self._grid, self._bearings, range_max=self._geom.range_max)


class DWAPredictiveOracleController(PredictiveDWAController):
    """Oracle-fed space-time DWA: perfect live velocities (the experimental ceiling).

    A deliberate cheat (exact velocities via the truth seam) that measures the
    achievable motion-aware ceiling for DWA. Walls are checked exactly as the lidar
    variant does (live lidar cloud), so the ONLY difference is the velocity source.
    """

    name = "dwa_predictive_oracle"
    wants_truth = True
    use_global_guidance = True

    def _make_tracker(self) -> Tracker:
        return OracleTracker()


# Self-register at import (mirrors dwa.py / d_star_lite_predictive.py). Imported
# after the class definitions so the registry sees the fully-defined classes.
from planners._grid import register  # noqa: E402

register("dwa_predictive", DWAPredictiveController)
register("dwa_predictive_oracle", DWAPredictiveOracleController)
