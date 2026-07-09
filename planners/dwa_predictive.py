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
   keeps the controller safe even when the tracker misses an obstacle.
2. **Space-time predictive LAYER** — :func:`planners._predict.trajectory_conflict`
   over the horizon: hard-reject candidates that collide with a track in matched
   time, and add a predicted-clearance term to the score so the robot yields early
   and smoothly rather than only at the last feasible instant.

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
    SAFETY_MARGIN,
    build_occupancy_grid,
    load_world,
)
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
    ROLLOUT_STEPS,
    DWAController,
)

# Weight on the space-time predicted-clearance term added to DWA's weighted-sum
# score. Same scale as the present-position CLEARANCE_WEIGHT (0.3): it rewards
# candidates that keep more matched-time margin from moving obstacles, so the robot
# starts easing away from a predicted conflict before a hard space-time rejection
# is forced. Tunable; a small default that biases toward foresight without
# overwhelming the goal-heading term.
PREDICTED_CLEARANCE_WEIGHT = 0.4


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
        # Static inflated occupancy grid (built in reset(); consumed only by the
        # LidarTracker's static-return subtraction, unused by the oracle).
        self._grid = None
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
        obstacle_points: np.ndarray,
    ) -> float | None:
        """Two-layer per-candidate evaluation: present floor + space-time layer.

        1. Present-position FLOOR + base score on the first ``ROLLOUT_STEPS`` (a
           forward prefix of the possibly-longer rollout) — identical to vanilla
           DWA, so with no tracks this returns exactly the base score.
        2. Space-time LAYER over the first ``horizon_steps``: hard-reject a
           matched-time collision with any track; otherwise add a capped
           predicted-clearance bonus so a candidate that keeps more space-time
           margin from movers scores higher.
        """
        del state  # the trajectory already encodes the robot's future poses

        # Base heading/clearance/speed on the first ROLLOUT_STEPS (byte-identical
        # to plain DWA — the extra rollout steps feed only the space-time check).
        base_trajectory = trajectory[:ROLLOUT_STEPS]
        clearance = self._trajectory_clearance(base_trajectory, obstacle_points)
        if clearance is None:
            return None
        base_score = self._score(base_trajectory, v, clearance)

        # No tracks -> no space-time layer; behaves as plain DWA (also the h0 path).
        if not self._tracks:
            return base_score

        assert self._robot_radius is not None  # narrowed by super().act()'s guard
        conflict = trajectory_conflict(
            trajectory,
            self._tracks,
            float(self._robot_radius),
            self._horizon_steps,
            PREDICT_DT,
            COLLISION_MARGIN,
        )
        if conflict.collides:
            # The robot body would meet a moving obstacle at a matched time: reject.
            return None

        # Predicted-clearance bonus: reward matched-time margin from movers, capped
        # like the present-position clearance term and normalized to [0, 1].
        predicted_clearance = min(max(conflict.min_gap, 0.0), CLEARANCE_CAP)
        return base_score + PREDICTED_CLEARANCE_WEIGHT * (predicted_clearance / CLEARANCE_CAP)


class DWAPredictiveController(PredictiveDWAController):
    """Lidar-fed space-time DWA: estimated velocities (the Mission-faithful, canonical key).

    Velocities come from frame-differencing the live lidar (``LidarTracker``), not
    the truth seam (``wants_truth=False``). Promoted to a canonical study planner.
    """

    name = "dwa_predictive"
    wants_truth = False

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

    def _make_tracker(self) -> Tracker:
        return OracleTracker()


# Self-register at import (mirrors dwa.py / d_star_lite_predictive.py). Imported
# after the class definitions so the registry sees the fully-defined classes.
from planners._grid import register  # noqa: E402

register("dwa_predictive", DWAPredictiveController)
register("dwa_predictive_oracle", DWAPredictiveOracleController)
