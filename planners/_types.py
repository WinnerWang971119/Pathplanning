from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np

from arena.dynamic import DynamicObstacleState

Path = tuple[np.ndarray, ...]  # ordered (2,)-shaped float64 world-frame waypoints; last == goal


@runtime_checkable
class Controller(Protocol):
    name: str  # e.g. "a_star_replan" - the FAMILY name; results label adds _k<K>

    # Opt-in live-truth flag. Only the oracle controller sets this True; the
    # runner gates its `observe_truth` call on `getattr(controller,
    # "wants_truth", False)`, so the existing concrete controllers (which never
    # define it) are unaffected.
    wants_truth: bool = False

    def reset(
        self,
        world_yaml: str,
        initial_snapshot: tuple[DynamicObstacleState, ...],  # t=0 view; () when traffic off
        lidar0: np.ndarray,                                  # (360,) float64; NaN = no return
        state0: np.ndarray,                                  # (3,) float64 [x, y, theta]
    ) -> None:
        """Build static substrate + t=0 plan. May raise ValueError/RuntimeError (no path)
        -> runner records planner_error."""
        ...

    def observe_truth(self, snapshot: tuple) -> None:
        """Receive the live dynamic-obstacle snapshot for the upcoming act() tick.
        Default no-op; only truth-consuming controllers (the oracle) override it."""
        ...

    def act(self, state: np.ndarray, lidar: np.ndarray) -> np.ndarray:
        """Return the next action, shape (2,1) float [[v],[w]]. Must NOT raise on a
        mid-episode replan failure (keep the last valid path)."""
        ...
