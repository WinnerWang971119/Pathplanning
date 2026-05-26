from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import irsim
import numpy as np


DEFAULT_TIMEOUT_S = 120.0
LIDAR_BEAM_COUNT = 360
ACTION_SHAPE = (2, 1)


class ArenaConfigError(ValueError):
    """Raised at Arena.__init__ for malformed config (e.g. lidar beam count mismatch)."""


class ArenaRuntimeError(RuntimeError):
    """Raised mid-episode for irsim contract violations (e.g. lidar dict missing 'ranges')."""


@dataclass(frozen=True)
class EpisodeInfo:
    sim_time: float
    step_idx: int
    crashed: bool
    timed_out: bool
    reached_goal: bool
    distance_to_goal: float
    wallclock_per_step: float
    dynamic_obstacle_count: int
    lidar_status: str


class Arena:
    """Static 50x50 arena wrapping irsim. Phase 0 = no dynamic obstacles."""

    def __init__(
        self,
        yaml_path: str | Path,
        seed: int,
        render: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._yaml_path = Path(yaml_path)
        self._render = bool(render)
        self._timeout_s = float(timeout_s)
        self._master_seed = int(seed)

        self._env = irsim.make(str(self._yaml_path), display=self._render)
        self._robot = self._env.robot_list[0]
        self._dt = float(self._env.step_time)
        self._goal_xy = self._robot.goal[:2, 0].astype(np.float64)

        scan = self._robot.get_lidar_scan()
        if not scan:
            raise ArenaConfigError("YAML robot has no working lidar2d sensor")
        if "ranges" not in scan:
            raise ArenaConfigError(
                f"YAML lidar2d scan dict missing 'ranges' key: keys={list(scan.keys())}"
            )
        beam_count = len(scan["ranges"])
        if beam_count != LIDAR_BEAM_COUNT:
            raise ArenaConfigError(
                f"lidar scan returned {beam_count} beams, expected {LIDAR_BEAM_COUNT}"
            )

        # traffic first, motion second — Phase 2 spawner consumes in this order
        ss = np.random.SeedSequence(self._master_seed)
        traffic_seed, motion_seed = ss.spawn(2)
        self._traffic_rng = np.random.default_rng(traffic_seed)
        self._motion_rng = np.random.default_rng(motion_seed)

        self._step_idx = 0
        self._done = False
        self._closed = False

    def reset(self) -> tuple[np.ndarray, np.ndarray, EpisodeInfo]:
        if self._closed:
            raise RuntimeError("Arena is closed")

        self._env.reset()
        # Defensive re-clear: irsim's reset() runs an internal warm-up step that
        # re-evaluates arrive/collision flags against the just-reset pose.
        self._robot.arrive_flag = False
        self._robot.collision_flag = False

        # traffic first, motion second — Phase 2 spawner consumes in this order
        ss = np.random.SeedSequence(self._master_seed)
        traffic_seed, motion_seed = ss.spawn(2)
        self._traffic_rng = np.random.default_rng(traffic_seed)
        self._motion_rng = np.random.default_rng(motion_seed)

        self._step_idx = 0
        self._done = False

        state = self._robot.state[:, 0].astype(np.float64)
        lidar, lidar_status = self._extract_lidar()

        info = EpisodeInfo(
            sim_time=0.0,
            step_idx=0,
            crashed=False,
            timed_out=False,
            reached_goal=False,
            distance_to_goal=float(np.linalg.norm(state[:2] - self._goal_xy)),
            wallclock_per_step=0.0,
            dynamic_obstacle_count=0,
            lidar_status=lidar_status,
        )
        return state, lidar, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, EpisodeInfo]:
        if self._done:
            raise RuntimeError("Episode is done; call reset() first.")
        if self._closed:
            raise RuntimeError("Arena is closed")

        if not isinstance(action, np.ndarray):
            raise ValueError(
                f"action must be np.ndarray, got {type(action).__name__}"
            )
        if action.shape != ACTION_SHAPE:
            raise ValueError(
                f"action shape must be {ACTION_SHAPE}, got {action.shape}"
            )
        if not np.issubdtype(action.dtype, np.floating):
            raise ValueError(f"action dtype must be float, got {action.dtype}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action contains NaN or Inf")

        # Snapshot flags BEFORE step: irsim's check_*_status overwrite them per tick
        # (see object_base.py:531-532), so harness-injected flags would be lost otherwise.
        pre_crashed = bool(getattr(self._robot, "collision_flag", False))
        pre_reached = bool(getattr(self._robot, "arrive_flag", False))

        start = time.perf_counter()
        self._env.step([action])
        wallclock = time.perf_counter() - start

        self._step_idx += 1

        state = self._robot.state[:, 0].astype(np.float64)
        lidar, lidar_status = self._extract_lidar()

        crashed = pre_crashed or bool(getattr(self._robot, "collision_flag", False))
        reached_goal = pre_reached or bool(getattr(self._robot, "arrive_flag", False))
        sim_time = self._step_idx * self._dt
        timed_out = sim_time >= self._timeout_s
        distance_to_goal = float(np.linalg.norm(state[:2] - self._goal_xy))

        info = EpisodeInfo(
            sim_time=sim_time,
            step_idx=self._step_idx,
            crashed=crashed,
            timed_out=timed_out,
            reached_goal=reached_goal,
            distance_to_goal=distance_to_goal,
            wallclock_per_step=wallclock,
            dynamic_obstacle_count=0,
            lidar_status=lidar_status,
        )

        done = crashed or timed_out or reached_goal
        self._done = bool(done)
        return state, lidar, self._done, info

    def _extract_lidar(self) -> tuple[np.ndarray, str]:
        scan = self._robot.get_lidar_scan()
        if not scan:
            return (
                np.full((LIDAR_BEAM_COUNT,), np.nan, dtype=np.float64),
                "missing",
            )
        if "ranges" not in scan:
            raise ArenaRuntimeError(
                f"irsim lidar returned a non-falsy scan without 'ranges' key: "
                f"keys={list(scan.keys())}"
            )
        ranges = np.asarray(scan["ranges"], dtype=np.float64)
        if ranges.shape != (LIDAR_BEAM_COUNT,):
            raise ArenaRuntimeError(
                f"irsim lidar returned ranges of shape {ranges.shape}, "
                f"expected ({LIDAR_BEAM_COUNT},)"
            )
        range_max = float(scan.get("range_max", np.inf))
        ranges = np.where(
            np.isfinite(ranges) & (ranges < range_max), ranges, np.nan
        )
        return ranges, "ok"

    @property
    def initial_dynamic_snapshot(self) -> tuple[Any, ...]:
        """Snapshot of dynamic obstacles at t=0. Empty in Phase 0; Phase 2 narrows the type."""
        return ()

    def close(self) -> None:
        if self._closed:
            return
        self._env.end()
        self._closed = True
