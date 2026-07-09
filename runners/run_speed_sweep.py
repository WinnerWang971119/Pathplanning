"""Sweep driver — run a planner set across the 4 named obstacle-speed regimes.

Phase 7 (Obstacle-speed-cap sweep, issue #11). The dynamic-obstacle speed band
is a swept parameter: we want to measure how the obstacle-speed cap drives
failure rate and time-to-goal per algorithm (specifically whether D* Lite's
failures floor to zero once no obstacle can outrun the robot). This module is the
top-level orchestrator for that sweep.

It is a sibling of ``runners.run_all``: where ``run_all`` loops the 11 canonical
planners and shells ``run_experiment`` per planner (two passes), this driver loops
the 4 speed regimes x a selectable planner set and shells ``run_experiment`` once
per (regime, planner). Going through a subprocess (rather than calling
``run_experiment.main()`` in-process) keeps each run isolated from the others'
``sys.path`` / irsim import side effects and mirrors the established two-tier
subprocess pattern, so per-episode byte-determinism (TC15/TC20/TC24) and the
spawn-based traffic/motion substreams carry over unchanged.

Per-regime results subtree:

    Each child is launched with ``--results-dir <root>/speed_<regime>``, so its
    ``episode_out_dir`` (which unconditionally appends ``<world_stem>/<label>``)
    lands files at ``<root>/speed_<regime>/<world_stem>/<label>/<seed>.json`` —
    exactly where ``runners.plot_speed_sweep`` reads via
    ``load_world_results("<root>/speed_<regime>", world_stem)``. This mirrors
    ``run_all``'s ``__wallclock__`` sibling-subtree convention: the stem is
    inserted ONCE by the child, never double-nested by this driver.

There is no wallclock pass (that is ``run_all``'s concern); the sweep cares about
failure rate + time-to-goal, both already in the per-seed metrics JSON.

CLI:
    python -m runners.run_speed_sweep \
        --world <yaml_path>      # required; e.g. arena/arena_v1.yaml
        [--algorithms focus|all] # default "focus" (the 4-planner hypothesis set)
        [--master-seed <int>]    # default DEFAULT_MASTER_SEED
        [--num-seeds <int>]      # default 50
        [--jobs <int>]           # default 1; forwarded to each child run_experiment
        [--results-dir <dir>]    # default "results" (the speed_<regime> subtrees' root)
        [--resume]               # skip seeds whose JSON already exists (forwarded)
        [--traffic|--no-traffic] # crossing traffic, default ON

Planner sets:
    --algorithms focus (default) — the 5 non-replan planners in FOCUS_SET
        (a_star_once, d_star_lite, dwa, dwa_predictive, apf): the static baseline,
        the incremental planner the hypothesis is about, the two reactive planners
        the issue calls out, and the space-time-predictive DWA. dwa_predictive is a
        predict family (carries --predict-horizon); the rest are non-replan/non-predict.
    --algorithms all — the 11 canonical labels, reused verbatim from
        run_all.canonical_planner_set() (so a registry change cannot desync the
        sweep from the study); the replan families carry --replan-k 5.

Exit codes:
    0 — every (regime, planner) child subprocess exited 0 (ran to completion)
    1 — >= 1 child exited non-zero (a runner/config fault, e.g. a wallclock-killed
        DNF seed); the sweep continues past it and lists the failures at the end
    2 — argparse error / up-front validation failure

Note: a child exit of 0 includes in-sim crashes, timeouts, and planner failures —
those are recorded inside the per-seed metrics JSON, not the exit code. Only a
non-zero child exit (a runner/config fault) counts as a runner failure here,
mirroring run_all / run_experiment.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Make repo root importable so `runners.run_experiment` / `runners.run_all` /
# `arena.speed_regimes` resolve when this module is invoked as
# `python -m runners.run_speed_sweep` from any cwd. Mirrors run_all.py:54-59.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arena.speed_regimes import SPEED_REGIMES  # noqa: E402
from planners import PREDICT_FAMILIES, algorithm_label  # noqa: E402
from runners._sweep_run import (  # noqa: E402
    SweepJob,
    add_common_sweep_args,
    run_jobs,
    summarize,
)
from runners.run_all import (  # noqa: E402
    DEFAULT_MASTER_SEED,
    DEFAULT_NUM_SEEDS,
    PREDICT_HORIZON,
    canonical_planner_set,
)


SPEED_SUBDIR_PREFIX = "speed_"           # results-dir suffix per regime; the child
                                         # re-inserts <world_stem>/<label> beneath it

# The hypothesis set: the static baseline (a_star_once), the incremental planner
# the D* Lite hypothesis is about, the two reactive planners whose degradation the
# issue calls out (dwa, apf), and the space-time-predictive DWA (issue #11: does
# true (x,y,t) prediction floor the crash rate as the obstacle-speed cap rises?).
# All non-replan, so replan_k is None; dwa_predictive is a predict family, so it
# carries the canonical PREDICT_HORIZON (the others carry None).
FOCUS_SET: tuple[str, ...] = ("a_star_once", "d_star_lite", "dwa", "dwa_predictive", "apf")


def focus_planner_set() -> list[tuple[str, int | None, int | None, str]]:
    """The FOCUS_SET as ``(algorithm, replan_k, predict_horizon, label)`` tuples, in order.

    Every focus planner is non-replan, so ``replan_k`` is ``None``. A predict-family
    focus planner (``dwa_predictive``) carries the canonical ``PREDICT_HORIZON`` and
    folds ``_h<H>`` into its label; every other carries ``None`` for the horizon and
    uses its bare key. Pure and side-effect-free, so it can be asserted on directly.
    The 4-tuple shape matches ``run_all.canonical_planner_set()`` so the two feed the
    same run loop.
    """
    planners: list[tuple[str, int | None, int | None, str]] = []
    for algorithm in FOCUS_SET:
        predict_horizon = PREDICT_HORIZON if algorithm in PREDICT_FAMILIES else None
        label = algorithm_label(algorithm, None, predict_horizon)
        planners.append((algorithm, None, predict_horizon, label))
    return planners


def selected_planner_set(algorithms: str) -> list[tuple[str, int | None, int | None, str]]:
    """Resolve ``--algorithms`` to its ``(algorithm, replan_k, predict_horizon, label)`` list.

    ``"focus"`` -> :func:`focus_planner_set` (the 5-planner hypothesis set);
    ``"all"`` -> ``run_all.canonical_planner_set()`` (the 13 canonical labels, with
    ``--replan-k 5`` folded into the replan families' labels and ``--predict-horizon``
    folded into the predict families'). Both return the same 4-tuple shape.
    """
    if algorithms == "focus":
        return focus_planner_set()
    if algorithms == "all":
        return canonical_planner_set()
    raise ValueError(f"unknown --algorithms set {algorithms!r}; expected 'focus' or 'all'")


def regime_results_dir(results_root: str, regime: str) -> str:
    """``<results_root>/speed_<regime>`` — the per-regime subtree root handed to a child.

    The child's ``episode_out_dir`` appends ``<world_stem>/<label>``, so files land
    at ``<results_root>/speed_<regime>/<world_stem>/<label>/`` — exactly where
    ``plot_speed_sweep`` reads. Pure string join (no I/O).
    """
    return str(Path(results_root) / f"{SPEED_SUBDIR_PREFIX}{regime}")


def build_experiment_cmd(
    algorithm: str,
    replan_k: int | None,
    predict_horizon: int | None,
    world: str,
    results_dir: str,
    regime: str,
    *,
    master_seed: int,
    num_seeds: int,
    jobs: int,
    traffic: bool,
    resume: bool,
) -> list[str]:
    """Construct one ``runners.run_experiment`` child command (no execution).

    Builds ``--algorithm <a> --world <abs> --results-dir <results_dir>
    --master-seed <m> --num-seeds <n> --jobs <j> --speed-regime <regime>``, then:

    - appends ``--replan-k <k>`` ONLY when ``replan_k`` is not None (the child
      rejects the flag for non-replan families),
    - appends ``--predict-horizon <h>`` ONLY when ``predict_horizon`` is not None
      (the child requires it for the predict families, rejects it otherwise),
    - appends ``--traffic`` or ``--no-traffic`` per the flag,
    - appends ``--resume`` only when requested.

    ``results_dir`` is the per-regime subtree root (``<root>/speed_<regime>``);
    pass it pre-built via :func:`regime_results_dir`. Pure (no I/O) so the
    command-builder TC can assert the forwarded argv directly.
    """
    cmd = [
        sys.executable,
        "-m",
        "runners.run_experiment",
        "--algorithm",
        algorithm,
        "--world",
        world,
        "--results-dir",
        results_dir,
        "--master-seed",
        str(master_seed),
        "--num-seeds",
        str(num_seeds),
        "--jobs",
        str(jobs),
        "--speed-regime",
        regime,
    ]
    if replan_k is not None:
        cmd.extend(["--replan-k", str(replan_k)])
    if predict_horizon is not None:
        cmd.extend(["--predict-horizon", str(predict_horizon)])
    cmd.append("--traffic" if traffic else "--no-traffic")
    if resume:
        cmd.append("--resume")
    return cmd


@dataclass(frozen=True)
class RunnerArgs:
    """Parsed CLI arguments — frozen so accidental mutation is impossible."""

    world: str
    algorithms: str
    master_seed: int
    num_seeds: int
    jobs: int
    results_dir: str
    resume: bool
    traffic: bool


def _parse_args(argv: list[str] | None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        prog="runners.run_speed_sweep",
        description="Run a planner set across the 4 named obstacle-speed regimes (Phase 7 sweep driver).",
    )
    add_common_sweep_args(
        parser,
        master_seed_default=DEFAULT_MASTER_SEED,
        num_seeds_default=DEFAULT_NUM_SEEDS,
        num_seeds_help=f"Seeds per (regime, planner) (default {DEFAULT_NUM_SEEDS}).",
        results_dir_help=(
            "Output directory root; each regime writes "
            "<results-dir>/speed_<regime>/<world_stem>/<label>/."
        ),
    )
    parser.add_argument(
        "--algorithms",
        choices=("focus", "all"),
        default="focus",
        help=(
            "Planner set: 'focus' (default) = the 5-planner hypothesis set "
            f"{', '.join(FOCUS_SET)}; 'all' = the 13 canonical planners."
        ),
    )
    ns = parser.parse_args(argv)
    return RunnerArgs(
        world=ns.world,
        algorithms=ns.algorithms,
        master_seed=int(ns.master_seed),
        num_seeds=int(ns.num_seeds),
        jobs=int(ns.jobs),
        results_dir=ns.results_dir,
        resume=bool(ns.resume),
        traffic=bool(ns.traffic),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the (regime x planner) sweep end-to-end. See module docstring for semantics."""
    args = _parse_args(argv)

    if args.num_seeds < 1:
        print(f"error: --num-seeds must be >= 1, got {args.num_seeds}", file=sys.stderr)
        return 2
    if args.jobs < 1:
        print(f"error: --jobs must be >= 1, got {args.jobs}", file=sys.stderr)
        return 2
    if args.master_seed < 0:
        print(f"error: --master-seed must be >= 0, got {args.master_seed}", file=sys.stderr)
        return 2

    # Resolve --world and the results-dir root to absolute paths ONCE (against this
    # process's cwd) so the children — launched with cwd=repo_root — agree on the
    # same trees. Mirrors run_all.main()'s resolution discipline.
    world_abs = Path(args.world).resolve()
    if not world_abs.exists():
        print(f"error: --world path does not exist: {world_abs}", file=sys.stderr)
        return 2
    world_abs_str = str(world_abs)

    results_root_abs = str(Path(args.results_dir).resolve())

    planners = selected_planner_set(args.algorithms)
    # Stable order: the regime dict's insertion order (slow/matched/current/fast)
    # x the planner-set order. Deterministic run to run so the [i/N] numbering and
    # the failure roster are reproducible.
    regimes = list(SPEED_REGIMES)

    print(
        f"run_speed_sweep: world_stem={world_abs.stem} algorithms={args.algorithms} "
        f"master_seed={args.master_seed} num_seeds={args.num_seeds} jobs={args.jobs} "
        f"regimes={','.join(regimes)} traffic={args.traffic} resume={args.resume}",
        flush=True,
    )

    # Build the (regime x planner) job list in the same nested order the [i/N]
    # numbering used to follow; build_experiment_cmd is pure (no I/O), so building
    # every job up front before launching any is behavior-preserving.
    jobs: list[SweepJob] = []
    for regime in regimes:
        regime_dir = regime_results_dir(results_root_abs, regime)
        for algorithm, replan_k, predict_horizon, label in planners:
            cmd = build_experiment_cmd(
                algorithm,
                replan_k,
                predict_horizon,
                world_abs_str,
                regime_dir,
                regime,
                master_seed=args.master_seed,
                num_seeds=args.num_seeds,
                jobs=args.jobs,
                traffic=args.traffic,
                resume=args.resume,
            )
            jobs.append(
                SweepJob(
                    display_label=f"regime={regime} {label}",
                    roster_label=f"[regime={regime}] {label}",
                    argv=cmd,
                )
            )

    outcomes = run_jobs(jobs, cwd=str(_REPO_ROOT))
    return summarize(outcomes, unit="(regime, planner)")


if __name__ == "__main__":
    raise SystemExit(main())
