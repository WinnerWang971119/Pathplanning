"""Sweep driver — run the predictive D* Lite family across a set of prediction horizons.

Phase 7 (Predictive / motion-aware D* Lite). The predictive D* Lite family stamps
each obstacle's predicted future footprint into the occupancy grid so the planner
routes *behind* moving traffic. The lookahead is the ``--predict-horizon`` knob
(integer steps; T seconds = steps x PREDICT_DT, PREDICT_DT = 0.1). This module is
the top-level orchestrator that sweeps that horizon for BOTH predictive keys — the
``d_star_lite_oracle`` (perfect velocities, the motion-aware ceiling) and the
``d_star_lite_predictive`` lidar estimator (frame-differencing velocities, the
realizable variant) — so ``plot_horizon_sweep`` can chart failure rate /
time-to-goal vs lookahead for each, the AC1 go/no-go deliverable.

It is a sibling of ``runners.run_speed_sweep``: where the speed sweep loops the 4
named speed regimes x a planner set, this driver loops the swept horizons x the
predictive planner set (both keys in :data:`SWEEP_ALGORITHMS`) and shells
``run_experiment`` once per (horizon, planner). Going through a subprocess (rather
than calling ``run_experiment.main()`` in-process) keeps each run isolated from the
others' ``sys.path`` / irsim import side effects and mirrors the established
two-tier subprocess pattern, so per-episode byte-determinism carries over unchanged.

No per-horizon subtree:

    Unlike the speed sweep (which hands each child a ``--results-dir
    <root>/speed_<regime>`` subtree), every horizon here is launched with the SAME
    ``--results-dir``. ``run_experiment``'s
    ``algorithm_label("d_star_lite_oracle", None, H)`` already folds the horizon
    into the label as ``d_star_lite_oracle_h<H>``, so each horizon lands in its own
    distinct dir ``<results-dir>/<world_stem>/d_star_lite_oracle_h<H>/``
    automatically. The driver invents no subtree.

There is no wallclock pass (that is ``run_all``'s concern); the sweep cares about
failure rate + time-to-goal, both already in the per-seed metrics JSON.

CLI:
    python -m runners.run_horizon_sweep \
        --world <yaml_path>      # required; e.g. arena/arena_v1.yaml
        [--master-seed <int>]    # default DEFAULT_MASTER_SEED
        [--num-seeds <int>]      # default 50
        [--jobs <int>]           # default 1; forwarded to each child run_experiment
        [--results-dir <dir>]    # default "results"
        [--resume]               # skip seeds whose JSON already exists (forwarded)
        [--traffic|--no-traffic] # crossing traffic, default ON
        [--horizons H ...]       # default 0 5 10 20 (steps); a list of ints

Swept planner set:
    SWEEP_ALGORITHMS lists which predictive keys to sweep. It holds both the
    ``d_star_lite_oracle`` (perfect-velocity ceiling) and the
    ``d_star_lite_predictive`` (lidar frame-differencing estimator) keys; the
    per-(horizon, planner) loop runs each at every horizon.

Exit codes:
    0 — every (horizon, planner) child subprocess exited 0 (ran to completion)
    1 — >= 1 child exited non-zero (a runner/config fault, e.g. a wallclock-killed
        DNF seed); the sweep continues past it and lists the failures at the end
    2 — argparse error / up-front validation failure

Note: a child exit of 0 includes in-sim crashes, timeouts, and planner failures —
those are recorded inside the per-seed metrics JSON, not the exit code. Only a
non-zero child exit (a runner/config fault) counts as a runner failure here,
mirroring run_all / run_experiment / run_speed_sweep.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Make repo root importable so `runners.run_experiment` / `runners.run_all`
# resolve when this module is invoked as `python -m runners.run_horizon_sweep`
# from any cwd. Mirrors run_speed_sweep.py:73-75.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runners._sweep_run import (  # noqa: E402
    SweepJob,
    add_common_sweep_args,
    run_jobs,
    summarize,
)
from runners.run_all import (  # noqa: E402
    DEFAULT_MASTER_SEED,
    DEFAULT_NUM_SEEDS,
)


# The horizon steps swept by default: 0/5/10/20 steps == 0.0/0.5/1.0/2.0 s at
# PREDICT_DT = 0.1. h0 stamps nothing and is the plain-D*-Lite baseline/ablation.
DEFAULT_HORIZONS: tuple[int, ...] = (0, 5, 10, 20)

# The predictive keys this driver sweeps: the two D* Lite (grid-stamping) variants
# and the two space-time DWA variants, each an oracle (perfect velocities) + a
# lidar estimator pair. The (horizon, planner) loop picks up every key
# automatically. Every predictive key is non-replan, so replan_k is None.
SWEEP_ALGORITHMS: tuple[str, ...] = (
    "d_star_lite_oracle",
    "d_star_lite_predictive",
    "dwa_predictive_oracle",
    "dwa_predictive",
)


def build_experiment_cmd(
    algorithm: str,
    world: str,
    results_dir: str,
    horizon: int,
    *,
    master_seed: int,
    num_seeds: int,
    jobs: int,
    traffic: bool,
    resume: bool,
) -> list[str]:
    """Construct one ``runners.run_experiment`` child command (no execution).

    Builds ``--algorithm <a> --predict-horizon <H> --world <abs>
    --results-dir <results_dir> --master-seed <m> --num-seeds <n> --jobs <j>``,
    then:

    - appends ``--traffic`` or ``--no-traffic`` per the flag,
    - appends ``--resume`` only when requested.

    Every swept planner is a predictive (non-replan) family, so ``--replan-k`` is
    never forwarded; ``--predict-horizon`` is ALWAYS forwarded (the child requires
    it for the predict family). ``results_dir`` is the SAME root for every horizon
    — ``run_experiment``'s ``algorithm_label`` folds the horizon into the
    ``_h<H>`` label dir, so the horizons do not collide. Pure (no I/O) so the
    command-builder TC can assert the forwarded argv directly.
    """
    cmd = [
        sys.executable,
        "-m",
        "runners.run_experiment",
        "--algorithm",
        algorithm,
        "--predict-horizon",
        str(horizon),
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
    ]
    cmd.append("--traffic" if traffic else "--no-traffic")
    if resume:
        cmd.append("--resume")
    return cmd


@dataclass(frozen=True)
class RunnerArgs:
    """Parsed CLI arguments — frozen so accidental mutation is impossible."""

    world: str
    master_seed: int
    num_seeds: int
    jobs: int
    results_dir: str
    resume: bool
    traffic: bool
    horizons: tuple[int, ...]


def _parse_args(argv: list[str] | None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        prog="runners.run_horizon_sweep",
        description=(
            "Run the predictive D* Lite family (the d_star_lite_oracle ceiling and "
            "the d_star_lite_predictive lidar estimator) across a set of prediction "
            "horizons (Phase 7 horizon sweep driver)."
        ),
    )
    add_common_sweep_args(
        parser,
        master_seed_default=DEFAULT_MASTER_SEED,
        num_seeds_default=DEFAULT_NUM_SEEDS,
        num_seeds_help=f"Seeds per (horizon, planner) (default {DEFAULT_NUM_SEEDS}).",
        results_dir_help=(
            "Output directory root; each horizon writes "
            "<results-dir>/<world_stem>/<key>_h<H>/ (the label folds in the "
            "horizon, so no per-horizon subtree is needed)."
        ),
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        metavar="H",
        help=(
            "Prediction horizons to sweep, in steps (default "
            f"{' '.join(str(h) for h in DEFAULT_HORIZONS)}). T seconds = steps x 0.1."
        ),
    )
    ns = parser.parse_args(argv)
    return RunnerArgs(
        world=ns.world,
        master_seed=int(ns.master_seed),
        num_seeds=int(ns.num_seeds),
        jobs=int(ns.jobs),
        results_dir=ns.results_dir,
        resume=bool(ns.resume),
        traffic=bool(ns.traffic),
        horizons=tuple(int(h) for h in ns.horizons),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the (horizon x planner) sweep end-to-end. See module docstring for semantics."""
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
    if not args.horizons:
        print("error: --horizons must list at least one horizon", file=sys.stderr)
        return 2
    if any(horizon < 0 for horizon in args.horizons):
        print(
            f"error: every --horizons value must be >= 0, got {list(args.horizons)}",
            file=sys.stderr,
        )
        return 2

    # Resolve --world and the results-dir root to absolute paths ONCE (against this
    # process's cwd) so the children — launched with cwd=repo_root — agree on the
    # same trees. Mirrors run_speed_sweep.main()'s resolution discipline.
    world_abs = Path(args.world).resolve()
    if not world_abs.exists():
        print(f"error: --world path does not exist: {world_abs}", file=sys.stderr)
        return 2
    world_abs_str = str(world_abs)

    results_root_abs = str(Path(args.results_dir).resolve())

    horizons = list(args.horizons)
    algorithms = list(SWEEP_ALGORITHMS)

    print(
        f"run_horizon_sweep: world_stem={world_abs.stem} "
        f"algorithms={','.join(algorithms)} master_seed={args.master_seed} "
        f"num_seeds={args.num_seeds} jobs={args.jobs} "
        f"horizons={','.join(str(h) for h in horizons)} "
        f"traffic={args.traffic} resume={args.resume}",
        flush=True,
    )

    # Build the (horizon x planner) job list in the same nested order the [i/N]
    # numbering used to follow; build_experiment_cmd is pure (no I/O), so building
    # every job up front before launching any is behavior-preserving.
    jobs: list[SweepJob] = []
    for horizon in horizons:
        for algorithm in algorithms:
            cmd = build_experiment_cmd(
                algorithm,
                world_abs_str,
                results_root_abs,
                horizon,
                master_seed=args.master_seed,
                num_seeds=args.num_seeds,
                jobs=args.jobs,
                traffic=args.traffic,
                resume=args.resume,
            )
            jobs.append(
                SweepJob(
                    display_label=f"horizon={horizon} {algorithm}",
                    roster_label=f"[horizon={horizon}] {algorithm}",
                    argv=cmd,
                )
            )

    outcomes = run_jobs(jobs, cwd=str(_REPO_ROOT))
    return summarize(outcomes, unit="(horizon, planner)")


if __name__ == "__main__":
    raise SystemExit(main())
