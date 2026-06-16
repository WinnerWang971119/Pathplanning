"""Batch driver — run EVERY canonical planner against the canonical seed stream.

Phase 5 (Comparison study). Mission.md's cross-algorithm scatter plot needs every
planner to face the SAME seeded traffic streams on the SAME world. This module is
the top-level orchestrator: for each of the 11 canonical planners it shells out to
the already-deterministic batch runner (`runners.run_experiment`) once, exactly as
`run_experiment` itself shells out to `run_episode` once per seed. Going through a
subprocess (rather than calling `run_experiment.main()` in-process) keeps each
planner isolated from the others' `sys.path` / import side effects and mirrors the
existing two-tier subprocess pattern.

Two passes:

1. Bulk pass — every planner over the full ``--num-seeds`` stream at ``--jobs``.
   Children write to ``<results-dir>/<world_stem>/<label>/`` (the plotter's main
   input).
2. Wallclock pass — every planner over a short ``--wallclock-seeds`` prefix at
   ``--jobs 1`` (serial → uncontended ``wallclock_per_step``). Children write to
   ``<results-dir>/__wallclock__/<world_stem>/<label>/`` (the plotter's B3 input).
   The results-dir handed to the children is ``<results-dir>/__wallclock__`` so
   that ``episode_out_dir``'s unconditional ``<world_stem>/<label>`` suffix lands
   the files where B3 reads — the stem is re-inserted ONCE, never double-nested.

CLI:
    python -m runners.run_all \
        --world <yaml_path>      # required; e.g. arena/arena_v1.yaml
        [--master-seed <int>]    # default DEFAULT_MASTER_SEED
        [--num-seeds <int>]      # default 50 (bulk pass)
        [--jobs <int>]           # default 1; bulk pass concurrency (wallclock is always 1)
        [--wallclock-seeds <int>]# default 5 (serial wallclock prefix)
        [--results-dir <dir>]    # default "results"
        [--resume]               # skip seeds whose JSON already exists (per pass)
        [--traffic|--no-traffic] # crossing traffic, default ON

Exit codes:
    0 — every child subprocess (bulk + wallclock) exited 0 (ran to completion)
    1 — >= 1 child exited non-zero (a runner/config fault); the batch continues
        past it and lists the failures at the end
    2 — argparse error / up-front validation failure

Note: a child exit of 0 includes in-sim crashes, timeouts, and planner failures —
those are recorded inside the per-seed metrics JSON, not the exit code. Only a
non-zero child exit (a runner/config fault) counts as a failure here, mirroring
`run_experiment`'s policy.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Make repo root importable so `runners.run_experiment` / `planners` resolve when
# this module is invoked as `python -m runners.run_all` from any cwd. Mirrors
# runners/run_experiment.py:69-74.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from planners import ALGORITHMS, algorithm_label  # noqa: E402
from planners._grid import REPLAN_FAMILIES  # noqa: E402


DEFAULT_MASTER_SEED = 20260605          # canonical experiment stream (matches run_experiment)
DEFAULT_NUM_SEEDS = 50                   # Mission.md: 50 seeds per algorithm (bulk pass)
DEFAULT_WALLCLOCK_SEEDS = 5              # short serial prefix for clean wallclock_per_step
DEFAULT_JOBS = 1                         # bulk-pass concurrency (wallclock is always serial)
DEFAULT_RESULTS_DIR = "results"
WALLCLOCK_SUBDIR = "__wallclock__"       # results-dir suffix for the serial pass; the child
                                         # re-inserts <world_stem>/<label> beneath it
REPLAN_K = 5                             # canonical cadence for every replan family

# Stable, hand-listed canonical order so the driver is reproducible run to run and
# the tally line numbering ([bulk i/11]) is deterministic. This is the authoritative
# ordering; it is asserted against ALGORITHMS at module import so a future registry
# change cannot silently drift the two apart.
_CANONICAL_ORDER: tuple[str, ...] = (
    "a_star_once",
    "a_star_replan",
    "dijkstra_once",
    "dijkstra_replan",
    "d_star_lite",
    "dwa",
    "apf",
    "rrt_once",
    "rrt_replan",
    "rrt_star_once",
    "rrt_star_replan",
)

# Fail loud at import if the hand-listed order ever falls out of sync with the
# registry (a new key added, a key renamed/removed). The driver's whole contract is
# "ALL canonical planners", so a silent mismatch would quietly drop or duplicate a
# planner from the study.
if set(_CANONICAL_ORDER) != set(ALGORITHMS):
    _missing = set(ALGORITHMS) - set(_CANONICAL_ORDER)
    _extra = set(_CANONICAL_ORDER) - set(ALGORITHMS)
    raise RuntimeError(
        "run_all._CANONICAL_ORDER is out of sync with planners.ALGORITHMS "
        f"(missing from order: {sorted(_missing)}, unknown in order: {sorted(_extra)})."
    )


def canonical_planner_set() -> list[tuple[str, int | None, str]]:
    """The 11 canonical ``(algorithm, replan_k, label)`` tuples, in canonical order.

    Replan-family keys (``planners._grid.REPLAN_FAMILIES``) carry the canonical
    cadence ``REPLAN_K``; every other planner carries ``None``. ``label`` is
    ``algorithm_label(algorithm, replan_k)`` so it matches the results-dir each
    child writes to (e.g. ``a_star_replan_k5`` vs the bare ``a_star_once``). Pure
    and side-effect-free — T5's TC-P10 imports and asserts on this directly.
    """
    planners: list[tuple[str, int | None, str]] = []
    for algorithm in _CANONICAL_ORDER:
        replan_k = REPLAN_K if algorithm in REPLAN_FAMILIES else None
        label = algorithm_label(algorithm, replan_k)
        planners.append((algorithm, replan_k, label))
    return planners


def build_experiment_cmd(
    algorithm: str,
    replan_k: int | None,
    world: str,
    results_dir: str,
    *,
    master_seed: int,
    num_seeds: int,
    jobs: int,
    traffic: bool,
    resume: bool,
) -> list[str]:
    """Construct one ``runners.run_experiment`` child command (no execution).

    Appends ``--replan-k <k>`` ONLY when ``replan_k`` is not None (the child
    rejects the flag for non-replan families), ``--traffic`` or ``--no-traffic``
    per the flag, and ``--resume`` only when requested. Pure so TC-P10 can assert
    that a replan family's command carries ``--replan-k 5`` and a non-replan
    family's does not.
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
    ]
    if replan_k is not None:
        cmd.extend(["--replan-k", str(replan_k)])
    cmd.append("--traffic" if traffic else "--no-traffic")
    if resume:
        cmd.append("--resume")
    return cmd


@dataclass(frozen=True)
class PlannerResult:
    """Outcome of one planner's child `run_experiment` subprocess in one pass."""

    pass_name: str          # "bulk" | "wallclock"
    algorithm: str
    label: str
    exit_code: int          # child return code; 0 == ran to completion
    ok: bool                # exit_code == 0


@dataclass(frozen=True)
class RunnerArgs:
    """Parsed CLI arguments — frozen so accidental mutation is impossible."""

    world: str
    master_seed: int
    num_seeds: int
    jobs: int
    wallclock_seeds: int
    results_dir: str
    resume: bool
    traffic: bool


def _parse_args(argv: list[str] | None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        prog="runners.run_all",
        description="Run every canonical planner against the canonical seed stream (Phase 5 driver).",
    )
    parser.add_argument(
        "--world",
        required=True,
        help="Path to the world YAML (e.g. arena/arena_v1.yaml).",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
        help=f"Master seed for the seed derivation (default {DEFAULT_MASTER_SEED}).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help=f"Seeds per planner in the bulk pass (default {DEFAULT_NUM_SEEDS}).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help="Bulk-pass concurrency forwarded to each child (default 1; the wallclock pass is always 1).",
    )
    parser.add_argument(
        "--wallclock-seeds",
        type=int,
        default=DEFAULT_WALLCLOCK_SEEDS,
        help=(
            f"Seeds per planner in the serial wallclock pass (default {DEFAULT_WALLCLOCK_SEEDS}); "
            "a prefix of the same canonical stream."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Output directory root; the bulk pass writes <results-dir>/<world_stem>/<label>/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip seeds whose <seed>.json already exists; scoped per pass (default: overwrite).",
    )
    traffic_group = parser.add_mutually_exclusive_group()
    traffic_group.add_argument(
        "--traffic", dest="traffic", action="store_true", help="Enable crossing traffic (default)."
    )
    traffic_group.add_argument(
        "--no-traffic", dest="traffic", action="store_false", help="Disable traffic."
    )
    parser.set_defaults(traffic=True)
    ns = parser.parse_args(argv)
    return RunnerArgs(
        world=ns.world,
        master_seed=int(ns.master_seed),
        num_seeds=int(ns.num_seeds),
        jobs=int(ns.jobs),
        wallclock_seeds=int(ns.wallclock_seeds),
        results_dir=ns.results_dir,
        resume=bool(ns.resume),
        traffic=bool(ns.traffic),
    )


def _run_pass(
    *,
    pass_name: str,
    results_dir: str,
    num_seeds: int,
    jobs: int,
    args: RunnerArgs,
    world_abs: str,
) -> list[PlannerResult]:
    """Run every canonical planner once in one pass; return per-planner outcomes.

    Each planner is a `run_experiment` subprocess launched with ``cwd=repo_root``
    (so the children agree on the absolute trees the parent resolved). A non-zero
    child exit is a runner/config fault: it is recorded and the pass continues to
    the next planner. The child's own stdout/stderr stream through to this console
    (not captured) so the per-seed progress and any traceback are visible live.
    """
    planners = canonical_planner_set()
    total = len(planners)
    results: list[PlannerResult] = []

    for index, (algorithm, replan_k, label) in enumerate(planners):
        cmd = build_experiment_cmd(
            algorithm,
            replan_k,
            world_abs,
            results_dir,
            master_seed=args.master_seed,
            num_seeds=num_seeds,
            jobs=jobs,
            traffic=args.traffic,
            resume=args.resume,
        )
        print(f"[{pass_name} {index + 1}/{total}] {label}: launching", flush=True)
        try:
            proc = subprocess.run(cmd, cwd=str(_REPO_ROOT))
            exit_code = proc.returncode
        except OSError as exc:
            # The child never started (e.g. a missing interpreter). Treat it like a
            # non-zero exit so the pass continues and the failure is reported.
            print(f"[{pass_name} {index + 1}/{total}] {label}: failed to spawn child: {exc}",
                  file=sys.stderr, flush=True)
            exit_code = 1
        ok = exit_code == 0
        print(f"[{pass_name} {index + 1}/{total}] {label} exit={exit_code}", flush=True)
        results.append(
            PlannerResult(
                pass_name=pass_name,
                algorithm=algorithm,
                label=label,
                exit_code=exit_code,
                ok=ok,
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    """Run the bulk + wallclock passes end-to-end. See module docstring for semantics."""
    args = _parse_args(argv)

    if args.num_seeds < 1:
        print(f"error: --num-seeds must be >= 1, got {args.num_seeds}", file=sys.stderr)
        return 2
    if args.wallclock_seeds < 1:
        print(f"error: --wallclock-seeds must be >= 1, got {args.wallclock_seeds}", file=sys.stderr)
        return 2
    if args.jobs < 1:
        print(f"error: --jobs must be >= 1, got {args.jobs}", file=sys.stderr)
        return 2
    if args.master_seed < 0:
        print(f"error: --master-seed must be >= 0, got {args.master_seed}", file=sys.stderr)
        return 2

    # Resolve --world and the results-dir root to absolute paths ONCE (against this
    # process's cwd) so the children — launched with cwd=repo_root — agree on the
    # same trees. Mirrors run_experiment.main()'s resolution discipline.
    world_abs = Path(args.world).resolve()
    if not world_abs.exists():
        print(f"error: --world path does not exist: {world_abs}", file=sys.stderr)
        return 2
    world_abs_str = str(world_abs)

    results_root_abs = Path(args.results_dir).resolve()
    bulk_results_dir = str(results_root_abs)
    # The child's episode_out_dir ALWAYS appends <world_stem>/<label>, so handing it
    # <results-dir>/__wallclock__ makes it write to
    # <results-dir>/__wallclock__/<world_stem>/<label> — exactly where the plotter's
    # B3 reads. Building <results-dir>/<world_stem>/__wallclock__ instead would
    # double-nest the stem and the plotter would never find it.
    wallclock_results_dir = str(results_root_abs / WALLCLOCK_SUBDIR)

    print(
        f"run_all: world_stem={world_abs.stem} master_seed={args.master_seed} "
        f"num_seeds={args.num_seeds} jobs={args.jobs} wallclock_seeds={args.wallclock_seeds} "
        f"traffic={args.traffic} resume={args.resume}",
        flush=True,
    )

    all_results: list[PlannerResult] = []

    print("=== bulk pass ===", flush=True)
    all_results.extend(
        _run_pass(
            pass_name="bulk",
            results_dir=bulk_results_dir,
            num_seeds=args.num_seeds,
            jobs=args.jobs,
            args=args,
            world_abs=world_abs_str,
        )
    )

    print(f"=== wallclock pass (serial, {args.wallclock_seeds} seeds) ===", flush=True)
    all_results.extend(
        _run_pass(
            pass_name="wallclock",
            results_dir=wallclock_results_dir,
            num_seeds=args.wallclock_seeds,
            jobs=1,  # serial regardless of --jobs, for an uncontended wallclock_per_step
            args=args,
            world_abs=world_abs_str,
        )
    )

    failures = [r for r in all_results if not r.ok]
    n_total = len(all_results)
    n_ok = n_total - len(failures)
    print(
        f"done: {n_ok}/{n_total} planner runs exited 0, {len(failures)} runner-failed "
        f"(across bulk + wallclock passes).",
        flush=True,
    )
    if failures:
        print("runner failures:", file=sys.stderr)
        for r in failures:
            print(f"  [{r.pass_name}] {r.label} exit={r.exit_code}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
