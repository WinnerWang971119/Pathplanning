"""Shared subprocess-driver helpers for the sweep orchestrators (speed + horizon).

`runners.run_speed_sweep` and `runners.run_horizon_sweep` both loop an axis (the 4
named speed regimes, or a set of prediction horizons) x a planner set and shell
`run_experiment` once per (axis-value, planner) child. The two drivers shared,
verbatim, the subprocess run loop, the progress / failure-roster printing, the
common argparse flags, and an outcome dataclass; that copy-paste lives here so a
change to the run/report contract happens in ONE place.

What stays in each driver: its own ``build_experiment_cmd`` (the per-axis child
argv), its own axis-specific flag (``--algorithms`` / ``--horizons``) and
validation, its own intro line, and the per-job label strings it hands to
:func:`run_jobs`.

HEADLESS (load-bearing): this module imports only the stdlib. It never imports
``runners.run_all`` (which pulls irsim) — the seed/num defaults are passed into
:func:`add_common_sweep_args` by the caller — nor matplotlib, so importing it has
no heavy side effects.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass


DEFAULT_JOBS = 1                         # per-child run_experiment concurrency
DEFAULT_RESULTS_DIR = "results"


@dataclass(frozen=True)
class SweepJob:
    """One sweep child to launch: its progress labels plus the argv to run.

    ``display_label`` is the per-line tag in the ``[sweep i/N] <display_label>:
    launching`` / ``... exit=<code>`` progress prints (e.g. ``"regime=fast
    a_star_once"`` or ``"horizon=10 d_star_lite_oracle"``). ``roster_label`` is the
    bracketed tag used in the end-of-run failure roster line
    ``  <roster_label> exit=<code>`` (e.g. ``"[regime=fast] a_star_once"``). Both
    are pre-formatted by the calling driver so the shared runner stays
    axis-agnostic. ``argv`` is the full child command (built by the driver's own
    ``build_experiment_cmd``).
    """

    display_label: str
    roster_label: str
    argv: list[str]


@dataclass(frozen=True)
class SweepOutcome:
    """Outcome of one sweep child subprocess (carries its labels for the roster)."""

    display_label: str
    roster_label: str
    exit_code: int          # child return code; 0 == ran to completion
    ok: bool                # exit_code == 0


def add_common_sweep_args(
    parser: argparse.ArgumentParser,
    *,
    master_seed_default: int,
    num_seeds_default: int,
    num_seeds_help: str,
    results_dir_help: str,
) -> None:
    """Register the flags shared by every sweep driver onto ``parser``.

    Adds ``--world`` (required), ``--master-seed``, ``--num-seeds``, ``--jobs``,
    ``--results-dir``, ``--resume``, and the ``--traffic`` / ``--no-traffic``
    mutually-exclusive pair (default traffic ON). The two help strings that read
    differently per driver — ``--num-seeds``'s "per (regime|horizon, planner)"
    phrasing and ``--results-dir``'s subtree description — are passed in so each
    driver keeps its own wording; every other flag is byte-identical across drivers.

    The seed/num defaults are passed in (rather than imported from
    ``runners.run_all``, which would pull irsim) so this module stays headless.
    """
    parser.add_argument(
        "--world",
        required=True,
        help="Path to the world YAML (e.g. arena/arena_v1.yaml).",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=master_seed_default,
        help=f"Master seed for the seed derivation (default {master_seed_default}).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=num_seeds_default,
        help=num_seeds_help,
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help="Concurrency forwarded to each child run_experiment (default 1).",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=results_dir_help,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip seeds whose <seed>.json already exists; forwarded to each child (default: overwrite).",
    )
    traffic_group = parser.add_mutually_exclusive_group()
    traffic_group.add_argument(
        "--traffic", dest="traffic", action="store_true", help="Enable crossing traffic (default)."
    )
    traffic_group.add_argument(
        "--no-traffic", dest="traffic", action="store_false", help="Disable traffic."
    )
    parser.set_defaults(traffic=True)


def run_jobs(jobs: Sequence[SweepJob], *, cwd: str) -> list[SweepOutcome]:
    """Run each sweep child subprocess in order, printing the shared progress lines.

    For job ``i`` of ``N`` (1-based), prints ``[sweep i/N] <display_label>:
    launching``, runs ``subprocess.run(job.argv, cwd=cwd)``, and prints
    ``[sweep i/N] <display_label> exit=<code>``. If the child fails to spawn
    (``OSError``, e.g. a missing interpreter) it is treated as exit code 1 and a
    ``... failed to spawn child: <exc>`` line is printed to stderr, so the sweep
    continues past it. Returns one :class:`SweepOutcome` per job, in job order.
    """
    outcomes: list[SweepOutcome] = []
    total = len(jobs)
    for index, job in enumerate(jobs, start=1):
        print(f"[sweep {index}/{total}] {job.display_label}: launching", flush=True)
        try:
            proc = subprocess.run(job.argv, cwd=cwd)
            exit_code = proc.returncode
        except OSError as exc:
            # The child never started (e.g. a missing interpreter). Treat it like a
            # non-zero exit so the sweep continues and the failure is reported.
            print(
                f"[sweep {index}/{total}] {job.display_label}: failed to spawn child: {exc}",
                file=sys.stderr,
                flush=True,
            )
            exit_code = 1
        ok = exit_code == 0
        print(f"[sweep {index}/{total}] {job.display_label} exit={exit_code}", flush=True)
        outcomes.append(
            SweepOutcome(
                display_label=job.display_label,
                roster_label=job.roster_label,
                exit_code=exit_code,
                ok=ok,
            )
        )
    return outcomes


def summarize(outcomes: Sequence[SweepOutcome], *, unit: str) -> int:
    """Print the ``done:`` line + failure roster and return the driver exit code.

    ``unit`` is the parenthesised pair phrase naming what each child run is
    (e.g. ``"(regime, planner)"`` or ``"(horizon, planner)"``), so the summary reads
    ``done: <n_ok>/<n_total> <unit> runs exited 0, <n_failed> runner-failed.``. When
    any child exited non-zero, prints a ``runner failures:`` roster (one
    ``  <roster_label> exit=<code>`` line per failure) to stderr and returns 1;
    otherwise returns 0.
    """
    failures = [outcome for outcome in outcomes if not outcome.ok]
    n_total = len(outcomes)
    n_ok = n_total - len(failures)
    print(
        f"done: {n_ok}/{n_total} {unit} runs exited 0, {len(failures)} runner-failed.",
        flush=True,
    )
    if failures:
        print("runner failures:", file=sys.stderr)
        for outcome in failures:
            print(f"  {outcome.roster_label} exit={outcome.exit_code}", file=sys.stderr)
        return 1
    return 0
