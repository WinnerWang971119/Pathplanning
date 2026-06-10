"""Read-only result plotter — loads the batch result JSONs and renders the study charts.

Phase 5 (Analysis). The batch runner (`runners.run_experiment`) writes one
`<seed>.json` metrics file per episode under
`<results-dir>/<world_stem>/<label>/`, plus a `_manifest.json` provenance
receipt. This module reads ONLY those JSONs (never irsim, never a sim) and turns
them into the cross-algorithm comparison charts Mission.md's analysis calls for.

This file is built in layers across T1/T2/T3/T5:
    T1 (this task) — the loader, the outcome classifier, the per-algorithm
        summary math, the summary CSV, and the argparse CLI skeleton. NO chart
        functions yet: the seven chart entry points are registered as stubs that
        raise NotImplementedError, and `--selfcheck` is a placeholder. Nothing
        here imports matplotlib until `ensure_matplotlib()` is called, so the
        loader/classifier unit tests stay headless.
    T2/T3 — fill in the seven chart functions (A1/A3/A4 and B1/B2/B3/B4).
    T5 — fill in `run_selfcheck()` and the unit tests (TC-P*).

CLI (once the chart layer lands):
    python -m runners.plot \
        --world <yaml_path>     # required; e.g. arena/arena_v1.yaml
        [--results-dir <dir>]   # default "results"
        [--replan-k <int>]      # default 5; cadence used to build replan labels
        [--charts a1,a3,...]    # default all of a1,a3,a4,b1,b2,b3,b4
        [--out-dir <dir>]       # default <results-dir>/<world_stem>/plots/
        [--selfcheck]           # run the self-check suite (T5) instead of plotting

Outputs (once the chart layer lands):
    <out-dir>/summary.csv           — per-algorithm tally (written by T1's loader)
    <out-dir>/<chart>.png           — one PNG per requested chart (T2/T3)

Exit codes:
    0 — charts/summary written (or selfcheck passed, once T5 lands)
    1 — matplotlib missing, no readable data, or a fatal render error
    2 — argparse / CLI validation error
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Make the repo root importable so `from planners import algorithm_label` resolves
# when this module is invoked as `python -m runners.plot` from any cwd. Mirrors
# runners/run_episode.py:54-58 and runners/run_experiment.py:69-74.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# `algorithm_label` (from `planners`) and `episode_out_dir` (from
# `runners._layout`) are imported LAZILY inside the functions that use them, NOT
# at module top level. Importing `planners` transitively pulls irsim and selects
# a matplotlib backend (planners -> manual_astar -> irsim -> matplotlib.pyplot /
# TkAgg). Keeping that import deferred is what lets `import runners.plot` stay
# headless (AC1), so the loader/classifier unit tests never touch matplotlib.


# --- Module constants -------------------------------------------------------

DEFAULT_RESULTS_DIR = "results"
DEFAULT_REPLAN_K = 5                 # cadence used to build the _replan label dirs (a_star_replan_k5, ...)
DEFAULT_EXPECTED_SEEDS = 50          # Mission.md: 50 seeds per algorithm
EPISODE_GLOB = "[0-9]*.json"         # numeric-stem episode files ONLY (skips _manifest.json)
MANIFEST_NAME = "_manifest.json"     # provenance receipt written by run_experiment
WALLCLOCK_SUBTREE = "__wallclock__"  # sibling of <world_stem> at the results ROOT; holds the B3 wallclock runs
SUMMARY_CSV_NAME = "summary.csv"     # per-algorithm tally written by write_summary_csv
PLOTS_DIR_NAME = "plots"             # default out-dir leaf under <results-dir>/<world_stem>/

# The seven charts this plotter produces. Order is the dispatch/registration
# order; the default --charts value is exactly this tuple.
CHART_KEYS = ("a1", "a3", "a4", "b1", "b2", "b3", "b4")

# Outcome buckets in precedence order (see classify_outcome). A record is exactly
# one of these.
OUTCOMES = ("success", "crash", "timeout", "planner_error")

# Euclidean (2,2) -> (48,48) straight-line distance: the unreachable lower bound
# every executed path is compared against (the robot must detour around walls).
STRAIGHT_LINE_IDEAL_M = 46.0 * (2.0 ** 0.5)   # ~= 65.05 m

# The canonical algorithm set, in the order they appear in every chart legend.
# Each tuple is (registry name, replan_k or None, family, display label). The
# concrete replan_k values here are placeholders; load_world_results() overrides
# them with the CLI's --replan-k so the labels match the dirs run_experiment
# actually wrote.
CANONICAL: list[tuple[str, int | None, str, str]] = [
    ("a_star_once",        None, "grid",        "A* once"),
    ("a_star_replan",      5,    "grid",        "A* replan (K=5)"),
    ("dijkstra_once",      None, "grid",        "Dijkstra once"),
    ("dijkstra_replan",    5,    "grid",        "Dijkstra replan (K=5)"),
    ("d_star_lite",        None, "incremental", "D* Lite"),
    ("dwa",                None, "reactive",    "DWA"),
    ("apf",                None, "reactive",    "APF"),
    ("rrt_once",           None, "sampling",    "RRT once"),
    ("rrt_replan",         5,    "sampling",    "RRT replan (K=5)"),
    ("rrt_star_once",      None, "sampling",    "RRT* once"),
    ("rrt_star_replan",    5,    "sampling",    "RRT* replan (K=5)"),
]


# --- matplotlib import guard (AC1) ------------------------------------------

def ensure_matplotlib():
    """Return the pyplot module, or print a friendly hint and exit non-zero if matplotlib is absent.

    The `importlib.util.find_spec` probe is the seam T5's TC-P8 patches to
    simulate a matplotlib-free environment, so it must run BEFORE any
    `import matplotlib` statement. On success the headless Agg backend is
    selected (no display required) and pyplot is returned.
    """
    import importlib.util
    import sys as _sys

    if importlib.util.find_spec("matplotlib") is None:
        print("error: matplotlib is required. Run: pip install -r requirements.txt", file=_sys.stderr)
        raise SystemExit(1)
    import matplotlib
    matplotlib.use("Agg")          # headless backend — no display needed
    import matplotlib.pyplot as plt
    return plt


# --- Outcome classifier (AC2) -----------------------------------------------

def classify_outcome(rec: dict) -> str:
    """Bucket one episode metrics record into exactly one of OUTCOMES.

    Precedence is load-bearing: a `reset()` failure writes `planner_error` with
    the other flags false and `time_to_goal` null, so check it first. The final
    fallthrough is defensive only — `run_episode` always writes one of the flags
    or a non-null `time_to_goal`, so it is unreachable from real output and only
    guards malformed/hand-authored records.
    """
    if rec.get("planner_error") is not None:
        return "planner_error"
    if rec.get("crashed"):
        return "crash"
    if rec.get("timed_out"):
        return "timeout"
    if rec.get("time_to_goal") is not None:
        return "success"
    return "planner_error"


# --- Data model -------------------------------------------------------------

@dataclass(frozen=True)
class AlgoSummary:
    """Per-algorithm aggregate over its present episode JSONs.

    `times` / `path_lengths` cover ONLY the successful episodes. `wallclocks` is
    sourced from the dedicated `__wallclock__` subtree when present (see
    `wallclock_from_subtree`); when that subtree is absent the tuple is empty and
    the flag is False, so B3 (built later) can fall back to the bulk dir's
    wallclock with a caveat. `failure_rate` is NaN when no episodes are present.
    """

    label: str                          # results dir label, e.g. "a_star_replan_k5"
    display: str                        # legend name, e.g. "A* replan (K=5)"
    family: str                         # "grid" | "incremental" | "reactive" | "sampling"
    n_present: int
    n_success: int
    n_crash: int
    n_timeout: int
    n_planner_error: int
    failure_rate: float                 # (crash+timeout+planner_error)/n_present; NaN if n_present == 0
    times: tuple[float, ...]            # successful time_to_goal values
    path_lengths: tuple[float, ...]     # path_length over successes
    wallclocks: tuple[float, ...]       # wallclock_per_step from the __wallclock__ subtree (empty if absent)
    per_seed: dict[int, str]            # seed -> outcome (for the B1 heatmap)
    median_time: float                  # median of times; NaN if n_success == 0
    mean_time: float                    # mean of times; NaN if n_success == 0
    wallclock_from_subtree: bool        # True iff wallclocks came from the __wallclock__ subtree


@dataclass(frozen=True)
class WorldResults:
    """Everything the loader produced for one world: the per-algorithm summaries
    plus the canonical seed-column order the B1 heatmap aligns every row to.

    `seed_order` is the manifest's `derived_seeds` when any label's
    `_manifest.json` was found, otherwise the sorted union of every numeric stem
    actually present on disk. `manifest_seed_order` records which of the two
    happened so a later chart can caveat a fallback ordering.
    """

    summaries: tuple[AlgoSummary, ...]
    seed_order: tuple[int, ...]
    manifest_seed_order: bool           # True iff seed_order came from a manifest's derived_seeds


# --- Loader (AC2 / AC3 / AC9 / AC11) ----------------------------------------

@dataclass
class _AlgoAccumulator:
    """Mutable scratch tally for one algorithm while scanning its JSONs."""

    n_present: int = 0
    n_success: int = 0
    n_crash: int = 0
    n_timeout: int = 0
    n_planner_error: int = 0
    times: list[float] = field(default_factory=list)
    path_lengths: list[float] = field(default_factory=list)
    per_seed: dict[int, str] = field(default_factory=dict)


def _read_json(path: Path) -> dict | None:
    """Parse one JSON file, or warn to stderr and return None on any read/parse error."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: skipping unreadable JSON {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"warning: skipping JSON {path}: expected an object, got {type(data).__name__}", file=sys.stderr)
        return None
    return data


def _seed_from_stem(path: Path) -> int | None:
    """Parse the integer seed from a `<seed>.json` stem, or warn + return None."""
    try:
        return int(path.stem)
    except ValueError:
        print(f"warning: skipping non-numeric episode file {path}", file=sys.stderr)
        return None


def _load_manifest_seed_order(label_dir: Path) -> tuple[int, ...] | None:
    """Return `derived_seeds` from this label dir's manifest, or None if absent/unusable."""
    manifest_path = label_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    data = _read_json(manifest_path)
    if data is None:
        return None
    derived = data.get("derived_seeds")
    if not isinstance(derived, list) or not derived:
        return None
    try:
        return tuple(int(seed) for seed in derived)
    except (TypeError, ValueError):
        print(f"warning: ignoring malformed derived_seeds in {manifest_path}", file=sys.stderr)
        return None


def _load_wallclocks(
    results_dir: str,
    world_stem: str,
    label: str,
) -> tuple[tuple[float, ...], bool]:
    """Load `wallclock_per_step` from the dedicated `__wallclock__` subtree.

    The subtree lives at `<results_dir>/__wallclock__/<world_stem>/<label>/` — a
    SIBLING of `<world_stem>` at the results root, NOT under the bulk world dir.
    Returns (values, present): an empty tuple + False when the subtree is missing
    (B3 falls back to the bulk wallclock with a caveat); a populated tuple + True
    otherwise. Never raises on a missing subtree (AC11).
    """
    subtree_dir = Path(results_dir).resolve() / WALLCLOCK_SUBTREE / world_stem / label
    if not subtree_dir.is_dir():
        return (), False

    values: list[float] = []
    for json_path in sorted(subtree_dir.glob(EPISODE_GLOB)):
        if _seed_from_stem(json_path) is None:
            continue
        rec = _read_json(json_path)
        if rec is None:
            continue
        wallclock = rec.get("wallclock_per_step")
        if isinstance(wallclock, (int, float)):
            values.append(float(wallclock))
    return tuple(values), True


def _accumulate_episode(acc: _AlgoAccumulator, seed: int, rec: dict) -> None:
    """Fold one parsed episode record into the accumulator (counts + per-success metrics)."""
    outcome = classify_outcome(rec)
    acc.n_present += 1
    acc.per_seed[seed] = outcome
    if outcome == "success":
        acc.n_success += 1
        time_to_goal = rec.get("time_to_goal")
        if isinstance(time_to_goal, (int, float)):
            acc.times.append(float(time_to_goal))
        path_length = rec.get("path_length")
        if isinstance(path_length, (int, float)):
            acc.path_lengths.append(float(path_length))
    elif outcome == "crash":
        acc.n_crash += 1
    elif outcome == "timeout":
        acc.n_timeout += 1
    else:  # "planner_error"
        acc.n_planner_error += 1


def _finalize_summary(
    *,
    label: str,
    display: str,
    family: str,
    acc: _AlgoAccumulator,
    wallclocks: tuple[float, ...],
    wallclock_from_subtree: bool,
) -> AlgoSummary:
    """Freeze an accumulator into an immutable AlgoSummary with the derived stats.

    `failure_rate`, `median_time`, and `mean_time` are NaN when their
    denominator is zero (n_present == 0 / n_success == 0). NaN is the chosen
    "no data" sentinel everywhere in this module so the dtype stays float and
    downstream charts can drop NaNs uniformly.
    """
    n_failed = acc.n_crash + acc.n_timeout + acc.n_planner_error
    failure_rate = (n_failed / acc.n_present) if acc.n_present > 0 else float("nan")
    median_time = statistics.median(acc.times) if acc.times else float("nan")
    mean_time = statistics.fmean(acc.times) if acc.times else float("nan")
    return AlgoSummary(
        label=label,
        display=display,
        family=family,
        n_present=acc.n_present,
        n_success=acc.n_success,
        n_crash=acc.n_crash,
        n_timeout=acc.n_timeout,
        n_planner_error=acc.n_planner_error,
        failure_rate=failure_rate,
        times=tuple(acc.times),
        path_lengths=tuple(acc.path_lengths),
        wallclocks=wallclocks,
        per_seed=dict(acc.per_seed),
        median_time=median_time,
        mean_time=mean_time,
        wallclock_from_subtree=wallclock_from_subtree,
    )


def load_world_results(
    results_dir: str,
    world_stem: str,
    *,
    replan_k: int = DEFAULT_REPLAN_K,
    expected: int = DEFAULT_EXPECTED_SEEDS,
) -> WorldResults:
    """Load every canonical algorithm's episodes for one world into summaries.

    For each CANONICAL entry, the dir label is recomputed via
    `algorithm_label(name, replan_k or None)` (so the CLI's `--replan-k` picks
    the `a_star_replan_k<K>` dirs) and the numeric-stem episode JSONs under
    `<results_dir>/<world_stem>/<label>/` are read, classified, and tallied.
    `_manifest.json` and any non-numeric-stem file are skipped. A missing label
    dir, a short count (< `expected`), or an unreadable file warns to stderr and
    is skipped — the loader NEVER raises (AC11).

    The B1 heatmap's seed-column order comes from the first manifest found (any
    label's `derived_seeds`); absent any manifest it falls back to the sorted
    union of every numeric stem present.
    """
    # Deferred to here so a bare `import runners.plot` stays headless (AC1):
    # importing `planners` pulls irsim, which selects a matplotlib backend
    # (planners -> manual_astar -> irsim -> matplotlib.pyplot / TkAgg).
    from planners import algorithm_label
    from runners._layout import episode_out_dir

    results_root = Path(results_dir).resolve()
    summaries: list[AlgoSummary] = []
    manifest_seed_order: tuple[int, ...] | None = None
    seen_seeds: set[int] = set()

    for name, default_k, family, display in CANONICAL:
        # A replan family takes the CLI cadence; everything else stays at None so
        # algorithm_label returns its bare key.
        effective_k = replan_k if default_k is not None else None
        label = algorithm_label(name, effective_k)
        label_dir = episode_out_dir(results_root, world_stem, label)

        acc = _AlgoAccumulator()

        if not label_dir.is_dir():
            print(
                f"warning: no result dir for {label} at {label_dir} (skipping)",
                file=sys.stderr,
            )
        else:
            # The first manifest we encounter fixes the canonical seed-column order
            # for the whole world (all manifests share the same derived_seeds).
            if manifest_seed_order is None:
                manifest_seed_order = _load_manifest_seed_order(label_dir)

            for json_path in sorted(label_dir.glob(EPISODE_GLOB)):
                seed = _seed_from_stem(json_path)
                if seed is None:
                    continue
                rec = _read_json(json_path)
                if rec is None:
                    continue
                _accumulate_episode(acc, seed, rec)
                seen_seeds.add(seed)

            if acc.n_present < expected:
                print(
                    f"warning: {label} has {acc.n_present} episodes (expected {expected})",
                    file=sys.stderr,
                )

        wallclocks, wallclock_from_subtree = _load_wallclocks(results_dir, world_stem, label)

        summaries.append(
            _finalize_summary(
                label=label,
                display=display,
                family=family,
                acc=acc,
                wallclocks=wallclocks,
                wallclock_from_subtree=wallclock_from_subtree,
            )
        )

    if manifest_seed_order is not None:
        seed_order = manifest_seed_order
        from_manifest = True
    else:
        seed_order = tuple(sorted(seen_seeds))
        from_manifest = False

    return WorldResults(
        summaries=tuple(summaries),
        seed_order=seed_order,
        manifest_seed_order=from_manifest,
    )


# --- Summary CSV (AC3) ------------------------------------------------------

SUMMARY_CSV_COLUMNS = (
    "label",
    "display",
    "family",
    "n_present",
    "n_success",
    "n_crash",
    "n_timeout",
    "n_planner_error",
    "failure_rate",
    "median_time",
    "mean_time",
)


def write_summary_csv(summaries: tuple[AlgoSummary, ...] | list[AlgoSummary], out_path: str | Path) -> None:
    """Write the per-algorithm tally as a CSV with SUMMARY_CSV_COLUMNS.

    One row per algorithm, in CANONICAL order. NaN floats (no episodes / no
    successes) are written as the literal "nan" by csv, which is the documented
    "no data" marker.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SUMMARY_CSV_COLUMNS)
        for summary in summaries:
            writer.writerow(
                [
                    summary.label,
                    summary.display,
                    summary.family,
                    summary.n_present,
                    summary.n_success,
                    summary.n_crash,
                    summary.n_timeout,
                    summary.n_planner_error,
                    summary.failure_rate,
                    summary.median_time,
                    summary.mean_time,
                ]
            )


# --- Chart stubs (filled by T2/T3) ------------------------------------------

# Each chart function takes the loaded WorldResults, the pyplot module, and the
# absolute output dir, and writes exactly one `<chart>.png`. T1 ships them as
# stubs so the dispatch seam is testable now; T2/T3 drop in the real bodies
# without touching the registry or main().

def _chart_a1(results: WorldResults, plt, out_dir: Path) -> Path:
    """A1 — (T2) success/failure outcome bar chart per algorithm."""
    raise NotImplementedError("chart a1 is implemented in T2")


def _chart_a3(results: WorldResults, plt, out_dir: Path) -> Path:
    """A3 — (T2) time-to-goal distribution per algorithm."""
    raise NotImplementedError("chart a3 is implemented in T2")


def _chart_a4(results: WorldResults, plt, out_dir: Path) -> Path:
    """A4 — (T2) path-length-vs-ideal distribution per algorithm."""
    raise NotImplementedError("chart a4 is implemented in T2")


def _chart_b1(results: WorldResults, plt, out_dir: Path) -> Path:
    """B1 — (T3) per-seed outcome heatmap (rows = algorithms, cols = seed_order)."""
    raise NotImplementedError("chart b1 is implemented in T3")


def _chart_b2(results: WorldResults, plt, out_dir: Path) -> Path:
    """B2 — (T3) failure-rate comparison across algorithms."""
    raise NotImplementedError("chart b2 is implemented in T3")


def _chart_b3(results: WorldResults, plt, out_dir: Path) -> Path:
    """B3 — (T3) wallclock-per-step comparison (from the __wallclock__ subtree)."""
    raise NotImplementedError("chart b3 is implemented in T3")


def _chart_b4(results: WorldResults, plt, out_dir: Path) -> Path:
    """B4 — (T3) speed/efficiency scatter across algorithms."""
    raise NotImplementedError("chart b4 is implemented in T3")


# Registry mapping each chart key to its renderer. Adding a real chart in T2/T3
# is a drop-in: implement the function body above; this table and main() need no
# change.
CHART_DISPATCH = {
    "a1": _chart_a1,
    "a3": _chart_a3,
    "a4": _chart_a4,
    "b1": _chart_b1,
    "b2": _chart_b2,
    "b3": _chart_b3,
    "b4": _chart_b4,
}


# --- Self-check (filled by T5) ----------------------------------------------

def run_selfcheck() -> int:
    """Run the plotter's self-check suite (TC-P*). Placeholder until T5."""
    raise SystemExit("selfcheck not yet implemented")


# --- CLI --------------------------------------------------------------------

@dataclass(frozen=True)
class PlotArgs:
    """Parsed CLI arguments — frozen so accidental mutation is impossible."""

    world: str
    results_dir: str
    replan_k: int
    charts: tuple[str, ...]
    out_dir: str | None
    selfcheck: bool


def _parse_charts(raw: str) -> tuple[str, ...]:
    """Parse a comma list of chart keys, validating each against CHART_KEYS.

    Raises ValueError (surfaced as an argparse error -> exit 2) on an unknown
    key. Order is preserved and duplicates are dropped, so `--charts a1,a1,b2`
    renders a1 then b2 once each.
    """
    keys = [token.strip() for token in raw.split(",") if token.strip()]
    if not keys:
        raise ValueError("no chart keys given")
    unknown = [key for key in keys if key not in CHART_KEYS]
    if unknown:
        raise ValueError(f"unknown chart key(s): {', '.join(unknown)} (valid: {', '.join(CHART_KEYS)})")
    seen: list[str] = []
    for key in keys:
        if key not in seen:
            seen.append(key)
    return tuple(seen)


def _parse_args(argv: list[str] | None) -> PlotArgs:
    parser = argparse.ArgumentParser(
        prog="runners.plot",
        description="Render the cross-algorithm comparison charts from the batch result JSONs (Phase 5).",
    )
    parser.add_argument(
        "--world",
        required=True,
        help="Path to the world YAML (e.g. arena/arena_v1.yaml); only its stem is used.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Results root to read from (default {DEFAULT_RESULTS_DIR!r}).",
    )
    parser.add_argument(
        "--replan-k",
        type=int,
        default=DEFAULT_REPLAN_K,
        help=f"Cadence used to build the _replan label dirs (default {DEFAULT_REPLAN_K}).",
    )
    parser.add_argument(
        "--charts",
        default=",".join(CHART_KEYS),
        help=f"Comma list of charts to render (default all: {','.join(CHART_KEYS)}).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output dir for the PNGs + summary.csv (default <results-dir>/<world_stem>/plots/).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="Run the self-check suite instead of plotting.",
    )
    ns = parser.parse_args(argv)
    try:
        charts = _parse_charts(ns.charts)
    except ValueError as exc:
        parser.error(str(exc))
    if ns.replan_k < 1:
        parser.error(f"--replan-k must be >= 1, got {ns.replan_k}")
    return PlotArgs(
        world=ns.world,
        results_dir=ns.results_dir,
        replan_k=int(ns.replan_k),
        charts=charts,
        out_dir=ns.out_dir,
        selfcheck=bool(ns.selfcheck),
    )


def _resolve_out_dir(args: PlotArgs, world_stem: str) -> Path:
    """Out-dir is the CLI override, else `<results-dir>/<world_stem>/plots/` (AC14)."""
    if args.out_dir is not None:
        return Path(args.out_dir).resolve()
    return Path(args.results_dir).resolve() / world_stem / PLOTS_DIR_NAME


def main(argv: list[str] | None = None) -> int:
    """Render the requested charts + summary.csv. See module docstring for CLI semantics."""
    args = _parse_args(argv)

    if args.selfcheck:
        # Placeholder until T5; run_selfcheck raises SystemExit with the message.
        return run_selfcheck()

    world_stem = Path(args.world).stem
    out_dir = _resolve_out_dir(args, world_stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt = ensure_matplotlib()

    results = load_world_results(
        args.results_dir,
        world_stem,
        replan_k=args.replan_k,
        expected=DEFAULT_EXPECTED_SEEDS,
    )

    # "No readable data at all" => every algorithm came back empty. There is
    # nothing to plot or summarize, so exit non-zero with a clear message (AC11).
    if all(summary.n_present == 0 for summary in results.summaries):
        print(
            f"error: nothing to plot - no readable episode JSONs under "
            f"{Path(args.results_dir).resolve() / world_stem}",
            file=sys.stderr,
        )
        return 1

    write_summary_csv(results.summaries, out_dir / SUMMARY_CSV_NAME)
    print(f"wrote {out_dir / SUMMARY_CSV_NAME}")

    for key in args.charts:
        render = CHART_DISPATCH[key]
        png_path = render(results, plt, out_dir)
        print(f"wrote {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
