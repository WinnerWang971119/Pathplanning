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


# --- Chart helpers (shared by the chart functions) --------------------------

# Fixed colors for the four outcome buckets, used by the A3 stacked-bar chart
# (and any later chart that colors by outcome). Keys match OUTCOMES exactly.
OUTCOME_COLORS = {
    "success": "#2ca02c",        # green
    "crash": "#d62728",          # red
    "timeout": "#ff7f0e",        # orange
    "planner_error": "#7f7f7f",  # grey
}

# Human-readable legend labels for the outcome buckets (A3 legend).
OUTCOME_DISPLAY = {
    "success": "success",
    "crash": "crash",
    "timeout": "timeout",
    "planner_error": "planner error",
}

# Marker shapes for the A1 centroid markers (one shape per statistic).
A1_MEAN_MARKER = "*"             # star = mean time
A1_MEDIAN_MARKER = "D"           # diamond = median time


def _algorithm_color_map(summaries, plt) -> dict[str, tuple]:
    """Map each algorithm's results label -> a stable RGBA color.

    Single source of truth for per-algorithm coloring across the A-charts: the
    same label always gets the same color regardless of which subset of
    algorithms a chart draws. Colors are sampled from matplotlib's `tab20`
    qualitative colormap in CANONICAL order (the order the loader produced
    `summaries`), which keeps the 11 algorithms visually distinct and groups
    neighbours (the families are listed contiguously in CANONICAL).
    """
    cmap = plt.get_cmap("tab20")
    color_map: dict[str, tuple] = {}
    for index, summary in enumerate(summaries):
        # tab20 has 20 discrete entries; modulo keeps it safe if the canonical
        # set ever grows past 20.
        color_map[summary.label] = cmap(index % cmap.N)
    return color_map


# --- Chart stubs (filled by T2/T3) ------------------------------------------

# Each chart function takes the loaded WorldResults, the pyplot module, and the
# absolute output dir, and writes exactly one `<chart>.png`. T1 ships them as
# stubs so the dispatch seam is testable now; T2/T3 drop in the real bodies
# without touching the registry or main().

def _chart_a1(results: WorldResults, plt, out_dir: Path) -> Path:
    """A1 — headline time-to-goal vs failure-rate scatter (AC4 / the Mission deliverable).

    X = successful per-seed time-to-goal (sim seconds), Y = the algorithm's
    failure_rate. Each algorithm's successes are scattered as dots at its
    failure_rate row (one color per algorithm), plus two larger edge-outlined
    centroid markers in the same color: a star at the MEAN success time and a
    diamond at the MEDIAN. A 0-success algorithm has no dots and NaN mean/median;
    it is represented by an annotation at its failure_rate row and never raises.
    "Down-left wins" (low time, low failure).
    """
    color_map = _algorithm_color_map(results.summaries, plt)

    fig, ax = plt.subplots(figsize=(11, 7))

    # A small, deterministic vertical jitter (no RNG) so dots that share a Y row
    # do not perfectly overlap; scaled tiny relative to the 0..1 failure axis.
    jitter_span = 0.012

    x_values_seen: list[float] = []
    for index, summary in enumerate(results.summaries):
        color = color_map[summary.label]
        failure_rate = summary.failure_rate
        # An all-empty algorithm (no episodes present) has a NaN failure_rate.
        # Pin it to the top row (failure_rate == 1.0) for the annotation so it is
        # still represented without polluting the numeric axis.
        row_y = failure_rate if failure_rate == failure_rate else 1.0  # NaN check

        if summary.times:
            n_times = len(summary.times)
            for dot_index, time_value in enumerate(summary.times):
                # Deterministic triangle-wave jitter in [-jitter_span, jitter_span].
                if n_times > 1:
                    frac = dot_index / (n_times - 1)
                else:
                    frac = 0.5
                jitter = (2.0 * frac - 1.0) * jitter_span
                ax.scatter(
                    time_value,
                    row_y + jitter,
                    color=color,
                    alpha=0.45,
                    s=28,
                    edgecolors="none",
                    zorder=2,
                )
                x_values_seen.append(time_value)

            # Centroid markers: mean (star) and median (diamond).
            ax.scatter(
                summary.mean_time,
                row_y,
                color=color,
                marker=A1_MEAN_MARKER,
                s=320,
                edgecolors="black",
                linewidths=1.1,
                zorder=4,
            )
            ax.scatter(
                summary.median_time,
                row_y,
                color=color,
                marker=A1_MEDIAN_MARKER,
                s=150,
                edgecolors="black",
                linewidths=1.1,
                zorder=4,
            )
            x_values_seen.append(summary.mean_time)
            x_values_seen.append(summary.median_time)
        else:
            # Zero successes: no dots, no finite centroid. Represent the algorithm
            # with an annotation at its failure_rate row, anchored to the right
            # edge of the axes so it is always visible.
            ax.annotate(
                f"{summary.display}: 0/{summary.n_present} success",
                xy=(1.0, row_y),
                xycoords=("axes fraction", "data"),
                xytext=(-6, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=8,
                color=color,
                fontweight="bold",
                zorder=5,
            )

    ax.set_xlabel("time to goal (sim seconds)")
    ax.set_ylabel("failure rate (0 = always solves, 1 = always fails)")
    ax.set_title("A1 - time-to-goal vs failure rate (down-left wins)")
    ax.set_ylim(-0.05, 1.08)
    if x_values_seen:
        x_lo = min(x_values_seen)
        x_hi = max(x_values_seen)
        pad = max(2.0, 0.05 * (x_hi - x_lo))
        ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.grid(True, linestyle=":", alpha=0.4, zorder=0)

    # "Down-left wins" guidance annotation in the lower-left corner.
    ax.annotate(
        "down-left wins\n(fast + reliable)",
        xy=(0.02, 0.04),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=9,
        style="italic",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#999999", alpha=0.85),
    )

    # Side legend: color -> algorithm display name, placed outside the axes.
    algo_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color_map[summary.label],
            markeredgecolor="none",
            markersize=8,
            label=summary.display,
        )
        for summary in results.summaries
    ]
    algo_legend = ax.legend(
        handles=algo_handles,
        title="algorithm",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        title_fontsize=9,
        borderaxespad=0.0,
    )
    ax.add_artist(algo_legend)

    # Shape legend: explains the mean vs median centroid markers.
    shape_handles = [
        plt.Line2D(
            [0], [0],
            marker=A1_MEAN_MARKER,
            linestyle="none",
            markerfacecolor="#cccccc",
            markeredgecolor="black",
            markersize=13,
            label="mean time",
        ),
        plt.Line2D(
            [0], [0],
            marker=A1_MEDIAN_MARKER,
            linestyle="none",
            markerfacecolor="#cccccc",
            markeredgecolor="black",
            markersize=9,
            label="median time",
        ),
    ]
    ax.legend(
        handles=shape_handles,
        title="centroid",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        fontsize=8,
        title_fontsize=9,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    out_path = out_dir / "a1_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _chart_a3(results: WorldResults, plt, out_dir: Path) -> Path:
    """A3 — per-algorithm failure-breakdown stacked bars (AC5).

    One stacked bar per algorithm (CANONICAL order). Segments are the COUNTS of
    success / crash / timeout / planner_error, summing to n_present, with fixed
    per-outcome colors and an outcome legend. An algorithm whose n_present
    differs from the expected 50 (partial data) is annotated above its bar.
    """
    summaries = results.summaries
    n_algos = len(summaries)
    x_positions = list(range(n_algos))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Per-outcome count series, one list aligned to x_positions.
    counts_by_outcome = {
        "success": [s.n_success for s in summaries],
        "crash": [s.n_crash for s in summaries],
        "timeout": [s.n_timeout for s in summaries],
        "planner_error": [s.n_planner_error for s in summaries],
    }

    # Running bottom for the stack, accumulated outcome by outcome.
    bottoms = [0.0] * n_algos
    for outcome in OUTCOMES:
        heights = counts_by_outcome[outcome]
        ax.bar(
            x_positions,
            heights,
            bottom=bottoms,
            color=OUTCOME_COLORS[outcome],
            label=OUTCOME_DISPLAY[outcome],
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        bottoms = [base + height for base, height in zip(bottoms, heights)]

    # Annotate any algorithm whose present-count differs from the expected 50.
    for x_pos, summary in zip(x_positions, summaries):
        if summary.n_present != DEFAULT_EXPECTED_SEEDS:
            ax.annotate(
                f"n={summary.n_present}",
                xy=(x_pos, summary.n_present),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
                fontweight="bold",
                zorder=3,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [summary.display for summary in summaries],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("episode count")
    ax.set_title(f"A3 - outcome breakdown per algorithm (expected {DEFAULT_EXPECTED_SEEDS} seeds)")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(title="outcome", loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, title_fontsize=9)

    fig.tight_layout()
    out_path = out_dir / "a3_failure_bars.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _chart_a4(results: WorldResults, plt, out_dir: Path) -> Path:
    """A4 — time-to-goal box/violin per algorithm (AC6).

    One box per algorithm showing the distribution of its SUCCESSFUL times,
    sorted by median success time ascending. An algorithm with 0 successes is
    placed last and annotated "no success"; one with a single success cannot form
    a box, so its lone point is scattered and annotated rather than boxed. Never
    raises on the degenerate cases.
    """
    color_map = _algorithm_color_map(results.summaries, plt)

    # Order: algorithms WITH >=1 success first, by ascending median time; the
    # zero-success algorithms go last in CANONICAL order. NaN medians (no
    # success) sort last via the (has_success, median) key.
    def _sort_key(summary):
        has_success = summary.n_success > 0
        # For no-success rows median_time is NaN; give them +inf so they trail.
        median = summary.median_time if has_success else float("inf")
        return (0 if has_success else 1, median)

    ordered = sorted(results.summaries, key=_sort_key)

    fig, ax = plt.subplots(figsize=(12, 7))

    positions = list(range(1, len(ordered) + 1))
    box_data: list[list[float]] = []
    box_positions: list[int] = []

    for position, summary in zip(positions, ordered):
        times = summary.times
        color = color_map[summary.label]
        if len(times) >= 2:
            box_data.append(list(times))
            box_positions.append(position)
        elif len(times) == 1:
            # Single success: a box is undefined, so plot the lone point and label it.
            ax.scatter(
                [position],
                [times[0]],
                color=color,
                s=40,
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )
            ax.annotate(
                "n=1",
                xy=(position, times[0]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                zorder=5,
            )
        else:
            # Zero successes: nothing to plot; annotate at the bottom of the axes.
            ax.annotate(
                "no success",
                xy=(position, 0.02),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
                color="#999999",
                fontweight="bold",
                zorder=5,
            )

    if box_data:
        boxes = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.6,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=3, markerfacecolor="#555555", markeredgecolor="none", alpha=0.5),
            medianprops=dict(color="black", linewidth=1.4),
        )
        # Tint each box with its algorithm color (box_positions aligns to the
        # subset of `ordered` that produced a box, in the same iteration order).
        boxed_summaries = [summary for summary in ordered if len(summary.times) >= 2]
        for patch, summary in zip(boxes["boxes"], boxed_summaries):
            patch.set_facecolor(color_map[summary.label])
            patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [summary.display for summary in ordered],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("time to goal (sim seconds)")
    ax.set_title("A4 - time-to-goal distribution per algorithm (sorted by median, fastest left)")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out_path = out_dir / "a4_time_box.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _success_times_by_seed(summary: AlgoSummary) -> dict[int, float]:
    """Reconstruct the per-seed success time_to_goal for one algorithm.

    The loader exposes `per_seed` (seed -> outcome) and a flat `times` tuple of
    success times, but no direct seed -> time map. `times` is appended in the
    SAME iteration order the loader scanned files — `sorted(label_dir.glob(...))`
    — which for `<seed>.json` files in one dir is lexicographic by filename. So
    re-sorting this algorithm's success seeds by their `"<seed>.json"` filename
    and zipping with `times` recovers the original mapping exactly.

    On any length mismatch (defensive — should not happen with loader output) the
    shorter of the two is zipped, so a malformed summary degrades to a partial
    map rather than raising.
    """
    success_seeds = [seed for seed, outcome in summary.per_seed.items() if outcome == "success"]
    ordered_seeds = sorted(success_seeds, key=lambda seed: f"{seed}.json")
    return {seed: time for seed, time in zip(ordered_seeds, summary.times)}


# Failure outcomes overlaid as flat categorical cells in the B1 heatmap (success
# is the continuous-cmap layer, so it is excluded here).
_B1_FAILURE_OUTCOMES = ("crash", "timeout", "planner_error")


def _chart_b1(results: WorldResults, plt, out_dir: Path) -> Path:
    """B1 — seed-difficulty heatmap: rows = algorithms (CANONICAL order), columns = the shared seed stream (AC7).

    Every row aligns to the same `results.seed_order` column order (a manifest's
    `derived_seeds`, else sorted stems), so reading down a column exposes
    universally-hard seeds. A SUCCESS cell is shaded by its time_to_goal on a
    continuous viridis colormap (colorbar "time to goal (s)"); a FAILURE cell is a
    flat categorical color per type (crash / timeout / planner_error, reusing
    OUTCOME_COLORS for parity with A3); an absent cell (no entry for that seed)
    keeps a neutral background. Never raises on missing seeds or 0-success rows.
    """
    import numpy as np

    summaries = results.summaries
    seed_order = results.seed_order
    n_rows = len(summaries)
    n_cols = len(seed_order)

    # Column index for each seed in the shared stream (first occurrence wins if a
    # manifest ever repeated a seed).
    col_of_seed: dict[int, int] = {}
    for col, seed in enumerate(seed_order):
        if seed not in col_of_seed:
            col_of_seed[seed] = col

    # Continuous layer: success times (NaN everywhere else so imshow renders the
    # bad/NaN color for non-success and absent cells).
    success_matrix = np.full((n_rows, max(n_cols, 1)), np.nan, dtype=float)
    # Categorical overlay: list of (row, col, color) for each failure cell.
    failure_cells: list[tuple[int, int, str]] = []

    for row, summary in enumerate(summaries):
        seed_times = _success_times_by_seed(summary)
        for seed, outcome in summary.per_seed.items():
            col = col_of_seed.get(seed)
            if col is None:
                # Seed not in the shared column order (e.g. a stray stem absent
                # from the manifest stream); skip rather than widen the matrix.
                continue
            if outcome == "success":
                time_value = seed_times.get(seed)
                if time_value is not None:
                    success_matrix[row, col] = time_value
            elif outcome in _B1_FAILURE_OUTCOMES:
                failure_cells.append((row, col, OUTCOME_COLORS[outcome]))

    fig, ax = plt.subplots(figsize=(max(12.0, 0.22 * max(n_cols, 1) + 4.0), 8))

    neutral_bg = "#eaeaea"
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(neutral_bg)   # NaN cells (non-success / absent) render neutral

    # Color limits from the finite success times only; guard the all-NaN case.
    finite_times = success_matrix[np.isfinite(success_matrix)]
    if finite_times.size > 0:
        vmin = float(finite_times.min())
        vmax = float(finite_times.max())
        if vmin == vmax:
            vmax = vmin + 1.0   # avoid a degenerate colorbar on a single time value
    else:
        vmin, vmax = 0.0, 1.0

    image = ax.imshow(
        success_matrix,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        origin="upper",
    )

    # Overlay each failure cell as a flat-colored unit rectangle. imshow centres
    # cell (row, col) on integer coords, so the patch spans [col-0.5, col+0.5].
    from matplotlib.patches import Rectangle

    for row, col, color in failure_cells:
        ax.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                1.0,
                1.0,
                facecolor=color,
                edgecolor="none",
                zorder=3,
            )
        )

    # Colorbar for the success layer.
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.12)
    colorbar.set_label("time to goal (s)")

    # Y ticks = algorithm display names, one per row.
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([summary.display for summary in summaries], fontsize=9)

    # X ticks = seed-column index, labelled sparsely (50 raw 64-bit seeds are far
    # too dense to print). Show at most ~12 ticks across the stream.
    if n_cols > 0:
        max_ticks = 12
        stride = max(1, n_cols // max_ticks)
        tick_cols = list(range(0, n_cols, stride))
        ax.set_xticks(tick_cols)
        ax.set_xticklabels([str(col) for col in tick_cols], fontsize=8)
    else:
        ax.set_xticks([])
    ax.set_xlabel("seed column (shared stream index)")

    order_source = "manifest derived_seeds" if results.manifest_seed_order else "sorted stems (no manifest)"
    ax.set_title(f"B1 - seed-difficulty heatmap (column order: {order_source})")

    # Legend mapping the 3 failure colors to their outcome labels, beside the bar.
    failure_handles = [
        plt.Line2D(
            [0], [0],
            marker="s",
            linestyle="none",
            markerfacecolor=OUTCOME_COLORS[outcome],
            markeredgecolor="none",
            markersize=10,
            label=OUTCOME_DISPLAY[outcome],
        )
        for outcome in _B1_FAILURE_OUTCOMES
    ]
    failure_handles.append(
        plt.Line2D(
            [0], [0],
            marker="s",
            linestyle="none",
            markerfacecolor=neutral_bg,
            markeredgecolor="#999999",
            markersize=10,
            label="absent (no entry)",
        )
    )
    ax.legend(
        handles=failure_handles,
        title="failure / absent",
        loc="upper left",
        bbox_to_anchor=(1.14, 1.0),
        fontsize=8,
        title_fontsize=9,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    out_path = out_dir / "b1_seed_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _chart_b2(results: WorldResults, plt, out_dir: Path) -> Path:
    """B2 — path-length box per algorithm over successful episodes, sorted by median ascending (AC8).

    One box per algorithm of its `path_lengths` over SUCCESSFUL episodes, sorted
    by median path length ascending. Mirrors A4's degenerate handling: 0
    successes -> annotate "no success"; exactly 1 -> scatter the lone point. A
    horizontal reference line marks the Euclidean lower bound, labelled so it is
    not read as an achievable target. Never raises on the degenerate cases.
    """
    color_map = _algorithm_color_map(results.summaries, plt)

    def _median_path(summary: AlgoSummary) -> float:
        return statistics.median(summary.path_lengths) if summary.path_lengths else float("inf")

    # Algorithms with >=1 success sort first by ascending median path length; the
    # zero-success ones trail (inf median) in CANONICAL order.
    def _sort_key(summary: AlgoSummary):
        has_path = len(summary.path_lengths) > 0
        return (0 if has_path else 1, _median_path(summary))

    ordered = sorted(results.summaries, key=_sort_key)

    fig, ax = plt.subplots(figsize=(12, 7))

    positions = list(range(1, len(ordered) + 1))
    box_data: list[list[float]] = []
    box_positions: list[int] = []

    for position, summary in zip(positions, ordered):
        path_lengths = summary.path_lengths
        color = color_map[summary.label]
        if len(path_lengths) >= 2:
            box_data.append(list(path_lengths))
            box_positions.append(position)
        elif len(path_lengths) == 1:
            ax.scatter(
                [position],
                [path_lengths[0]],
                color=color,
                s=40,
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )
            ax.annotate(
                "n=1",
                xy=(position, path_lengths[0]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                zorder=5,
            )
        else:
            ax.annotate(
                "no success",
                xy=(position, 0.02),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
                color="#999999",
                fontweight="bold",
                zorder=5,
            )

    if box_data:
        boxes = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.6,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=3, markerfacecolor="#555555", markeredgecolor="none", alpha=0.5),
            medianprops=dict(color="black", linewidth=1.4),
        )
        boxed_summaries = [summary for summary in ordered if len(summary.path_lengths) >= 2]
        for patch, summary in zip(boxes["boxes"], boxed_summaries):
            patch.set_facecolor(color_map[summary.label])
            patch.set_alpha(0.6)

    # Euclidean lower bound reference line, labelled as unreachable.
    ax.axhline(
        STRAIGHT_LINE_IDEAL_M,
        color="#444444",
        linestyle="--",
        linewidth=1.2,
        zorder=1,
    )
    ax.annotate(
        f"Euclidean lower bound (unreachable through walls) = {STRAIGHT_LINE_IDEAL_M:.2f} m",
        xy=(0.01, STRAIGHT_LINE_IDEAL_M),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#444444",
        fontweight="bold",
        zorder=5,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [summary.display for summary in ordered],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("path length (m)")
    ax.set_title("B2 - path-length distribution per algorithm (sorted by median, shortest left)")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out_path = out_dir / "b2_pathlen_box.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _chart_b3(results: WorldResults, plt, out_dir: Path) -> Path:
    """B3 — compute-cost bars: mean wallclock_per_step per algorithm, sorted ascending (AC9).

    One bar per algorithm = mean `wallclock_per_step`, sourced from the
    `wallclocks` the loader populated from the `__wallclock__` subtree. The
    figure footnote states the source AND its --jobs sensitivity: if ANY
    algorithm's samples came from that dedicated serial subtree
    (`wallclock_from_subtree` True) the footnote credits the "serial --jobs 1
    pass"; otherwise (the subtree was absent for every algorithm) it caveats that
    the numbers are from the parallel bulk pass and are perturbed by --jobs
    contention. Algorithms with no wallclock samples are annotated "no data" and
    get no bar. Never raises.
    """
    color_map = _algorithm_color_map(results.summaries, plt)

    # Mean wallclock per algorithm; None where there are no samples.
    means: list[tuple[AlgoSummary, float | None]] = []
    for summary in results.summaries:
        mean_wallclock = statistics.fmean(summary.wallclocks) if summary.wallclocks else None
        means.append((summary, mean_wallclock))

    # Sort: algorithms WITH samples first by ascending mean, no-data ones trail in
    # CANONICAL order.
    def _sort_key(item: tuple[AlgoSummary, float | None]):
        _summary, mean_wallclock = item
        if mean_wallclock is None:
            return (1, float("inf"))
        return (0, mean_wallclock)

    ordered = sorted(means, key=_sort_key)

    fig, ax = plt.subplots(figsize=(12, 7))

    positions = list(range(len(ordered)))
    for position, (summary, mean_wallclock) in zip(positions, ordered):
        if mean_wallclock is None:
            ax.annotate(
                "no data",
                xy=(position, 0.02),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
                color="#999999",
                fontweight="bold",
                zorder=5,
            )
            continue
        ax.bar(
            position,
            mean_wallclock,
            color=color_map[summary.label],
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )

    # Footnote source: serial subtree if ANY algorithm drew from it, else the
    # bulk-pass caveat. Either way the footnote names the --jobs sensitivity (AC9).
    any_subtree = any(summary.wallclock_from_subtree for summary in results.summaries)
    if any_subtree:
        footnote = (
            "wallclock from serial --jobs 1 pass (__wallclock__ subtree); "
            "wallclock_per_step is --jobs-sensitive, so these serial numbers are the headline values."
        )
    else:
        footnote = (
            "wallclock from parallel bulk pass - perturbed by --jobs contention; approximate. "
            "wallclock_per_step is --jobs-sensitive; rerun the serial __wallclock__ pass for headline numbers."
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [summary.display for summary, _ in ordered],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("mean wallclock per step (s)")
    ax.set_title("B3 - compute cost per step per algorithm (sorted by mean, cheapest left)")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.text(0.01, 0.01, footnote, fontsize=8, color="#555555", ha="left", va="bottom", style="italic")

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    out_path = out_dir / "b3_compute_bars.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _chart_b4(results: WorldResults, plt, out_dir: Path) -> Path:
    """B4 — family-contrast panels isolating the three designed experiments (AC10).

    Small-multiple subplots, each a grouped-bar comparison of failure_rate (left
    axis) and median time (right axis) for a designed contrast:
      1. A* vs Dijkstra (the heuristic question).
      2. once vs replan within each family that has both (the replanning question).
      3. reactive vs global (the reactivity question).
    NaN medians (0-success algorithms) are drawn as a 0-height median bar and
    annotated "no median (0 success)" so they read as missing, not fast. Never
    raises when an algorithm is absent (it is simply skipped from its panel).
    """
    by_label = {summary.label: summary for summary in results.summaries}

    # The panels reference algorithms by their registry name; map each to the
    # loader's actual label (which folds in --replan-k) so a CLI K other than the
    # CANONICAL placeholder still resolves. The label order in CANONICAL is the
    # same order the loader produced `summaries`, so name -> label is by position.
    name_to_label = {name: summary.label for (name, *_), summary in zip(CANONICAL, results.summaries)}

    def _summary_for(name: str) -> AlgoSummary | None:
        label = name_to_label.get(name)
        if label is None:
            return None
        return by_label.get(label)

    # Each panel is (title, [algorithm registry names in display order]).
    panels: list[tuple[str, list[str]]] = [
        (
            "heuristic: A* vs Dijkstra",
            ["a_star_once", "dijkstra_once", "a_star_replan", "dijkstra_replan"],
        ),
        (
            "replanning: once vs replan",
            [
                "a_star_once", "a_star_replan",
                "dijkstra_once", "dijkstra_replan",
                "rrt_once", "rrt_replan",
                "rrt_star_once", "rrt_star_replan",
            ],
        ),
        (
            "reactivity: reactive vs global",
            ["dwa", "apf", "d_star_lite", "a_star_replan"],
        ),
    ]

    import numpy as np

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 6.5))
    if len(panels) == 1:
        axes = [axes]

    failure_color = OUTCOME_COLORS["crash"]   # red bar = failure_rate
    median_color = "#1f77b4"                  # blue bar = median time

    for ax_left, (title, names) in zip(axes, panels):
        # Keep only the algorithms present in this world; skip absent ones.
        present = [(name, _summary_for(name)) for name in names]
        present = [(name, summary) for name, summary in present if summary is not None and summary.n_present > 0]

        ax_right = ax_left.twinx()

        if not present:
            ax_left.annotate(
                "no data",
                xy=(0.5, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=11,
                color="#999999",
                fontweight="bold",
            )
            ax_left.set_title(title, fontsize=11)
            ax_left.set_xticks([])
            continue

        indices = np.arange(len(present))
        bar_half = 0.2

        failure_rates = [
            summary.failure_rate if summary.failure_rate == summary.failure_rate else 0.0
            for _name, summary in present
        ]
        ax_left.bar(
            indices - bar_half,
            failure_rates,
            width=2 * bar_half,
            color=failure_color,
            edgecolor="white",
            linewidth=0.4,
            label="failure rate",
            zorder=2,
        )

        # Median time: NaN (0-success) -> draw nothing, annotate instead.
        median_heights = []
        for offset, (_name, summary) in zip(indices, present):
            median = summary.median_time
            if median == median:   # finite
                ax_right.bar(
                    offset + bar_half,
                    median,
                    width=2 * bar_half,
                    color=median_color,
                    edgecolor="white",
                    linewidth=0.4,
                    label="median time",
                    zorder=2,
                )
                median_heights.append(median)
            else:
                ax_right.annotate(
                    "no median\n(0 success)",
                    xy=(offset + bar_half, 0.0),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#777777",
                    zorder=5,
                )

        ax_left.set_xticks(indices)
        ax_left.set_xticklabels(
            [summary.display for _name, summary in present],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        ax_left.set_ylim(0.0, 1.05)
        ax_left.set_ylabel("failure rate", color=failure_color)
        ax_left.tick_params(axis="y", labelcolor=failure_color)
        ax_right.set_ylabel("median time to goal (s)", color=median_color)
        ax_right.tick_params(axis="y", labelcolor=median_color)
        if median_heights:
            ax_right.set_ylim(0.0, 1.15 * max(median_heights))
        ax_left.set_title(title, fontsize=11)
        ax_left.set_axisbelow(True)
        ax_left.grid(True, axis="y", linestyle=":", alpha=0.3)

    # One shared legend for the two bar series (colors are identical per panel).
    legend_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=failure_color, markeredgecolor="none", markersize=10, label="failure rate (left axis)"),
        plt.Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=median_color, markeredgecolor="none", markersize=10, label="median time (right axis)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=9, frameon=True)

    fig.suptitle("B4 - family-contrast panels (the three designed experiments)", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.96))
    out_path = out_dir / "b4_family_panels.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


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


# --- Self-check (T5) --------------------------------------------------------
#
# The suite below runs TC-P1..TC-P10 against synthetic result trees built in a
# TemporaryDirectory — no irsim, no real episodes. Each TC is a plain function
# that asserts its invariants; `_run_selfcheck_suite` catches per-TC exceptions
# so one failure never aborts the rest, prints a PASS/FAIL line each, and returns
# an int exit code (0 = all passed, 1 = any failed). The fixture helper writes
# `<seed>.json` records with the 7 metric fields plus a `_manifest.json` carrying
# `derived_seeds`, and is parametrizable so each TC can inject the edge cases
# (0-success algos, short seed counts, malformed JSON, decoy files, a
# `__wallclock__` subtree). Charts are rendered through the real chart functions
# under the headless Agg backend selected by `ensure_matplotlib()`.

# The 7 metric fields run_episode writes (mirrors runners/run_episode.py:78-84).
_SELFCHECK_METRIC_KEYS = (
    "time_to_goal",
    "crashed",
    "timed_out",
    "path_length",
    "mean_speed",
    "wallclock_per_step",
    "planner_error",
)


def _make_record(
    outcome: str,
    *,
    time_to_goal: float | None = None,
    path_length: float | None = None,
    wallclock_per_step: float = 0.01,
) -> dict:
    """Build one synthetic metrics record for the given outcome.

    `outcome` is one of OUTCOMES. A "success" record carries a non-null
    `time_to_goal` (and a `path_length`); every other outcome sets its flag (or
    `planner_error` string) and leaves `time_to_goal` null, matching exactly what
    run_episode writes. `path_length` defaults to a value derived from the time so
    the B2 boxes have spread without the caller spelling it out.
    """
    record = {
        "time_to_goal": None,
        "crashed": False,
        "timed_out": False,
        "path_length": 0.0,
        "mean_speed": 0.0,
        "wallclock_per_step": float(wallclock_per_step),
        "planner_error": None,
    }
    if outcome == "success":
        if time_to_goal is None:
            raise ValueError("a success record needs a time_to_goal")
        path = path_length if path_length is not None else (STRAIGHT_LINE_IDEAL_M + time_to_goal)
        record["time_to_goal"] = float(time_to_goal)
        record["path_length"] = float(path)
        record["mean_speed"] = float(path) / float(time_to_goal) if time_to_goal else 0.0
    elif outcome == "crash":
        record["crashed"] = True
    elif outcome == "timeout":
        record["timed_out"] = True
    elif outcome == "planner_error":
        record["planner_error"] = "synthetic planner failure"
    else:
        raise ValueError(f"unknown outcome {outcome!r}")
    return record


def _outcomes_for_algo(
    *,
    n_success: int,
    n_crash: int = 0,
    n_timeout: int = 0,
    n_planner_error: int = 0,
) -> list[str]:
    """Expand per-outcome counts into the per-seed outcome list (success first)."""
    return (
        ["success"] * n_success
        + ["crash"] * n_crash
        + ["timeout"] * n_timeout
        + ["planner_error"] * n_planner_error
    )


def _write_algo_tree(
    *,
    results_root: Path,
    world_stem: str,
    label: str,
    seeds: list[int],
    outcomes: list[str],
    derived_seeds: list[int] | None,
    success_times: list[float] | None = None,
    wallclock_per_step: float = 0.01,
    write_manifest: bool = True,
    decoy_files: bool = False,
    malformed_seed: int | None = None,
) -> None:
    """Write one algorithm's synthetic result dir under `<results_root>/<world_stem>/<label>/`.

    Writes one `<seed>.json` per (seed, outcome) pair (so `len(seeds)` ==
    `len(outcomes)`), an optional `_manifest.json` carrying `derived_seeds`, and
    optional decoy non-episode files (`notes.txt`, a stray `_manifest.json` is
    always skipped by the loader regardless). `success_times` supplies the
    `time_to_goal` for the success records in order; absent, a deterministic ramp
    is used. `malformed_seed`, if set, overwrites that seed's JSON with invalid
    bytes (to exercise the loader's skip-with-warning path).
    """
    if len(seeds) != len(outcomes):
        raise ValueError("seeds and outcomes must be the same length")

    label_dir = results_root / world_stem / label
    label_dir.mkdir(parents=True, exist_ok=True)

    success_iter = iter(success_times) if success_times is not None else None
    ramp = 10.0
    for seed, outcome in zip(seeds, outcomes):
        if outcome == "success":
            if success_iter is not None:
                time_value = next(success_iter)
            else:
                time_value = ramp
                ramp += 5.0
            record = _make_record(
                "success",
                time_to_goal=time_value,
                wallclock_per_step=wallclock_per_step,
            )
        else:
            record = _make_record(outcome, wallclock_per_step=wallclock_per_step)
        (label_dir / f"{seed}.json").write_text(
            json.dumps(record, sort_keys=True), encoding="utf-8"
        )

    if malformed_seed is not None:
        (label_dir / f"{malformed_seed}.json").write_text(
            "{ this is not valid json ", encoding="utf-8"
        )

    if write_manifest and derived_seeds is not None:
        manifest = {
            "master_seed": 20260605,
            "num_seeds": len(derived_seeds),
            "derived_seeds": list(derived_seeds),
        }
        (label_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )

    if decoy_files:
        (label_dir / "notes.txt").write_text("scratch notes, not an episode\n", encoding="utf-8")


def _write_wallclock_subtree(
    *,
    results_root: Path,
    world_stem: str,
    label: str,
    seeds: list[int],
    wallclock_per_step: float,
) -> None:
    """Write the B3 `__wallclock__` subtree at the EXACT path the loader reads.

    Path: `<results_root>/__wallclock__/<world_stem>/<label>/<seed>.json`. Each
    record is a success with the given sentinel `wallclock_per_step` so TC-P9 can
    prove the loader sources B3's wallclock from this subtree, not the bulk dir.
    """
    subtree_dir = results_root / WALLCLOCK_SUBTREE / world_stem / label
    subtree_dir.mkdir(parents=True, exist_ok=True)
    for index, seed in enumerate(seeds):
        record = _make_record(
            "success",
            time_to_goal=10.0 + index,
            wallclock_per_step=wallclock_per_step,
        )
        (subtree_dir / f"{seed}.json").write_text(
            json.dumps(record, sort_keys=True), encoding="utf-8"
        )


def _build_full_fixture(
    results_root: Path,
    world_stem: str,
    *,
    seeds: list[int],
    replan_k: int = DEFAULT_REPLAN_K,
) -> None:
    """Write an 11-label fixture covering every CANONICAL algorithm.

    Used by the chart-smoke and alignment TCs. Deliberately includes the two edge
    cases the charts must survive: a 0-success algorithm (every seed crashes) and
    a <50-seed algorithm (only a 12-seed prefix present). Every other algorithm
    gets a mix of success / crash / timeout / planner_error so the stacked bars,
    boxes, and heatmap all have content. `derived_seeds` is the full `seeds` list
    so every row aligns to the same column order.
    """
    # Import the label deriver lazily (mirrors load_world_results). This pulls
    # planners; the selfcheck is allowed to do so once it is actually running.
    from planners import algorithm_label

    n_seeds = len(seeds)
    for index, (name, default_k, _family, _display) in enumerate(CANONICAL):
        effective_k = replan_k if default_k is not None else None
        label = algorithm_label(name, effective_k)

        if name == "dwa":
            # 0-success algorithm: every present seed crashes (the experimental
            # signal the charts must render without dropping the algorithm).
            outcomes = ["crash"] * n_seeds
            algo_seeds = list(seeds)
            derived = list(seeds)
        elif name == "apf":
            # <50-seed algorithm: only a 12-seed prefix is present on disk. Its
            # manifest still carries the full derived_seeds stream.
            short = min(12, n_seeds)
            algo_seeds = list(seeds[:short])
            outcomes = _outcomes_for_algo(
                n_success=max(short - 3, 0),
                n_crash=min(2, short),
                n_timeout=min(1, max(short - 2, 0)),
            )
            # Pad/truncate so outcomes lines up with algo_seeds exactly.
            outcomes = (outcomes + ["crash"] * short)[:short]
            derived = list(seeds)
        else:
            # A varied mix: most succeed, a couple fail across the three failure
            # buckets, so every chart layer has data. Rotate the failing seeds by
            # the algorithm index so the heatmap columns are not all identical.
            outcomes = ["success"] * n_seeds
            algo_seeds = list(seeds)
            if n_seeds >= 3:
                outcomes[(index) % n_seeds] = "crash"
                outcomes[(index + 1) % n_seeds] = "timeout"
                outcomes[(index + 2) % n_seeds] = "planner_error"
            derived = list(seeds)

        _write_algo_tree(
            results_root=results_root,
            world_stem=world_stem,
            label=label,
            seeds=algo_seeds,
            outcomes=outcomes,
            derived_seeds=derived,
        )


# --- The test cases (TC-P1 .. TC-P10) ---------------------------------------
#
# Each returns a short detail string on success and raises AssertionError (with a
# message) on failure. `_run_selfcheck_suite` turns that into the PASS/FAIL line.


def _tc_p1_classify_precedence(_tmp: Path) -> str:
    """TC-P1: classify_outcome precedence + the defensive all-false branch."""
    assert classify_outcome(_make_record("success", time_to_goal=12.0)) == "success", \
        "a clean success record must classify success"
    assert classify_outcome(_make_record("crash")) == "crash", "crash flag must classify crash"
    assert classify_outcome(_make_record("timeout")) == "timeout", "timeout flag must classify timeout"
    assert classify_outcome(_make_record("planner_error")) == "planner_error", \
        "planner_error must classify planner_error"

    # planner_error set AND crashed=True -> planner_error wins (precedence).
    both = _make_record("crash")
    both["planner_error"] = "boom"
    assert classify_outcome(both) == "planner_error", \
        "planner_error must take precedence over crashed"

    # planner_error set AND time_to_goal non-null -> still planner_error.
    err_with_time = _make_record("success", time_to_goal=5.0)
    err_with_time["planner_error"] = "boom"
    assert classify_outcome(err_with_time) == "planner_error", \
        "planner_error must take precedence over a non-null time_to_goal"

    # All flags false / null and no time -> defensive planner_error fallthrough.
    blank = {
        "time_to_goal": None,
        "crashed": False,
        "timed_out": False,
        "path_length": 0.0,
        "mean_speed": 0.0,
        "wallclock_per_step": 0.0,
        "planner_error": None,
    }
    assert classify_outcome(blank) == "planner_error", \
        "an all-false/null record must hit the defensive planner_error branch"
    # An empty dict (every .get returns None/falsey) also takes the defensive branch.
    assert classify_outcome({}) == "planner_error", \
        "an empty record must hit the defensive planner_error branch"
    return "precedence + defensive branch correct"


def _tc_p2_loader_over_tree(tmp: Path) -> str:
    """TC-P2: loader reads numeric stems, skips _manifest.json + notes.txt, counts a short dir."""
    from planners import algorithm_label

    results_root = tmp / "tc_p2"
    world_stem = "w"
    seeds = [101, 202, 303]
    # a_star_once: 3 present episodes + decoy files + a manifest.
    _write_algo_tree(
        results_root=results_root,
        world_stem=world_stem,
        label=algorithm_label("a_star_once", None),
        seeds=seeds,
        outcomes=["success", "crash", "timeout"],
        derived_seeds=seeds,
        success_times=[10.0],
        decoy_files=True,
    )

    results = load_world_results(str(results_root), world_stem, replan_k=DEFAULT_REPLAN_K)
    by_label = {summary.label: summary for summary in results.summaries}
    a_star = by_label[algorithm_label("a_star_once", None)]

    assert a_star.n_present == 3, f"expected 3 episode files, counted {a_star.n_present}"
    # The decoy notes.txt and the _manifest.json must not have been counted.
    assert a_star.n_success == 1 and a_star.n_crash == 1 and a_star.n_timeout == 1, \
        "the 3 episodes must classify as one success/crash/timeout each (decoys skipped)"
    # The label dir physically holds notes.txt + _manifest.json beside the 3 JSONs.
    label_dir = results_root / world_stem / algorithm_label("a_star_once", None)
    assert (label_dir / "notes.txt").is_file(), "decoy notes.txt should have been written"
    assert (label_dir / MANIFEST_NAME).is_file(), "manifest should have been written"
    # Manifest seed order picked up.
    assert results.manifest_seed_order is True, "seed order must come from the manifest"
    assert results.seed_order == tuple(seeds), "seed order must equal derived_seeds"
    return "n_present=3 over numeric stems; decoys + manifest skipped"


def _tc_p3_summary_math(tmp: Path) -> str:
    """TC-P3: failure_rate / median / mean / n_success on a known 3-success + crash + timeout set."""
    from planners import algorithm_label

    results_root = tmp / "tc_p3"
    world_stem = "w"
    seeds = [1, 2, 3, 4, 5]
    outcomes = ["success", "success", "success", "crash", "timeout"]
    _write_algo_tree(
        results_root=results_root,
        world_stem=world_stem,
        label=algorithm_label("a_star_once", None),
        seeds=seeds,
        outcomes=outcomes,
        derived_seeds=seeds,
        success_times=[10.0, 20.0, 60.0],
    )

    results = load_world_results(str(results_root), world_stem, replan_k=DEFAULT_REPLAN_K)
    by_label = {summary.label: summary for summary in results.summaries}
    summary = by_label[algorithm_label("a_star_once", None)]

    assert summary.n_present == 5, f"n_present should be 5, got {summary.n_present}"
    assert summary.n_success == 3, f"n_success should be 3, got {summary.n_success}"
    assert abs(summary.failure_rate - 0.4) < 1e-9, \
        f"failure_rate should be 2/5=0.4, got {summary.failure_rate}"
    assert abs(summary.median_time - 20.0) < 1e-9, \
        f"median_time should be 20, got {summary.median_time}"
    assert abs(summary.mean_time - 30.0) < 1e-9, \
        f"mean_time should be 30, got {summary.mean_time}"
    return "failure_rate=0.4, median=20, mean=30, n_success=3"


def _tc_p4_partial_missing(tmp: Path) -> str:
    """TC-P4: a missing label dir is present in summaries with n_present 0; nothing raises."""
    from planners import algorithm_label

    results_root = tmp / "tc_p4"
    world_stem = "w"
    seeds = [7, 8, 9]
    # Only write a_star_once; every other canonical label dir is absent on disk.
    _write_algo_tree(
        results_root=results_root,
        world_stem=world_stem,
        label=algorithm_label("a_star_once", None),
        seeds=seeds,
        outcomes=["success", "success", "crash"],
        derived_seeds=seeds,
        success_times=[12.0, 18.0],
    )

    # Must NOT raise even though 10 of 11 label dirs are missing.
    results = load_world_results(str(results_root), world_stem, replan_k=DEFAULT_REPLAN_K)

    # Every canonical algorithm still appears in the summaries (count-charts need them).
    assert len(results.summaries) == len(CANONICAL), \
        f"all {len(CANONICAL)} canonical algos must appear, got {len(results.summaries)}"
    by_label = {summary.label: summary for summary in results.summaries}
    # A missing algorithm (dijkstra_once dir was never written) reports n_present 0.
    missing = by_label[algorithm_label("dijkstra_once", None)]
    assert missing.n_present == 0, \
        f"a missing label dir must report n_present 0, got {missing.n_present}"
    # And its derived stats are the NaN sentinels, not a crash.
    assert missing.failure_rate != missing.failure_rate, "missing algo failure_rate must be NaN"
    assert missing.median_time != missing.median_time, "missing algo median_time must be NaN"
    # The present one is intact.
    present = by_label[algorithm_label("a_star_once", None)]
    assert present.n_present == 3 and present.n_success == 2, \
        "the present algorithm must still load correctly alongside missing ones"
    return "missing dirs present with n_present 0; no exception in the load path"


def _tc_p5_chart_smoke(tmp: Path) -> str:
    """TC-P5: render all 7 charts from an 11-label fixture; each PNG must be non-empty."""
    plt = ensure_matplotlib()

    results_root = tmp / "tc_p5"
    world_stem = "w"
    seeds = list(range(1000, 1015))  # 15 seeds
    _build_full_fixture(results_root, world_stem, seeds=seeds)

    results = load_world_results(str(results_root), world_stem, replan_k=DEFAULT_REPLAN_K)
    # Sanity: the fixture really does contain a 0-success and a short algo.
    by_label = {s.label: s for s in results.summaries}
    from planners import algorithm_label
    dwa = by_label[algorithm_label("dwa", None)]
    apf = by_label[algorithm_label("apf", None)]
    assert dwa.n_success == 0, "fixture must include a 0-success algorithm (dwa)"
    assert apf.n_present < len(seeds), "fixture must include a <full-seed algorithm (apf)"

    out_dir = tmp / "tc_p5_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in CHART_KEYS:
        render = CHART_DISPATCH[key]
        png_path = render(results, plt, out_dir)
        assert png_path.is_file(), f"{key}: chart did not write {png_path}"
        size = png_path.stat().st_size
        assert size > 0, f"{key}: chart PNG {png_path} is empty ({size} bytes)"
    return f"all {len(CHART_KEYS)} charts rendered non-empty PNGs"


def _tc_p6_b1_alignment(tmp: Path) -> str:
    """TC-P6: B1 matrix is 11 rows x len(seed_order) cols; every row shares the seed->col index."""
    import numpy as np

    results_root = tmp / "tc_p6"
    world_stem = "w"
    seeds = list(range(2000, 2012))  # 12 seeds
    _build_full_fixture(results_root, world_stem, seeds=seeds)

    results = load_world_results(str(results_root), world_stem, replan_k=DEFAULT_REPLAN_K)

    # The seed-column order is the manifest stream.
    assert results.manifest_seed_order is True, "seed order must come from the manifest"
    seed_order = results.seed_order
    n_rows = len(results.summaries)
    n_cols = len(seed_order)
    assert n_rows == len(CANONICAL), f"expected {len(CANONICAL)} rows, got {n_rows}"
    assert n_cols == len(seeds), f"expected {len(seeds)} columns, got {n_cols}"

    # Reconstruct B1's shared seed->column map exactly as _chart_b1 does.
    col_of_seed: dict[int, int] = {}
    for col, seed in enumerate(seed_order):
        if seed not in col_of_seed:
            col_of_seed[seed] = col

    # Build the same categorical/continuous layers B1 builds, and assert that for
    # every algorithm row, each per_seed entry lands at the SAME column index that
    # the shared seed_order assigns — i.e. the alignment is row-independent.
    failure_cell_seen = False
    for row, summary in enumerate(results.summaries):
        for seed, outcome in summary.per_seed.items():
            col = col_of_seed.get(seed)
            assert col is not None, f"seed {seed} from row {row} missing from the shared column order"
            # The column index must be identical to what the manifest order dictates,
            # regardless of which algorithm row we are on.
            assert seed_order[col] == seed, \
                f"row {row} seed {seed} maps to column {col} which holds {seed_order[col]}"
            if outcome in _B1_FAILURE_OUTCOMES:
                # A failure cell carries a concrete categorical color from OUTCOME_COLORS.
                color = OUTCOME_COLORS[outcome]
                assert isinstance(color, str) and color.startswith("#"), \
                    f"failure cell for {outcome} must map to a categorical hex color"
                failure_cell_seen = True

    assert failure_cell_seen, "the fixture must contain at least one failure cell for B1"

    # Cross-check by reconstructing the success matrix shape the way B1 does.
    success_matrix = np.full((n_rows, max(n_cols, 1)), np.nan, dtype=float)
    for row, summary in enumerate(results.summaries):
        seed_times = _success_times_by_seed(summary)
        for seed, outcome in summary.per_seed.items():
            col = col_of_seed[seed]
            if outcome == "success":
                time_value = seed_times.get(seed)
                if time_value is not None:
                    success_matrix[row, col] = time_value
    assert success_matrix.shape == (n_rows, n_cols), \
        f"B1 matrix shape {success_matrix.shape} != ({n_rows}, {n_cols})"
    return f"B1 matrix {n_rows}x{n_cols}; rows share the manifest seed->column index"


def _tc_p7_malformed_and_no_data(tmp: Path) -> str:
    """TC-P7: a malformed JSON is skipped with a warning; a no-data world exits non-zero."""
    from planners import algorithm_label

    # Part A: a malformed JSON file in an otherwise-populated dir is skipped, not raised.
    results_root = tmp / "tc_p7a"
    world_stem = "w"
    seeds = [11, 22, 33]
    _write_algo_tree(
        results_root=results_root,
        world_stem=world_stem,
        label=algorithm_label("a_star_once", None),
        seeds=seeds,
        outcomes=["success", "success", "crash"],
        derived_seeds=seeds,
        success_times=[10.0, 20.0],
        malformed_seed=44,  # writes 44.json with invalid bytes
    )
    # Must not raise; the malformed file is skipped, the 3 valid ones load.
    results = load_world_results(str(results_root), world_stem, replan_k=DEFAULT_REPLAN_K)
    by_label = {s.label: s for s in results.summaries}
    a_star = by_label[algorithm_label("a_star_once", None)]
    assert a_star.n_present == 3, \
        f"malformed 44.json must be skipped, expected 3 present, got {a_star.n_present}"

    # Part B: a world stem with zero readable episode files must exit non-zero with
    # the "nothing to plot" message, via main()'s code path.
    empty_root = tmp / "tc_p7b_empty"
    empty_root.mkdir(parents=True, exist_ok=True)
    try:
        code = main([
            "--world", "arena/arena_v1.yaml",      # only its stem ("arena_v1") matters
            "--results-dir", str(empty_root),
            "--charts", "a1",
        ])
    except SystemExit as exc:
        code = exc.code
    assert code is not None and code != 0, \
        f"a no-data world must exit non-zero, got {code!r}"
    return "malformed JSON skipped; no-data world exits non-zero"


def _tc_p8_matplotlib_guard(_tmp: Path) -> str:
    """TC-P8: ensure_matplotlib() with find_spec patched to None exits non-zero + prints the hint."""
    import importlib.util
    import io
    import contextlib

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        # Narrow: only "matplotlib" disappears; every other name resolves for real.
        if name == "matplotlib":
            return None
        return real_find_spec(name, *args, **kwargs)

    stderr_capture = io.StringIO()
    importlib.util.find_spec = fake_find_spec
    try:
        with contextlib.redirect_stderr(stderr_capture):
            try:
                ensure_matplotlib()
            except SystemExit as exc:
                code = exc.code
            else:
                raise AssertionError("ensure_matplotlib must raise SystemExit when matplotlib is absent")
    finally:
        importlib.util.find_spec = real_find_spec

    assert code is not None and code != 0, f"the guard must exit non-zero, got {code!r}"
    message = stderr_capture.getvalue()
    assert "pip install -r requirements.txt" in message, \
        f"the guard must print the pip hint, stderr was: {message!r}"
    return "find_spec=None -> SystemExit non-zero + pip hint printed"


def _tc_p9_wallclock_source(tmp: Path) -> str:
    """TC-P9: B3 sources wallclock from the __wallclock__ subtree, then falls back when it is gone."""
    from planners import algorithm_label

    results_root = tmp / "tc_p9"
    world_stem = "w"
    seeds = [1, 2, 3]
    label = algorithm_label("a_star_once", None)

    # Bulk dir with a DIFFERENT sentinel wallclock (0.001) so we can prove B3 does
    # NOT read it when the subtree exists.
    _write_algo_tree(
        results_root=results_root,
        world_stem=world_stem,
        label=label,
        seeds=seeds,
        outcomes=["success", "success", "success"],
        derived_seeds=seeds,
        success_times=[10.0, 20.0, 30.0],
        wallclock_per_step=0.001,
    )
    # The __wallclock__ subtree with the sentinel B3 must surface (0.999).
    _write_wallclock_subtree(
        results_root=results_root,
        world_stem=world_stem,
        label=label,
        seeds=seeds,
        wallclock_per_step=0.999,
    )

    results = load_world_results(str(results_root), world_stem, replan_k=DEFAULT_REPLAN_K)
    summary = {s.label: s for s in results.summaries}[label]
    assert summary.wallclock_from_subtree is True, \
        "wallclock_from_subtree must be True when the subtree exists"
    assert len(summary.wallclocks) == len(seeds), \
        f"expected {len(seeds)} subtree wallclock samples, got {len(summary.wallclocks)}"
    assert all(abs(value - 0.999) < 1e-9 for value in summary.wallclocks), \
        f"B3 wallclock must reflect the subtree sentinel 0.999, got {summary.wallclocks}"

    # Now delete the subtree and reload: the loader must fall back (empty + False).
    import shutil
    shutil.rmtree(results_root / WALLCLOCK_SUBTREE)
    results2 = load_world_results(str(results_root), world_stem, replan_k=DEFAULT_REPLAN_K)
    summary2 = {s.label: s for s in results2.summaries}[label]
    assert summary2.wallclock_from_subtree is False, \
        "wallclock_from_subtree must be False after the subtree is removed"
    assert summary2.wallclocks == (), \
        f"fallback must leave wallclocks empty, got {summary2.wallclocks}"

    # B3 must footnote the caveat in the fallback case. The footnote text is built
    # from `any(... wallclock_from_subtree)`; with all False, it is the bulk caveat.
    any_subtree = any(s.wallclock_from_subtree for s in results2.summaries)
    assert any_subtree is False, "no algorithm should claim the subtree after deletion"
    return "subtree sentinel 0.999 used; fallback to empty + caveat after removal"


def _tc_p10_run_all_canonical(_tmp: Path) -> str:
    """TC-P10: run_all.canonical_planner_set + build_experiment_cmd derivation (pure, no subprocess)."""
    from runners.run_all import canonical_planner_set, build_experiment_cmd

    planners = canonical_planner_set()
    assert len(planners) == 11, f"canonical_planner_set must return 11 tuples, got {len(planners)}"

    replan_families = {"a_star_replan", "dijkstra_replan", "rrt_replan", "rrt_star_replan"}
    seen_replan = set()
    for algorithm, replan_k, label in planners:
        if algorithm in replan_families:
            seen_replan.add(algorithm)
            assert replan_k == 5, f"{algorithm} must carry replan_k=5, got {replan_k}"
            assert label.endswith("_k5"), f"{algorithm} label must end _k5, got {label!r}"
        else:
            assert replan_k is None, f"{algorithm} must carry replan_k=None, got {replan_k}"
            assert not label.endswith("_k5"), f"{algorithm} label must not end _k5, got {label!r}"
    assert seen_replan == replan_families, \
        f"all 4 replan families must appear, missing {replan_families - seen_replan}"

    # build_experiment_cmd: a replan family includes --replan-k 5; a non-replan omits it.
    replan_cmd = build_experiment_cmd(
        "a_star_replan", 5, "arena/arena_v1.yaml", "results",
        master_seed=20260605, num_seeds=50, jobs=1, traffic=True, resume=False,
    )
    assert "--replan-k" in replan_cmd, "a replan family's command must include --replan-k"
    k_index = replan_cmd.index("--replan-k")
    assert replan_cmd[k_index + 1] == "5", \
        f"--replan-k value must be 5, got {replan_cmd[k_index + 1]!r}"

    once_cmd = build_experiment_cmd(
        "a_star_once", None, "arena/arena_v1.yaml", "results",
        master_seed=20260605, num_seeds=50, jobs=1, traffic=True, resume=False,
    )
    assert "--replan-k" not in once_cmd, "a non-replan family's command must omit --replan-k"
    return "11 tuples; 4 replan families k=5/_k5; command builder gates --replan-k"


# Ordered registry of the 10 test cases. Each entry is (id, callable).
_SELFCHECK_CASES = (
    ("TC-P1", _tc_p1_classify_precedence),
    ("TC-P2", _tc_p2_loader_over_tree),
    ("TC-P3", _tc_p3_summary_math),
    ("TC-P4", _tc_p4_partial_missing),
    ("TC-P5", _tc_p5_chart_smoke),
    ("TC-P6", _tc_p6_b1_alignment),
    ("TC-P7", _tc_p7_malformed_and_no_data),
    ("TC-P8", _tc_p8_matplotlib_guard),
    ("TC-P9", _tc_p9_wallclock_source),
    ("TC-P10", _tc_p10_run_all_canonical),
)


def run_selfcheck() -> int:
    """Run the plotter's self-check suite (TC-P1..TC-P10). Return 0 if all pass, else 1.

    Builds every fixture inside a single TemporaryDirectory (each TC namespaces its
    own subdir), runs each TC in isolation so one failure never aborts the rest,
    prints a per-TC PASS/FAIL line, and ends with an "N/10 passed" summary. No
    irsim, no real episodes — synthetic JSON trees only. Charts are rendered
    through the real chart functions under the headless Agg backend.
    """
    import tempfile

    n_passed = 0
    n_total = len(_SELFCHECK_CASES)

    with tempfile.TemporaryDirectory(prefix="plot_selfcheck_") as tmp_name:
        tmp = Path(tmp_name)
        for tc_id, tc_func in _SELFCHECK_CASES:
            try:
                detail = tc_func(tmp)
            except Exception as exc:  # noqa: BLE001 — one TC must not abort the suite
                # Surface the failure reason (assertion message or unexpected error)
                # on the TC's line rather than crashing the whole run.
                print(f"{tc_id}: FAIL - {type(exc).__name__}: {exc}")
            else:
                n_passed += 1
                print(f"{tc_id}: PASS - {detail}")

    print(f"selfcheck: {n_passed}/{n_total} passed")
    return 0 if n_passed == n_total else 1


# --- CLI --------------------------------------------------------------------

@dataclass(frozen=True)
class PlotArgs:
    """Parsed CLI arguments — frozen so accidental mutation is impossible."""

    world: str | None
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
        required=False,
        default=None,
        help="Path to the world YAML (e.g. arena/arena_v1.yaml); only its stem is used. "
        "Required unless --selfcheck is given.",
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
        # Selfcheck ignores --world entirely; run it before any --world
        # validation so the documented `python -m runners.plot --selfcheck`
        # command (no --world) works.
        return run_selfcheck()

    # --world is only required for the normal plotting path. Validate it here
    # (after the selfcheck gate) rather than via argparse required=True, and
    # mirror the repo's other up-front validation-failure exits (exit code 2).
    if args.world is None:
        print(
            "error: --world is required unless --selfcheck is given",
            file=sys.stderr,
        )
        return 2

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
