"""Read-only horizon-sweep plotter — charts failure rate and time-to-goal vs the prediction horizon.

Phase 7 (Predictive / motion-aware D* Lite). The horizon sweep driver
(`runners.run_horizon_sweep`) runs `d_star_lite_oracle` at several prediction
horizons, each landing in its own label dir:

    <results-dir>/<world_stem>/d_star_lite_oracle_h<steps>/<seed>.json

This module reads ONLY those JSONs (never irsim, never a sim) and turns them into
two cross-horizon line charts: **failure rate vs horizon** (the AC1 deliverable)
and **median time-to-goal vs horizon**, with the swept horizons as distinct x
positions in SECONDS (T = steps x PREDICT_DT, PREDICT_DT = 0.1, so 0/5/10/20 steps
-> 0.0/0.5/1.0/2.0 s). Reading the failure-rate line down the horizons shows
directly whether anticipating obstacle motion lowers the crash+timeout rate below
plain D* Lite (the h0 ablation).

This is a SIBLING of `runners.plot`: it reuses that module's per-label loading
machinery (`_AlgoAccumulator`, the manifest/roster/glob accumulators,
`_finalize_summary`, `_load_wallclocks`), data model (`AlgoSummary`), matplotlib
import guard (`ensure_matplotlib`), and color map (`_algorithm_color_map`). It does
NOT call `plot.load_world_results` (that loader iterates the CANONICAL-11 set, and
the predictive oracle is EXPERIMENTAL — not canonical — so it would never load the
`_h<steps>` label dirs). Instead it loads each horizon's label dir DIRECTLY, the
SAME way `load_world_results` loads one canonical entry (manifest-roster-
authoritative when present, glob fallback otherwise; warn-not-raise on a missing
or short dir so the plotter never crashes on partial sweep data).

HEADLESS (load-bearing — gotcha: importing planners pulls irsim + matplotlib):
nothing here imports irsim or matplotlib at module top. `algorithm_label` (from
`planners`) is imported LAZILY inside the function that needs it; matplotlib is
reached ONLY through `ensure_matplotlib()` inside the render functions. `runners.plot`
is itself headless at import (it defers its `planners`/irsim and matplotlib imports
into functions), so importing its loader helpers here does NOT pull irsim. The
verification `python -c "import runners.plot_horizon_sweep; ..."` must report that
neither irsim nor matplotlib is in `sys.modules` after import.

CLI:
    python -m runners.plot_horizon_sweep \
        --world <yaml_path>          # required unless --selfcheck; only its stem is used
        [--results-dir <dir>]        # ROOT (default "results"); reads <root>/<world_stem>/d_star_lite_oracle_h<H>/
        [--horizons H ...]           # default 0 5 10 20 (steps); a list of ints
        [--out-dir <dir>]            # default <results-dir>/<world_stem>/horizon_sweep_plots/
        [--selfcheck]                # run the headless self-check suite instead of plotting

Outputs:
    <out-dir>/failure_rate_vs_horizon.png   — failure rate vs horizon (AC1 deliverable)
    <out-dir>/median_time_vs_horizon.png    — median time-to-goal vs horizon
    <out-dir>/horizon_sweep_summary.csv     — one row per (algorithm, horizon)

Exit codes:
    0 — charts/summary written (or selfcheck passed)
    1 — matplotlib missing, or no readable data under any horizon label dir
    2 — argparse / CLI validation error
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the repo root importable so `from runners.plot import ...` and
# `from runners._layout import ...` resolve when this module is invoked as
# `python -m runners.plot_horizon_sweep` from any cwd. Mirrors
# runners/plot_speed_sweep.py:62-64.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the sibling plotter's per-label loading machinery + data model + matplotlib
# guard + color map. `runners.plot` is itself headless at import (it defers the
# `planners`/irsim and matplotlib imports into functions), so importing these here
# does NOT pull irsim or matplotlib at module top — the headless boundary holds.
from runners.plot import (
    DEFAULT_EXPECTED_SEEDS,
    AlgoSummary,
    _accumulate_from_glob,
    _accumulate_from_roster,
    _algorithm_color_map,
    _AlgoAccumulator,
    _load_manifest,
    _load_wallclocks,
    _manifest_episodes,
    _finalize_summary,
    ensure_matplotlib,
)
from runners._layout import episode_out_dir


# --- Module constants -------------------------------------------------------

DEFAULT_RESULTS_DIR = "results"

# The single oracle key this plotter charts. T13 may add a second predictive key
# (d_star_lite_predictive); the loader/series logic already handles a multi-key
# dict, so adding it is a one-line change to this tuple.
SWEEP_ALGORITHMS: tuple[str, ...] = ("d_star_lite_oracle",)

# Display names for the swept predictive keys (kept local so the headless plotter
# never imports `planners` at module top just to resolve a label).
ALGO_DISPLAY = {
    "d_star_lite_oracle": "D* Lite (oracle)",
    "d_star_lite_predictive": "D* Lite (predictive)",
}

# The horizon steps charted by default: 0/5/10/20 steps == 0.0/0.5/1.0/2.0 s.
DEFAULT_HORIZONS: tuple[int, ...] = (0, 5, 10, 20)

# Seconds per prediction step (matches planners._predict.PREDICT_DT / irsim
# step_time). Duplicated as a stdlib-only constant so the headless plotter never
# imports `planners._predict` (which pulls numpy + the planner stack) just to read
# one float. AC: x positions == steps * PREDICT_DT.
PREDICT_DT = 0.1

SUMMARY_CSV_NAME = "horizon_sweep_summary.csv"
PLOTS_DIR_NAME = "horizon_sweep_plots"     # default out-dir leaf under <results-dir>/<world_stem>/
FAILURE_CHART_NAME = "failure_rate_vs_horizon.png"
MEDIAN_CHART_NAME = "median_time_vs_horizon.png"

# Marker shape used for the per-algorithm line points (one shape; color carries
# the algorithm identity via the shared color map).
LINE_MARKER = "o"


def horizon_seconds(horizon_steps: int) -> float:
    """T seconds for a horizon expressed in steps: ``steps * PREDICT_DT``."""
    return horizon_steps * PREDICT_DT


# --- Per-horizon loader (reuses plot.py's per-label machinery) ---------------

def _algorithm_label(name: str, predict_horizon: int) -> str:
    """Resolve a predictive key + horizon to its results label (lazy `planners` import).

    Importing `planners` pulls irsim; this is deferred to call time so a bare
    `import runners.plot_horizon_sweep` stays headless. Returns
    e.g. ``d_star_lite_oracle_h10`` via the registry's own labeller, so the dir
    name never drifts from what `run_experiment` actually wrote.
    """
    from planners import algorithm_label

    return algorithm_label(name, None, predict_horizon)


def _load_label_dir_summary(
    label_dir: Path,
    *,
    label: str,
    display: str,
    family: str,
    expected: int,
) -> AlgoSummary:
    """Tally ONE label dir into an `AlgoSummary`, reusing plot.py's accumulators.

    Mirrors `runners.plot.load_world_results`'s per-entry body exactly (so the
    classify/median/failure-rate math is single-sourced with plot.py): prefer the
    manifest's `episodes` roster when present (DNF-accounting authoritative),
    otherwise glob the present numeric-stem JSONs. A missing label dir, a short
    count (< `expected`), or an unreadable file warns to stderr and is skipped —
    this NEVER raises, so the plotter degrades a missing horizon to an all-empty
    summary (`n_present == 0`) rather than crashing on partial sweep data.

    `_load_wallclocks` is called for parity with the canonical loader (so the
    AlgoSummary is fully populated), though the horizon charts do not read it.
    """
    acc = _AlgoAccumulator()
    seen_seeds: set[int] = set()

    if not label_dir.is_dir():
        print(
            f"warning: no result dir for {label} at {label_dir} (skipping)",
            file=sys.stderr,
        )
    else:
        manifest = _load_manifest(label_dir)
        episodes = _manifest_episodes(manifest)
        if episodes is not None:
            _accumulate_from_roster(acc, label_dir, episodes, label, seen_seeds)
        else:
            _accumulate_from_glob(acc, label_dir, seen_seeds)

        if acc.n_present < expected:
            print(
                f"warning: {label} has {acc.n_present} episodes (expected {expected})",
                file=sys.stderr,
            )

    # Wallclock subtree is keyed by (results_dir, world_stem, label); we do not
    # have those here and the charts ignore wallclocks, so pass an empty source.
    # (Keeping the AlgoSummary field populated avoids a None where plot.py uses a
    # tuple.)
    return _finalize_summary(
        label=label,
        display=display,
        family=family,
        acc=acc,
        wallclocks=(),
        wallclock_from_subtree=False,
    )


def load_horizon_results(
    results_dir: str,
    world_stem: str,
    horizons: tuple[int, ...] | list[int],
    *,
    algorithms: tuple[str, ...] | list[str] = SWEEP_ALGORITHMS,
    expected: int = DEFAULT_EXPECTED_SEEDS,
) -> dict[str, dict[int, AlgoSummary]]:
    """Load every (algorithm, horizon) label dir into an `algorithm -> {horizon: AlgoSummary}` map.

    For each swept `algorithm` and each `horizon` step H:
      1. build the label via `algorithm_label(algorithm, None, H)` (lazy import) —
         e.g. ``d_star_lite_oracle_h10``,
      2. build `label_dir = episode_out_dir(<results_dir>, world_stem, label)`,
      3. load that ONE label dir into an `AlgoSummary` REUSING plot.py's accumulators
         (`_load_label_dir_summary`), the SAME way `load_world_results` builds one
         summary per canonical entry.

    A missing horizon dir degrades to an all-empty summary (warn-not-raise), so a
    partial sweep still charts the horizons that ran. Horizons are kept in the
    given order; the returned inner dict is keyed by the integer horizon step.
    """
    results: dict[str, dict[int, AlgoSummary]] = {}
    results_root = Path(results_dir).resolve()

    for algorithm in algorithms:
        display = ALGO_DISPLAY.get(algorithm, algorithm)
        per_horizon: dict[int, AlgoSummary] = {}
        for horizon in horizons:
            label = _algorithm_label(algorithm, horizon)
            label_dir = episode_out_dir(results_root, world_stem, label)
            per_horizon[horizon] = _load_label_dir_summary(
                label_dir,
                label=label,
                display=display,
                family="incremental",
                expected=expected,
            )
        results[algorithm] = per_horizon

    return results


# --- Per-algorithm cross-horizon series -------------------------------------

@dataclass(frozen=True)
class AlgoSeries:
    """One algorithm's data points across the swept horizons.

    The parallel lists are aligned by index: for the horizon `horizons[i]` (steps)
    the algorithm has lookahead `seconds[i]`, failure rate `failure_rates[i]`,
    median time `median_times[i]`, and present-count `n_presents[i]`. A horizon
    where the algorithm has no present episodes (or a NaN failure_rate) is a GAP
    and is simply absent from these lists, so each chart skips it rather than
    drawing a 0.

    `present_anywhere` is True iff the algorithm had >=1 present episode in at
    least one horizon; the plotter filters on it so an absent algorithm draws no
    empty line.
    """

    algorithm: str
    display: str
    horizons: tuple[int, ...]
    seconds: tuple[float, ...]
    failure_rates: tuple[float, ...]
    median_times: tuple[float, ...]
    n_presents: tuple[int, ...]
    present_anywhere: bool


def _is_nan(value: float) -> bool:
    """True iff `value` is NaN (the loader's "no data" sentinel for floats)."""
    return value != value


def build_series(
    horizon_results: dict[str, dict[int, AlgoSummary]],
    horizons: tuple[int, ...] | list[int],
    *,
    algorithms: tuple[str, ...] | list[str] = SWEEP_ALGORITHMS,
) -> list[AlgoSeries]:
    """Build the per-algorithm cross-horizon series, filtered to the present set.

    For every swept algorithm, walks the horizons in the given order and collects
    `(seconds, failure_rate, median_time)` for each horizon where the algorithm has
    `n_present > 0` AND a finite failure_rate; the median_time may be NaN (a horizon
    where the algorithm was present but never succeeded) — that point is still kept
    for the failure-rate chart, and the median-time chart skips it via its own NaN
    filter at draw time. Algorithms absent in every horizon are dropped so the chart
    draws no empty line.
    """
    series: list[AlgoSeries] = []

    for algorithm in algorithms:
        per_horizon = horizon_results.get(algorithm)
        if per_horizon is None:
            continue
        display = ALGO_DISPLAY.get(algorithm, algorithm)

        horizons_kept: list[int] = []
        seconds: list[float] = []
        failure_rates: list[float] = []
        median_times: list[float] = []
        n_presents: list[int] = []
        present_anywhere = False

        for horizon in horizons:
            summary = per_horizon.get(horizon)
            if summary is None:
                continue
            if summary.n_present > 0:
                present_anywhere = True
            # A horizon where the algorithm has no present episodes, or a NaN
            # failure_rate, is a gap — skip the point entirely.
            if summary.n_present == 0 or _is_nan(summary.failure_rate):
                continue
            horizons_kept.append(horizon)
            seconds.append(horizon_seconds(horizon))
            failure_rates.append(summary.failure_rate)
            median_times.append(summary.median_time)  # may be NaN (present but 0 success)
            n_presents.append(summary.n_present)

        if not present_anywhere:
            continue

        series.append(
            AlgoSeries(
                algorithm=algorithm,
                display=display,
                horizons=tuple(horizons_kept),
                seconds=tuple(seconds),
                failure_rates=tuple(failure_rates),
                median_times=tuple(median_times),
                n_presents=tuple(n_presents),
                present_anywhere=present_anywhere,
            )
        )

    return series


def _series_color_map(series: list[AlgoSeries], plt) -> dict[str, tuple]:
    """Per-algorithm color map keyed by the algorithm registry name.

    Reuses `runners.plot._algorithm_color_map`, which colors by enumerate-index
    over the objects it is handed and reads each object's `.label`. We feed it tiny
    shims whose `.label` is the algorithm name, so each predictive key gets a
    stable color. With a single oracle line this is cosmetic, but it keeps the
    color path identical to plot_speed_sweep and stays correct once T13 adds a
    second line.
    """
    shims = [type("_C", (), {"label": algo.algorithm})() for algo in series]
    raw = _algorithm_color_map(shims, plt)
    return {algo.algorithm: raw.get(algo.algorithm, "#333333") for algo in series}


# --- Charts -----------------------------------------------------------------

def _render_line_chart(
    series: list[AlgoSeries],
    color_map: dict[str, tuple],
    plt,
    out_dir: Path,
    horizons: tuple[int, ...] | list[int],
    *,
    value_attr: str,
    out_name: str,
    title: str,
    y_label: str,
    y_lim: tuple[float, float] | None,
) -> Path:
    """Render one per-algorithm line chart (x = horizon seconds) and save it.

    `value_attr` selects the y series on each `AlgoSeries` (`"failure_rates"` or
    `"median_times"`). Points whose y value is NaN are skipped (a horizon where the
    algorithm was present but never succeeded has a NaN median time), so a line is
    drawn only through the horizons that have a finite value. The x ticks are the
    full swept horizon set (in seconds) regardless of which points each line has,
    so a gap reads as a gap. A side legend maps each line color to its algorithm
    display name, mirroring plot_speed_sweep's legend placement.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    # The x ticks are every swept horizon in seconds, labelled "<T>s\n(<steps> st)".
    ordered_horizons = list(horizons)
    tick_positions = [horizon_seconds(h) for h in ordered_horizons]
    tick_labels = [f"{horizon_seconds(h):g}s\n({h} st)" for h in ordered_horizons]

    drawn_handles = []
    for algo in series:
        color = color_map.get(algo.algorithm, "#333333")
        values = getattr(algo, value_attr)

        # Keep only the finite (seconds, value) points; a NaN value (e.g. NaN
        # median time at a present-but-0-success horizon) is a gap in THIS line.
        xs: list[float] = []
        ys: list[float] = []
        for seconds, value in zip(algo.seconds, values):
            if _is_nan(value):
                continue
            xs.append(seconds)
            ys.append(value)

        if not xs:
            # No finite points for this chart (e.g. an algo that never succeeded
            # in any horizon on the median-time chart): omit its line entirely.
            continue

        (line,) = ax.plot(
            xs,
            ys,
            marker=LINE_MARKER,
            markersize=7,
            linewidth=1.8,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=algo.display,
            zorder=3,
        )
        drawn_handles.append(line)

    ax.set_xlabel("prediction horizon T (seconds; steps x 0.1)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Side legend (color -> algorithm), placed outside the axes like plot_speed_sweep.
    if drawn_handles:
        legend = ax.legend(
            handles=drawn_handles,
            title="algorithm",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8,
            title_fontsize=9,
            borderaxespad=0.0,
        )
        extra = (legend,)
    else:
        # No drawable series for this chart; annotate so the PNG is not blank.
        ax.annotate(
            "no data to plot",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=12,
            color="#999999",
            fontweight="bold",
        )
        extra = ()

    fig.tight_layout()
    out_path = out_dir / out_name
    # The legend sits OUTSIDE the axes; pass it to savefig so bbox_inches="tight"
    # includes it (it may otherwise clip out-of-axes artists).
    fig.savefig(out_path, dpi=150, bbox_inches="tight", bbox_extra_artists=extra)
    plt.close(fig)
    return out_path


def chart_failure_rate_vs_horizon(
    series: list[AlgoSeries],
    color_map: dict[str, tuple],
    plt,
    out_dir: Path,
    horizons: tuple[int, ...] | list[int],
) -> Path:
    """Failure-rate-vs-horizon line chart (the AC1 deliverable).

    One line per present algorithm: x = the prediction horizon in seconds
    (0.0/0.5/1.0/2.0), y = failure rate (0..1). Reading the oracle's line across
    the horizons answers whether anticipating obstacle motion lowers the
    crash+timeout rate below the h0 (plain D* Lite) ablation.
    """
    return _render_line_chart(
        series,
        color_map,
        plt,
        out_dir,
        horizons,
        value_attr="failure_rates",
        out_name=FAILURE_CHART_NAME,
        title="Failure rate vs prediction horizon (predictive D* Lite)",
        y_label="failure rate (0 = always solves, 1 = always fails)",
        y_lim=(-0.05, 1.05),
    )


def chart_median_time_vs_horizon(
    series: list[AlgoSeries],
    color_map: dict[str, tuple],
    plt,
    out_dir: Path,
    horizons: tuple[int, ...] | list[int],
) -> Path:
    """Median-time-to-goal-vs-horizon line chart.

    One line per present algorithm: x = the prediction horizon in seconds, y =
    median time-to-goal over successful episodes (sim seconds). A horizon where an
    algorithm was present but never succeeded has a NaN median and is a gap in its
    line.
    """
    return _render_line_chart(
        series,
        color_map,
        plt,
        out_dir,
        horizons,
        value_attr="median_times",
        out_name=MEDIAN_CHART_NAME,
        title="Median time-to-goal vs prediction horizon (predictive D* Lite)",
        y_label="median time to goal (sim seconds)",
        y_lim=None,
    )


# --- Summary CSV ------------------------------------------------------------

SUMMARY_CSV_COLUMNS = (
    "label",
    "horizon_steps",
    "horizon_seconds",
    "failure_rate",
    "median_time",
    "n_present",
)


def write_summary_csv(series: list[AlgoSeries], out_path: str | Path) -> None:
    """Write one row per (algorithm, horizon) point to a CSV with SUMMARY_CSV_COLUMNS.

    Rows are emitted in swept-algorithm order, then by ascending horizon within
    each algorithm. Only the horizons that survived as points (present, finite
    failure_rate) are written; a NaN median_time is written as the literal "nan".
    The `label` column is the algorithm registry name (e.g. "d_star_lite_oracle"),
    so a downstream reader can pair it with the `_h<steps>` label dir via the steps.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SUMMARY_CSV_COLUMNS)
        for algo in series:
            for horizon, seconds, failure_rate, median_time, n_present in zip(
                algo.horizons,
                algo.seconds,
                algo.failure_rates,
                algo.median_times,
                algo.n_presents,
            ):
                writer.writerow(
                    [
                        algo.algorithm,
                        horizon,
                        seconds,
                        failure_rate,
                        median_time,
                        n_present,
                    ]
                )


# --- Self-check (headless; mirrors plot_speed_sweep's TC-S pattern) ----------
#
# The suite below runs TC-S1..TC-S7 against synthetic per-horizon result trees
# built in a TemporaryDirectory — no irsim, no real episodes. Each TC is a plain
# function that asserts its invariants; `run_selfcheck` catches per-TC exceptions
# so one failure never aborts the rest, prints a PASS/FAIL line each, and returns
# an int exit code (0 = all passed, 1 = any failed).

# The 7 metric fields run_episode writes (mirrors runners/plot.py:_SELFCHECK_METRIC_KEYS).
_SELFCHECK_METRIC_KEYS = (
    "time_to_goal",
    "crashed",
    "timed_out",
    "path_length",
    "mean_speed",
    "wallclock_per_step",
    "planner_error",
)

# The oracle key used throughout the selfcheck fixtures.
_ORACLE = "d_star_lite_oracle"


def _make_record(
    outcome: str,
    *,
    time_to_goal: float | None = None,
) -> dict:
    """Build one synthetic 7-key metrics record for the given outcome.

    Mirrors `runners.plot._make_record` but kept local so the selfcheck stays
    self-contained (and so a future plot.py refactor cannot silently break it). A
    "success" record carries a non-null `time_to_goal`; every other outcome sets
    its flag (or `planner_error` string) with `time_to_goal` null.
    """
    record = {
        "time_to_goal": None,
        "crashed": False,
        "timed_out": False,
        "path_length": 0.0,
        "mean_speed": 0.0,
        "wallclock_per_step": 0.01,
        "planner_error": None,
    }
    if outcome == "success":
        if time_to_goal is None:
            raise ValueError("a success record needs a time_to_goal")
        record["time_to_goal"] = float(time_to_goal)
        record["path_length"] = 65.0 + float(time_to_goal)
        record["mean_speed"] = record["path_length"] / float(time_to_goal)
    elif outcome == "crash":
        record["crashed"] = True
    elif outcome == "timeout":
        record["timed_out"] = True
    elif outcome == "planner_error":
        record["planner_error"] = "synthetic planner failure"
    else:
        raise ValueError(f"unknown outcome {outcome!r}")
    return record


def _write_horizon_algo_tree(
    *,
    results_root: Path,
    world_stem: str,
    algorithm: str,
    horizon: int,
    seeds: list[int],
    outcomes: list[str],
    success_times: list[float] | None = None,
) -> None:
    """Write one (algorithm, horizon)'s synthetic JSONs under its `_h<H>` label dir.

    The label dir is `<root>/<world_stem>/<algorithm_label(algorithm, None, H)>/`
    — exactly the dir `run_experiment` writes and `load_horizon_results` reads. One
    `<seed>.json` per (seed, outcome) pair. `success_times` supplies the
    `time_to_goal` for the success records in order; absent, a deterministic ramp
    is used. No manifest is written (the loader falls back to globbing the present
    numeric-stem JSONs, which is exactly the synthetic-fixture path).
    """
    import json

    if len(seeds) != len(outcomes):
        raise ValueError("seeds and outcomes must be the same length")

    label = _algorithm_label(algorithm, horizon)
    label_dir = results_root / world_stem / label
    label_dir.mkdir(parents=True, exist_ok=True)

    success_iter = iter(success_times) if success_times is not None else None
    ramp = 20.0
    for seed, outcome in zip(seeds, outcomes):
        if outcome == "success":
            if success_iter is not None:
                time_value = next(success_iter)
            else:
                time_value = ramp
                ramp += 5.0
            record = _make_record("success", time_to_goal=time_value)
        else:
            record = _make_record(outcome)
        (label_dir / f"{seed}.json").write_text(
            json.dumps(record, sort_keys=True), encoding="utf-8"
        )


def _tc_s1_loader_aggregates(tmp: Path) -> str:
    """TC-S1: the per-horizon loader aggregates failure_rate / median_time correctly."""
    results_root = tmp / "tc_s1"
    world_stem = "w"

    # One horizon (h5): 5 seeds = 3 success + 1 crash + 1 timeout.
    _write_horizon_algo_tree(
        results_root=results_root,
        world_stem=world_stem,
        algorithm=_ORACLE,
        horizon=5,
        seeds=[1, 2, 3, 4, 5],
        outcomes=["success", "success", "success", "crash", "timeout"],
        success_times=[10.0, 20.0, 60.0],
    )

    horizon_results = load_horizon_results(str(results_root), world_stem, (5,))
    summary = horizon_results[_ORACLE][5]

    assert summary.n_present == 5, f"expected 5 present, got {summary.n_present}"
    assert summary.n_success == 3, f"expected 3 success, got {summary.n_success}"
    assert abs(summary.failure_rate - 0.4) < 1e-9, \
        f"failure_rate should be 2/5=0.4, got {summary.failure_rate}"
    assert abs(summary.median_time - 20.0) < 1e-9, \
        f"median_time should be 20, got {summary.median_time}"
    return "loader aggregates failure_rate=0.4, median=20 over a horizon label dir"


def _tc_s2_x_positions_are_seconds(tmp: Path) -> str:
    """TC-S2: the series x positions equal steps * PREDICT_DT for the present horizons."""
    results_root = tmp / "tc_s2"
    world_stem = "w"
    horizons = (0, 5, 10, 20)

    # Present at every horizon with a success + a crash each.
    for horizon in horizons:
        _write_horizon_algo_tree(
            results_root=results_root,
            world_stem=world_stem,
            algorithm=_ORACLE,
            horizon=horizon,
            seeds=[1, 2],
            outcomes=["success", "crash"],
            success_times=[30.0],
        )

    horizon_results = load_horizon_results(str(results_root), world_stem, horizons)
    series = build_series(horizon_results, horizons)
    by_algo = {algo.algorithm: algo for algo in series}
    assert _ORACLE in by_algo, f"{_ORACLE} must be present in the series"
    algo = by_algo[_ORACLE]

    expected_seconds = tuple(horizon_seconds(h) for h in horizons)
    assert algo.seconds == expected_seconds, \
        f"x positions must equal steps*PREDICT_DT {expected_seconds}, got {algo.seconds}"
    # The exact AC numbers: 0/5/10/20 steps -> 0.0/0.5/1.0/2.0 s.
    assert algo.seconds == (0.0, 0.5, 1.0, 2.0), \
        f"horizon seconds must be (0.0, 0.5, 1.0, 2.0), got {algo.seconds}"
    assert list(algo.seconds) == sorted(algo.seconds), "seconds must be ascending"
    return "series x positions == steps*0.1 == (0.0, 0.5, 1.0, 2.0)"


def _tc_s3_both_charts_render(tmp: Path) -> str:
    """TC-S3: both line charts render under Agg without raising; PNGs are non-empty."""
    plt = ensure_matplotlib()

    results_root = tmp / "tc_s3"
    world_stem = "w"
    horizons = (0, 5, 10, 20)

    # h0 is the baseline (high failure); the failure rate falls as the horizon
    # grows, then the longest horizon is present-but-0-success (NaN median -> a gap
    # in the median chart but a finite point on the failure chart).
    plans = {
        0: (["crash", "crash", "success"], [40.0]),
        5: (["crash", "success", "success"], [35.0, 38.0]),
        10: (["success", "success", "success"], [30.0, 32.0, 34.0]),
        20: (["crash", "timeout"], None),
    }
    for horizon, (outcomes, times) in plans.items():
        _write_horizon_algo_tree(
            results_root=results_root,
            world_stem=world_stem,
            algorithm=_ORACLE,
            horizon=horizon,
            seeds=list(range(1, len(outcomes) + 1)),
            outcomes=outcomes,
            success_times=times,
        )

    horizon_results = load_horizon_results(str(results_root), world_stem, horizons)
    series = build_series(horizon_results, horizons)
    color_map = _series_color_map(series, plt)

    out_dir = tmp / "tc_s3_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    failure_png = chart_failure_rate_vs_horizon(series, color_map, plt, out_dir, horizons)
    median_png = chart_median_time_vs_horizon(series, color_map, plt, out_dir, horizons)
    for png in (failure_png, median_png):
        assert png.is_file(), f"chart did not write {png}"
        assert png.stat().st_size > 0, f"chart PNG {png} is empty"

    # The failure-rate series must span all four present horizons (failure_rate is
    # finite even when the horizon is 0-success).
    by_algo = {algo.algorithm: algo for algo in series}
    assert _ORACLE in by_algo, "the failure-rate series must include the oracle line"
    oracle = by_algo[_ORACLE]
    assert len(oracle.failure_rates) == len(horizons), \
        f"the oracle must have a failure point at all {len(horizons)} horizons, " \
        f"got {len(oracle.failure_rates)}"
    # The 0-success longest horizon contributes a NaN median (a median-chart gap).
    assert _is_nan(oracle.median_times[-1]), \
        "the present-but-0-success horizon must carry a NaN median (a median-chart gap)"
    return "both charts render non-empty; oracle failure line spans all 4 horizons"


def _tc_s4_missing_horizon_is_gap(tmp: Path) -> str:
    """TC-S4: a missing horizon dir degrades to a gap, not a crash."""
    plt = ensure_matplotlib()

    results_root = tmp / "tc_s4"
    world_stem = "w"
    horizons = (0, 5, 10, 20)

    # Write ONLY h0 and h10; the h5 and h20 label dirs are absent on disk entirely.
    for horizon in (0, 10):
        _write_horizon_algo_tree(
            results_root=results_root,
            world_stem=world_stem,
            algorithm=_ORACLE,
            horizon=horizon,
            seeds=[1, 2, 3],
            outcomes=["success", "success", "crash"],
            success_times=[18.0, 28.0],
        )

    # Must NOT raise even though two horizon dirs are missing.
    horizon_results = load_horizon_results(str(results_root), world_stem, horizons)
    series = build_series(horizon_results, horizons)
    by_algo = {algo.algorithm: algo for algo in series}
    assert _ORACLE in by_algo, f"{_ORACLE} must still be present from the two written horizons"
    algo = by_algo[_ORACLE]

    # Only the two present horizons contribute points; the two absent ones are gaps.
    assert algo.horizons == (0, 10), \
        f"only the present horizons must contribute points, got {algo.horizons}"
    assert algo.seconds == (0.0, 1.0), \
        f"seconds must be the two present horizons in order, got {algo.seconds}"

    # The two absent horizon summaries exist but are empty (n_present == 0).
    assert horizon_results[_ORACLE][5].n_present == 0, "the absent h5 must be empty, not raise"
    assert horizon_results[_ORACLE][20].n_present == 0, "the absent h20 must be empty, not raise"

    # And the charts still render from the present horizons.
    out_dir = tmp / "tc_s4_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    color_map = _series_color_map(series, plt)
    failure_png = chart_failure_rate_vs_horizon(series, color_map, plt, out_dir, horizons)
    assert failure_png.is_file() and failure_png.stat().st_size > 0, \
        "the failure chart must still render from the present horizons"
    return "missing horizon dir -> gap (2/4 horizons); charts still render"


def _tc_s5_summary_csv(tmp: Path) -> str:
    """TC-S5: the summary CSV has one header + one row per present (algorithm, horizon)."""
    import csv as _csv

    results_root = tmp / "tc_s5"
    world_stem = "w"
    horizons = (0, 5, 10)

    for horizon in horizons:
        _write_horizon_algo_tree(
            results_root=results_root,
            world_stem=world_stem,
            algorithm=_ORACLE,
            horizon=horizon,
            seeds=[1, 2],
            outcomes=["success", "crash"],
            success_times=[25.0],
        )

    horizon_results = load_horizon_results(str(results_root), world_stem, horizons)
    series = build_series(horizon_results, horizons)

    csv_path = tmp / "tc_s5_out" / SUMMARY_CSV_NAME
    write_summary_csv(series, csv_path)
    assert csv_path.is_file(), "the summary CSV must be written"

    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = list(_csv.reader(fh))
    assert rows[0] == list(SUMMARY_CSV_COLUMNS), \
        f"CSV header must be {SUMMARY_CSV_COLUMNS}, got {rows[0]}"
    # One data row per present horizon (3 horizons, all present).
    assert len(rows) == 1 + len(horizons), \
        f"expected header + {len(horizons)} rows, got {len(rows)}"
    # Spot-check the first data row: oracle, h0, 0.0 s, failure_rate 0.5, n_present 2.
    first = rows[1]
    assert first[0] == _ORACLE, f"label column must be the oracle key, got {first[0]!r}"
    assert int(first[1]) == 0, f"horizon_steps must be 0, got {first[1]!r}"
    assert abs(float(first[2]) - 0.0) < 1e-9, f"horizon_seconds must be 0.0, got {first[2]!r}"
    assert abs(float(first[3]) - 0.5) < 1e-9, f"failure_rate must be 0.5, got {first[3]!r}"
    assert int(first[5]) == 2, f"n_present must be 2, got {first[5]!r}"
    return "summary CSV: header + one row per present (algorithm, horizon)"


def _tc_s6_manifest_roster_dnf(tmp: Path) -> str:
    """TC-S6: a manifest `episodes` roster is authoritative — a runner_error seed folds in as a DNF."""
    import json

    results_root = tmp / "tc_s6"
    world_stem = "w"
    horizon = 10
    label = _algorithm_label(_ORACLE, horizon)
    label_dir = results_root / world_stem / label
    label_dir.mkdir(parents=True, exist_ok=True)

    # Two seeds have JSON (1 success, 1 crash); a third was killed at the wallclock
    # wall (status=runner_error, no JSON) -> it must count as a DNF failure.
    (label_dir / "1.json").write_text(
        json.dumps(_make_record("success", time_to_goal=22.0), sort_keys=True),
        encoding="utf-8",
    )
    (label_dir / "2.json").write_text(
        json.dumps(_make_record("crash"), sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "derived_seeds": [1, 2, 3],
        "episodes": [
            {"seed": 1, "exit_code": 0, "status": "ok"},
            {"seed": 2, "exit_code": 0, "status": "ok"},
            {"seed": 3, "exit_code": 124, "status": "runner_error"},
        ],
    }
    (label_dir / "_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    horizon_results = load_horizon_results(str(results_root), world_stem, (horizon,))
    summary = horizon_results[_ORACLE][horizon]

    assert summary.n_present == 3, f"roster must count all 3 seeds, got {summary.n_present}"
    assert summary.n_success == 1, f"expected 1 success, got {summary.n_success}"
    assert summary.n_crash == 1, f"expected 1 crash, got {summary.n_crash}"
    assert summary.n_dnf == 1, f"the runner_error seed must be a DNF, got {summary.n_dnf}"
    # failure_rate = (crash + dnf) / present = 2/3.
    assert abs(summary.failure_rate - (2.0 / 3.0)) < 1e-9, \
        f"failure_rate should be 2/3, got {summary.failure_rate}"
    return "manifest roster authoritative: runner_error seed folds in as a DNF (2/3 failure)"


def _tc_s7_matplotlib_guard(_tmp: Path) -> str:
    """TC-S7: the matplotlib-absent guard exits non-zero (patch find_spec, like plot.py's TC-P8)."""
    import importlib.util
    import io
    import contextlib

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
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


# Ordered registry of the self-check cases. Each entry is (id, callable).
_SELFCHECK_CASES = (
    ("TC-S1", _tc_s1_loader_aggregates),
    ("TC-S2", _tc_s2_x_positions_are_seconds),
    ("TC-S3", _tc_s3_both_charts_render),
    ("TC-S4", _tc_s4_missing_horizon_is_gap),
    ("TC-S5", _tc_s5_summary_csv),
    ("TC-S6", _tc_s6_manifest_roster_dnf),
    ("TC-S7", _tc_s7_matplotlib_guard),
)


def run_selfcheck() -> int:
    """Run the horizon-sweep plotter's self-check suite (TC-S1..TC-S7). Return 0 if all pass, else 1.

    Builds every fixture inside a single TemporaryDirectory (each TC namespaces its
    own subdir), runs each TC in isolation so one failure never aborts the rest,
    prints a per-TC PASS/FAIL line, and ends with an "N/M passed" summary. No
    irsim, no real episodes — synthetic per-horizon JSON trees only. Charts are
    rendered through the real chart functions under the headless Agg backend
    selected by `ensure_matplotlib()`.
    """
    import tempfile

    n_passed = 0
    n_total = len(_SELFCHECK_CASES)

    with tempfile.TemporaryDirectory(prefix="horizon_sweep_selfcheck_") as tmp_name:
        tmp = Path(tmp_name)
        for tc_id, tc_func in _SELFCHECK_CASES:
            try:
                detail = tc_func(tmp)
            except Exception as exc:  # noqa: BLE001 — one TC must not abort the suite
                print(f"{tc_id}: FAIL - {type(exc).__name__}: {exc}")
            else:
                n_passed += 1
                print(f"{tc_id}: PASS - {detail}")

    print(f"selfcheck: {n_passed}/{n_total} passed")
    return 0 if n_passed == n_total else 1


# --- CLI --------------------------------------------------------------------

@dataclass(frozen=True)
class HorizonPlotArgs:
    """Parsed CLI arguments — frozen so accidental mutation is impossible."""

    world: str | None
    results_dir: str
    horizons: tuple[int, ...]
    out_dir: str | None
    selfcheck: bool


def _parse_args(argv: list[str] | None) -> HorizonPlotArgs:
    parser = argparse.ArgumentParser(
        prog="runners.plot_horizon_sweep",
        description=(
            "Chart predictive D* Lite failure rate and median time-to-goal vs the "
            "prediction horizon, reading the per-horizon oracle label dirs (Phase 7)."
        ),
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
        help=f"Results ROOT to read from (default {DEFAULT_RESULTS_DIR!r}); "
        "the plotter reads <root>/<world_stem>/d_star_lite_oracle_h<H>/.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        metavar="H",
        help=(
            "Prediction horizons to chart, in steps (default "
            f"{' '.join(str(h) for h in DEFAULT_HORIZONS)}). T seconds = steps x 0.1."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output dir for the PNGs + summary CSV "
        f"(default <results-dir>/<world_stem>/{PLOTS_DIR_NAME}/).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="Run the self-check suite instead of plotting.",
    )
    ns = parser.parse_args(argv)
    return HorizonPlotArgs(
        world=ns.world,
        results_dir=ns.results_dir,
        horizons=tuple(int(h) for h in ns.horizons),
        out_dir=ns.out_dir,
        selfcheck=bool(ns.selfcheck),
    )


def _resolve_out_dir(args: HorizonPlotArgs, world_stem: str) -> Path:
    """Out-dir is the CLI override, else `<results-dir>/<world_stem>/horizon_sweep_plots/`."""
    if args.out_dir is not None:
        return Path(args.out_dir).resolve()
    return Path(args.results_dir).resolve() / world_stem / PLOTS_DIR_NAME


def main(argv: list[str] | None = None) -> int:
    """Render the two horizon charts + summary CSV. See module docstring for CLI semantics."""
    args = _parse_args(argv)

    if args.selfcheck:
        # Selfcheck ignores --world entirely; run it BEFORE any --world validation
        # so the documented `python -m runners.plot_horizon_sweep --selfcheck`
        # command (no --world) works (mirrors plot_speed_sweep's main() gating).
        return run_selfcheck()

    # --world is only required for the normal plotting path. Validate it here
    # (after the selfcheck gate), mirroring plot_speed_sweep's up-front exit code 2.
    if args.world is None:
        print(
            "error: --world is required unless --selfcheck is given",
            file=sys.stderr,
        )
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

    world_stem = Path(args.world).stem
    out_dir = _resolve_out_dir(args, world_stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt = ensure_matplotlib()

    horizon_results = load_horizon_results(
        args.results_dir,
        world_stem,
        args.horizons,
    )

    # "No readable data at all" => every (algorithm, horizon) came back empty.
    # Nothing to plot or summarize, so exit non-zero with a clear message.
    any_present = any(
        summary.n_present > 0
        for per_horizon in horizon_results.values()
        for summary in per_horizon.values()
    )
    if not any_present:
        root = Path(args.results_dir).resolve()
        horizon_labels = ", ".join(
            f"{_algorithm_label(algorithm, h)}"
            for algorithm in SWEEP_ALGORITHMS
            for h in args.horizons
        )
        print(
            f"error: nothing to plot - no readable episode JSONs under any of "
            f"{root}/{world_stem}/{{{horizon_labels}}}",
            file=sys.stderr,
        )
        return 1

    series = build_series(horizon_results, args.horizons)
    color_map = _series_color_map(series, plt)

    csv_path = out_dir / SUMMARY_CSV_NAME
    write_summary_csv(series, csv_path)
    print(f"wrote {csv_path}")

    failure_png = chart_failure_rate_vs_horizon(series, color_map, plt, out_dir, args.horizons)
    print(f"wrote {failure_png}")
    median_png = chart_median_time_vs_horizon(series, color_map, plt, out_dir, args.horizons)
    print(f"wrote {median_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
