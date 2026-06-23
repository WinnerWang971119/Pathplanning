"""Read-only sweep plotter — charts failure rate and median time vs the obstacle-speed cap.

Phase 7 (the obstacle-speed-cap sweep, issue #11). The sweep driver
(`runners.run_speed_sweep`) runs each planner against the four named speed
regimes (slow, matched, current, fast), partitioning the per-episode metrics
JSONs into a per-regime results subtree:

    <results-dir>/speed_<regime>/<world_stem>/<label>/<seed>.json

This module reads ONLY those JSONs (never irsim, never a sim) and turns them
into two cross-regime line charts: per-algorithm **failure rate vs speed cap**
and **median time-to-goal vs speed cap**, one line per algorithm, with the four
regimes as distinct x positions (the regime's max-cap factor 0.7/1.0/1.5/2.0).
The failure-rate chart is the hypothesis deliverable — fed the full sweep it
shows directly whether D* Lite's failures floor toward zero once no obstacle can
outrun the robot.

This is a SIBLING of `runners.plot`: it reuses that module's loader
(`load_world_results`), data model (`AlgoSummary` / `WorldResults`), matplotlib
import guard (`ensure_matplotlib`), canonical algorithm set (`CANONICAL`), and
color map (`_algorithm_color_map`) wholesale, adding only the "load four regime
subtrees, line them up by speed cap, draw two line charts" logic on top.

HEADLESS (load-bearing, AC11): nothing here imports irsim or matplotlib at module
top. The speed constants come from the PURE `arena.speed_regimes` module (never
`arena.dynamic`, which pulls irsim); the `all` label list comes from
`runners.plot.CANONICAL` (never `runners.run_all`, which pulls irsim); matplotlib
is reached ONLY through `ensure_matplotlib()` inside the render functions. The
verification `python -c "import runners.plot_speed_sweep; ..."` must report that
neither irsim nor matplotlib is in `sys.modules` after import.

CLI:
    python -m runners.plot_speed_sweep \
        --world <yaml_path>          # required unless --selfcheck; only its stem is used
        [--results-dir <dir>]        # ROOT (default "results"); reads <root>/speed_<regime>/...
        [--algorithms {focus,all}]   # default "focus" (a_star_once, d_star_lite, dwa, apf)
        [--replan-k <int>]           # default 5; cadence used to build replan labels
        [--out-dir <dir>]            # default <results-dir>/<world_stem>/speed_sweep_plots/
        [--selfcheck]                # run the headless self-check suite instead of plotting

Outputs:
    <out-dir>/failure_rate_vs_cap.png   — per-algorithm failure rate vs speed cap
    <out-dir>/median_time_vs_cap.png    — per-algorithm median time-to-goal vs speed cap
    <out-dir>/speed_sweep_summary.csv   — one row per algorithm x regime

Exit codes:
    0 — charts/summary written (or selfcheck passed)
    1 — matplotlib missing, or no readable data under any regime subtree
    2 — argparse / CLI validation error
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the repo root importable so `from runners.plot import ...` and the pure
# `from arena.speed_regimes import ...` resolve when this module is invoked as
# `python -m runners.plot_speed_sweep` from any cwd. Mirrors runners/plot.py:50-52.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the sibling plotter's loader + data model + matplotlib guard + canonical
# set + color map. `runners.plot` is itself headless at import (it defers the
# `planners`/irsim and matplotlib imports into functions), so importing it here
# does NOT pull irsim or matplotlib at module top — the AC11 boundary holds.
from runners.plot import (
    CANONICAL,
    AlgoSummary,
    WorldResults,
    _algorithm_color_map,
    ensure_matplotlib,
    load_world_results,
)

# Speed-regime constants come from the PURE module (stdlib-only, no irsim). NEVER
# import these from arena.dynamic (which pulls numpy and the irsim-backed arena
# package members), so the plotter and its --selfcheck stay headless.
from arena.speed_regimes import SPEED_REGIMES, SPEED_REGIME_CAP


# --- Module constants -------------------------------------------------------

DEFAULT_RESULTS_DIR = "results"
DEFAULT_REPLAN_K = 5                       # cadence used to build the _replan label dirs
SPEED_SUBTREE_PREFIX = "speed_"            # per-regime subtree: <root>/speed_<regime>/...
SUMMARY_CSV_NAME = "speed_sweep_summary.csv"
PLOTS_DIR_NAME = "speed_sweep_plots"       # default out-dir leaf under <results-dir>/<world_stem>/
FAILURE_CHART_NAME = "failure_rate_vs_cap.png"
MEDIAN_CHART_NAME = "median_time_vs_cap.png"

# The four regimes, in ascending-cap order (the x-axis ordering). Derived from the
# pure table so a regime added there is picked up without editing this list.
REGIME_ORDER: tuple[str, ...] = tuple(
    sorted(SPEED_REGIMES, key=lambda name: SPEED_REGIME_CAP[name])
)

# The focus algorithm set: the static baseline + the incremental planner the
# hypothesis is about + the two reactive planners whose degradation the issue
# calls out. All four are non-replan (replan_k None), so their labels are bare
# registry keys. Display names are resolved from CANONICAL so they never drift.
FOCUS_NAMES: tuple[str, ...] = ("a_star_once", "d_star_lite", "dwa", "apf")

# Marker shape used for the per-algorithm line points (one shape, color carries
# the algorithm identity via the shared color map).
LINE_MARKER = "o"


# --- Per-algorithm cross-regime series --------------------------------------

@dataclass(frozen=True)
class AlgoSeries:
    """One algorithm's data points across the swept regimes.

    `label` / `display` / `family` mirror `AlgoSummary`. The three parallel lists
    are aligned by index to `regimes`: for regime `regimes[i]` the algorithm has
    speed cap `caps[i]`, failure rate `failure_rates[i]`, median time
    `median_times[i]`, and present-count `n_presents[i]`. A regime where the
    algorithm has no present episodes (or a NaN stat) is a GAP and is simply
    absent from these lists, so each chart skips it rather than drawing a 0.

    `present_anywhere` is True iff the algorithm had >=1 present episode in at
    least one regime; the plotter filters on it so a `focus` run does not draw
    empty lines for the absent planners.
    """

    label: str
    display: str
    family: str
    regimes: tuple[str, ...]
    caps: tuple[float, ...]
    failure_rates: tuple[float, ...]
    median_times: tuple[float, ...]
    n_presents: tuple[int, ...]
    present_anywhere: bool


def _is_nan(value: float) -> bool:
    """True iff `value` is NaN (the loader's "no data" sentinel for floats)."""
    return value != value


def load_regime_results(
    results_dir: str,
    world_stem: str,
    *,
    replan_k: int = DEFAULT_REPLAN_K,
) -> dict[str, WorldResults]:
    """Load every regime subtree into a regime -> WorldResults map.

    For each regime in :data:`REGIME_ORDER`, reads
    `<results_dir>/speed_<regime>/<world_stem>/<label>/` via
    `runners.plot.load_world_results`. That loader warns-not-raises on a missing
    label dir, so an entirely absent regime subtree degrades to all-empty
    summaries (every `n_present == 0`) rather than crashing — the caller treats
    such a regime as a gap. Every returned `WorldResults` carries the full
    CANONICAL set of summaries in CANONICAL order (the loader always emits all 11).
    """
    regime_results: dict[str, WorldResults] = {}
    for regime in REGIME_ORDER:
        regime_dir = f"{results_dir}/{SPEED_SUBTREE_PREFIX}{regime}"
        regime_results[regime] = load_world_results(
            regime_dir,
            world_stem,
            replan_k=replan_k,
        )
    return regime_results


def _resolve_drawn_labels(algorithms: str) -> set[str]:
    """The set of result labels to draw for the `--algorithms` mode.

    `focus` selects the four FOCUS_NAMES; `all` selects every CANONICAL label.
    All FOCUS_NAMES are non-replan, so their labels equal their registry keys —
    no `replan_k` folding is needed to resolve them.
    """
    if algorithms == "all":
        return {name for (name, *_rest) in CANONICAL}
    if algorithms == "focus":
        return set(FOCUS_NAMES)
    raise ValueError(f"unknown algorithms mode {algorithms!r}; expected 'focus' or 'all'")


def build_series(
    regime_results: dict[str, WorldResults],
    *,
    algorithms: str = "focus",
) -> list[AlgoSeries]:
    """Build the per-algorithm cross-regime series, filtered to the drawn + present set.

    For every CANONICAL algorithm, walks the regimes in ascending-cap order and
    collects `(cap, failure_rate, median_time)` for each regime where the
    algorithm has `n_present > 0` AND a finite failure_rate; the median_time may
    be NaN (a regime where the algorithm was present but never succeeded) — that
    point is still collected for the failure-rate chart, and the median-time
    chart skips it via its own NaN filter at draw time.

    The result keeps ONLY the algorithms in the `--algorithms` selection that
    were present in at least one regime, in CANONICAL order. The per-regime
    `WorldResults.summaries` are all in CANONICAL order, so indexing them by
    position keeps `label`/`display`/`family` consistent across regimes.
    """
    drawn_labels = _resolve_drawn_labels(algorithms)

    # CANONICAL order is shared by every regime's summaries; iterate by index so
    # one canonical algorithm lines up across all regimes.
    n_canonical = len(CANONICAL)
    series: list[AlgoSeries] = []

    for index in range(n_canonical):
        # Pull this algorithm's summary from each regime (all summaries lists are
        # CANONICAL-ordered and full-length, so position `index` is the same algo).
        per_regime: dict[str, AlgoSummary] = {}
        label: str | None = None
        display: str | None = None
        family: str | None = None
        for regime in REGIME_ORDER:
            world = regime_results.get(regime)
            if world is None or index >= len(world.summaries):
                continue
            summary = world.summaries[index]
            per_regime[regime] = summary
            # The label/display/family are identical across regimes for one algo;
            # capture them from whichever regime we see first.
            if label is None:
                label = summary.label
                display = summary.display
                family = summary.family

        if label is None:
            # No regime produced a summary at this index (should not happen — the
            # loader always emits the full CANONICAL set — but stay defensive).
            continue

        # Filter to the requested algorithm set by label.
        if label not in drawn_labels:
            continue

        regimes_kept: list[str] = []
        caps: list[float] = []
        failure_rates: list[float] = []
        median_times: list[float] = []
        n_presents: list[int] = []
        present_anywhere = False

        for regime in REGIME_ORDER:
            summary = per_regime.get(regime)
            if summary is None:
                continue
            if summary.n_present > 0:
                present_anywhere = True
            # A regime where the algorithm has no present episodes, or a NaN
            # failure_rate, is a gap — skip the point entirely.
            if summary.n_present == 0 or _is_nan(summary.failure_rate):
                continue
            regimes_kept.append(regime)
            caps.append(SPEED_REGIME_CAP[regime])
            failure_rates.append(summary.failure_rate)
            median_times.append(summary.median_time)  # may be NaN (present but 0 success)
            n_presents.append(summary.n_present)

        if not present_anywhere:
            # Absent in every regime — drop it so the chart draws no empty line.
            continue

        series.append(
            AlgoSeries(
                label=label,
                display=display if display is not None else label,
                family=family if family is not None else "",
                regimes=tuple(regimes_kept),
                caps=tuple(caps),
                failure_rates=tuple(failure_rates),
                median_times=tuple(median_times),
                n_presents=tuple(n_presents),
                present_anywhere=present_anywhere,
            )
        )

    return series


def _stable_color_map(regime_results: dict[str, WorldResults], plt) -> dict[str, tuple]:
    """Build the per-algorithm color map ONCE from the FULL CANONICAL-ordered summaries.

    Color stability is load-bearing (the spec's "Color stability"): the shared
    `_algorithm_color_map` colors by enumerate-index over the summaries it is
    handed, so it MUST be built from a full CANONICAL-ordered summary list (all 11
    labels, in order) and only THEN filtered to the drawn algorithms — building it
    from a pre-filtered subset would shift the indices and recolor lines per
    regime. Any regime's `summaries` is the full CANONICAL list in order, so we
    pass the first available one. If somehow no regime loaded, fall back to an
    empty map (the caller will have already exited on the no-data path).
    """
    for regime in REGIME_ORDER:
        world = regime_results.get(regime)
        if world is not None and world.summaries:
            return _algorithm_color_map(world.summaries, plt)
    return {}


# --- x-axis (speed-cap) helpers ---------------------------------------------

def _regime_tick_positions() -> tuple[list[float], list[str]]:
    """The shared x ticks for both charts: one tick per regime at its speed cap.

    Returns `(positions, labels)` where `positions` are the regime caps in
    ascending order and each label pairs the regime name with its cap factor
    (e.g. "slow\\n(0.7x)"), so the four regimes read as distinct, named x
    positions per AC14.
    """
    positions = [SPEED_REGIME_CAP[regime] for regime in REGIME_ORDER]
    labels = [f"{regime}\n({SPEED_REGIME_CAP[regime]:g}x)" for regime in REGIME_ORDER]
    return positions, labels


# --- Charts -----------------------------------------------------------------

def _render_line_chart(
    series: list[AlgoSeries],
    color_map: dict[str, tuple],
    plt,
    out_dir: Path,
    *,
    value_attr: str,
    out_name: str,
    title: str,
    y_label: str,
    y_lim: tuple[float, float] | None,
) -> Path:
    """Render one per-algorithm line chart (x = speed cap) and save it.

    `value_attr` selects the y series on each `AlgoSeries`
    (`"failure_rates"` or `"median_times"`). Points whose y value is NaN are
    skipped (a regime where the algorithm was present but never succeeded has a
    NaN median time), so a line is drawn only through the regimes that have a
    finite value. A side legend maps each line color to its algorithm display
    name, mirroring `plot.py`'s A1 legend placement.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    tick_positions, tick_labels = _regime_tick_positions()

    drawn_handles = []
    for algo in series:
        color = color_map.get(algo.label, "#333333")
        values = getattr(algo, value_attr)

        # Keep only the finite (cap, value) points; a NaN value (e.g. NaN median
        # time at a present-but-0-success regime) is a gap in THIS chart's line.
        xs: list[float] = []
        ys: list[float] = []
        for cap, value in zip(algo.caps, values):
            if _is_nan(value):
                continue
            xs.append(cap)
            ys.append(value)

        if not xs:
            # No finite points for this chart (e.g. an algo that never succeeded
            # in any regime on the median-time chart): omit its line entirely.
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

    ax.set_xlabel("obstacle speed cap (x robot top speed)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Side legend (color -> algorithm), placed outside the axes like plot.py's A1.
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


def chart_failure_rate_vs_cap(
    series: list[AlgoSeries],
    color_map: dict[str, tuple],
    plt,
    out_dir: Path,
) -> Path:
    """Failure-rate-vs-speed-cap line chart (AC10 / the AC14 hypothesis deliverable).

    One line per present algorithm: x = the regime's speed cap (0.7/1.0/1.5/2.0),
    y = failure rate (0..1). Fed the full sweep, reading D* Lite's line down the
    caps directly answers whether its failures floor toward zero once no obstacle
    can outrun the robot.
    """
    return _render_line_chart(
        series,
        color_map,
        plt,
        out_dir,
        value_attr="failure_rates",
        out_name=FAILURE_CHART_NAME,
        title="Failure rate vs obstacle speed cap (per algorithm)",
        y_label="failure rate (0 = always solves, 1 = always fails)",
        y_lim=(-0.05, 1.05),
    )


def chart_median_time_vs_cap(
    series: list[AlgoSeries],
    color_map: dict[str, tuple],
    plt,
    out_dir: Path,
) -> Path:
    """Median-time-to-goal-vs-speed-cap line chart (AC10).

    One line per present algorithm: x = the regime's speed cap, y = median
    time-to-goal over successful episodes (sim seconds). A regime where an
    algorithm was present but never succeeded has a NaN median and is a gap in its
    line.
    """
    return _render_line_chart(
        series,
        color_map,
        plt,
        out_dir,
        value_attr="median_times",
        out_name=MEDIAN_CHART_NAME,
        title="Median time-to-goal vs obstacle speed cap (per algorithm)",
        y_label="median time to goal (sim seconds)",
        y_lim=None,
    )


# --- Summary CSV ------------------------------------------------------------

SUMMARY_CSV_COLUMNS = (
    "label",
    "display",
    "regime",
    "cap",
    "failure_rate",
    "median_time",
    "n_present",
)


def write_summary_csv(series: list[AlgoSeries], out_path: str | Path) -> None:
    """Write one row per (algorithm, regime) point to a CSV with SUMMARY_CSV_COLUMNS.

    Rows are emitted in CANONICAL algorithm order, then by ascending speed cap
    within each algorithm. Only the regimes that survived as points (present, finite
    failure_rate) are written; a NaN median_time is written as the literal "nan".
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SUMMARY_CSV_COLUMNS)
        for algo in series:
            for regime, cap, failure_rate, median_time, n_present in zip(
                algo.regimes,
                algo.caps,
                algo.failure_rates,
                algo.median_times,
                algo.n_presents,
            ):
                writer.writerow(
                    [
                        algo.label,
                        algo.display,
                        regime,
                        cap,
                        failure_rate,
                        median_time,
                        n_present,
                    ]
                )


# --- Self-check (headless; mirrors plot.py's TC-P pattern) ------------------
#
# The suite below runs TC-S1..TC-S7 against synthetic per-regime result trees
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


def _write_regime_algo_tree(
    *,
    results_root: Path,
    regime: str,
    world_stem: str,
    label: str,
    seeds: list[int],
    outcomes: list[str],
    success_times: list[float] | None = None,
) -> None:
    """Write one algorithm's synthetic JSONs under `<root>/speed_<regime>/<world_stem>/<label>/`.

    One `<seed>.json` per (seed, outcome) pair. `success_times` supplies the
    `time_to_goal` for the success records in order; absent, a deterministic ramp
    is used. No manifest is written (the loader falls back to globbing the present
    numeric-stem JSONs, which is exactly the synthetic-fixture path).
    """
    import json

    if len(seeds) != len(outcomes):
        raise ValueError("seeds and outcomes must be the same length")

    label_dir = results_root / f"{SPEED_SUBTREE_PREFIX}{regime}" / world_stem / label
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


def _label_for(name: str, replan_k: int | None) -> str:
    """Resolve a registry name to its results label (lazy `planners` import).

    Importing `planners` pulls irsim; the selfcheck is allowed to do so once it is
    actually RUNNING (it just must not happen at module import time), mirroring
    `runners.plot`'s deferred import discipline.
    """
    from planners import algorithm_label

    return algorithm_label(name, replan_k)


def _tc_s1_loader_aggregates(tmp: Path) -> str:
    """TC-S1: the per-regime loader aggregates failure_rate / median_time correctly."""
    results_root = tmp / "tc_s1"
    world_stem = "w"
    label = _label_for("a_star_once", None)

    # One regime (slow): 5 seeds = 3 success + 1 crash + 1 timeout.
    _write_regime_algo_tree(
        results_root=results_root,
        regime="slow",
        world_stem=world_stem,
        label=label,
        seeds=[1, 2, 3, 4, 5],
        outcomes=["success", "success", "success", "crash", "timeout"],
        success_times=[10.0, 20.0, 60.0],
    )

    regime_results = load_regime_results(str(results_root), world_stem, replan_k=DEFAULT_REPLAN_K)
    slow = regime_results["slow"]
    by_label = {s.label: s for s in slow.summaries}
    summary = by_label[label]

    assert summary.n_present == 5, f"expected 5 present, got {summary.n_present}"
    assert summary.n_success == 3, f"expected 3 success, got {summary.n_success}"
    assert abs(summary.failure_rate - 0.4) < 1e-9, \
        f"failure_rate should be 2/5=0.4, got {summary.failure_rate}"
    assert abs(summary.median_time - 20.0) < 1e-9, \
        f"median_time should be 20, got {summary.median_time}"
    return "loader aggregates failure_rate=0.4, median=20 over a regime subtree"


def _tc_s2_x_positions_are_caps(tmp: Path) -> str:
    """TC-S2: the series x positions equal SPEED_REGIME_CAP for the present regimes."""
    results_root = tmp / "tc_s2"
    world_stem = "w"
    label = _label_for("a_star_once", None)

    # Present in all four regimes with a single success each.
    for regime in REGIME_ORDER:
        _write_regime_algo_tree(
            results_root=results_root,
            regime=regime,
            world_stem=world_stem,
            label=label,
            seeds=[1, 2],
            outcomes=["success", "crash"],
            success_times=[30.0],
        )

    regime_results = load_regime_results(str(results_root), world_stem, replan_k=DEFAULT_REPLAN_K)
    series = build_series(regime_results, algorithms="focus")
    by_label = {algo.label: algo for algo in series}
    assert label in by_label, f"{label} must be present in the focus series"
    algo = by_label[label]

    expected_caps = tuple(SPEED_REGIME_CAP[regime] for regime in REGIME_ORDER)
    assert algo.caps == expected_caps, \
        f"x positions must equal the regime caps {expected_caps}, got {algo.caps}"
    # And they must be ascending (the chart's x ordering).
    assert list(algo.caps) == sorted(algo.caps), "caps must be in ascending order"
    return f"series x positions == SPEED_REGIME_CAP {expected_caps}"


def _tc_s3_focus_draws_only_present(tmp: Path) -> str:
    """TC-S3: a focus run draws only the present focus algorithms (absent ones dropped)."""
    results_root = tmp / "tc_s3"
    world_stem = "w"

    # Write only a_star_once and d_star_lite; dwa and apf are absent everywhere.
    for name in ("a_star_once", "d_star_lite"):
        label = _label_for(name, None)
        for regime in REGIME_ORDER:
            _write_regime_algo_tree(
                results_root=results_root,
                regime=regime,
                world_stem=world_stem,
                label=label,
                seeds=[1, 2, 3],
                outcomes=["success", "success", "crash"],
                success_times=[15.0, 25.0],
            )

    regime_results = load_regime_results(str(results_root), world_stem, replan_k=DEFAULT_REPLAN_K)
    series = build_series(regime_results, algorithms="focus")
    drawn = {algo.label for algo in series}

    a_star = _label_for("a_star_once", None)
    d_star = _label_for("d_star_lite", None)
    dwa = _label_for("dwa", None)
    apf = _label_for("apf", None)
    assert drawn == {a_star, d_star}, \
        f"focus must draw only the present {{a_star_once, d_star_lite}}, got {drawn}"
    assert dwa not in drawn and apf not in drawn, \
        "absent focus planners (dwa, apf) must NOT produce an empty line"
    # And it must NOT pull in any of the 7 non-focus canonical planners.
    assert _label_for("dijkstra_once", None) not in drawn, \
        "a focus run must not draw non-focus planners"
    return "focus draws only present focus algos; absent ones dropped"


def _tc_s4_color_map_stable(tmp: Path) -> str:
    """TC-S4: the color map is built from the FULL CANONICAL order, then filtered.

    The drawn algorithms must get the SAME color they would in a full-CANONICAL
    map (so colors stay stable whether or not the other 7 planners are present).
    """
    plt = ensure_matplotlib()

    results_root = tmp / "tc_s4"
    world_stem = "w"

    # Only the focus set is present, but the color map must still be keyed off the
    # full CANONICAL ordering.
    for name in FOCUS_NAMES:
        label = _label_for(name, None)
        for regime in REGIME_ORDER:
            _write_regime_algo_tree(
                results_root=results_root,
                regime=regime,
                world_stem=world_stem,
                label=label,
                seeds=[1, 2],
                outcomes=["success", "crash"],
                success_times=[40.0],
            )

    regime_results = load_regime_results(str(results_root), world_stem, replan_k=DEFAULT_REPLAN_K)
    color_map = _stable_color_map(regime_results, plt)

    # The reference: the color map a full CANONICAL-ordered summary list yields.
    # Any regime's summaries IS that full list (the loader emits all 11 in order).
    reference_world = regime_results[REGIME_ORDER[0]]
    reference_map = _algorithm_color_map(reference_world.summaries, plt)

    for name in FOCUS_NAMES:
        label = _label_for(name, None)
        assert label in color_map, f"{label} must be in the stable color map"
        assert color_map[label] == reference_map[label], \
            f"{label} color must match the full-CANONICAL-order map (stability)"

    # The map must cover all 11 canonical labels, not just the 4 drawn ones — the
    # proof it was built from the FULL order, not a pre-filtered subset.
    assert len(color_map) == len(CANONICAL), \
        f"color map must cover all {len(CANONICAL)} canonical labels, got {len(color_map)}"
    return "color map built from full CANONICAL order then filtered; drawn colors stable"


def _tc_s5_both_charts_render(tmp: Path) -> str:
    """TC-S5: both line charts render under Agg without raising; PNGs are non-empty."""
    plt = ensure_matplotlib()

    results_root = tmp / "tc_s5"
    world_stem = "w"

    # A mix: a_star_once present everywhere with successes; d_star_lite present but
    # 0-success in the fast regime (NaN median -> a gap in the median chart).
    a_label = _label_for("a_star_once", None)
    d_label = _label_for("d_star_lite", None)
    for index, regime in enumerate(REGIME_ORDER):
        _write_regime_algo_tree(
            results_root=results_root,
            regime=regime,
            world_stem=world_stem,
            label=a_label,
            seeds=[1, 2, 3],
            outcomes=["success", "success", "crash"],
            success_times=[20.0 + 5.0 * index, 30.0 + 5.0 * index],
        )
        if regime == "fast":
            # d_star_lite present but never succeeds at the fast cap.
            _write_regime_algo_tree(
                results_root=results_root,
                regime=regime,
                world_stem=world_stem,
                label=d_label,
                seeds=[1, 2],
                outcomes=["crash", "timeout"],
            )
        else:
            _write_regime_algo_tree(
                results_root=results_root,
                regime=regime,
                world_stem=world_stem,
                label=d_label,
                seeds=[1, 2],
                outcomes=["success", "crash"],
                success_times=[50.0],
            )

    regime_results = load_regime_results(str(results_root), world_stem, replan_k=DEFAULT_REPLAN_K)
    series = build_series(regime_results, algorithms="focus")
    color_map = _stable_color_map(regime_results, plt)

    out_dir = tmp / "tc_s5_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    failure_png = chart_failure_rate_vs_cap(series, color_map, plt, out_dir)
    median_png = chart_median_time_vs_cap(series, color_map, plt, out_dir)
    for png in (failure_png, median_png):
        assert png.is_file(), f"chart did not write {png}"
        assert png.stat().st_size > 0, f"chart PNG {png} is empty"

    # AC14: the failure-rate series must include a D* Lite line spanning all four
    # present regimes (failure_rate is finite even when 0-success).
    by_label = {algo.label: algo for algo in series}
    assert d_label in by_label, "the failure-rate series must include a D* Lite line"
    d_series = by_label[d_label]
    assert len(d_series.failure_rates) == len(REGIME_ORDER), \
        f"D* Lite must have a failure point in all {len(REGIME_ORDER)} regimes, got {len(d_series.failure_rates)}"
    return "both charts render non-empty; D* Lite failure line spans all 4 regimes"


def _tc_s6_missing_regime_is_gap(tmp: Path) -> str:
    """TC-S6: a missing regime subtree degrades to a gap, not a crash."""
    plt = ensure_matplotlib()

    results_root = tmp / "tc_s6"
    world_stem = "w"
    label = _label_for("a_star_once", None)

    # Write ONLY the slow and current regimes; matched and fast subtrees are absent
    # on disk entirely.
    for regime in ("slow", "current"):
        _write_regime_algo_tree(
            results_root=results_root,
            regime=regime,
            world_stem=world_stem,
            label=label,
            seeds=[1, 2, 3],
            outcomes=["success", "success", "crash"],
            success_times=[18.0, 28.0],
        )

    # Must NOT raise even though two regime subtrees are missing.
    regime_results = load_regime_results(str(results_root), world_stem, replan_k=DEFAULT_REPLAN_K)
    series = build_series(regime_results, algorithms="focus")
    by_label = {algo.label: algo for algo in series}
    assert label in by_label, f"{label} must still be present from the two written regimes"
    algo = by_label[label]

    # Only the two present regimes contribute points; the two absent ones are gaps.
    assert set(algo.regimes) == {"slow", "current"}, \
        f"only the present regimes must contribute points, got {algo.regimes}"
    assert algo.caps == (SPEED_REGIME_CAP["slow"], SPEED_REGIME_CAP["current"]), \
        f"caps must be the two present regimes' caps in order, got {algo.caps}"

    # And the charts still render from the present regimes.
    out_dir = tmp / "tc_s6_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    color_map = _stable_color_map(regime_results, plt)
    failure_png = chart_failure_rate_vs_cap(series, color_map, plt, out_dir)
    assert failure_png.is_file() and failure_png.stat().st_size > 0, \
        "the failure chart must still render from the present regimes"
    return "missing regime subtree -> gap (2/4 regimes); charts still render"


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
    ("TC-S2", _tc_s2_x_positions_are_caps),
    ("TC-S3", _tc_s3_focus_draws_only_present),
    ("TC-S4", _tc_s4_color_map_stable),
    ("TC-S5", _tc_s5_both_charts_render),
    ("TC-S6", _tc_s6_missing_regime_is_gap),
    ("TC-S7", _tc_s7_matplotlib_guard),
)


def run_selfcheck() -> int:
    """Run the sweep plotter's self-check suite (TC-S1..TC-S7). Return 0 if all pass, else 1.

    Builds every fixture inside a single TemporaryDirectory (each TC namespaces its
    own subdir), runs each TC in isolation so one failure never aborts the rest,
    prints a per-TC PASS/FAIL line, and ends with an "N/M passed" summary. No
    irsim, no real episodes — synthetic per-regime JSON trees only. Charts are
    rendered through the real chart functions under the headless Agg backend
    selected by `ensure_matplotlib()`.
    """
    import tempfile

    n_passed = 0
    n_total = len(_SELFCHECK_CASES)

    with tempfile.TemporaryDirectory(prefix="speed_sweep_selfcheck_") as tmp_name:
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
class SweepPlotArgs:
    """Parsed CLI arguments — frozen so accidental mutation is impossible."""

    world: str | None
    results_dir: str
    algorithms: str
    replan_k: int
    out_dir: str | None
    selfcheck: bool


def _parse_args(argv: list[str] | None) -> SweepPlotArgs:
    parser = argparse.ArgumentParser(
        prog="runners.plot_speed_sweep",
        description=(
            "Chart per-algorithm failure rate and median time-to-goal vs the "
            "obstacle-speed cap, reading the four speed-regime result subtrees (Phase 7)."
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
        "the plotter reads <root>/speed_<regime>/<world_stem>/<label>/.",
    )
    parser.add_argument(
        "--algorithms",
        choices=("focus", "all"),
        default="focus",
        help="Which algorithm set to draw: 'focus' (a_star_once, d_star_lite, dwa, apf) "
        "or 'all' (the 11 canonical planners). Default 'focus'.",
    )
    parser.add_argument(
        "--replan-k",
        type=int,
        default=DEFAULT_REPLAN_K,
        help=f"Cadence used to build the _replan label dirs (default {DEFAULT_REPLAN_K}).",
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
    if ns.replan_k < 1:
        parser.error(f"--replan-k must be >= 1, got {ns.replan_k}")
    return SweepPlotArgs(
        world=ns.world,
        results_dir=ns.results_dir,
        algorithms=ns.algorithms,
        replan_k=int(ns.replan_k),
        out_dir=ns.out_dir,
        selfcheck=bool(ns.selfcheck),
    )


def _resolve_out_dir(args: SweepPlotArgs, world_stem: str) -> Path:
    """Out-dir is the CLI override, else `<results-dir>/<world_stem>/speed_sweep_plots/`."""
    if args.out_dir is not None:
        return Path(args.out_dir).resolve()
    return Path(args.results_dir).resolve() / world_stem / PLOTS_DIR_NAME


def main(argv: list[str] | None = None) -> int:
    """Render the two sweep charts + summary CSV. See module docstring for CLI semantics."""
    args = _parse_args(argv)

    if args.selfcheck:
        # Selfcheck ignores --world entirely; run it BEFORE any --world validation
        # so the documented `python -m runners.plot_speed_sweep --selfcheck`
        # command (no --world) works (mirrors plot.py's main() gating).
        return run_selfcheck()

    # --world is only required for the normal plotting path. Validate it here
    # (after the selfcheck gate), mirroring plot.py's up-front exit code 2.
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

    regime_results = load_regime_results(
        args.results_dir,
        world_stem,
        replan_k=args.replan_k,
    )

    # "No readable data at all" => every algorithm came back empty in every
    # regime. Nothing to plot or summarize, so exit non-zero with a clear message.
    any_present = any(
        summary.n_present > 0
        for world in regime_results.values()
        for summary in world.summaries
    )
    if not any_present:
        root = Path(args.results_dir).resolve()
        regime_dirs = ", ".join(
            f"{SPEED_SUBTREE_PREFIX}{regime}/{world_stem}" for regime in REGIME_ORDER
        )
        print(
            f"error: nothing to plot - no readable episode JSONs under any of "
            f"{root} / {{{regime_dirs}}}",
            file=sys.stderr,
        )
        return 1

    series = build_series(regime_results, algorithms=args.algorithms)
    color_map = _stable_color_map(regime_results, plt)

    csv_path = out_dir / SUMMARY_CSV_NAME
    write_summary_csv(series, csv_path)
    print(f"wrote {csv_path}")

    failure_png = chart_failure_rate_vs_cap(series, color_map, plt, out_dir)
    print(f"wrote {failure_png}")
    median_png = chart_median_time_vs_cap(series, color_map, plt, out_dir)
    print(f"wrote {median_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
