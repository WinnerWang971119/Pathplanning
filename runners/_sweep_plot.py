"""Shared headless helpers for the read-only sweep plotters (speed + horizon).

`runners.plot_speed_sweep` and `runners.plot_horizon_sweep` both draw the same
shape of chart — one per-algorithm line over a swept x axis (the obstacle-speed cap
vs the prediction horizon) — and ran a near-identical headless self-check harness.
The genuinely shared pieces live here so a change to the line-chart render or the
self-check scaffold happens in ONE place.

What stays in each plotter: its own ``AlgoSeries`` dataclass (different fields), its
own loader (``load_regime_results`` vs ``load_horizon_results``), its own
``build_series``, its own ``write_summary_csv`` column set, its own ``_tc_s*``
fixtures, its own color map, and its own CLI ``main``. Only the axis-agnostic render
body, the NaN sentinel test, the synthetic metrics-record builder, the
matplotlib-absent guard TC, and the self-check loop are shared.

HEADLESS (load-bearing — gotcha: importing planners pulls irsim + matplotlib):
nothing here imports irsim, matplotlib, or `planners` at module top. The only
heavy-looking import is ``ensure_matplotlib`` from ``runners.plot``, which is itself
headless at import (it defers its planners/irsim and matplotlib imports into
functions); matplotlib is reached ONLY through that seam, at call time. So
``import runners.plot_speed_sweep`` / ``import runners.plot_horizon_sweep`` keep
neither irsim nor matplotlib in ``sys.modules`` after import.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Sequence

# Make the repo root importable so `from runners.plot import ...` resolves when a
# plotter imports this module from any cwd. MUST sit above the `runners.plot`
# import below (a `from runners.X import` over an unresolved repo root would fail
# with "runners is not a package"). Mirrors runners/plot_speed_sweep.py:62-64.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# `runners.plot` is headless at import (it defers its planners/irsim and matplotlib
# imports into functions), so importing `ensure_matplotlib` here does NOT pull
# irsim or matplotlib at module top — the headless boundary the plotters rely on
# holds.
from runners.plot import ensure_matplotlib  # noqa: E402


# Marker shape used for the per-algorithm line points (one shape; color carries the
# algorithm identity via each plotter's own color map).
LINE_MARKER = "o"


def is_nan(value: float) -> bool:
    """True iff `value` is NaN (the loader's "no data" sentinel for floats)."""
    return value != value


# --- Shared line-chart render -----------------------------------------------

def render_line_chart(
    series: Sequence[Any],
    color_map: dict[str, tuple],
    plt: Any,
    out_dir: Path,
    *,
    tick_positions: Sequence[float],
    tick_labels: Sequence[str],
    x_label: str,
    x_getter: Callable[[Any], Sequence[float]],
    color_key: Callable[[Any], str],
    value_attr: str,
    out_name: str,
    title: str,
    y_label: str,
    y_lim: tuple[float, float] | None,
) -> Path:
    """Render one per-algorithm line chart and save it under ``out_dir / out_name``.

    Shared by both sweep plotters; the axis-specific bits are injected:

    - ``tick_positions`` / ``tick_labels`` / ``x_label`` describe the swept x axis
      (speed cap vs horizon seconds),
    - ``x_getter(algo)`` pulls that algorithm's per-point x values (caps vs seconds),
    - ``color_key(algo)`` pulls the algorithm's color-map key (label vs registry name).

    ``value_attr`` selects the y series on each series object (``"failure_rates"`` or
    ``"median_times"``). Points whose y value is NaN are skipped (a position where the
    algorithm was present but never succeeded has a NaN median time), so a line is
    drawn only through the x positions that have a finite value. A side legend maps
    each line color to its ``algo.display`` name, mirroring plot.py's A1 legend
    placement; when no series is drawable the chart is annotated "no data to plot"
    so the PNG is not blank.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    drawn_handles = []
    for algo in series:
        color = color_map.get(color_key(algo), "#333333")
        values = getattr(algo, value_attr)

        # Keep only the finite (x, value) points; a NaN value (e.g. a NaN median
        # time at a present-but-0-success x position) is a gap in THIS chart's line.
        xs: list[float] = []
        ys: list[float] = []
        for x_value, value in zip(x_getter(algo), values):
            if is_nan(value):
                continue
            xs.append(x_value)
            ys.append(value)

        if not xs:
            # No finite points for this chart (e.g. an algo that never succeeded in
            # any position on the median-time chart): omit its line entirely.
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

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(list(tick_positions))
    ax.set_xticklabels(list(tick_labels), fontsize=9)
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


# --- Shared self-check scaffolding ------------------------------------------

def make_selfcheck_record(
    outcome: str,
    *,
    time_to_goal: float | None = None,
) -> dict:
    """Build one synthetic 7-key metrics record for the given outcome.

    Mirrors `runners.plot._make_record` but kept here so the sweep self-checks stay
    self-contained (and so a future plot.py refactor cannot silently break them). A
    "success" record carries a non-null `time_to_goal`; every other outcome sets its
    flag (or `planner_error` string) with `time_to_goal` null.
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


def tc_matplotlib_guard(_tmp: Path) -> str:
    """Shared TC: the matplotlib-absent guard exits non-zero (patch find_spec, like plot.py's TC-P8)."""
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


def run_selfcheck_suite(
    cases: Sequence[tuple[str, Callable[[Path], str]]],
    *,
    tempdir_prefix: str,
) -> int:
    """Run a sweep plotter's self-check suite. Return 0 if all pass, else 1.

    Builds every fixture inside a single TemporaryDirectory (each TC namespaces its
    own subdir), runs each TC in isolation so one failure never aborts the rest,
    prints a per-TC PASS/FAIL line, and ends with an "N/M passed" summary. No irsim,
    no real episodes — synthetic JSON trees only. Charts are rendered through the
    real chart functions under the headless Agg backend selected by
    `ensure_matplotlib()`.
    """
    import tempfile

    n_passed = 0
    n_total = len(cases)

    with tempfile.TemporaryDirectory(prefix=tempdir_prefix) as tmp_name:
        tmp = Path(tmp_name)
        for tc_id, tc_func in cases:
            try:
                detail = tc_func(tmp)
            except Exception as exc:  # noqa: BLE001 — one TC must not abort the suite
                print(f"{tc_id}: FAIL - {type(exc).__name__}: {exc}")
            else:
                n_passed += 1
                print(f"{tc_id}: PASS - {detail}")

    print(f"selfcheck: {n_passed}/{n_total} passed")
    return 0 if n_passed == n_total else 1
