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
        [--filter|--no-filter]  # apply the degenerate-seed sidecar if fresh (default on; issue #9)
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
from dataclasses import dataclass, field, replace
from pathlib import Path

# Make the repo root importable so `from planners import algorithm_label` resolves
# when this module is invoked as `python -m runners.plot` from any cwd. Mirrors
# runners/run_episode.py:54-58 and runners/run_experiment.py:69-74.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The degenerate-seed filter core (issue #9). This module is STDLIB-ONLY — it
# imports nothing from `planners`/irsim/matplotlib — so importing it at module
# top keeps `import runners.plot` headless (AC12, verified by the AC12 subprocess
# check). These five names are the whole read-side contract: read the sidecar,
# verify it is fresh + covers the plotted labels, know its filename, and quote the
# criterion id in the A1 footnote.
from runners._seed_filter import (  # noqa: E402 — must follow the sys.path shim above
    CRITERION_ID,
    SEED_FILTER_NAME,
    read_seed_filter,
    sidecar_covers,
    sidecar_is_fresh,
)

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

# Outcome buckets. The first four are the per-record buckets `classify_outcome`
# returns (a JSON record is exactly one of those). "dnf" is a fifth FAILURE
# subtype that has NO record at all — it is detected at the manifest level (an
# episode the batch runner killed at the 600s wallclock wall, recorded as
# `status="runner_error"` with no `<seed>.json`), never from a record. It is
# listed last so the A3 stack draws it on top of the other failures.
OUTCOMES = ("success", "crash", "timeout", "planner_error", "dnf")

# Euclidean (2,2) -> (48,48) straight-line distance: the unreachable lower bound
# every executed path is compared against (the robot must detour around walls).
STRAIGHT_LINE_IDEAL_M = 46.0 * (2.0 ** 0.5)   # ~= 65.05 m

# The algorithm set charted on the scatter, in legend order. Each tuple is
# (registry name, replan_k or None, predict_horizon or None, family, display).
# load_world_results() folds the replan_k / predict_horizon into the dir label
# (a_star_replan_k5 / d_star_lite_predictive_h10) so the label matches the dir
# run_experiment wrote. These are exactly the 13 canonical Mission planners run_all
# writes — the same set, and the same order, as run_all._CANONICAL_ORDER. The two
# canonical predictive keys — d_star_lite_predictive (grid-stamping) and
# dwa_predictive_paper (the space-time braking-only policy) — fold their canonical
# horizon (h10) into the label. The experimental motion-aware keys
# (d_star_lite_oracle, dwa_predictive, dwa_predictive_oracle,
# dwa_predictive_paper_oracle) are EXCLUDED here on purpose: they are held out of
# run_all's canonical set (EXPERIMENTAL_KEYS), so their `_h<steps>` label dirs are
# never written by the documented run_all -> plot workflow. Charting them here would
# draw a data-less line on every chart (the headline A1 scatter included). The
# oracle/global-guidance results live on the horizon-sweep charts
# (`runners.plot_horizon_sweep`), their proper experimental venue.
CANONICAL: list[tuple[str, int | None, int | None, str, str]] = [
    ("a_star_once",            None, None, "grid",        "A* once"),
    ("a_star_replan",          5,    None, "grid",        "A* replan (K=5)"),
    ("dijkstra_once",          None, None, "grid",        "Dijkstra once"),
    ("dijkstra_replan",        5,    None, "grid",        "Dijkstra replan (K=5)"),
    ("d_star_lite",            None, None, "incremental", "D* Lite"),
    ("d_star_lite_predictive", None, 10,   "predictive",  "D* Lite predictive (h10)"),
    ("dwa",                    None, None, "reactive",    "DWA"),
    ("dwa_predictive_paper",   None, 10,   "predictive",  "DWA predictive (h10)"),
    ("apf",                    None, None, "reactive",    "APF"),
    ("rrt_once",               None, None, "sampling",    "RRT once"),
    ("rrt_replan",             5,    None, "sampling",    "RRT replan (K=5)"),
    ("rrt_star_once",          None, None, "sampling",    "RRT* once"),
    ("rrt_star_replan",        5,    None, "sampling",    "RRT* replan (K=5)"),
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

    Degenerate-seed filter (issue #9): `n_present`/`n_*`/`failure_rate` are the
    HEADLINE (filtered) values — every dropped degenerate seed this algorithm
    saw is removed (a degenerate seed is a crash, so each drop removes 1 present
    + 1 crash). `n_present_raw`/`failure_rate_raw` keep the pre-drop values for
    the summary.csv audit trail. When nothing is dropped they are equal.
    `per_seed` stays COMPLETE so B1 still renders the dropped seeds.
    """

    label: str                          # results dir label, e.g. "a_star_replan_k5"
    display: str                        # legend name, e.g. "A* replan (K=5)"
    family: str                         # "grid" | "incremental" | "reactive" | "sampling"
    n_present: int                      # FILTERED roster size (raw minus this algo's dropped seeds); the chart denominator
    n_success: int                      # counts below are FILTERED (dropped seeds removed; a degenerate seed is always a crash)
    n_crash: int
    n_timeout: int
    n_planner_error: int
    n_dnf: int                          # episodes the batch killed at the wallclock wall (manifest status=runner_error, no JSON)
    failure_rate: float                 # FILTERED (crash+timeout+planner_error+dnf)/n_present; NaN if n_present == 0
    n_present_raw: int                  # pre-drop present count (== n_present when nothing dropped); summary.csv audit column
    failure_rate_raw: float             # pre-drop failure rate over n_present_raw; NaN if n_present_raw == 0
    times: tuple[float, ...]            # successful time_to_goal values (unfiltered — dropped seeds are crashes, never successes)
    path_lengths: tuple[float, ...]     # path_length over successes (unfiltered, same reason)
    wallclocks: tuple[float, ...]       # wallclock_per_step from the __wallclock__ subtree (empty if absent)
    per_seed: dict[int, str]            # seed -> outcome, COMPLETE incl. dropped seeds (for the B1 heatmap)
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

    Degenerate-seed filter (issue #9): `degenerate_seeds` is the dropped set
    intersected with `seed_order` (excluded from every numeric stat but still
    drawn, marked, on B1); `n_dropped` is its length; `filter_status` is how the
    drop was decided ("applied"/"stale"/"indeterminate"/"absent"/"off"), used by
    the A1 footnote; `filter_missing_labels` names the required labels the sidecar
    could not find (the A1 footnote's "missing:" list in the indeterminate case).
    The loader fills `degenerate_seeds`/`n_dropped` from its `dropped_seeds`
    argument; `main` sets `filter_status`/`filter_missing_labels` from the sidecar
    checks (via `dataclasses.replace`). The three status fields default to the
    unfiltered values so the sweep plotters, which never filter, are unaffected.
    """

    summaries: tuple[AlgoSummary, ...]
    seed_order: tuple[int, ...]
    manifest_seed_order: bool           # True iff seed_order came from a manifest's derived_seeds
    degenerate_seeds: tuple[int, ...] = ()   # dropped seeds ∩ seed_order (issue #9); () when unfiltered
    filter_status: str = "off"               # "applied"|"stale"|"indeterminate"|"absent"|"off"
    n_dropped: int = 0                       # len(degenerate_seeds)
    filter_missing_labels: tuple[str, ...] = ()  # sidecar's missing_labels (A1 footnote, indeterminate case)


# --- Loader (AC2 / AC3 / AC9 / AC11) ----------------------------------------

@dataclass
class _AlgoAccumulator:
    """Mutable scratch tally for one algorithm while scanning its JSONs."""

    n_present: int = 0
    n_success: int = 0
    n_crash: int = 0
    n_timeout: int = 0
    n_planner_error: int = 0
    n_dnf: int = 0
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


def _load_manifest(label_dir: Path) -> dict | None:
    """Return this label dir's parsed `_manifest.json`, or None if absent/unreadable."""
    manifest_path = label_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    return _read_json(manifest_path)


def _manifest_seed_order_from(manifest: dict | None, label_dir: Path) -> tuple[int, ...] | None:
    """Return `derived_seeds` from an already-parsed manifest, or None if absent/unusable."""
    if manifest is None:
        return None
    derived = manifest.get("derived_seeds")
    if not isinstance(derived, list) or not derived:
        return None
    try:
        return tuple(int(seed) for seed in derived)
    except (TypeError, ValueError):
        print(f"warning: ignoring malformed derived_seeds in {label_dir / MANIFEST_NAME}", file=sys.stderr)
        return None


def _manifest_episodes(manifest: dict | None) -> list[dict] | None:
    """Return the manifest's `episodes` roster list, or None if absent/not a list.

    None signals "no usable roster" so the loader falls back to globbing present
    JSONs (synthetic fixtures and single-episode dirs have no `episodes` key). An
    EMPTY list is treated as "no roster" too (there is nothing to drive the
    roster path with), so the fallback still runs.
    """
    if manifest is None:
        return None
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        return None
    return episodes


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


def _accumulate_dnf(acc: _AlgoAccumulator, seed: int) -> None:
    """Fold one did-not-finish (wallclock-killed) episode into the accumulator.

    A DNF has no `<seed>.json` to classify — it is a roster-level failure subtype
    (manifest `status="runner_error"`). It counts toward `n_present` (so the
    denominator stays at the full seed count) and toward `n_dnf`, and marks its
    seed "dnf" in `per_seed` so the B1 heatmap colors it.
    """
    acc.n_present += 1
    acc.n_dnf += 1
    acc.per_seed[seed] = "dnf"


def _finalize_summary(
    *,
    label: str,
    display: str,
    family: str,
    acc: _AlgoAccumulator,
    wallclocks: tuple[float, ...],
    wallclock_from_subtree: bool,
    dropped_seeds: frozenset[int] = frozenset(),
) -> AlgoSummary:
    """Freeze an accumulator into an immutable AlgoSummary with the derived stats.

    `failure_rate`, `median_time`, and `mean_time` are NaN when their
    denominator is zero (n_present == 0 / n_success == 0). NaN is the chosen
    "no data" sentinel everywhere in this module so the dtype stays float and
    downstream charts can drop NaNs uniformly.

    `dropped_seeds` is the degenerate-seed set (issue #9). The RAW counts are the
    pre-drop tally (`n_present_raw`/`failure_rate_raw`); the headline
    `n_present`/`n_*`/`failure_rate` fields are the FILTERED tally with every
    dropped seed this accumulator actually saw removed. A degenerate seed is a
    crash, so each drop removes 1 present + 1 crash (i.e. 1 present + 1 failure);
    the general per-outcome decrement below is defensive. `per_seed`, `times`, and
    `path_lengths` stay COMPLETE (B1 still renders the dropped seeds; dropped seeds
    are never successes so the success stats never move). With `dropped_seeds`
    empty every filtered value equals its raw value (AC13) — which is exactly how
    the sweep plotters, which call this without the argument, keep working.
    """
    # Raw (pre-drop) tally — the summary.csv audit columns.
    n_present_raw = acc.n_present
    n_failed_raw = acc.n_crash + acc.n_timeout + acc.n_planner_error + acc.n_dnf
    failure_rate_raw = (n_failed_raw / n_present_raw) if n_present_raw > 0 else float("nan")

    # Filtered tally: drop each degenerate seed this algorithm actually has a
    # per-seed outcome for. `per_seed` only carries PRESENT seeds, so a dropped
    # seed absent from this algorithm (partial data) is a natural no-op here.
    n_present = n_present_raw
    n_success = acc.n_success
    n_crash = acc.n_crash
    n_timeout = acc.n_timeout
    n_planner_error = acc.n_planner_error
    n_dnf = acc.n_dnf
    for seed in dropped_seeds:
        outcome = acc.per_seed.get(seed)
        if outcome is None:
            continue
        n_present -= 1
        if outcome == "success":
            n_success -= 1
        elif outcome == "crash":
            n_crash -= 1
        elif outcome == "timeout":
            n_timeout -= 1
        elif outcome == "planner_error":
            n_planner_error -= 1
        elif outcome == "dnf":
            n_dnf -= 1

    n_failed = n_crash + n_timeout + n_planner_error + n_dnf
    failure_rate = (n_failed / n_present) if n_present > 0 else float("nan")
    median_time = statistics.median(acc.times) if acc.times else float("nan")
    mean_time = statistics.fmean(acc.times) if acc.times else float("nan")
    return AlgoSummary(
        label=label,
        display=display,
        family=family,
        n_present=n_present,
        n_success=n_success,
        n_crash=n_crash,
        n_timeout=n_timeout,
        n_planner_error=n_planner_error,
        n_dnf=n_dnf,
        failure_rate=failure_rate,
        n_present_raw=n_present_raw,
        failure_rate_raw=failure_rate_raw,
        times=tuple(acc.times),
        path_lengths=tuple(acc.path_lengths),
        wallclocks=wallclocks,
        per_seed=dict(acc.per_seed),
        median_time=median_time,
        mean_time=mean_time,
        wallclock_from_subtree=wallclock_from_subtree,
    )


def _accumulate_from_glob(acc: _AlgoAccumulator, label_dir: Path, seen_seeds: set[int]) -> None:
    """Tally one label dir by globbing its present numeric-stem JSONs (the fallback path).

    This is the pre-DNF behavior: every readable `<seed>.json` is classified and
    folded; `n_present` ends up equal to the JSON count and there are no DNFs.
    Used when a label has no manifest with an `episodes` roster (synthetic
    fixtures, single-episode dirs, manifest-less trees).
    """
    for json_path in sorted(label_dir.glob(EPISODE_GLOB)):
        seed = _seed_from_stem(json_path)
        if seed is None:
            continue
        rec = _read_json(json_path)
        if rec is None:
            continue
        _accumulate_episode(acc, seed, rec)
        seen_seeds.add(seed)


def _accumulate_from_roster(
    acc: _AlgoAccumulator,
    label_dir: Path,
    episodes: list[dict],
    label: str,
    seen_seeds: set[int],
) -> None:
    """Tally one label dir using its manifest `episodes` list as the authoritative roster.

    Per episode entry (`{"seed", "exit_code", "status"}`):
      - "ok"           -> read `<seed>.json` and classify it (the normal path). If
                          the JSON is unexpectedly missing or malformed, warn and
                          treat the episode as a DNF (defensive — an "ok" status
                          promises a JSON, so its absence is itself a failure).
      - "runner_error" -> a wallclock-killed episode with NO JSON: count as "dnf".
      - "skipped"      -> a `--resume` no-op. Count it ONLY if its JSON exists on
                          disk (resume case: the data is from an earlier run);
                          otherwise drop it from the roster entirely (it was never
                          actually run, so it must not inflate the denominator).

    `n_present` (the denominator) thus equals ok + runner_error (+ any skipped
    that carried data), keeping every planner's denominator at the full seed
    count whether or not its slow seeds finished.

    Episodes are processed in lexicographic `"<seed>.json"` order (the SAME order
    the glob path scans files), so the success `times` list stays append-aligned
    with `_success_times_by_seed`'s filename-sorted reconstruction — the roster's
    own derivation order would otherwise desync that seed->time recovery.
    """
    # Pre-parse seeds so a non-integer / malformed entry is reported once, then
    # process the survivors in lexicographic filename order (glob-path parity).
    parsed: list[tuple[int, str | None]] = []
    for episode in episodes:
        if not isinstance(episode, dict):
            print(f"warning: skipping malformed episode entry in {label} roster: {episode!r}", file=sys.stderr)
            continue
        raw_seed = episode.get("seed")
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError):
            print(f"warning: skipping roster entry with non-integer seed in {label}: {raw_seed!r}", file=sys.stderr)
            continue
        status = episode.get("status")
        parsed.append((seed, status if isinstance(status, str) else None))

    for seed, status in sorted(parsed, key=lambda item: f"{item[0]}.json"):
        if status == "runner_error":
            _accumulate_dnf(acc, seed)
            seen_seeds.add(seed)
            continue

        if status == "skipped":
            # A skipped seed only contributes if it left data behind (resume);
            # a skipped-without-data seed never ran, so it leaves the roster.
            json_path = label_dir / f"{seed}.json"
            if not json_path.is_file():
                continue
            rec = _read_json(json_path)
            if rec is None:
                continue
            _accumulate_episode(acc, seed, rec)
            seen_seeds.add(seed)
            continue

        if status == "ok":
            json_path = label_dir / f"{seed}.json"
            rec = _read_json(json_path) if json_path.is_file() else None
            if rec is None:
                # An "ok" status promises a readable JSON; its absence/corruption
                # is itself a did-not-finish (defensive — should not happen).
                print(
                    f"warning: {label} seed {seed} is status=ok but its JSON is "
                    f"missing/unreadable; counting it as DNF",
                    file=sys.stderr,
                )
                _accumulate_dnf(acc, seed)
                seen_seeds.add(seed)
                continue
            _accumulate_episode(acc, seed, rec)
            seen_seeds.add(seed)
            continue

        # An unknown status string is treated like a skipped-without-data seed:
        # not counted, but warned so a schema drift is visible.
        print(
            f"warning: {label} seed {seed} has unknown manifest status {status!r}; "
            f"dropping it from the roster",
            file=sys.stderr,
        )


def load_world_results(
    results_dir: str,
    world_stem: str,
    *,
    replan_k: int = DEFAULT_REPLAN_K,
    expected: int = DEFAULT_EXPECTED_SEEDS,
    dropped_seeds: frozenset[int] = frozenset(),
) -> WorldResults:
    """Load every canonical algorithm's episodes for one world into summaries.

    For each CANONICAL entry the dir label is recomputed via
    `algorithm_label(name, replan_k or None)` (so the CLI's `--replan-k` picks
    the `a_star_replan_k<K>` dirs). How that label dir is tallied depends on
    whether its `_manifest.json` carries an `episodes` roster:

      - WITH a roster: the manifest is authoritative. Every roster episode is
        counted so the denominator stays at the full seed count — an "ok" seed
        reads + classifies its `<seed>.json`, a "runner_error" seed (killed at
        the batch wallclock wall, no JSON) counts as a "dnf" failure, and a
        "skipped" seed counts only if its JSON exists (resume). See
        `_accumulate_from_roster`.
      - WITHOUT a roster (synthetic fixtures, single-episode dirs, manifest-less
        trees): the pre-DNF fallback — glob the present numeric-stem JSONs,
        `n_present` == JSON count, zero DNF. See `_accumulate_from_glob`.

    `_manifest.json` and any non-numeric-stem file are skipped by the glob path.
    A missing label dir, a short count (< `expected`), or an unreadable file
    warns to stderr and is skipped — the loader NEVER raises (AC11).

    The B1 heatmap's seed-column order comes from the first manifest found (any
    label's `derived_seeds`); absent any manifest it falls back to the sorted
    union of every numeric stem present.

    `dropped_seeds` (issue #9) is the degenerate-seed set to exclude from the
    FILTERED per-algorithm counts (`n_present`/`failure_rate`/`n_*`); it is passed
    straight through to `_finalize_summary`. `WorldResults.degenerate_seeds` is set
    to `dropped_seeds ∩ seed_order` (and `n_dropped` to its length); `per_seed`
    stays complete so B1 still marks those seeds. `main` decides `dropped_seeds`
    from the sidecar and later stamps `filter_status`/`filter_missing_labels` onto
    the returned `WorldResults` via `dataclasses.replace`. The default empty set is
    a true no-op (AC13), so callers that never filter get the raw numbers.
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

    for name, default_k, default_h, family, display in CANONICAL:
        # A replan family takes the CLI cadence; a predict family folds in its
        # horizon (default_h); everything else stays bare. algorithm_label handles
        # all three from the (replan_k, predict_horizon) pair.
        effective_k = replan_k if default_k is not None else None
        label = algorithm_label(name, effective_k, default_h)
        label_dir = episode_out_dir(results_root, world_stem, label)

        acc = _AlgoAccumulator()

        if not label_dir.is_dir():
            print(
                f"warning: no result dir for {label} at {label_dir} (skipping)",
                file=sys.stderr,
            )
        else:
            manifest = _load_manifest(label_dir)

            # The first manifest we encounter fixes the canonical seed-column order
            # for the whole world (all manifests share the same derived_seeds).
            if manifest_seed_order is None:
                manifest_seed_order = _manifest_seed_order_from(manifest, label_dir)

            # When the manifest carries an `episodes` roster it is authoritative
            # (counts DNF seeds and keeps the denominator at the full seed count);
            # otherwise fall back to the pre-DNF glob behavior exactly.
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

        wallclocks, wallclock_from_subtree = _load_wallclocks(results_dir, world_stem, label)

        summaries.append(
            _finalize_summary(
                label=label,
                display=display,
                family=family,
                acc=acc,
                wallclocks=wallclocks,
                wallclock_from_subtree=wallclock_from_subtree,
                dropped_seeds=dropped_seeds,
            )
        )

    if manifest_seed_order is not None:
        seed_order = manifest_seed_order
        from_manifest = True
    else:
        seed_order = tuple(sorted(seen_seeds))
        from_manifest = False

    # Degenerate seeds actually present in this world's shared stream (issue #9).
    degenerate_seeds = tuple(sorted(set(dropped_seeds) & set(seed_order)))

    return WorldResults(
        summaries=tuple(summaries),
        seed_order=seed_order,
        manifest_seed_order=from_manifest,
        degenerate_seeds=degenerate_seeds,
        n_dropped=len(degenerate_seeds),
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
    "n_dnf",
    "failure_rate",
    "n_present_raw",
    "failure_rate_raw",
    "n_dropped",
    "median_time",
    "mean_time",
)


def write_summary_csv(summaries: tuple[AlgoSummary, ...] | list[AlgoSummary], out_path: str | Path) -> None:
    """Write the per-algorithm tally as a CSV with SUMMARY_CSV_COLUMNS.

    One row per algorithm, in CANONICAL order. NaN floats (no episodes / no
    successes) are written as the literal "nan" by csv, which is the documented
    "no data" marker. The degenerate-seed filter (issue #9) adds three audit
    columns after `failure_rate`: `n_present_raw`/`failure_rate_raw` (the pre-drop
    values) and per-row `n_dropped` (this algorithm's dropped count, i.e.
    `n_present_raw - n_present`, so `n_present + n_dropped == n_present_raw`).
    Without a filter they are the raw values with `n_dropped == 0` (AC9/AC13).
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
                    summary.n_dnf,
                    summary.failure_rate,
                    summary.n_present_raw,
                    summary.failure_rate_raw,
                    summary.n_present_raw - summary.n_present,
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
    "dnf": "#000000",            # black — wallclock-killed; distinct from crash-red/timeout-orange/error-grey
}

# Human-readable legend labels for the outcome buckets (A3 legend).
OUTCOME_DISPLAY = {
    "success": "success",
    "crash": "crash",
    "timeout": "timeout",
    "planner_error": "planner error",
    "dnf": "DNF (wallclock)",
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
    `summaries`), which keeps the 12 algorithms visually distinct and groups
    neighbours (the families are listed contiguously in CANONICAL).
    """
    cmap = plt.get_cmap("tab20")
    color_map: dict[str, tuple] = {}
    for index, summary in enumerate(summaries):
        # tab20 has 20 discrete entries; modulo keeps it safe if the canonical
        # set ever grows past 20.
        color_map[summary.label] = cmap(index % cmap.N)
    return color_map


def _filter_status_footnote(results: WorldResults) -> str:
    """One-line A1 footnote describing how the degenerate-seed filter (issue #9)
    was applied. Pinned to A1 only (the headline scatter), never the other
    figures. Derived purely from `WorldResults` so the chart signature stays
    `(results, plt, out_dir)`. Never raises.
    """
    status = results.filter_status
    if status == "applied":
        return (
            f"filter: applied - {results.n_dropped} degenerate seed(s) dropped "
            f"(criterion {CRITERION_ID}); failure rates are CONDITIONAL on "
            f"non-degenerate seeds; raw full-set rates in summary.csv (failure_rate_raw)"
        )
    if status == "stale":
        return (
            "filter: STALE, ignored - re-run runners.filter_seeds; "
            "showing unfiltered full-set failure rates"
        )
    if status == "indeterminate":
        missing = ", ".join(results.filter_missing_labels) or "(unnamed)"
        return (
            "filter present but indeterminate - nothing dropped; "
            f"missing: {missing}"
        )
    if status == "off":
        return "filter: off - showing unfiltered full-set failure rates"
    # "absent" (and any unexpected value) — no sidecar in play.
    return "filter: absent - showing unfiltered full-set failure rates"


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

    When the degenerate-seed filter (issue #9) is applied, the plotted
    `failure_rate` is CONDITIONAL (degenerate seeds excluded); a bottom-left
    figure footnote states that, the drop count + criterion, and where the raw
    full-set rates live (summary.csv). The footnote is on A1 only.
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
    centroid_legend = ax.legend(
        handles=shape_handles,
        title="centroid",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        fontsize=8,
        title_fontsize=9,
        borderaxespad=0.0,
    )

    # Degenerate-seed filter status footnote (issue #9) — pinned to A1 only. It
    # states the drop count + criterion (or the stale/absent/off/indeterminate
    # reason) and flags that the plotted rates are conditional when applied.
    fig.text(
        0.01,
        0.005,
        _filter_status_footnote(results),
        fontsize=8,
        color="#555555",
        ha="left",
        va="bottom",
        style="italic",
    )

    fig.tight_layout()
    out_path = out_dir / "a1_scatter.png"
    # Both legends sit OUTSIDE the axes (bbox_to_anchor), and bbox_inches="tight"
    # alone does not reliably include out-of-axes artists — pass them explicitly
    # so the longest legend entries are not clipped at the right edge.
    fig.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
        bbox_extra_artists=(algo_legend, centroid_legend),
    )
    plt.close(fig)
    return out_path


def _chart_a3(results: WorldResults, plt, out_dir: Path) -> Path:
    """A3 — per-algorithm failure-breakdown stacked bars (AC5).

    One stacked bar per algorithm (CANONICAL order). Segments are the COUNTS of
    success / crash / timeout / planner_error / dnf, summing to n_present, with
    fixed per-outcome colors and an outcome legend. DNF (wallclock-killed) is its
    own black segment on top of the other failures. An algorithm whose n_present
    differs from the expected 50 (partial data) is annotated above its bar.
    """
    summaries = results.summaries
    n_algos = len(summaries)
    x_positions = list(range(n_algos))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Per-outcome count series, one list aligned to x_positions. Keys cover every
    # OUTCOMES bucket so the `for outcome in OUTCOMES` stack below never misses one.
    counts_by_outcome = {
        "success": [s.n_success for s in summaries],
        "crash": [s.n_crash for s in summaries],
        "timeout": [s.n_timeout for s in summaries],
        "planner_error": [s.n_planner_error for s in summaries],
        "dnf": [s.n_dnf for s in summaries],
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
    outcome_legend = ax.legend(title="outcome", loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, title_fontsize=9)

    fig.tight_layout()
    out_path = out_dir / "a3_failure_bars.png"
    # The outcome legend sits OUTSIDE the axes; pass it to savefig so the tight
    # bbox includes it (bbox_inches="tight" alone may clip out-of-axes artists).
    fig.savefig(out_path, dpi=150, bbox_inches="tight", bbox_extra_artists=(outcome_legend,))
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
# is the continuous-cmap layer, so it is excluded here). DNF is included so a
# wallclock-killed seed colors its cell like the other failure types.
_B1_FAILURE_OUTCOMES = ("crash", "timeout", "planner_error", "dnf")


def _chart_b1(results: WorldResults, plt, out_dir: Path) -> Path:
    """B1 — seed-difficulty heatmap: rows = algorithms (CANONICAL order), columns = the shared seed stream (AC7).

    Every row aligns to the same `results.seed_order` column order (a manifest's
    `derived_seeds`, else sorted stems), so reading down a column exposes
    universally-hard seeds. A SUCCESS cell is shaded by its time_to_goal on a
    continuous viridis colormap (colorbar "time to goal (s)"); a FAILURE cell is a
    flat categorical color per type (crash / timeout / planner_error / dnf,
    reusing OUTCOME_COLORS for parity with A3); an absent cell (no entry for that
    seed) keeps a neutral background. Never raises on missing seeds or 0-success
    rows.
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
    from matplotlib.patches import Patch, Rectangle

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

    # Degenerate-seed columns (issue #9): the filter culled these seeds from every
    # numeric stat, but B1 still renders them (their real per_seed outcome shows
    # through). Mark the whole column with a hatched, black-edged full-height
    # overlay so a culled seed reads as visually distinct from a counted crash.
    degenerate_cols = [
        col_of_seed[seed] for seed in results.degenerate_seeds if seed in col_of_seed
    ]
    for col in degenerate_cols:
        ax.add_patch(
            Rectangle(
                (col - 0.5, -0.5),
                1.0,
                max(n_rows, 1),
                facecolor="none",
                edgecolor="#000000",
                hatch="////",
                linewidth=1.0,
                zorder=4,
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
    if results.degenerate_seeds:
        # Hatched swatch matching the full-height degenerate-column overlay above.
        failure_handles.append(
            Patch(
                facecolor="none",
                edgecolor="#000000",
                hatch="////",
                label="degenerate (filtered)",
            )
        )
    failure_legend = ax.legend(
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
    # The failure/absent legend sits OUTSIDE the axes; pass it to savefig so the
    # tight bbox includes it (bbox_inches="tight" alone may clip out-of-axes artists).
    fig.savefig(out_path, dpi=150, bbox_inches="tight", bbox_extra_artists=(failure_legend,))
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


# --- Self-check (T5 + issue #9) ---------------------------------------------
#
# The suite below runs TC-P1..TC-P16 against synthetic result trees built in a
# TemporaryDirectory — no irsim, no real episodes. Each TC is a plain function
# that asserts its invariants; `_run_selfcheck_suite` catches per-TC exceptions
# so one failure never aborts the rest, prints a PASS/FAIL line each, and returns
# an int exit code (0 = all passed, 1 = any failed). The fixture helper writes
# `<seed>.json` records with the 7 metric fields plus a `_manifest.json` carrying
# `derived_seeds` (and, for the DNF roster TC, an `episodes` list), and is
# parametrizable so each TC can inject the edge cases (0-success algos, short
# seed counts, malformed JSON, decoy files, a `__wallclock__` subtree, a
# wallclock-killed roster). TC-P12..TC-P16 (issue #9) additionally write a
# `_seed_filter.json` sidecar via `_write_seed_filter_sidecar` and assert the
# drop/stale/absent/off/indeterminate paths through the data layer (reconstructed
# summaries + `WorldResults.degenerate_seeds`), never by inspecting a PNG. Charts
# are rendered through the real chart functions under the headless Agg backend
# selected by `ensure_matplotlib()`.

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
    """Write a 12-label fixture covering every CANONICAL algorithm.

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
    for index, (name, default_k, default_h, _family, _display) in enumerate(CANONICAL):
        effective_k = replan_k if default_k is not None else None
        label = algorithm_label(name, effective_k, default_h)

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


def _build_filter_tree(
    results_root: Path,
    world_stem: str,
    *,
    seeds: list[int],
    degenerate_seed: int,
    replan_k: int = DEFAULT_REPLAN_K,
) -> list[str]:
    """Write the 12 canonical label dirs for the degenerate-seed filter TCs.

    `degenerate_seed` CRASHES in every algorithm (so dropping it removes exactly
    1 present + 1 crash from each), every other seed SUCCEEDS, and each dir's
    `_manifest.json` carries `derived_seeds == seeds`. Returns the 12 labels in
    CANONICAL order (also the plotter's `_plotted_labels(replan_k)` output).
    """
    from planners import algorithm_label

    labels: list[str] = []
    for name, default_k, default_h, _family, _display in CANONICAL:
        effective_k = replan_k if default_k is not None else None
        label = algorithm_label(name, effective_k, default_h)
        labels.append(label)
        outcomes = ["crash" if seed == degenerate_seed else "success" for seed in seeds]
        _write_algo_tree(
            results_root=results_root,
            world_stem=world_stem,
            label=label,
            seeds=list(seeds),
            outcomes=outcomes,
            derived_seeds=list(seeds),
        )
    return labels


def _write_seed_filter_sidecar(
    *,
    results_root: Path,
    world_stem: str,
    required_labels: list[str],
    seed_order: list[int],
    dropped_seeds: list[int],
    consulted: list[tuple[str, int]],
    global_status: str = "ok",
    missing_labels: list[str] | None = None,
    absent_files: list[str] | None = None,
    replan_k: int = DEFAULT_REPLAN_K,
    predict_horizon: int = 10,
) -> Path:
    """Write a `_seed_filter.json` sidecar at `<results_root>/<world_stem>/`.

    Hashes each `consulted` (label, seed) metrics file AS IT EXISTS NOW (so a
    later mutation trips `sidecar_is_fresh`), records `absent_files` verbatim
    (they must genuinely be absent for the sidecar to read fresh), and uses
    `seed_order` as the recorded roster (which must equal the first canonical
    label's manifest `derived_seeds`). Returns the sidecar path. The write-side
    `_seed_filter` symbols are imported here (test-only) rather than at module top
    to keep the runtime top-level import to the read-side contract.
    """
    from runners._seed_filter import (
        SCHEMA_VERSION,
        PlannerEvidence,
        SeedFilter,
        SeedVerdictRow,
        file_sha256,
        write_seed_filter,
    )

    world_dir = results_root / world_stem
    consulted_hashes: dict[str, str] = {}
    for label, seed in consulted:
        digest = file_sha256(world_dir / label / f"{seed}.json")
        assert digest is not None, f"consulted fixture file {label}/{seed}.json must exist"
        consulted_hashes[f"{label}/{seed}.json"] = digest

    dropped = set(dropped_seeds)
    rows = tuple(
        SeedVerdictRow(
            seed=seed,
            verdict="degenerate" if seed in dropped else "kept",
            planners=(PlannerEvidence(label=required_labels[0], status="instant_crash" if seed in dropped else "survived", crash_step=1 if seed in dropped else None),),
        )
        for seed in seed_order
    )

    obj = SeedFilter(
        schema_version=SCHEMA_VERSION,
        criterion_id=CRITERION_ID,
        world_stem=world_stem,
        window_seconds=4.0,
        step_time=0.1,
        window_steps=40,
        replan_k=replan_k,
        predict_horizon=predict_horizon,
        traffic=True,
        speed_regime="current",
        speed_min_factor=0.3,
        speed_max_factor=1.5,
        required_labels=tuple(required_labels),
        roster_is_canonical=True,
        git_sha=None,
        global_status=global_status,
        missing_labels=tuple(missing_labels or ()),
        seed_order=tuple(seed_order),
        dropped_seeds=tuple(sorted(dropped)),
        rows=rows,
        consulted_hashes=consulted_hashes,
        absent_files=tuple(absent_files or ()),
    )
    sidecar_path = world_dir / SEED_FILTER_NAME
    write_seed_filter(obj, sidecar_path)
    return sidecar_path


# --- The test cases (TC-P1 .. TC-P16) ---------------------------------------
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

    # Must NOT raise even though 11 of 12 label dirs are missing.
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
    """TC-P5: render all 7 charts from a 12-label fixture; each PNG must be non-empty."""
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
    """TC-P6: B1 matrix is 12 rows x len(seed_order) cols; every row shares the seed->col index."""
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
    assert len(planners) == 13, f"canonical_planner_set must return 13 tuples, got {len(planners)}"

    replan_families = {"a_star_replan", "dijkstra_replan", "rrt_replan", "rrt_star_replan"}
    predict_canonical = {"d_star_lite_predictive", "dwa_predictive_paper"}
    seen_replan = set()
    seen_predictive = set()
    for algorithm, replan_k, predict_horizon, label in planners:
        if algorithm in replan_families:
            seen_replan.add(algorithm)
            assert replan_k == 5, f"{algorithm} must carry replan_k=5, got {replan_k}"
            assert predict_horizon is None, f"{algorithm} must carry predict_horizon=None, got {predict_horizon}"
            assert label.endswith("_k5"), f"{algorithm} label must end _k5, got {label!r}"
        elif algorithm in predict_canonical:
            seen_predictive.add(algorithm)
            assert replan_k is None, f"{algorithm} must carry replan_k=None, got {replan_k}"
            assert predict_horizon == 10, f"{algorithm} must carry predict_horizon=10, got {predict_horizon}"
            assert label == f"{algorithm}_h10", f"{algorithm} label must be {algorithm}_h10, got {label!r}"
        else:
            assert replan_k is None, f"{algorithm} must carry replan_k=None, got {replan_k}"
            assert predict_horizon is None, f"{algorithm} must carry predict_horizon=None, got {predict_horizon}"
            assert not label.endswith("_k5"), f"{algorithm} label must not end _k5, got {label!r}"
    assert seen_replan == replan_families, \
        f"all 4 replan families must appear, missing {replan_families - seen_replan}"
    assert seen_predictive == predict_canonical, \
        f"both canonical predict families must appear, missing {predict_canonical - seen_predictive}"

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
    assert "--predict-horizon" not in once_cmd, "a non-predict family's command must omit --predict-horizon"

    # The canonical predictive carries --predict-horizon 10 and no --replan-k.
    predict_cmd = build_experiment_cmd(
        "d_star_lite_predictive", None, "arena/arena_v1.yaml", "results",
        master_seed=20260605, num_seeds=50, jobs=1, traffic=True, resume=False,
        predict_horizon=10,
    )
    assert "--predict-horizon" in predict_cmd, "the canonical predictive command must include --predict-horizon"
    h_index = predict_cmd.index("--predict-horizon")
    assert predict_cmd[h_index + 1] == "10", \
        f"--predict-horizon value must be 10, got {predict_cmd[h_index + 1]!r}"
    assert "--replan-k" not in predict_cmd, "the predictive command must omit --replan-k"
    return "12 tuples; 4 replan families k=5/_k5; predictive h10/_h10; command builder gates --replan-k/--predict-horizon"


def _tc_p11_dnf_roster(tmp: Path) -> str:
    """TC-P11: a manifest `episodes` roster counts wallclock-killed seeds as DNF failures.

    Build one label dir whose `_manifest.json` carries an `episodes` list with 3
    "ok" seeds (JSONs present: 2 success + 1 crash) and 2 "runner_error" seeds
    (NO JSON — the batch killed them at the wallclock wall). The loader must use
    the roster as the authoritative denominator: n_total == 5, n_dnf == 2,
    n_success == 2, failure_rate == 3/5 (1 crash + 2 dnf), and the two
    runner_error seeds marked "dnf" in per_seed.
    """
    from planners import algorithm_label

    results_root = tmp / "tc_p11"
    world_stem = "w"
    label = algorithm_label("d_star_lite", None)
    label_dir = results_root / world_stem / label
    label_dir.mkdir(parents=True, exist_ok=True)

    ok_seeds = [101, 202, 303]
    dnf_seeds = [404, 505]
    derived = ok_seeds + dnf_seeds

    # The 3 "ok" seeds leave JSONs on disk: 2 success + 1 crash.
    (label_dir / "101.json").write_text(
        json.dumps(_make_record("success", time_to_goal=10.0), sort_keys=True), encoding="utf-8"
    )
    (label_dir / "202.json").write_text(
        json.dumps(_make_record("success", time_to_goal=20.0), sort_keys=True), encoding="utf-8"
    )
    (label_dir / "303.json").write_text(
        json.dumps(_make_record("crash"), sort_keys=True), encoding="utf-8"
    )
    # The 2 "runner_error" seeds leave NO JSON — only a roster entry.

    manifest = {
        "master_seed": 20260605,
        "num_seeds": len(derived),
        "derived_seeds": list(derived),
        "episodes": [
            {"seed": 101, "exit_code": 0, "status": "ok"},
            {"seed": 202, "exit_code": 0, "status": "ok"},
            {"seed": 303, "exit_code": 0, "status": "ok"},
            {"seed": 404, "exit_code": 124, "status": "runner_error"},
            {"seed": 505, "exit_code": 124, "status": "runner_error"},
        ],
    }
    (label_dir / MANIFEST_NAME).write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    results = load_world_results(str(results_root), world_stem, replan_k=DEFAULT_REPLAN_K)
    summary = {s.label: s for s in results.summaries}[label]

    assert summary.n_present == 5, f"roster denominator must be 5 (3 ok + 2 runner_error), got {summary.n_present}"
    assert summary.n_dnf == 2, f"the 2 runner_error seeds must count as DNF, got n_dnf={summary.n_dnf}"
    assert summary.n_success == 2, f"the 2 success JSONs must count, got n_success={summary.n_success}"
    assert summary.n_crash == 1, f"the 1 crash JSON must count, got n_crash={summary.n_crash}"
    assert abs(summary.failure_rate - 3.0 / 5.0) < 1e-9, \
        f"failure_rate must be (1 crash + 2 dnf)/5 = 0.6, got {summary.failure_rate}"
    assert summary.per_seed.get(404) == "dnf", "seed 404 (runner_error) must be marked dnf in per_seed"
    assert summary.per_seed.get(505) == "dnf", "seed 505 (runner_error) must be marked dnf in per_seed"
    # The DNF seeds carry no time and no success entry.
    assert summary.per_seed.get(101) == "success" and summary.per_seed.get(303) == "crash", \
        "the ok-seed JSONs must classify normally alongside the DNF roster entries"
    return "roster: n_total=5, n_dnf=2, n_success=2, failure_rate=3/5; 404/505 marked dnf"


def _tc_p12_filter_applied(tmp: Path) -> str:
    """TC-P12: a fresh, covering, global-'ok' sidecar drops one degenerate seed everywhere.

    Data-layer assertions only (mirroring TC-P6, never inspecting a PNG): the
    dropped seed leaves every algorithm's FILTERED `n_present`/`failure_rate` and
    the A3 count layer (`n_crash`), the pre-drop values survive as
    `n_present_raw`/`failure_rate_raw`, `per_seed` stays complete for B1, and
    `WorldResults.degenerate_seeds`/`n_dropped` name the culled seed.
    """
    results_root = tmp / "tc_p12"
    world_stem = "arena_v1"
    seeds = [11, 22, 33, 44, 55]
    degenerate = 33
    labels = _build_filter_tree(results_root, world_stem, seeds=seeds, degenerate_seed=degenerate)

    # Consult the degenerate seed's metrics in every canonical label (what the
    # real filter reads to decide degeneracy), hashed fresh into the sidecar.
    _write_seed_filter_sidecar(
        results_root=results_root,
        world_stem=world_stem,
        required_labels=labels,
        seed_order=seeds,
        dropped_seeds=[degenerate],
        consulted=[(label, degenerate) for label in labels],
    )

    plotted = _plotted_labels(DEFAULT_REPLAN_K)
    status, dropped, missing = _decide_seed_filter(
        results_dir=str(results_root),
        world_stem=world_stem,
        plotted_labels=plotted,
        apply_filter=True,
    )
    assert status == "applied", f"a fresh+covering ok sidecar must apply, got {status!r}"
    assert dropped == frozenset({degenerate}), f"must drop the declared seed, got {dropped}"
    assert missing == (), "an applied filter has no missing labels"

    results = load_world_results(
        str(results_root), world_stem, replan_k=DEFAULT_REPLAN_K, dropped_seeds=dropped
    )
    assert results.degenerate_seeds == (degenerate,), \
        f"degenerate_seeds must hold the dropped seed, got {results.degenerate_seeds}"
    assert results.n_dropped == 1, f"n_dropped must be 1, got {results.n_dropped}"

    n_seeds = len(seeds)
    for summary in results.summaries:
        assert summary.n_present == n_seeds - 1, \
            f"{summary.label}: filtered n_present must be {n_seeds - 1}, got {summary.n_present}"
        assert summary.n_crash == 0, \
            f"{summary.label}: the dropped crash must leave the A3 layer, got n_crash={summary.n_crash}"
        assert abs(summary.failure_rate - 0.0) < 1e-9, \
            f"{summary.label}: filtered failure_rate must be 0, got {summary.failure_rate}"
        assert summary.n_present_raw == n_seeds, \
            f"{summary.label}: n_present_raw must be {n_seeds}, got {summary.n_present_raw}"
        assert abs(summary.failure_rate_raw - 1.0 / n_seeds) < 1e-9, \
            f"{summary.label}: failure_rate_raw must be 1/{n_seeds}, got {summary.failure_rate_raw}"
        assert summary.per_seed.get(degenerate) == "crash", \
            f"{summary.label}: per_seed must keep the degenerate seed's real outcome"
    return f"1 degenerate seed dropped from every algo; raw preserved (fr_raw=1/{n_seeds})"


def _tc_p13_filter_stale(tmp: Path) -> str:
    """TC-P13: a mutated hash, a reappeared absent file, or an uncovered label set
    each yields no drop, filter_status='stale', a stderr reason, and still renders."""
    import contextlib
    import io

    plt = ensure_matplotlib()
    plotted = _plotted_labels(DEFAULT_REPLAN_K)
    seeds = [11, 22, 33, 44, 55]
    degenerate = 33

    def _decide_capturing(root: Path):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status, dropped, missing = _decide_seed_filter(
                results_dir=str(root),
                world_stem="arena_v1",
                plotted_labels=plotted,
                apply_filter=True,
            )
        return status, dropped, missing, stderr.getvalue()

    # (a) A consulted metrics file mutated after the sidecar was written.
    root_a = tmp / "tc_p13a"
    labels = _build_filter_tree(root_a, "arena_v1", seeds=seeds, degenerate_seed=degenerate)
    _write_seed_filter_sidecar(
        results_root=root_a, world_stem="arena_v1", required_labels=labels,
        seed_order=seeds, dropped_seeds=[degenerate],
        consulted=[(label, degenerate) for label in labels],
    )
    (root_a / "arena_v1" / labels[0] / f"{degenerate}.json").write_text(
        json.dumps(_make_record("timeout"), sort_keys=True), encoding="utf-8"
    )
    status, dropped, _missing, msg = _decide_capturing(root_a)
    assert status == "stale" and dropped == frozenset(), \
        f"a mutated consulted hash must go stale with no drop, got {status!r}/{dropped}"
    assert "hash changed" in msg and "STALE" in msg, \
        f"the stale warning must name the changed hash, stderr was: {msg!r}"

    # (b) A recorded-absent file that now exists.
    root_b = tmp / "tc_p13b"
    labels = _build_filter_tree(root_b, "arena_v1", seeds=seeds, degenerate_seed=degenerate)
    _write_seed_filter_sidecar(
        results_root=root_b, world_stem="arena_v1", required_labels=labels,
        seed_order=seeds, dropped_seeds=[degenerate],
        consulted=[(label, degenerate) for label in labels],
        absent_files=[f"{labels[0]}/9999.json"],
    )
    (root_b / "arena_v1" / labels[0] / "9999.json").write_text(
        json.dumps(_make_record("success", time_to_goal=5.0), sort_keys=True), encoding="utf-8"
    )
    status, dropped, _missing, msg = _decide_capturing(root_b)
    assert status == "stale" and dropped == frozenset(), \
        f"a reappeared absent file must go stale, got {status!r}/{dropped}"
    assert "now exists" in msg, f"the warning must name the reappeared file, stderr was: {msg!r}"

    # (c) The plotted label set is not covered by required_labels.
    root_c = tmp / "tc_p13c"
    labels = _build_filter_tree(root_c, "arena_v1", seeds=seeds, degenerate_seed=degenerate)
    from runners._seed_filter import build_required_labels

    required_k7 = build_required_labels(10, 7, "canonical")
    _write_seed_filter_sidecar(
        results_root=root_c, world_stem="arena_v1", required_labels=required_k7,
        seed_order=seeds, dropped_seeds=[degenerate],
        consulted=[(labels[0], degenerate)],
        replan_k=7,
    )
    status, dropped, _missing, msg = _decide_capturing(root_c)
    assert status == "stale" and dropped == frozenset(), \
        f"an uncovered label set must go stale, got {status!r}/{dropped}"
    assert "not covered" in msg, f"the warning must name the coverage miss, stderr was: {msg!r}"

    # Charts still render on a stale (unfiltered) load.
    results = replace(
        load_world_results(str(root_c), "arena_v1", replan_k=DEFAULT_REPLAN_K),
        filter_status="stale",
    )
    out_dir = tmp / "tc_p13_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = _chart_a1(results, plt, out_dir)
    assert png.is_file() and png.stat().st_size > 0, "A1 must still render on a stale filter"
    return "mutated-hash / reappeared-absent / uncovered-labels each stale, no drop, still renders"


def _tc_p14_absent_vs_no_filter(tmp: Path) -> str:
    """TC-P14: absent-sidecar and --no-filter are byte-identical + numeric==raw, n_dropped==0 (AC13)."""
    root = tmp / "tc_p14"
    world_stem = "arena_v1"
    seeds = [11, 22, 33, 44, 55]
    _build_filter_tree(root, world_stem, seeds=seeds, degenerate_seed=33)
    # NO sidecar written: the "absent" path.
    plotted = _plotted_labels(DEFAULT_REPLAN_K)

    status_absent, dropped_absent, _ = _decide_seed_filter(
        results_dir=str(root), world_stem=world_stem, plotted_labels=plotted, apply_filter=True,
    )
    status_off, dropped_off, _ = _decide_seed_filter(
        results_dir=str(root), world_stem=world_stem, plotted_labels=plotted, apply_filter=False,
    )
    assert status_absent == "absent" and dropped_absent == frozenset(), \
        f"no sidecar must read absent with no drop, got {status_absent!r}/{dropped_absent}"
    assert status_off == "off" and dropped_off == frozenset(), \
        f"--no-filter must read off with no drop, got {status_off!r}/{dropped_off}"

    results_absent = load_world_results(
        str(root), world_stem, replan_k=DEFAULT_REPLAN_K, dropped_seeds=dropped_absent
    )
    results_off = load_world_results(
        str(root), world_stem, replan_k=DEFAULT_REPLAN_K, dropped_seeds=dropped_off
    )

    assert results_absent.n_dropped == 0 and results_off.n_dropped == 0, "no drop expected"
    for summary in results_absent.summaries:
        assert summary.n_present == summary.n_present_raw, "n_present must equal raw when unfiltered"
        fr, fr_raw = summary.failure_rate, summary.failure_rate_raw
        both_nan = fr != fr and fr_raw != fr_raw
        assert both_nan or abs(fr - fr_raw) < 1e-12, "failure_rate must equal raw when unfiltered"

    # Byte-identical output: the two runs' summary.csv are identical bytes.
    out_a = tmp / "tc_p14_absent.csv"
    out_b = tmp / "tc_p14_off.csv"
    write_summary_csv(results_absent.summaries, out_a)
    write_summary_csv(results_off.summaries, out_b)
    assert out_a.read_bytes() == out_b.read_bytes(), \
        "absent-sidecar and --no-filter summary.csv must be byte-identical (AC13)"
    return "absent==off byte-identical; numeric==raw; n_dropped==0"


def _tc_p15_summary_csv_columns(tmp: Path) -> str:
    """TC-P15: summary.csv carries n_present_raw / failure_rate_raw / n_dropped, filtered + unfiltered."""
    import csv as _csv

    root = tmp / "tc_p15"
    world_stem = "arena_v1"
    seeds = [11, 22, 33, 44, 55]
    degenerate = 33
    _build_filter_tree(root, world_stem, seeds=seeds, degenerate_seed=degenerate)
    n_seeds = len(seeds)

    # The three audit columns must sit immediately after failure_rate.
    fr_index = SUMMARY_CSV_COLUMNS.index("failure_rate")
    assert SUMMARY_CSV_COLUMNS[fr_index + 1: fr_index + 4] == (
        "n_present_raw", "failure_rate_raw", "n_dropped"
    ), f"the 3 audit columns must follow failure_rate, got {SUMMARY_CSV_COLUMNS}"

    # Filtered run.
    results = load_world_results(
        str(root), world_stem, replan_k=DEFAULT_REPLAN_K, dropped_seeds=frozenset({degenerate})
    )
    filtered_csv = tmp / "tc_p15_filtered.csv"
    write_summary_csv(results.summaries, filtered_csv)
    with open(filtered_csv, newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    assert rows, "the filtered CSV must have rows"
    for row in rows:
        assert int(row["n_present"]) == n_seeds - 1, f"filtered n_present must be {n_seeds - 1}"
        assert int(row["n_present_raw"]) == n_seeds, f"n_present_raw must be {n_seeds}"
        assert int(row["n_dropped"]) == 1, "n_dropped must be 1"
        assert abs(float(row["failure_rate"]) - 0.0) < 1e-9, "filtered failure_rate must be 0"
        assert abs(float(row["failure_rate_raw"]) - 1.0 / n_seeds) < 1e-9, "failure_rate_raw must be 1/n"

    # Unfiltered run: raw == filtered, n_dropped == 0.
    results_u = load_world_results(str(root), world_stem, replan_k=DEFAULT_REPLAN_K)
    unfiltered_csv = tmp / "tc_p15_unfiltered.csv"
    write_summary_csv(results_u.summaries, unfiltered_csv)
    with open(unfiltered_csv, newline="", encoding="utf-8") as fh:
        rows_u = list(_csv.DictReader(fh))
    for row in rows_u:
        assert int(row["n_present"]) == int(row["n_present_raw"]), "unfiltered n_present must equal raw"
        assert int(row["n_dropped"]) == 0, "unfiltered n_dropped must be 0"
        assert float(row["failure_rate"]) == float(row["failure_rate_raw"]), \
            "unfiltered failure_rate must equal raw"
    return "summary.csv: n_present_raw/failure_rate_raw/n_dropped correct for filtered + unfiltered"


def _tc_p16_filter_indeterminate(tmp: Path) -> str:
    """TC-P16: a fresh, covering, global-'indeterminate' sidecar drops nothing; footnote confesses."""
    plt = ensure_matplotlib()
    root = tmp / "tc_p16"
    world_stem = "arena_v1"
    seeds = [11, 22, 33, 44, 55]
    labels = _build_filter_tree(root, world_stem, seeds=seeds, degenerate_seed=33)

    # required_labels = all13 (12 canonical + 1 experimental oracle). Only the
    # oracle dir is absent on disk (the 12 canonical dirs, incl. the now-canonical
    # d_star_lite_predictive_h10, were written by _build_filter_tree), so the
    # recorded absent_file stays absent (still fresh) and the oracle label surfaces
    # as missing.
    from runners._seed_filter import build_required_labels

    required = build_required_labels(10, DEFAULT_REPLAN_K, "all13")
    missing = [required[-1]]  # d_star_lite_oracle_h10 (the only non-canonical required label)
    _write_seed_filter_sidecar(
        results_root=root, world_stem=world_stem, required_labels=required,
        seed_order=seeds, dropped_seeds=[],
        consulted=[(labels[0], seeds[0])],
        global_status="indeterminate",
        missing_labels=missing,
        absent_files=[f"{missing[0]}/{seeds[0]}.json"],
    )

    plotted = _plotted_labels(DEFAULT_REPLAN_K)
    status, dropped, missing_out = _decide_seed_filter(
        results_dir=str(root), world_stem=world_stem, plotted_labels=plotted, apply_filter=True,
    )
    assert status == "indeterminate", f"an indeterminate sidecar must not apply, got {status!r}"
    assert dropped == frozenset(), "indeterminate must drop nothing"
    assert set(missing_out) == set(missing), f"missing labels must surface, got {missing_out}"

    results = replace(
        load_world_results(str(root), world_stem, replan_k=DEFAULT_REPLAN_K, dropped_seeds=dropped),
        filter_status=status,
        filter_missing_labels=missing_out,
    )
    assert results.n_dropped == 0, "indeterminate leaves n_dropped 0"

    # The A1 footnote confesses + names the missing labels; the render must not raise.
    footnote = _filter_status_footnote(results)
    assert "indeterminate" in footnote and missing[0] in footnote, \
        f"the A1 footnote must confess indeterminate + name a missing label, got {footnote!r}"
    out_dir = tmp / "tc_p16_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = _chart_a1(results, plt, out_dir)
    assert png.is_file() and png.stat().st_size > 0, "A1 must render on an indeterminate filter"
    return "indeterminate sidecar: no drop, footnote names missing labels, renders"


# Ordered registry of the 16 test cases. Each entry is (id, callable).
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
    ("TC-P11", _tc_p11_dnf_roster),
    ("TC-P12", _tc_p12_filter_applied),
    ("TC-P13", _tc_p13_filter_stale),
    ("TC-P14", _tc_p14_absent_vs_no_filter),
    ("TC-P15", _tc_p15_summary_csv_columns),
    ("TC-P16", _tc_p16_filter_indeterminate),
)


def run_selfcheck() -> int:
    """Run the plotter's self-check suite (TC-P1..TC-P16). Return 0 if all pass, else 1.

    Builds every fixture inside a single TemporaryDirectory (each TC namespaces its
    own subdir), runs each TC in isolation so one failure never aborts the rest,
    prints a per-TC PASS/FAIL line, and ends with an "N/16 passed" summary. No
    irsim, no real episodes — synthetic JSON trees only (TC-P12..TC-P16 add a
    `_seed_filter.json` sidecar). Charts are rendered through the real chart
    functions under the headless Agg backend.
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
    filter: bool                        # apply a fresh+covering degenerate-seed sidecar (default True; --no-filter disables)


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
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--filter",
        dest="filter",
        action="store_true",
        help="Apply a fresh, covering degenerate-seed sidecar if present (default; issue #9).",
    )
    filter_group.add_argument(
        "--no-filter",
        dest="filter",
        action="store_false",
        help="Ignore any degenerate-seed sidecar; chart the full seed set.",
    )
    parser.set_defaults(filter=True)
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
        filter=bool(ns.filter),
    )


def _resolve_out_dir(args: PlotArgs, world_stem: str) -> Path:
    """Out-dir is the CLI override, else `<results-dir>/<world_stem>/plots/` (AC14)."""
    if args.out_dir is not None:
        return Path(args.out_dir).resolve()
    return Path(args.results_dir).resolve() / world_stem / PLOTS_DIR_NAME


# --- Degenerate-seed filter (issue #9) --------------------------------------

def _plotted_labels(replan_k: int) -> list[str]:
    """The results-dir labels `load_world_results` will actually load for the
    CANONICAL set, derived the SAME way (via `algorithm_label`) so the sidecar
    coverage check (`sidecar_covers`) tests the exact label strings the plotter
    charts. Imports `planners` lazily (like `load_world_results`), so it never
    runs at module import — `import runners.plot` stays headless.
    """
    from planners import algorithm_label

    labels: list[str] = []
    for name, default_k, default_h, _family, _display in CANONICAL:
        effective_k = replan_k if default_k is not None else None
        labels.append(algorithm_label(name, effective_k, default_h))
    return labels


def _decide_seed_filter(
    *,
    results_dir: str,
    world_stem: str,
    plotted_labels: list[str],
    apply_filter: bool,
) -> tuple[str, frozenset[int], tuple[str, ...]]:
    """Decide how the degenerate-seed filter (issue #9) applies to this plot run.

    Returns `(filter_status, dropped_seeds, missing_labels)`:
      - `--no-filter` (apply_filter False)         -> ("off", ∅, ())
      - no readable sidecar                         -> ("absent", ∅, ())
      - sidecar not fresh OR not covering           -> ("stale", ∅, ()) + loud stderr
      - fresh + covering, global "ok"               -> ("applied", set(dropped_seeds), ())
      - fresh + covering, global "indeterminate"    -> ("indeterminate", ∅, missing_labels)
      - fresh + covering, any other global          -> ("indeterminate", ∅, missing_labels)

    Never partial-applies and never raises: a `read_seed_filter` that returns
    `None` (absent/unreadable/schema mismatch) degrades to "absent"; any
    freshness OR coverage miss degrades to "stale" with a stderr warning naming
    every failing reason (AC8). Only a fresh, covering, "ok" sidecar drops seeds.
    """
    if not apply_filter:
        return "off", frozenset(), ()

    results_root = Path(results_dir).resolve()
    sidecar_path = results_root / world_stem / SEED_FILTER_NAME
    obj = read_seed_filter(sidecar_path)
    if obj is None:
        return "absent", frozenset(), ()

    # A sidecar built for a different world must never apply here: every world
    # shares the same 50 seeds, so a mis-placed sidecar would otherwise re-hash
    # its own (unchanged) files, pass freshness, and drop the wrong world's seeds.
    world_matches = obj.world_stem == world_stem
    fresh, reasons = sidecar_is_fresh(obj, results_root)
    covers = sidecar_covers(obj, plotted_labels)
    if not world_matches or not fresh or not covers:
        problems = list(reasons)
        if not world_matches:
            problems.append(
                f"sidecar world_stem {obj.world_stem!r} does not match the plotted world {world_stem!r}"
            )
        if not covers:
            problems.append(
                "plotted label set not covered by sidecar required_labels "
                f"(plotted={plotted_labels})"
            )
        print(
            "warning: seed filter sidecar is STALE, ignoring (re-run "
            f"runners.filter_seeds): {'; '.join(problems)}",
            file=sys.stderr,
        )
        return "stale", frozenset(), ()

    if obj.global_status == "ok":
        return "applied", frozenset(obj.dropped_seeds), ()

    # "indeterminate" (or any non-"ok" status): fresh + covering, but the filter
    # itself determined nothing droppable. No drop; surface its missing labels.
    return "indeterminate", frozenset(), tuple(obj.missing_labels)


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

    # Decide the degenerate-seed drop (issue #9) BEFORE loading so the filtered
    # counts land in every chart. The sidecar must be fresh AND cover the exact
    # label set we are about to load; otherwise no seed is dropped.
    plotted_labels = _plotted_labels(args.replan_k)
    filter_status, dropped_seeds, missing_labels = _decide_seed_filter(
        results_dir=args.results_dir,
        world_stem=world_stem,
        plotted_labels=plotted_labels,
        apply_filter=args.filter,
    )

    results = load_world_results(
        args.results_dir,
        world_stem,
        replan_k=args.replan_k,
        expected=DEFAULT_EXPECTED_SEEDS,
        dropped_seeds=dropped_seeds,
    )
    # Stamp how the drop was decided onto the frozen result (the loader filled
    # degenerate_seeds/n_dropped from dropped_seeds; these two are main's call).
    results = replace(
        results,
        filter_status=filter_status,
        filter_missing_labels=missing_labels,
    )

    # "No readable data at all" => every algorithm came back empty ON DISK. Test
    # the RAW (pre-filter) count so an all-degenerate world (every seed dropped)
    # does not misreport as "no readable JSONs" — nothing to plot or summarize, so
    # exit non-zero with a clear message (AC11).
    if all(summary.n_present_raw == 0 for summary in results.summaries):
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
