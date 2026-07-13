"""Pure, stdlib-only classification core for the degenerate-seed filter (issue #9).

This module implements the "every algo dies instantly" criterion that flags a
traffic seed as degenerate: the sidecar schema it is written/read as, the
content-hash + roster freshness check, and the pure-stdlib required-label
builder. It has exactly THREE consumers: `runners/filter_seeds.py` (T2, the CLI
labeler that writes the sidecar), `runners/plot.py` (T3, which reads and
freshness/coverage-verifies it), and this module's own future callers.

**Headless import discipline (load-bearing).** Importing the `planners` package
transitively pulls irsim AND matplotlib (`planners -> manual_astar -> irsim ->
matplotlib.pyplot / TkAgg`), and `runners.run_all` imports `planners` at module
top, so even a *lazy* `from runners.run_all import canonical_planner_set` pulls
irsim the moment it runs. This module therefore imports NOTHING from
`planners`, `runners.run_all`, `arena.*`, `irsim`, `numpy`, or `matplotlib` —
stdlib only (`json`, `hashlib`, `sys`, `dataclasses`, `pathlib`, `typing`). The
12 canonical planner names, the replan-family `_k<K>` fold, and the canonical
predict-family `_h<H>` fold (`d_star_lite_predictive`) are hand-mirrored below as
`_CANONICAL_ORDER` / `_REPLAN_FAMILIES` / `_PREDICT_CANONICAL`; a subprocess test in
`filter_seeds.py --selfcheck` (where importing `planners` is permitted) asserts
they stay in sync with the real registry (AC15).

**Freshness scope — metrics only.** `sidecar_is_fresh` hashes only the
consulted `<seed>.json` *metrics* files, never trace JSONL. `wallclock_per_step`
is a `perf_counter` mean, so any re-run of any episode changes its metrics
bytes and trips the hash; a trace is a deterministic function of the same
episode and cannot change without its metrics changing too, so hashing metrics
alone is sufficient to detect a re-run.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence


# --- Module constants --------------------------------------------------------

SEED_FILTER_NAME = "_seed_filter.json"   # world-level sidecar (a SIBLING of the label dirs)
DEFAULT_STEP_TIME = 0.1                  # s; mirrors irsim step_time (documented in the plan's Notes)
DEFAULT_WINDOW_SECONDS = 4.0
SCHEMA_VERSION = 1
CRITERION_ID = "all_required_instant_crash/v1"

# Private: the manifest filename `sidecar_is_fresh` re-reads to re-derive the
# roster. Hand-copied (not imported from `runners.run_experiment`/`plot.py`) to
# keep this module free of any sibling-module import beyond the stdlib.
_MANIFEST_NAME = "_manifest.json"

# Documentation-only aliases (plain `str` at runtime — these are not NewType,
# just names for the literal value sets the schema uses).
PlannerStatus = str   # "instant_crash" | "survived" | "missing"
SeedVerdict = str     # "degenerate"    | "kept"     | "indeterminate"


# --- Required-label construction (CR7 / AC15) --------------------------------

# Frozen mirror of `run_all._CANONICAL_ORDER` (the 12 canonical planner keys, in
# `run_all`'s hand-listed order). Must NOT be imported from `run_all` per the
# headless-import discipline above; AC15's subprocess parity test is what keeps
# this copy honest against registry drift.
_CANONICAL_ORDER: tuple[str, ...] = (
    "a_star_once",
    "a_star_replan",
    "dijkstra_once",
    "dijkstra_replan",
    "d_star_lite",
    "d_star_lite_predictive",
    "dwa",
    "dwa_predictive_paper",
    "apf",
    "rrt_once",
    "rrt_replan",
    "rrt_star_once",
    "rrt_star_replan",
)

# Frozen mirror of `planners._grid.REPLAN_FAMILIES` — the four keys whose
# results dir label folds in `_k<replan_k>` (`planners._grid.algorithm_label`).
_REPLAN_FAMILIES = frozenset(
    {"a_star_replan", "dijkstra_replan", "rrt_replan", "rrt_star_replan"}
)

# The canonical predict-family keys — folded with `_h<predict_horizon>` the same
# way `planners._grid.algorithm_label` folds a PREDICT_FAMILIES key. The canonical
# predictive variants (d_star_lite_predictive, dwa_predictive_paper — the braking-only
# DWA policy) fold at h10; the oracles (d_star_lite_oracle, dwa_predictive_oracle),
# the global-guidance dwa_predictive, and dwa_predictive_paper_oracle stay
# experimental. The "all13" set appends the d_star_lite_oracle label separately in
# `build_required_labels` (the other experimental keys are not part of the roster).
_PREDICT_CANONICAL = frozenset({"d_star_lite_predictive", "dwa_predictive_paper"})


def _canonical_label(name: str, replan_k: int, predict_horizon: int) -> str:
    """Results-dir label for a canonical-order key — mirrors `algorithm_label`.

    Folds `_k<replan_k>` for a replan family, `_h<predict_horizon>` for the
    canonical predict family (d_star_lite_predictive), else the bare key.
    """
    if name in _REPLAN_FAMILIES:
        return f"{name}_k{replan_k}"
    if name in _PREDICT_CANONICAL:
        return f"{name}_h{predict_horizon}"
    return name


def build_required_labels(predict_horizon: int, replan_k: int, planner_set: str) -> list[str]:
    """Build the list of results-dir labels the filter requires evidence from.

    PURE stdlib string logic — no `planners` import (see the module docstring).
    Rebuilds the 12 canonical labels from `_CANONICAL_ORDER`, folding
    `_k<replan_k>` for the four replan families and `_h<predict_horizon>` for the
    canonical predict family (`d_star_lite_predictive`), mirroring
    `planners._grid.algorithm_label`'s fold. When `planner_set == "all13"` the one
    remaining experimental `_h<predict_horizon>` label is appended
    (`d_star_lite_oracle_h<H>`); any other `planner_set` value (in practice
    `"canonical"`) returns just the 12.

    AC15's subprocess test cross-checks this output against
    `run_all.canonical_planner_set()` labels plus
    `algorithm_label("d_star_lite_oracle", None, H)` (where importing `planners`
    is permitted), so a future registry change fails loud instead of silently
    checking the wrong labels.
    """
    labels = [
        _canonical_label(name, replan_k, predict_horizon) for name in _CANONICAL_ORDER
    ]
    if planner_set == "all13":
        labels.append(f"d_star_lite_oracle_h{predict_horizon}")
    return labels


# --- Data model ---------------------------------------------------------------

@dataclass(frozen=True)
class PlannerEvidence:
    """One required planner's classification for one seed."""

    label: str              # results-dir label, e.g. "a_star_replan_k5"
    status: str              # PlannerStatus
    crash_step: int | None   # first crashed trace step when status == "instant_crash", else None


@dataclass(frozen=True)
class SeedVerdictRow:
    """One seed's classification plus the per-planner evidence it rests on."""

    seed: int
    verdict: str                            # SeedVerdict
    planners: tuple[PlannerEvidence, ...]   # sorted by label


@dataclass(frozen=True)
class SeedFilter:
    """The sidecar's in-memory shape — the whole `_seed_filter.json` payload.

    Written by `write_seed_filter` (T2's `filter_seeds.py`), read back by
    `read_seed_filter` (T3's `plot.py`). Every field here is part of the
    on-disk contract; see the plan's "Sidecar JSON" section for the exact
    serialization (dataclasses -> dicts, tuples -> JSON arrays, the per-seed
    field named `rows`).
    """

    schema_version: int
    criterion_id: str
    world_stem: str
    window_seconds: float
    step_time: float
    window_steps: int
    replan_k: int
    predict_horizon: int
    traffic: bool
    speed_regime: str | None
    speed_min_factor: float
    speed_max_factor: float
    required_labels: tuple[str, ...]
    roster_is_canonical: bool        # True iff roster came from a manifest's derived_seeds
    git_sha: str | None
    global_status: str               # "ok" | "indeterminate"
    missing_labels: tuple[str, ...]
    seed_order: tuple[int, ...]
    dropped_seeds: tuple[int, ...]
    rows: tuple[SeedVerdictRow, ...]
    consulted_hashes: dict[str, str]  # "<label>/<seed>.json" -> sha256 (present files)
    absent_files: tuple[str, ...]     # "<label>/<seed>.json" that were missing at write time


# --- Classification primitives ------------------------------------------------

def window_steps(window_seconds: float, step_time: float) -> int:
    """Number of sim steps in the instant-crash window (rounded to nearest int)."""
    return round(window_seconds / step_time)


def crash_step_from_trace(lines: Iterable[dict]) -> int | None:
    """Return the `step` of the first trace line whose `crashed` is truthy.

    `lines` is an iterable of already-parsed trace-line dicts (one per JSONL
    line); this function does no file I/O and no JSON parsing. Returns `None`
    if no line has `crashed` truthy. Step 0 is the runner's post-reset sentinel
    with `crashed` hardcoded False, so it can never be returned here.
    """
    for line in lines:
        if line.get("crashed"):
            # A crashed line missing `step` yields None here; downstream
            # classify_planner then treats None as "cannot confirm instant" ->
            # survived, which conservatively blocks a drop. This is intentional.
            return line.get("step")
    return None


def classify_planner(
    *, present: bool, crashed: bool, crash_step: int | None, window_steps: int
) -> str:
    """Classify one (seed, planner) pair into a `PlannerStatus`.

    `missing` iff the metrics file itself was absent (`present` is False —
    NEVER set for a crashed-but-untraceable record, see `filter_seeds`'s
    Error Handling). Otherwise `instant_crash` iff the episode crashed AND its
    first crashed trace step falls in `[1, window_steps]`; else `survived`
    (goal reached, timed out, crashed after the window, planner_error, or a DNF
    roster entry).
    """
    if not present:
        return "missing"
    if crashed and crash_step is not None and 1 <= crash_step <= window_steps:
        return "instant_crash"
    return "survived"


def classify_seed(evidence: Sequence[PlannerEvidence]) -> str:
    """Classify one seed into a `SeedVerdict` from its per-planner evidence.

    Precedence: a single `survived` planner is dispositive -> `kept`, even if
    another required planner is `missing`. Otherwise any `missing` planner
    makes the seed `indeterminate` (never dropped). Only when every required
    planner is `instant_crash` is the seed `degenerate`.
    """
    statuses = [e.status for e in evidence]
    if "survived" in statuses:
        return "kept"
    if "missing" in statuses:
        return "indeterminate"
    return "degenerate"


# --- Hashing -------------------------------------------------------------------

def file_sha256(path: str | Path) -> str | None:
    """Hex sha256 digest of a file's bytes, or `None` if it cannot be read."""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


# --- Sidecar read / write -------------------------------------------------------

def write_seed_filter(obj: SeedFilter, path: str | Path) -> None:
    """Write `obj` as the `_seed_filter.json` sidecar.

    JSON with `sort_keys=True`, 2-space indent (matching `run_episode`'s and
    `run_experiment`'s metrics/manifest writers), and a trailing newline. No
    timestamp is ever written, and dict key sorting makes the output
    independent of any incidental construction order, so two writes of an
    EQUAL `SeedFilter` over an unchanged tree are byte-identical (AC5).
    """
    data = asdict(obj)
    with open(Path(path), "w", encoding="utf-8") as fh:
        json.dump(data, fh, sort_keys=True, indent=2)
        fh.write("\n")


def read_seed_filter(path: str | Path) -> SeedFilter | None:
    """Read and reconstruct a `SeedFilter` written by `write_seed_filter`.

    Returns `None` (after a stderr warning) when the file is absent,
    unreadable, malformed, or its `schema_version` does not match
    `SCHEMA_VERSION`. Callers (the plotter) treat a `None` return as "no
    usable sidecar" and fall back to running unfiltered — never as a hard
    error.
    """
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: could not read seed filter sidecar {path}: {exc}", file=sys.stderr)
        return None

    # Valid JSON that is not an object (null, list, number, string) has no .get;
    # fail closed to None so the plotter degrades to unfiltered rather than crashing.
    if not isinstance(data, dict):
        print(
            f"warning: seed filter sidecar {path} is not a JSON object; ignoring",
            file=sys.stderr,
        )
        return None

    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        print(
            f"warning: seed filter sidecar {path} has schema_version={version!r}, "
            f"expected {SCHEMA_VERSION!r}; ignoring",
            file=sys.stderr,
        )
        return None

    try:
        rows = tuple(
            SeedVerdictRow(
                seed=int(row["seed"]),
                verdict=str(row["verdict"]),
                planners=tuple(
                    PlannerEvidence(
                        label=str(p["label"]),
                        status=str(p["status"]),
                        crash_step=None if p["crash_step"] is None else int(p["crash_step"]),
                    )
                    for p in row["planners"]
                ),
            )
            for row in data["rows"]
        )
        return SeedFilter(
            schema_version=int(data["schema_version"]),
            criterion_id=str(data["criterion_id"]),
            world_stem=str(data["world_stem"]),
            window_seconds=float(data["window_seconds"]),
            step_time=float(data["step_time"]),
            window_steps=int(data["window_steps"]),
            replan_k=int(data["replan_k"]),
            predict_horizon=int(data["predict_horizon"]),
            traffic=bool(data["traffic"]),
            speed_regime=None if data["speed_regime"] is None else str(data["speed_regime"]),
            speed_min_factor=float(data["speed_min_factor"]),
            speed_max_factor=float(data["speed_max_factor"]),
            required_labels=tuple(str(x) for x in data["required_labels"]),
            roster_is_canonical=bool(data["roster_is_canonical"]),
            git_sha=None if data["git_sha"] is None else str(data["git_sha"]),
            global_status=str(data["global_status"]),
            missing_labels=tuple(str(x) for x in data["missing_labels"]),
            seed_order=tuple(int(x) for x in data["seed_order"]),
            dropped_seeds=tuple(int(x) for x in data["dropped_seeds"]),
            rows=rows,
            consulted_hashes=dict(data["consulted_hashes"]),
            absent_files=tuple(str(x) for x in data["absent_files"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        print(f"warning: seed filter sidecar {path} is malformed: {exc}", file=sys.stderr)
        return None


# --- Freshness / coverage checks (consumer: plot.py) ---------------------------

def _canonical_order_labels(
    required_labels: Sequence[str], replan_k: int, predict_horizon: int
) -> list[str]:
    """The subset of `required_labels` that are canonical-order labels, walked
    in `_CANONICAL_ORDER` sequence (folding `_k<replan_k>` for the replan
    families and `_h<predict_horizon>` for the canonical predict family, matching
    `build_required_labels`'s first 12 entries).

    Used only to pick the manifest-lookup order for the roster freshness check
    below — the same order `plot.load_world_results` and the plan's "Seed
    roster" data-model note both use.
    """
    ordered = []
    for name in _CANONICAL_ORDER:
        label = _canonical_label(name, replan_k, predict_horizon)
        if label in required_labels:
            ordered.append(label)
    return ordered


def _read_roster_seeds(canonical_labels: Sequence[str], world_dir: Path) -> list[int] | None:
    """`derived_seeds` from the first present, readable manifest among
    `canonical_labels` (already in canonical order), or `None` if none of them
    has a readable manifest with a `derived_seeds` list.
    """
    for label in canonical_labels:
        manifest_path = world_dir / label / _MANIFEST_NAME
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        derived = manifest.get("derived_seeds")
        if isinstance(derived, list):
            # Coerce to int so the roster comparison is symmetric with
            # read_seed_filter's int-coerced obj.seed_order.
            try:
                return [int(x) for x in derived]
            except (TypeError, ValueError):
                continue
    return None


def sidecar_is_fresh(obj: SeedFilter, results_root: str | Path) -> tuple[bool, list[str]]:
    """Verify a loaded `SeedFilter` is still valid against the on-disk result tree.

    `world_stem` is taken from `obj` (not passed separately). Three checks, all
    run (never short-circuited) so every failure is reported:

    1. Re-hash each `consulted_hashes` path (relative to
       `<results_root>/<obj.world_stem>/`) and compare to the recorded digest.
    2. Confirm every `absent_files` path is STILL absent.
    3. Re-derive the seed roster from the first canonical-order required
       manifest (see `_read_roster_seeds`) and confirm it still equals
       `obj.seed_order`.

    Traces are NEVER hashed here (see the module docstring's freshness-scope
    note). Returns `(True, [])` when every check passes, else `(False,
    reasons)` naming each failing check — the caller decides whether to apply
    the filter; this function only reports.
    """
    results_root = Path(results_root)
    world_dir = results_root / obj.world_stem
    reasons: list[str] = []

    for rel_path, recorded_hash in obj.consulted_hashes.items():
        current_hash = file_sha256(world_dir / Path(rel_path))
        if current_hash != recorded_hash:
            reasons.append(f"hash changed: {rel_path}")

    for rel_path in obj.absent_files:
        if (world_dir / Path(rel_path)).exists():
            reasons.append(f"recorded-absent file now exists: {rel_path}")

    canonical_labels = _canonical_order_labels(obj.required_labels, obj.replan_k, obj.predict_horizon)
    current_roster = _read_roster_seeds(canonical_labels, world_dir)
    if current_roster is None:
        reasons.append("roster manifest not found or unreadable among required labels")
    elif tuple(current_roster) != obj.seed_order:
        reasons.append("roster changed: derived_seeds no longer matches recorded seed_order")

    return (len(reasons) == 0, reasons)


def sidecar_covers(obj: SeedFilter, plotted_labels: Sequence[str]) -> bool:
    """True iff every label in `plotted_labels` is in `obj.required_labels`.

    Used by the plotter to refuse a drop when it is charting a label set the
    sidecar was never built against (e.g. a different `--replan-k` /
    `--predict-horizon`).
    """
    required = set(obj.required_labels)
    return all(label in required for label in plotted_labels)
