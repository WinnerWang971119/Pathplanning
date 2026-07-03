"""CLI labeler for the degenerate-seed filter (issue #9).

Reads one world's result tree, classifies every roster seed against a declared
set of required planner labels using the "every required planner dies
instantly" criterion, and writes the `_seed_filter.json` sidecar
(`runners._seed_filter.SEED_FILTER_NAME`) that `runners.plot` later reads to
drop degenerate seeds from the headline `failure_rate`.

This module imports ONLY `runners._seed_filter` (stdlib-backed) plus plain
stdlib (`argparse`, `json`, `sys`, `subprocess`, `pathlib`, `tempfile`) on its
runtime path. Importing `planners` or `runners.run_all` transitively pulls
irsim AND matplotlib (`planners -> manual_astar -> irsim ->
matplotlib.pyplot`), and `runners.run_all` imports `planners` at module top,
so even a lazy import of either at call time would contaminate this module's
headless guarantee (AC1). The ONLY place either is imported is inside the
TC-F15 subprocess script string, where it is explicitly permitted so a future
registry drift is caught loud. Verified by TC-F12's subprocess check.

CLI:
    python -m runners.filter_seeds \
        --world <yaml_path>          # required unless --selfcheck; only its stem is used
        [--results-dir <dir>]        # default "results"
        [--replan-k <int>]           # default 5; folds _k<K> into the replan-family labels
        [--predict-horizon <int>]    # default 10; folds _h<H> into the predictive labels
        [--window-seconds <float>]   # default 4.0; instant-crash window in sim seconds
        [--step-time <float>]        # default 0.1; sim seconds per tick
        [--planners {all13,canonical}]  # default all13
        [--selfcheck]                # run the TC-F self-check suite instead

Outputs:
    <results-dir>/<world_stem>/_seed_filter.json  (world-level sidecar; a
    SIBLING of the label dirs, never inside one, so the plotter's numeric-stem
    episode glob never sees it).

Exit codes:
    0 — sidecar written and the roster was determinable (global_status == "ok")
    1 — no result tree / no seed roster found; nothing written
    2 — CLI / validation error (before any file work)
    3 — sidecar written but global_status == "indeterminate" (a required label
        dir is entirely absent, or a cross-manifest consensus mismatch) — the
        loud no-op: filtering ran but decided nothing can be safely dropped
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Make the repo root importable so `runners._seed_filter` resolves when this
# module is invoked as `python -m runners.filter_seeds` from any cwd, or run
# directly as a script. Mirrors runners/run_experiment.py:76-81 /
# runners/plot.py:47-52.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runners._seed_filter import (  # noqa: E402
    CRITERION_ID,
    DEFAULT_STEP_TIME,
    DEFAULT_WINDOW_SECONDS,
    SCHEMA_VERSION,
    SEED_FILTER_NAME,
    PlannerEvidence,
    SeedFilter,
    SeedVerdictRow,
    build_required_labels,
    classify_planner,
    classify_seed,
    crash_step_from_trace,
    file_sha256,
    read_seed_filter,
    sidecar_is_fresh,
    window_steps,
    write_seed_filter,
)


# --- Module constants --------------------------------------------------------

DEFAULT_RESULTS_DIR = "results"
DEFAULT_REPLAN_K = 5                 # canonical cadence; matches run_all.REPLAN_K
DEFAULT_PREDICT_HORIZON = 10         # matches the shipped d_star_lite_oracle/predictive h10 runs
MANIFEST_NAME = "_manifest.json"     # hand-copied (not imported) to keep this module's only
                                      # sibling-module import scoped to runners._seed_filter
GIT_SHA_TIMEOUT_S = 10.0             # hard wall on the git-sha probe (mirrors run_experiment.py)

# Best-effort provenance defaults used ONLY when no required-label manifest at
# all is found (the roster came from the numeric-stem fallback). Match
# arena.speed_regimes.SPEED_REGIMES["current"] / DEFAULT_REGIME (the Mission
# baseline) without importing that module, keeping this file's dependency
# surface limited to runners._seed_filter + stdlib.
_FALLBACK_TRAFFIC = True
_FALLBACK_SPEED_REGIME = "current"
_FALLBACK_SPEED_MIN_FACTOR = 0.3
_FALLBACK_SPEED_MAX_FACTOR = 1.5

# The manifest fields that must agree across every required label's manifest
# (CR6 / AC14). `git_sha` is checked separately and is warning-only.
_CONSENSUS_FIELDS = (
    "derived_seeds",
    "world_stem",
    "traffic",
    "speed_regime",
    "speed_min_factor",
    "speed_max_factor",
)


# --- Manifest helpers ---------------------------------------------------------

def _read_manifest(world_dir: Path, label: str) -> dict | None:
    """Parse `<world_dir>/<label>/_manifest.json`, or None if absent/unreadable/not-an-object."""
    manifest_path = world_dir / label / MANIFEST_NAME
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _find_base_manifest(world_dir: Path, canonical_labels: list[str]) -> tuple[str, dict] | None:
    """First (label, manifest) in canonical order carrying a `derived_seeds` list."""
    for label in canonical_labels:
        manifest = _read_manifest(world_dir, label)
        if manifest is None:
            continue
        derived = manifest.get("derived_seeds")
        if isinstance(derived, list):
            return label, manifest
    return None


def _find_any_manifest(world_dir: Path, required_labels: list[str]) -> dict | None:
    """First readable manifest among ALL required labels, for best-effort provenance
    when no canonical-order manifest carries a usable roster."""
    for label in required_labels:
        manifest = _read_manifest(world_dir, label)
        if manifest is not None:
            return manifest
    return None


def _check_consensus(
    world_dir: Path,
    required_labels: list[str],
    base_label: str,
    base_manifest: dict,
    warnings: list[str],
) -> bool:
    """Every OTHER found required-label manifest must agree with the base on
    `_CONSENSUS_FIELDS` (CR6). A `git_sha` mismatch is warning-only and never
    affects the returned bool. Returns True iff no real mismatch was found."""
    ok = True
    for label in required_labels:
        if label == base_label:
            continue
        manifest = _read_manifest(world_dir, label)
        if manifest is None:
            continue  # a missing manifest is not a consensus concern here (see missing_labels)
        mismatched = [f for f in _CONSENSUS_FIELDS if manifest.get(f) != base_manifest.get(f)]
        if mismatched:
            warnings.append(
                f"manifest consensus mismatch: {label!r} disagrees with {base_label!r} on "
                f"{', '.join(mismatched)}"
            )
            ok = False
        if manifest.get("git_sha") != base_manifest.get("git_sha"):
            warnings.append(
                f"git_sha mismatch (informational only): {label!r} vs {base_label!r}"
            )
    return ok


def _fallback_roster(world_dir: Path, required_labels: list[str]) -> list[int]:
    """Sorted union of numeric `<seed>.json` stems across every required label dir.

    Used only when no required-label manifest carries a usable `derived_seeds`
    list; sets `roster_is_canonical=False` in the caller.
    """
    if not world_dir.is_dir():
        return []
    seeds: set[int] = set()
    for label in required_labels:
        label_dir = world_dir / label
        if not label_dir.is_dir():
            continue
        for path in label_dir.glob("[0-9]*.json"):
            try:
                seeds.add(int(path.stem))
            except ValueError:
                continue
    return sorted(seeds)


def _git_sha(repo_root: Path) -> str | None:
    """Best-effort local `git rev-parse HEAD`; None if not a git checkout / git absent.

    This is the CURRENT checkout's sha recorded on the sidecar for provenance —
    distinct from the per-manifest `git_sha` FIELDS compared in `_check_consensus`.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=GIT_SHA_TIMEOUT_S,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


# --- Evidence collection -------------------------------------------------------

def _read_trace_lines(trace_path: Path, warnings: list[str], label: str, seed: int) -> list[dict] | None:
    """Parsed trace-line dicts, or None if the trace is absent/unreadable (warns)."""
    if not trace_path.exists():
        warnings.append(f"missing trace for crashed episode {label}/{seed}.trace.jsonl")
        return None
    lines: list[dict] = []
    try:
        with open(trace_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                parsed = json.loads(raw_line)
                if isinstance(parsed, dict):
                    lines.append(parsed)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"unreadable trace {label}/{seed}.trace.jsonl: {exc}")
        return None
    return lines


def _evidence_for(
    seed: int,
    label: str,
    world_dir: Path,
    window_steps_val: int,
    consulted_hashes: dict[str, str],
    absent_files: list[str],
    warnings: list[str],
) -> PlannerEvidence:
    """One (seed, label) pair's `PlannerEvidence`, per the plan's evidence-collection steps."""
    label_dir = world_dir / label
    metrics_path = label_dir / f"{seed}.json"
    rel_metrics = f"{label}/{seed}.json"

    if not metrics_path.exists():
        # A DNF roster entry (the batch killed this seed past the wallclock wall,
        # so it provably ran past the instant-crash window) counts as survived.
        manifest = _read_manifest(world_dir, label)
        if manifest is not None:
            for episode in manifest.get("episodes", []):
                if (
                    isinstance(episode, dict)
                    and episode.get("seed") == seed
                    and episode.get("status") == "runner_error"
                ):
                    return PlannerEvidence(label=label, status="survived", crash_step=None)
        absent_files.append(rel_metrics)
        return PlannerEvidence(label=label, status="missing", crash_step=None)

    try:
        with open(metrics_path, "r", encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"unreadable metrics file {rel_metrics}: {exc}")
        absent_files.append(rel_metrics)
        return PlannerEvidence(label=label, status="missing", crash_step=None)

    # Valid JSON that is not an object (e.g. [] or null) has no .get; treat it
    # like an unreadable metrics file -> missing, rather than crashing.
    if not isinstance(record, dict):
        warnings.append(f"metrics file {rel_metrics} is not a JSON object; treating as missing")
        absent_files.append(rel_metrics)
        return PlannerEvidence(label=label, status="missing", crash_step=None)

    digest = file_sha256(metrics_path)
    if digest is not None:
        consulted_hashes[rel_metrics] = digest

    crashed = bool(record.get("crashed"))
    if not crashed:
        return PlannerEvidence(label=label, status="survived", crash_step=None)

    trace_path = label_dir / f"{seed}.trace.jsonl"
    lines = _read_trace_lines(trace_path, warnings, label, seed)
    if lines is None:
        # Cannot confirm instant -> conservative "survived" (blocks a drop).
        return PlannerEvidence(label=label, status="survived", crash_step=None)

    crash_step = crash_step_from_trace(lines)
    status = classify_planner(
        present=True, crashed=True, crash_step=crash_step, window_steps=window_steps_val
    )
    return PlannerEvidence(
        label=label,
        status=status,
        crash_step=crash_step if status == "instant_crash" else None,
    )


# --- Core pipeline -------------------------------------------------------------

def _label_effectively_absent(world_dir: Path, label: str) -> bool:
    """True if a required label dir cannot contribute any evidence.

    Absent when the dir does not exist, OR it exists but holds neither a readable
    manifest nor any numeric-stem `<seed>.json` — an empty dir would leave every
    seed indeterminate, so treat it like a missing dir so the loud no-op (exit 3)
    fires instead of a silent ok/exit-0.
    """
    label_dir = world_dir / label
    if not label_dir.is_dir():
        return True
    if _read_manifest(world_dir, label) is not None:
        return False
    return not any(label_dir.glob("[0-9]*.json"))


def run_filter(
    *,
    world_stem: str,
    results_dir: str,
    replan_k: int,
    predict_horizon: int,
    window_seconds: float,
    step_time: float,
    planners: str,
) -> tuple[SeedFilter | None, int]:
    """Label every roster seed and write the sidecar. Returns (obj_or_None, exit_code).

    `obj` is None only in the "no result tree / no seed roster" case (exit 1);
    an error is already printed to stderr in that case and nothing is written.
    Every other path writes the sidecar to
    `<results_dir>/<world_stem>/_seed_filter.json` before returning.
    """
    results_root = Path(results_dir).resolve()
    world_dir = results_root / world_stem

    canonical_labels = build_required_labels(predict_horizon, replan_k, "canonical")
    required_labels = build_required_labels(predict_horizon, replan_k, planners)
    window_steps_val = window_steps(window_seconds, step_time)

    warnings: list[str] = []

    found_base = _find_base_manifest(world_dir, canonical_labels)
    base_seed_order: tuple[int, ...] | None = None
    if found_base is not None:
        base_label, base_manifest = found_base
        try:
            base_seed_order = tuple(int(s) for s in base_manifest["derived_seeds"])
        except (TypeError, ValueError):
            warnings.append(
                f"manifest for {base_label} has non-integer derived_seeds; "
                "falling back to the numeric-stem roster"
            )
            base_seed_order = None

    if base_seed_order:
        seed_order = base_seed_order
        roster_is_canonical = True
        consensus_ok = _check_consensus(world_dir, required_labels, base_label, base_manifest, warnings)
        provenance_manifest: dict | None = base_manifest
    else:
        seed_order = tuple(_fallback_roster(world_dir, required_labels))
        roster_is_canonical = False
        consensus_ok = True
        provenance_manifest = _find_any_manifest(world_dir, required_labels)
        if seed_order:
            warnings.append(
                "no canonical-order manifest with derived_seeds found; roster is the sorted "
                "union of numeric-stem <seed>.json files present (roster_is_canonical=False)"
            )

    if not seed_order:
        print(
            f"error: no seed roster found under {world_dir} (no required-label manifest with "
            "derived_seeds, and no numeric-stem <seed>.json files under any required label dir)",
            file=sys.stderr,
        )
        return None, 1

    missing_labels = tuple(label for label in required_labels if _label_effectively_absent(world_dir, label))
    for label in missing_labels:
        warnings.append(f"required label dir entirely absent: {label}")

    global_status = "ok" if (not missing_labels and consensus_ok) else "indeterminate"

    if provenance_manifest is not None:
        traffic = bool(provenance_manifest.get("traffic", _FALLBACK_TRAFFIC))
        speed_regime = provenance_manifest.get("speed_regime")
        speed_min_factor = float(provenance_manifest.get("speed_min_factor", _FALLBACK_SPEED_MIN_FACTOR))
        speed_max_factor = float(provenance_manifest.get("speed_max_factor", _FALLBACK_SPEED_MAX_FACTOR))
    else:
        traffic = _FALLBACK_TRAFFIC
        speed_regime = _FALLBACK_SPEED_REGIME
        speed_min_factor = _FALLBACK_SPEED_MIN_FACTOR
        speed_max_factor = _FALLBACK_SPEED_MAX_FACTOR

    rows: list[SeedVerdictRow] = []
    consulted_hashes: dict[str, str] = {}
    absent_files: list[str] = []
    dropped_seeds: list[int] = []

    for seed in seed_order:
        evidence = [
            _evidence_for(seed, label, world_dir, window_steps_val, consulted_hashes, absent_files, warnings)
            for label in required_labels
        ]
        evidence_sorted = tuple(sorted(evidence, key=lambda e: e.label))
        raw_verdict = classify_seed(evidence_sorted)
        # A globally indeterminate run (missing required dir, or a cross-manifest
        # consensus mismatch) never drops a seed, even one whose raw per-planner
        # evidence would otherwise be a clean "every planner crashed instantly"
        # (AC14's "no drop"); a would-be "kept" verdict is left untouched (AC6).
        verdict = "indeterminate" if (raw_verdict == "degenerate" and global_status == "indeterminate") else raw_verdict
        if verdict == "degenerate":
            dropped_seeds.append(seed)
        rows.append(SeedVerdictRow(seed=seed, verdict=verdict, planners=evidence_sorted))

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    obj = SeedFilter(
        schema_version=SCHEMA_VERSION,
        criterion_id=CRITERION_ID,
        world_stem=world_stem,
        window_seconds=float(window_seconds),
        step_time=float(step_time),
        window_steps=window_steps_val,
        replan_k=replan_k,
        predict_horizon=predict_horizon,
        traffic=traffic,
        speed_regime=speed_regime,
        speed_min_factor=speed_min_factor,
        speed_max_factor=speed_max_factor,
        required_labels=tuple(required_labels),
        roster_is_canonical=roster_is_canonical,
        git_sha=_git_sha(_REPO_ROOT),
        global_status=global_status,
        missing_labels=missing_labels,
        seed_order=seed_order,
        dropped_seeds=tuple(dropped_seeds),
        rows=tuple(rows),
        consulted_hashes=consulted_hashes,
        absent_files=tuple(absent_files),
    )

    world_dir.mkdir(parents=True, exist_ok=True)
    write_seed_filter(obj, world_dir / SEED_FILTER_NAME)

    exit_code = 3 if global_status == "indeterminate" else 0
    return obj, exit_code


def _print_verdict(obj: SeedFilter, exit_code: int, sidecar_path: Path) -> None:
    n_kept = sum(1 for row in obj.rows if row.verdict == "kept")
    n_indeterminate = sum(1 for row in obj.rows if row.verdict == "indeterminate")
    n_degenerate = len(obj.dropped_seeds)
    print(
        f"filter_seeds: world_stem={obj.world_stem} required_labels={len(obj.required_labels)} "
        f"roster_is_canonical={obj.roster_is_canonical} seeds={len(obj.seed_order)} "
        f"degenerate={n_degenerate} kept={n_kept} indeterminate={n_indeterminate} "
        f"global_status={obj.global_status}"
    )
    if obj.missing_labels:
        print(
            f"filter_seeds: missing required label dirs: {', '.join(obj.missing_labels)}",
            file=sys.stderr,
        )
    if obj.global_status == "indeterminate":
        print(
            "filter_seeds: nothing determinable - sidecar written but no seed can be safely "
            "dropped (see the missing/mismatch warnings above)",
            file=sys.stderr,
        )
    print(f"wrote {sidecar_path}")


# --- CLI ------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterArgs:
    """Parsed CLI arguments - frozen so accidental mutation is impossible."""

    world: str | None
    results_dir: str
    replan_k: int
    predict_horizon: int
    window_seconds: float
    step_time: float
    planners: str
    selfcheck: bool


def _parse_args(argv: list[str] | None) -> FilterArgs:
    parser = argparse.ArgumentParser(
        prog="runners.filter_seeds",
        description=(
            "Label degenerate traffic seeds (every required planner crashes instantly) and "
            "write the _seed_filter.json sidecar (issue #9)."
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
        help=f"Results root to read from (default {DEFAULT_RESULTS_DIR!r}).",
    )
    parser.add_argument(
        "--replan-k",
        type=int,
        default=DEFAULT_REPLAN_K,
        help=f"Replan cadence folded into the replan-family labels (default {DEFAULT_REPLAN_K}).",
    )
    parser.add_argument(
        "--predict-horizon",
        type=int,
        default=DEFAULT_PREDICT_HORIZON,
        help=(
            "Prediction horizon folded into the d_star_lite_oracle/d_star_lite_predictive "
            f"labels (default {DEFAULT_PREDICT_HORIZON})."
        ),
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help=f"Instant-crash window in sim seconds (default {DEFAULT_WINDOW_SECONDS}).",
    )
    parser.add_argument(
        "--step-time",
        type=float,
        default=DEFAULT_STEP_TIME,
        help=f"Sim seconds per tick, for converting --window-seconds to a step count "
        f"(default {DEFAULT_STEP_TIME}).",
    )
    parser.add_argument(
        "--planners",
        choices=["all13", "canonical"],
        default="all13",
        help="Required planner set: all13 (11 canonical + 2 experimental, default) or "
        "canonical (the 11 only).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="Run the TC-F self-check suite instead of labeling a world.",
    )
    ns = parser.parse_args(argv)

    if ns.replan_k < 1:
        parser.error(f"--replan-k must be >= 1, got {ns.replan_k}")
    if ns.predict_horizon < 0:
        parser.error(f"--predict-horizon must be >= 0, got {ns.predict_horizon}")
    if ns.window_seconds <= 0:
        parser.error(f"--window-seconds must be > 0, got {ns.window_seconds}")
    if ns.step_time <= 0:
        parser.error(f"--step-time must be > 0, got {ns.step_time}")

    return FilterArgs(
        world=ns.world,
        results_dir=ns.results_dir,
        replan_k=int(ns.replan_k),
        predict_horizon=int(ns.predict_horizon),
        window_seconds=float(ns.window_seconds),
        step_time=float(ns.step_time),
        planners=ns.planners,
        selfcheck=bool(ns.selfcheck),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the labeler end-to-end. See module docstring for CLI semantics."""
    args = _parse_args(argv)

    if args.selfcheck:
        # Selfcheck ignores --world entirely; run it before any --world
        # validation so the documented `python -m runners.filter_seeds
        # --selfcheck` command (no --world) works.
        return run_selfcheck()

    if args.world is None:
        print("error: --world is required unless --selfcheck is given", file=sys.stderr)
        return 2

    world_stem = Path(args.world).stem
    obj, exit_code = run_filter(
        world_stem=world_stem,
        results_dir=args.results_dir,
        replan_k=args.replan_k,
        predict_horizon=args.predict_horizon,
        window_seconds=args.window_seconds,
        step_time=args.step_time,
        planners=args.planners,
    )
    if obj is None:
        return exit_code

    sidecar_path = Path(args.results_dir).resolve() / world_stem / SEED_FILTER_NAME
    _print_verdict(obj, exit_code, sidecar_path)
    return exit_code


# --- Self-check (TC-F1..TC-F15) --------------------------------------------
#
# Synthetic result trees built in a TemporaryDirectory - no irsim, no real
# episodes - mirroring runners/plot.py's run_selfcheck. Each TC is a plain
# function that asserts its invariants and returns a short success detail
# string; `run_selfcheck` catches per-TC exceptions so one failure never
# aborts the rest, prints a PASS/FAIL line each, and returns an int exit code.

def _write_metrics(
    label_dir: Path,
    seed: int,
    *,
    crashed: bool,
    time_to_goal: float | None = None,
    timed_out: bool = False,
    planner_error: str | None = None,
) -> None:
    """Write one synthetic `<seed>.json` metrics record (the 7 run_episode fields)."""
    label_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "time_to_goal": time_to_goal,
        "crashed": crashed,
        "timed_out": timed_out,
        "path_length": 0.0,
        "mean_speed": 0.0,
        "wallclock_per_step": 0.01,
        "planner_error": planner_error,
    }
    (label_dir / f"{seed}.json").write_text(json.dumps(record, sort_keys=True), encoding="utf-8")


def _write_trace(label_dir: Path, seed: int, crash_step: int | None) -> None:
    """Write a tiny synthetic `<seed>.trace.jsonl`: step-0 sentinel plus, when
    `crash_step` is given, steps up to and including the crash (crashed=True
    only on the final line); a short 2-step non-crashing trace otherwise."""
    label_dir.mkdir(parents=True, exist_ok=True)
    last_step = 1 if crash_step is None else crash_step
    lines = []
    for step in range(0, last_step + 1):
        crashed_flag = crash_step is not None and step == crash_step
        lines.append(
            {
                "step": step,
                "state": [0.0, 0.0, 0.0],
                "action": [0.0, 0.0],
                "crashed": crashed_flag,
                "reached_goal": False,
                "done": crashed_flag,
                "lidar_sha256": "0" * 64,
            }
        )
    text = "\n".join(json.dumps(line, sort_keys=True) for line in lines) + "\n"
    (label_dir / f"{seed}.trace.jsonl").write_text(text, encoding="utf-8")


def _write_manifest(
    label_dir: Path,
    *,
    derived_seeds: list[int],
    world_stem: str = "w",
    traffic: bool = True,
    speed_regime: str | None = "current",
    speed_min_factor: float = 0.3,
    speed_max_factor: float = 1.5,
    git_sha: str | None = "abc123",
    episodes: list[dict] | None = None,
) -> None:
    """Write one synthetic `_manifest.json` (the run_experiment provenance shape)."""
    label_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "master_seed": 1,
        "num_seeds": len(derived_seeds),
        "world_stem": world_stem,
        "traffic": traffic,
        "speed_regime": speed_regime,
        "speed_min_factor": speed_min_factor,
        "speed_max_factor": speed_max_factor,
        "git_sha": git_sha,
        "derived_seeds": list(derived_seeds),
        "episodes": (
            episodes
            if episodes is not None
            else [{"seed": s, "exit_code": 0, "status": "ok"} for s in derived_seeds]
        ),
    }
    (label_dir / MANIFEST_NAME).write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _write_manifests_for_labels(
    world_dir: Path, labels: list[str], derived_seeds: list[int], **overrides
) -> None:
    """Write an identical manifest to every given label dir (a consistent baseline)."""
    for label in labels:
        _write_manifest(world_dir / label, derived_seeds=derived_seeds, **overrides)


def _default_outcomes(required_labels: list[str], *, crash_step: int) -> dict[str, dict]:
    """Every required label crashes at `crash_step` - the "everyone dies instantly" baseline."""
    return {label: {"crashed": True, "crash_step": crash_step} for label in required_labels}


def _write_seed_fixture(world_dir: Path, seed: int, label_outcomes: dict[str, dict]) -> None:
    """Write one seed's metrics (+ trace, if crashed) for each (label -> outcome spec) pair.

    outcome spec keys: `crashed` (bool), `crash_step` (int|None, only meaningful
    when crashed), `time_to_goal` (float, for a success), `timed_out` (bool),
    `planner_error` (str), `missing` (bool - write nothing for this label at all).
    """
    for label, spec in label_outcomes.items():
        if spec.get("missing"):
            continue
        label_dir = world_dir / label
        crashed = bool(spec.get("crashed", False))
        _write_metrics(
            label_dir,
            seed,
            crashed=crashed,
            time_to_goal=spec.get("time_to_goal"),
            timed_out=spec.get("timed_out", False),
            planner_error=spec.get("planner_error"),
        )
        if crashed:
            _write_trace(label_dir, seed, spec.get("crash_step"))


def _tc_f1_all_instant_crash(tmp: Path) -> str:
    """TC-F1: every required planner crashes at step<=40 -> degenerate, in dropped_seeds."""
    world_dir = tmp / "tc_f1" / "w"
    required = build_required_labels(10, 5, "all13")
    seed = 1
    _write_manifests_for_labels(world_dir, required, [seed])
    _write_seed_fixture(world_dir, seed, _default_outcomes(required, crash_step=5))

    obj, exit_code = run_filter(
        world_stem="w", results_dir=str(tmp / "tc_f1"), replan_k=5, predict_horizon=10,
        window_seconds=DEFAULT_WINDOW_SECONDS, step_time=DEFAULT_STEP_TIME, planners="all13",
    )
    assert obj is not None, "a fixture with a determinable roster must not return None"
    row = obj.rows[0]
    assert row.verdict == "degenerate", f"expected degenerate, got {row.verdict}"
    assert seed in obj.dropped_seeds, f"seed must be in dropped_seeds, got {obj.dropped_seeds}"
    assert exit_code == 0, f"a determinable all-crash seed must exit 0, got {exit_code}"
    return f"seed {seed} degenerate across {len(required)} required planners; exit=0"


def _tc_f2_one_survivor_goal(tmp: Path) -> str:
    """TC-F2: one planner reaches goal on that seed -> verdict kept, not dropped."""
    world_dir = tmp / "tc_f2" / "w"
    required = build_required_labels(10, 5, "all13")
    seed = 2
    _write_manifests_for_labels(world_dir, required, [seed])
    outcomes = _default_outcomes(required, crash_step=3)
    outcomes[required[0]] = {"crashed": False, "time_to_goal": 50.0}
    _write_seed_fixture(world_dir, seed, outcomes)

    obj, _ = run_filter(
        world_stem="w", results_dir=str(tmp / "tc_f2"), replan_k=5, predict_horizon=10,
        window_seconds=DEFAULT_WINDOW_SECONDS, step_time=DEFAULT_STEP_TIME, planners="all13",
    )
    row = obj.rows[0]
    assert row.verdict == "kept", f"expected kept, got {row.verdict}"
    assert seed not in obj.dropped_seeds
    return f"seed {seed} kept via a goal-reaching survivor ({required[0]})"


def _tc_f3_late_crash_survives(tmp: Path) -> str:
    """TC-F3: one planner crashes at step 41 (late) -> that planner survived; seed kept."""
    world_dir = tmp / "tc_f3" / "w"
    required = build_required_labels(10, 5, "all13")
    seed = 3
    _write_manifests_for_labels(world_dir, required, [seed])
    outcomes = _default_outcomes(required, crash_step=1)
    outcomes[required[1]] = {"crashed": True, "crash_step": 41}
    _write_seed_fixture(world_dir, seed, outcomes)

    obj, _ = run_filter(
        world_stem="w", results_dir=str(tmp / "tc_f3"), replan_k=5, predict_horizon=10,
        window_seconds=DEFAULT_WINDOW_SECONDS, step_time=DEFAULT_STEP_TIME, planners="all13",
    )
    row = obj.rows[0]
    late_evidence = next(e for e in row.planners if e.label == required[1])
    assert late_evidence.status == "survived", f"step 41 must be survived, got {late_evidence.status}"
    assert row.verdict == "kept", f"expected kept, got {row.verdict}"
    return "crash at step 41 classified survived (late); seed kept"


def _tc_f4_missing_metrics(tmp: Path) -> str:
    """TC-F4: one required planner has no <seed>.json (not DNF) -> missing; absent_files; indeterminate."""
    world_dir = tmp / "tc_f4" / "w"
    required = build_required_labels(10, 5, "all13")
    seed = 4
    _write_manifests_for_labels(world_dir, required, [seed])  # default episodes roster: all "ok"
    outcomes = _default_outcomes(required, crash_step=2)
    outcomes[required[2]] = {"missing": True}
    _write_seed_fixture(world_dir, seed, outcomes)

    obj, _ = run_filter(
        world_stem="w", results_dir=str(tmp / "tc_f4"), replan_k=5, predict_horizon=10,
        window_seconds=DEFAULT_WINDOW_SECONDS, step_time=DEFAULT_STEP_TIME, planners="all13",
    )
    row = obj.rows[0]
    missing_evidence = next(e for e in row.planners if e.label == required[2])
    assert missing_evidence.status == "missing", f"expected missing, got {missing_evidence.status}"
    assert row.verdict == "indeterminate", f"expected indeterminate, got {row.verdict}"
    rel = f"{required[2]}/{seed}.json"
    assert rel in obj.absent_files, f"{rel} must be recorded in absent_files, got {obj.absent_files}"
    return f"missing (non-DNF) metrics for {required[2]} -> indeterminate; recorded in absent_files"


def _tc_f5_window_boundary(_tmp: Path) -> str:
    """TC-F5: window boundary - step 40 instant, step 41 not, step-0 sentinel never a crash."""
    lines_step0_only = [{"step": 0, "crashed": False}]
    assert crash_step_from_trace(lines_step0_only) is None, "the step-0 sentinel must never be a crash"

    w = window_steps(DEFAULT_WINDOW_SECONDS, DEFAULT_STEP_TIME)
    assert w == 40, f"expected window_steps=40 for 4.0s/0.1s, got {w}"

    lines_40 = [{"step": 0, "crashed": False}, {"step": 40, "crashed": True}]
    step_40 = crash_step_from_trace(lines_40)
    status_40 = classify_planner(present=True, crashed=True, crash_step=step_40, window_steps=w)
    assert status_40 == "instant_crash", f"step==window_steps must be instant, got {status_40}"

    lines_41 = [{"step": 0, "crashed": False}, {"step": 41, "crashed": True}]
    step_41 = crash_step_from_trace(lines_41)
    status_41 = classify_planner(present=True, crashed=True, crash_step=step_41, window_steps=w)
    assert status_41 == "survived", f"step 41 must not be instant, got {status_41}"

    return "step 40 instant, step 41 not, step-0 sentinel never a crash"


def _tc_f6_planner_error_survives(tmp: Path) -> str:
    """TC-F6: a planner_error record (no trace, crashed=False) -> survived (blocks a drop)."""
    world_dir = tmp / "tc_f6" / "w"
    required = build_required_labels(10, 5, "all13")
    seed = 6
    _write_manifests_for_labels(world_dir, required, [seed])
    outcomes = _default_outcomes(required, crash_step=1)
    outcomes[required[3]] = {"crashed": False, "planner_error": "synthetic planner failure"}
    _write_seed_fixture(world_dir, seed, outcomes)

    obj, _ = run_filter(
        world_stem="w", results_dir=str(tmp / "tc_f6"), replan_k=5, predict_horizon=10,
        window_seconds=DEFAULT_WINDOW_SECONDS, step_time=DEFAULT_STEP_TIME, planners="all13",
    )
    row = obj.rows[0]
    evidence = next(e for e in row.planners if e.label == required[3])
    assert evidence.status == "survived", f"expected survived, got {evidence.status}"
    assert row.verdict == "kept", f"expected kept, got {row.verdict}"
    return f"planner_error record for {required[3]} classified survived; seed kept"


def _tc_f6b_dnf_roster_survives(tmp: Path) -> str:
    """TC-F6b: a DNF roster entry (manifest status=='runner_error', no JSON) -> survived, kept."""
    world_dir = tmp / "tc_f6b" / "w"
    required = build_required_labels(10, 5, "all13")
    seed = 7
    dnf_label = required[4]
    other_labels = [label for label in required if label != dnf_label]
    _write_manifests_for_labels(world_dir, other_labels, [seed])
    _write_manifest(
        world_dir / dnf_label,
        derived_seeds=[seed],
        episodes=[{"seed": seed, "exit_code": 124, "status": "runner_error"}],
    )

    outcomes = _default_outcomes(required, crash_step=1)
    outcomes[dnf_label] = {"missing": True}
    _write_seed_fixture(world_dir, seed, outcomes)

    obj, _ = run_filter(
        world_stem="w", results_dir=str(tmp / "tc_f6b"), replan_k=5, predict_horizon=10,
        window_seconds=DEFAULT_WINDOW_SECONDS, step_time=DEFAULT_STEP_TIME, planners="all13",
    )
    row = obj.rows[0]
    evidence = next(e for e in row.planners if e.label == dnf_label)
    assert evidence.status == "survived", f"a DNF roster entry must be survived, got {evidence.status}"
    assert row.verdict == "kept", f"expected kept, got {row.verdict}"
    rel = f"{dnf_label}/{seed}.json"
    assert rel not in obj.absent_files, "a DNF roster hit must not be recorded in absent_files"
    return f"DNF roster entry for {dnf_label} classified survived; seed kept"


def _tc_f7_missing_label_dir(tmp: Path) -> str:
    """TC-F7: a whole required label dir absent (AC6) - global_status indeterminate,
    missing_labels set, distinct exit code, dropped_seeds==[], a real survivor still kept."""
    world_dir = tmp / "tc_f7" / "w"
    required = build_required_labels(10, 5, "canonical")
    missing_label = required[0]
    present_labels = required[1:]
    seeds = [10, 11]

    # missing_label's dir is never created at all - no manifest, no metrics.
    _write_manifests_for_labels(world_dir, present_labels, seeds)

    # seed 10: every present planner crashes instantly -> would be degenerate if
    # the missing dir did not block it.
    _write_seed_fixture(world_dir, 10, _default_outcomes(present_labels, crash_step=1))

    # seed 11: one present planner survives -> must still classify kept.
    outcomes_11 = _default_outcomes(present_labels, crash_step=1)
    outcomes_11[present_labels[0]] = {"crashed": False, "time_to_goal": 33.0}
    _write_seed_fixture(world_dir, 11, outcomes_11)

    obj, exit_code = run_filter(
        world_stem="w", results_dir=str(tmp / "tc_f7"), replan_k=5, predict_horizon=10,
        window_seconds=DEFAULT_WINDOW_SECONDS, step_time=DEFAULT_STEP_TIME, planners="canonical",
    )
    assert obj.global_status == "indeterminate", f"expected indeterminate, got {obj.global_status}"
    assert missing_label in obj.missing_labels, f"{missing_label} must be in missing_labels"
    assert obj.dropped_seeds == (), f"a missing required dir must block every drop, got {obj.dropped_seeds}"
    row10 = next(r for r in obj.rows if r.seed == 10)
    row11 = next(r for r in obj.rows if r.seed == 11)
    assert row10.verdict == "indeterminate", f"seed 10 must be indeterminate, got {row10.verdict}"
    assert row11.verdict == "kept", f"seed 11 (has a survivor) must still be kept, got {row11.verdict}"
    assert exit_code == 3, f"a missing required label dir must exit with the distinct code, got {exit_code}"
    return "missing label dir -> global_status indeterminate, dropped_seeds==[], exit=3; survivor still kept"


def _tc_f8_roundtrip(tmp: Path) -> str:
    """TC-F8: read_seed_filter(write(...)) equals the original; a schema_version mismatch -> None."""
    world_dir = tmp / "tc_f8" / "w"
    required = build_required_labels(10, 5, "canonical")
    seed = 20
    _write_manifests_for_labels(world_dir, required, [seed])
    outcomes = _default_outcomes(required, crash_step=1)
    outcomes[required[0]] = {"crashed": False, "time_to_goal": 12.0}
    _write_seed_fixture(world_dir, seed, outcomes)

    obj, _ = run_filter(
        world_stem="w", results_dir=str(tmp / "tc_f8"), replan_k=5, predict_horizon=10,
        window_seconds=DEFAULT_WINDOW_SECONDS, step_time=DEFAULT_STEP_TIME, planners="canonical",
    )
    sidecar_path = world_dir / SEED_FILTER_NAME
    reloaded = read_seed_filter(sidecar_path)
    assert reloaded == obj, "read_seed_filter(write(obj)) must equal the original SeedFilter"

    bad_path = tmp / "tc_f8" / "bad_seed_filter.json"
    data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    data["schema_version"] = SCHEMA_VERSION + 1
    bad_path.write_text(json.dumps(data), encoding="utf-8")
    assert read_seed_filter(bad_path) is None, "a schema_version mismatch must return None"
    return "round-trip equality holds; a schema_version mismatch returns None"


def _tc_f9_determinism(tmp: Path) -> str:
    """TC-F9: two runs over one unchanged fixture tree -> byte-identical sidecar."""
    world_dir = tmp / "tc_f9" / "w"
    required = build_required_labels(10, 5, "all13")
    seed = 30
    _write_manifests_for_labels(world_dir, required, [seed])
    outcomes = _default_outcomes(required, crash_step=2)
    outcomes[required[5]] = {"crashed": False, "time_to_goal": 77.0}
    _write_seed_fixture(world_dir, seed, outcomes)

    run_filter(
        world_stem="w", results_dir=str(tmp / "tc_f9"), replan_k=5, predict_horizon=10,
        window_seconds=DEFAULT_WINDOW_SECONDS, step_time=DEFAULT_STEP_TIME, planners="all13",
    )
    sidecar_path = world_dir / SEED_FILTER_NAME
    first_bytes = sidecar_path.read_bytes()

    run_filter(
        world_stem="w", results_dir=str(tmp / "tc_f9"), replan_k=5, predict_horizon=10,
        window_seconds=DEFAULT_WINDOW_SECONDS, step_time=DEFAULT_STEP_TIME, planners="all13",
    )
    second_bytes = sidecar_path.read_bytes()

    assert first_bytes == second_bytes, "two runs over an unchanged tree must be byte-identical"
    return f"two runs -> {len(first_bytes)}-byte sidecar, byte-identical"


def _tc_f10_canonical_only(tmp: Path) -> str:
    """TC-F10: --planners canonical on an 11-only tree - required set is the 11; a seed can drop."""
    world_dir = tmp / "tc_f10" / "w"
    required = build_required_labels(10, 5, "canonical")
    assert len(required) == 11, f"expected 11 canonical labels, got {len(required)}"
    seed = 40
    _write_manifests_for_labels(world_dir, required, [seed])
    _write_seed_fixture(world_dir, seed, _default_outcomes(required, crash_step=1))

    obj, exit_code = run_filter(
        world_stem="w", results_dir=str(tmp / "tc_f10"), replan_k=5, predict_horizon=10,
        window_seconds=DEFAULT_WINDOW_SECONDS, step_time=DEFAULT_STEP_TIME, planners="canonical",
    )
    assert obj.required_labels == tuple(required), "recorded required_labels must equal the 11"
    assert len(obj.required_labels) == 11
    assert seed in obj.dropped_seeds, "a clean all-crash seed on an 11-only tree must still drop"
    assert exit_code == 0
    return "canonical-only 11-label tree: a seed drops; required_labels recorded as the 11"


def _tc_f11_freshness_and_posix_keys(tmp: Path) -> str:
    """TC-F11: POSIX-relative keys; mutate/absent-now-present/roster-change each trip sidecar_is_fresh."""
    world_dir = tmp / "tc_f11" / "w"
    required = build_required_labels(10, 5, "canonical")
    seed = 50
    _write_manifests_for_labels(world_dir, required, [seed])
    outcomes = _default_outcomes(required, crash_step=1)
    outcomes[required[0]] = {"crashed": False, "time_to_goal": 5.0}
    outcomes[required[1]] = {"missing": True}
    _write_seed_fixture(world_dir, seed, outcomes)

    results_dir = tmp / "tc_f11"
    obj, _ = run_filter(
        world_stem="w", results_dir=str(results_dir), replan_k=5, predict_horizon=10,
        window_seconds=DEFAULT_WINDOW_SECONDS, step_time=DEFAULT_STEP_TIME, planners="canonical",
    )

    for key in list(obj.consulted_hashes) + list(obj.absent_files):
        assert "/" in key and "\\" not in key, f"key must be POSIX-relative, got {key!r}"

    fresh, reasons = sidecar_is_fresh(obj, str(results_dir))
    assert fresh, f"an unchanged tree must be fresh, got reasons={reasons}"

    consulted_rel = next(iter(obj.consulted_hashes))
    mutated_path = world_dir / consulted_rel
    original = mutated_path.read_text(encoding="utf-8")
    mutated_path.write_text(original + " ", encoding="utf-8")
    fresh2, reasons2 = sidecar_is_fresh(obj, str(results_dir))
    assert not fresh2 and reasons2, "a mutated consulted file must trip freshness"
    mutated_path.write_text(original, encoding="utf-8")

    assert obj.absent_files, "this fixture must have recorded at least one absent file"
    absent_rel = obj.absent_files[0]
    absent_path = world_dir / absent_rel
    absent_path.parent.mkdir(parents=True, exist_ok=True)
    absent_path.write_text("{}", encoding="utf-8")
    fresh3, reasons3 = sidecar_is_fresh(obj, str(results_dir))
    assert not fresh3 and reasons3, "a newly-present recorded-absent file must trip freshness"
    absent_path.unlink()

    base_label = required[0]
    manifest_path = world_dir / base_label / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["derived_seeds"] = manifest["derived_seeds"] + [999]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    fresh4, reasons4 = sidecar_is_fresh(obj, str(results_dir))
    assert not fresh4 and reasons4, "a roster change must trip freshness"

    return "POSIX keys confirmed; hash/absent/roster mutations each trip sidecar_is_fresh"


def _tc_f12_headless_guard(_tmp: Path) -> str:
    """TC-F12: in a fresh subprocess, import + build_required_labels leaves matplotlib/irsim unimported (AC1)."""
    script = (
        "import sys\n"
        "import runners.filter_seeds as f\n"
        "f.build_required_labels(10, 5, 'all13')\n"
        "print('matplotlib' in sys.modules, 'irsim' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    assert result.returncode == 0, f"subprocess failed (exit {result.returncode}): {result.stderr}"
    output = result.stdout.strip()
    assert output == "False False", f"expected 'False False', got {output!r} (stderr: {result.stderr})"
    return "fresh subprocess: import + build_required_labels leaves matplotlib/irsim unimported"


def _tc_f13_window_override(tmp: Path) -> str:
    """TC-F13: --window-seconds / --step-time override the resolved window (and its boundary effect)."""
    world_dir = tmp / "tc_f13" / "w"
    required = build_required_labels(10, 5, "canonical")
    seed = 60
    _write_manifests_for_labels(world_dir, required, [seed])
    # A crash at step 8 is instant under a wide default window but late under a
    # narrow one - and instant again once step_time widens the same window.
    _write_seed_fixture(world_dir, seed, _default_outcomes(required, crash_step=8))

    obj, _ = run_filter(
        world_stem="w", results_dir=str(tmp / "tc_f13"), replan_k=5, predict_horizon=10,
        window_seconds=0.5, step_time=0.1, planners="canonical",
    )
    assert obj.window_seconds == 0.5 and obj.step_time == 0.1
    assert obj.window_steps == 5, f"expected window_steps=5 for 0.5s/0.1s, got {obj.window_steps}"
    row = obj.rows[0]
    assert row.verdict == "kept", f"crash at step 8 must be late under a 5-step window, got {row.verdict}"

    obj2, _ = run_filter(
        world_stem="w", results_dir=str(tmp / "tc_f13"), replan_k=5, predict_horizon=10,
        window_seconds=0.5, step_time=0.05, planners="canonical",
    )
    assert obj2.window_steps == 10, f"expected window_steps=10 for 0.5s/0.05s, got {obj2.window_steps}"
    row2 = obj2.rows[0]
    assert row2.verdict == "degenerate", f"crash at step 8 must be instant under a 10-step window, got {row2.verdict}"
    return "resolved window_seconds/step_time/window_steps recorded; boundary shift verified both ways"


def _tc_f14_cross_manifest_consensus(tmp: Path) -> str:
    """TC-F14 (AC14): a real mismatch -> indeterminate + no drop; a git_sha-only mismatch -> ok, warning only."""
    world_dir = tmp / "tc_f14" / "w"
    required = build_required_labels(10, 5, "canonical")
    seeds = [70, 71]

    _write_manifests_for_labels(world_dir, required, seeds)
    odd_label = required[3]
    manifest_path = world_dir / odd_label / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["traffic"] = not manifest["traffic"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    for seed in seeds:
        _write_seed_fixture(world_dir, seed, _default_outcomes(required, crash_step=1))

    obj, exit_code = run_filter(
        world_stem="w", results_dir=str(tmp / "tc_f14"), replan_k=5, predict_horizon=10,
        window_seconds=DEFAULT_WINDOW_SECONDS, step_time=DEFAULT_STEP_TIME, planners="canonical",
    )
    assert obj.global_status == "indeterminate", "a traffic mismatch must force indeterminate"
    assert exit_code == 3
    assert obj.dropped_seeds == (), "a consensus mismatch must block every drop"

    world_dir2 = tmp / "tc_f14b" / "w"
    _write_manifests_for_labels(world_dir2, required, seeds)
    odd_label2 = required[5]
    manifest_path2 = world_dir2 / odd_label2 / MANIFEST_NAME
    manifest2 = json.loads(manifest_path2.read_text(encoding="utf-8"))
    manifest2["git_sha"] = "deadbeef"
    manifest_path2.write_text(json.dumps(manifest2, sort_keys=True), encoding="utf-8")
    for seed in seeds:
        _write_seed_fixture(world_dir2, seed, _default_outcomes(required, crash_step=1))

    obj2, exit_code2 = run_filter(
        world_stem="w", results_dir=str(tmp / "tc_f14b"), replan_k=5, predict_horizon=10,
        window_seconds=DEFAULT_WINDOW_SECONDS, step_time=DEFAULT_STEP_TIME, planners="canonical",
    )
    assert obj2.global_status == "ok", f"a git_sha-only mismatch must stay ok, got {obj2.global_status}"
    assert exit_code2 == 0
    assert obj2.dropped_seeds == tuple(seeds), "with real consensus intact, the clean seeds must still drop"
    return "real field mismatch -> indeterminate/no-drop/exit3; git_sha-only mismatch -> ok/warning/drops"


def _tc_f15_label_parity(_tmp: Path) -> str:
    """TC-F15 (AC15, CR7): build_required_labels equals canonical_planner_set() + the 2 experimental labels."""
    script = (
        "from runners.filter_seeds import build_required_labels\n"
        "from runners.run_all import canonical_planner_set\n"
        "from planners import algorithm_label\n"
        "expected = [label for _n, _k, label in canonical_planner_set()] + [\n"
        "    algorithm_label('d_star_lite_oracle', None, 10),\n"
        "    algorithm_label('d_star_lite_predictive', None, 10),\n"
        "]\n"
        "actual = build_required_labels(10, 5, 'all13')\n"
        "assert actual == expected, (actual, expected)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    assert result.returncode == 0, f"subprocess failed (exit {result.returncode}): {result.stderr}"
    assert result.stdout.strip() == "OK", f"unexpected output: {result.stdout!r} / {result.stderr!r}"
    return "build_required_labels(10,5,'all13') matches canonical_planner_set() + the 2 experimental labels"


# Ordered registry of the self-check cases. Each entry is (id, callable).
_SELFCHECK_CASES = (
    ("TC-F1", _tc_f1_all_instant_crash),
    ("TC-F2", _tc_f2_one_survivor_goal),
    ("TC-F3", _tc_f3_late_crash_survives),
    ("TC-F4", _tc_f4_missing_metrics),
    ("TC-F5", _tc_f5_window_boundary),
    ("TC-F6", _tc_f6_planner_error_survives),
    ("TC-F6b", _tc_f6b_dnf_roster_survives),
    ("TC-F7", _tc_f7_missing_label_dir),
    ("TC-F8", _tc_f8_roundtrip),
    ("TC-F9", _tc_f9_determinism),
    ("TC-F10", _tc_f10_canonical_only),
    ("TC-F11", _tc_f11_freshness_and_posix_keys),
    ("TC-F12", _tc_f12_headless_guard),
    ("TC-F13", _tc_f13_window_override),
    ("TC-F14", _tc_f14_cross_manifest_consensus),
    ("TC-F15", _tc_f15_label_parity),
)


def run_selfcheck() -> int:
    """Run the labeler's self-check suite (TC-F1..TC-F15, incl. TC-F6b). Return 0 if all pass, else 1.

    Builds every fixture inside a single TemporaryDirectory (each TC namespaces
    its own subdir), runs each TC in isolation so one failure never aborts the
    rest, prints a per-TC PASS/FAIL line, and ends with an "N/16 passed"
    summary. No irsim, no real episodes - synthetic JSON trees only.
    """
    n_passed = 0
    n_total = len(_SELFCHECK_CASES)

    with tempfile.TemporaryDirectory(prefix="filter_seeds_selfcheck_") as tmp_name:
        tmp = Path(tmp_name)
        for tc_id, tc_func in _SELFCHECK_CASES:
            try:
                detail = tc_func(tmp)
            except Exception as exc:  # noqa: BLE001 - one TC must not abort the suite
                print(f"{tc_id}: FAIL - {type(exc).__name__}: {exc}")
            else:
                n_passed += 1
                print(f"{tc_id}: PASS - {detail}")

    print(f"selfcheck: {n_passed}/{n_total} passed")
    return 0 if n_passed == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
