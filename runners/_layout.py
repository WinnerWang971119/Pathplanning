"""Shared on-disk results layout for the episode + experiment runners.

Single source of truth for the ``<results-dir>/<world_stem>/<algorithm>/`` tree
so the single-episode runner, the batch runner's manifest/resume probe, and the
paths handed to child subprocesses can never drift. That drift is exactly what
let a relative ``--results-dir`` split the manifest from the episode JSONs: the
batch parent built the path against its own cwd while each child rebuilt it
against ``cwd=repo_root``.
"""
from __future__ import annotations

from pathlib import Path


def episode_out_dir(results_dir: str | Path, world_stem: str, algorithm: str) -> Path:
    """Absolute output dir for one (world, algorithm): ``<results_dir>/<world_stem>/<algorithm>``.

    ``results_dir`` is resolved to an absolute path so a batch runner invoked
    from any cwd and its child episode subprocesses (launched with
    ``cwd=repo_root``) agree on the same tree. The batch runner resolves the
    results-dir root ONCE and forwards that absolute path to its children;
    resolving an already-absolute path here is a no-op, so a standalone
    ``run_episode`` call with a relative dir still works against its own cwd.
    """
    return Path(results_dir).resolve() / world_stem / algorithm
