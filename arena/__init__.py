"""arena package — reusable irsim test environment for path-planning experiments."""
from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["Arena", "EpisodeInfo", "ArenaConfigError", "ArenaRuntimeError"]

if TYPE_CHECKING:  # keep static type/IDE resolution without an import-time cost
    from arena.arena import Arena, EpisodeInfo, ArenaConfigError, ArenaRuntimeError


def __getattr__(name: str):
    # Lazily forward the public names to arena.arena (which imports irsim) only
    # when actually accessed, so importing pure submodules like
    # arena.speed_regimes stays irsim-free (AC11 headless guarantee).
    if name in __all__:
        from arena import arena as _arena_mod
        return getattr(_arena_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
