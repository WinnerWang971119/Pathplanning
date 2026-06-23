"""Arena exception types, isolated to break the arena.arena <-> arena.dynamic cycle.

`arena.arena` imports `arena.dynamic` (for the traffic substrate) and
`arena.dynamic` needs `ArenaRuntimeError`. Defining these plain exception
subclasses here — a leaf module with no project imports — lets `arena.dynamic`
import them without reaching back into `arena.arena`, so neither module is left
partially initialized. `arena.arena` re-exports both names, so existing callers
that `from arena.arena import ArenaConfigError, ArenaRuntimeError` keep working.
"""

from __future__ import annotations


class ArenaConfigError(ValueError):
    """Raised at Arena.__init__ for malformed config (e.g. lidar beam count mismatch)."""


class ArenaRuntimeError(RuntimeError):
    """Raised mid-episode for irsim contract violations (e.g. lidar dict missing 'ranges')."""
