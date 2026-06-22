"""Pure (stdlib-only) speed-regime table, resolver, and shared CLI plumbing.

This module is the single source of truth for the dynamic-obstacle speed band
used by the obstacle-speed-cap sweep (issue #11). It is deliberately
**headless**: it imports ONLY the Python standard library (``argparse``), and
must NEVER import ``arena.arena``, ``arena.dynamic``, ``irsim``, ``numpy``, or
``matplotlib`` — anything that transitively pulls irsim into the process.

The reason is concrete: ``arena/dynamic.py`` does
``from arena.arena import ArenaRuntimeError``, and ``arena/arena.py`` does
``import irsim``. The headless plotter (``runners/plot_speed_sweep.py``) and its
``--selfcheck`` must read the speed-regime constants without standing up irsim,
so they import them from THIS module rather than from ``arena.dynamic``.

Each regime maps a name to ``(min_factor, max_factor)`` — factors of the robot's
top speed. ``"current"`` reproduces ``arena/dynamic.py``'s ``SPEED_MIN_FACTOR`` /
``SPEED_MAX_FACTOR`` constants exactly (the Mission baseline); a ``--check`` TC
asserts the two agree.
"""

from __future__ import annotations

import argparse


# Each value is (min_factor, max_factor) of robot top speed. "current"
# reproduces the existing arena/dynamic.py SPEED_MIN_FACTOR / SPEED_MAX_FACTOR
# constants exactly (a TC asserts the two agree).
SPEED_REGIMES: dict[str, tuple[float, float]] = {
    "slow": (0.3, 0.7),
    "matched": (0.3, 1.0),
    "current": (0.3, 1.5),  # the Mission baseline
    "fast": (0.5, 2.0),
}

# x positions for plot_speed_sweep: the "cap" is the max factor.
SPEED_REGIME_CAP: dict[str, float] = {k: v[1] for k, v in SPEED_REGIMES.items()}

# The default regime when no speed flag is given (the Mission baseline).
DEFAULT_REGIME = "current"


def resolve_speed_factors(
    regime: str | None,
    min_override: float | None,
    max_override: float | None,
) -> tuple[float, float]:
    """Resolve a ``(min_factor, max_factor)`` speed band from a regime name or a
    pair of raw overrides.

    - If BOTH overrides are given, validate ``0 < min_override <= max_override``
      and return ``(min_override, max_override)``.
    - Otherwise look up ``regime`` (``None`` ⇒ :data:`DEFAULT_REGIME`) in
      :data:`SPEED_REGIMES`; an unknown key raises ``ValueError`` listing the
      valid keys.

    This function does NOT enforce the regime-vs-override conflict nor the
    both-or-neither override rule — those are CLI-layer concerns handled by
    :func:`resolve_speed_args`. Here, a partial override (exactly one of the two
    set) is ignored and the ``regime`` path is taken.
    """
    if min_override is not None and max_override is not None:
        if not (0.0 < min_override <= max_override):
            raise ValueError(
                "speed override bounds must satisfy 0 < min <= max, got "
                f"min={min_override}, max={max_override}"
            )
        return (float(min_override), float(max_override))

    key = DEFAULT_REGIME if regime is None else regime
    if key not in SPEED_REGIMES:
        valid = ", ".join(sorted(SPEED_REGIMES))
        raise ValueError(f"unknown speed regime {key!r}; valid regimes: {valid}")
    return SPEED_REGIMES[key]


def add_speed_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared speed-band CLI flags on ``parser``.

    Three PLAIN args (NOT a mutually-exclusive group, since argparse cannot
    express "regime OR (min AND max together)"):

    - ``--speed-regime`` (``choices`` = the regime keys, ``default=None`` so an
      explicit ``--speed-regime current`` is distinguishable from the unset
      default),
    - ``--speed-min-factor`` (float, ``default=None``),
    - ``--speed-max-factor`` (float, ``default=None``).

    The conflict / both-or-neither / bounds validation is performed afterwards by
    :func:`resolve_speed_args`.
    """
    parser.add_argument(
        "--speed-regime",
        choices=list(SPEED_REGIMES),
        default=None,
        help=(
            "named dynamic-obstacle speed band (factors of robot top speed); "
            f"one of {', '.join(SPEED_REGIMES)}. Defaults to "
            f"{DEFAULT_REGIME!r} (the Mission baseline) when unset. Mutually "
            "exclusive with --speed-min-factor/--speed-max-factor."
        ),
    )
    parser.add_argument(
        "--speed-min-factor",
        type=float,
        default=None,
        help=(
            "raw lower speed factor of robot top speed (for off-menu single "
            "runs); must be given together with --speed-max-factor and is "
            "mutually exclusive with --speed-regime."
        ),
    )
    parser.add_argument(
        "--speed-max-factor",
        type=float,
        default=None,
        help=(
            "raw upper speed factor of robot top speed (for off-menu single "
            "runs); must be given together with --speed-min-factor and is "
            "mutually exclusive with --speed-regime."
        ),
    )


def resolve_speed_args(
    parser: argparse.ArgumentParser,
    ns: argparse.Namespace,
) -> tuple[float, float]:
    """Validate the parsed speed flags and resolve the ``(min, max)`` band.

    Manual post-parse validation (calling ``parser.error(...)``, which exits 2,
    on any violation — BEFORE any Arena is constructed, so a bound error never
    surfaces as a mid-run traceback):

    (a) reject an explicitly-set ``--speed-regime`` together with either raw
        override;
    (b) require both raw overrides together or neither (one alone ⇒ error);
    (c) if overrides given, require ``0 < min <= max``;
    (d) default a ``None`` regime to :data:`DEFAULT_REGIME`.

    Returns the resolved ``(min_factor, max_factor)`` tuple.
    """
    regime: str | None = ns.speed_regime
    min_override: float | None = ns.speed_min_factor
    max_override: float | None = ns.speed_max_factor

    has_min = min_override is not None
    has_max = max_override is not None

    # (a) explicit regime conflicts with either raw override.
    if regime is not None and (has_min or has_max):
        parser.error(
            "--speed-regime cannot be combined with --speed-min-factor/"
            "--speed-max-factor; give a regime OR a min/max pair, not both"
        )

    # (b) both overrides together or neither.
    if has_min != has_max:
        parser.error(
            "--speed-min-factor and --speed-max-factor must be given together "
            "(specify both or neither)"
        )

    # (c) + (d): delegate to resolve_speed_factors for the bounds check and the
    # regime lookup, converting any ValueError into a parser.error (exit 2).
    try:
        return resolve_speed_factors(regime, min_override, max_override)
    except ValueError as exc:
        parser.error(str(exc))
        raise  # unreachable: parser.error exits, but keeps the type checker happy
