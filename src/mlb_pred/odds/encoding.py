"""Compact, exact encoding for lines and prices -- and the guard that keeps it exact.

Lines are stored **doubled** as ``SMALLINT``: ``8.5 -> 17``, ``-1.5 -> -3``.
That is exact and costs 2 bytes against ~12 for ``NUMERIC``, and the tick table
is the one place in this project where row width actually matters.

**The encoding is only valid while every quote lands on a half-point.** The
skill this follows is explicit that getting a lossy encoding into millions of
rows is expensive to undo, so the increment was measured before committing:

    32 MLB games sampled across the 2025 season, 6,923 ticks
      totals   -> {4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5}
      run line -> {-3.5, -2.5, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5, 3.5}
    every value a multiple of 0.5, including the alternate run lines

A 32-game sample is evidence, not proof, so :func:`encode_line` **raises** on a
value it cannot represent exactly rather than rounding it away. If SBR ever
starts quoting quarter-point totals, the load fails loudly on the first one and
the scale changes by one constant -- instead of silently storing wrong numbers.

Decoding happens in exactly one place, so no downstream module ever has to
remember the encoding.
"""

from __future__ import annotations

import math

#: Multiplier that makes a valid quote an exact integer.
LINE_SCALE = 2

#: SBR writes this where a book had no price up. Left in, it is a catastrophic
#: outlier that survives into every mean, std and devig -- so it becomes NULL
#: once, here at the storage boundary, and never downstream where each consumer
#: would have to remember it.
OFF_THE_BOARD_SENTINEL = -100_000

#: American prices outside this band are not real quotes.
MIN_PLAUSIBLE_PRICE = -100_000
MAX_PLAUSIBLE_PRICE = 100_000

#: SMALLINT bounds; encoded values must fit.
_SMALLINT_MIN = -32_768
_SMALLINT_MAX = 32_767


class LineEncodingError(ValueError):
    """A line cannot be represented exactly at :data:`LINE_SCALE`."""


def encode_line(value: float | None) -> int | None:
    """``8.5 -> 17``. Raises rather than rounding a value it cannot represent."""
    if value is None:
        return None

    scaled = float(value) * LINE_SCALE
    nearest = round(scaled)
    if not math.isclose(scaled, nearest, abs_tol=1e-9):
        raise LineEncodingError(
            f"Line {value!r} is not a multiple of {1 / LINE_SCALE}. The "
            "SMALLINT encoding assumes half-point quotes; if SBR has started "
            "quoting a finer increment, raise LINE_SCALE (and migrate the "
            "stored values) rather than rounding here."
        )
    if not _SMALLINT_MIN <= nearest <= _SMALLINT_MAX:
        raise LineEncodingError(f"Encoded line {nearest} does not fit in SMALLINT.")
    return int(nearest)


def decode_line(value: int | None) -> float | None:
    """``17 -> 8.5``. The single place the encoding is undone."""
    if value is None:
        return None
    return float(value) / LINE_SCALE


def encode_price(value: int | float | None) -> int | None:
    """American price -> SMALLINT-safe int, with the sentinel nulled.

    Unlike lines, an out-of-range price is *nulled* rather than raising: a
    single absurd price is a bad quote on one side of one tick, and dropping
    the field while keeping the row loses less than refusing the whole load.
    """
    if value is None:
        return None
    try:
        price = int(value)
    except (TypeError, ValueError):
        return None

    if price == OFF_THE_BOARD_SENTINEL:
        return None
    if not MIN_PLAUSIBLE_PRICE < price < MAX_PLAUSIBLE_PRICE:
        return None
    # A real American price is never between -100 and +100 exclusive.
    if -100 < price < 100:
        return None
    if not _SMALLINT_MIN <= price <= _SMALLINT_MAX:
        return None
    return price


def decode_price(value: int | None) -> int | None:
    return None if value is None else int(value)


def american_to_probability(price: int | None) -> float | None:
    """Implied (vigged) probability of an American price."""
    if price is None:
        return None
    numeric_price = float(price)
    if numeric_price > 0:
        return 100.0 / (numeric_price + 100.0)
    if numeric_price < 0:
        return -numeric_price / (-numeric_price + 100.0)
    return None


def devig_two_way(
    left_price: int | None, right_price: int | None
) -> tuple[float | None, float | None, float | None]:
    """Two American prices -> ``(fair_left, fair_right, overround)``.

    The overround is kept as its own quantity rather than discarded: a book can
    move its *price* without moving its *line*, and how much juice it is
    charging is part of the market view.
    """
    left = american_to_probability(left_price)
    right = american_to_probability(right_price)
    if left is None or right is None:
        return None, None, None

    total = left + right
    if total <= 0:
        return None, None, None
    return left / total, right / total, total - 1.0
