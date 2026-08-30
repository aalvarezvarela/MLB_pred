"""Small shared helpers. Mirrors ``nba_ou.utils.general_utils``."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

BEFORE_SUFFIX = "_BEFORE"


def with_before_suffix(name: str) -> str:
    """Tag a column as leakage-safe / pre-game.

    Same convention as the NBA repo: only ``_BEFORE`` columns are eligible for
    training-column selection.
    """
    return f"{name}{BEFORE_SUFFIX}"


def get_season_year_from_date(value: datetime | date | str) -> int:
    """Return the season's start year for a date.

    An MLB season is contained in a single calendar year, so this is trivially
    the year -- unlike the NBA, where the season straddles New Year and the
    helper has to subtract one before October. The named function exists anyway
    so that "which season is this date in?" is one concept in one place: season
    boundaries show up in rolling resets, previous-season fallbacks, partition
    keys and roster windows, and two conventions in one codebase is a whole
    class of off-by-one bugs.
    """
    ts = pd.to_datetime(value)
    if pd.isna(ts):
        raise ValueError(f"Cannot derive a season year from {value!r}.")
    return int(ts.year)


def season_range(first_season: int, last_season: int) -> list[int]:
    """Inclusive list of season years."""
    if last_season < first_season:
        raise ValueError(
            f"last_season ({last_season}) is before first_season ({first_season})."
        )
    return list(range(first_season, last_season + 1))


def as_id(value: object) -> str | None:
    """Normalise a provider id to TEXT.

    All entity ids are stored as strings so that a missing value never
    coerces an id column to float and silently reformats every id in it
    (``147`` -> ``147.0``).
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None
