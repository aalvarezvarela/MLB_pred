"""Season-level helpers. Mirrors ``nba_ou.utils.seasons``."""

from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from mlb_pred.config.constants import (
    GAME_TYPE_MAP,
    INGESTED_GAME_TYPES,
    POSTSEASON_GAME_TYPES,
)
from mlb_pred.utils.general_utils import get_season_year_from_date

# The Stats API accepts a date range per request, and one request comfortably
# returns a whole season. These bounds are deliberately wide: they must cover
# the earliest spring-training date and the latest possible World Series game
# so that no season is ever silently truncated at the edges.
SEASON_WINDOW_START = (2, 1)  # February 1
SEASON_WINDOW_END = (12, 15)  # December 15
MLB_TIMEZONE = ZoneInfo("America/New_York")


def mlb_slate_date(now_utc: datetime | None = None) -> date:
    """Current official MLB slate date, independent of host timezone."""
    now_utc = now_utc or datetime.now(UTC)
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    return now_utc.astimezone(MLB_TIMEZONE).date()


def classify_game_type(game_type: str) -> str:
    """Human-readable season type for a Stats API ``gameType`` code.

    Filters elsewhere key off the raw code, not off this string -- the code is
    structural and the label is editorial.
    """
    return GAME_TYPE_MAP.get(game_type, "Unknown")


def is_postseason(game_type: str) -> bool:
    return game_type in POSTSEASON_GAME_TYPES


def is_ingested_game_type(game_type: str) -> bool:
    """True for regular-season and postseason games only."""
    return game_type in INGESTED_GAME_TYPES


def season_date_bounds(season_year: int) -> tuple[date, date]:
    """Wide (start, end) date window guaranteed to contain the whole season."""
    return (
        date(season_year, *SEASON_WINDOW_START),
        date(season_year, *SEASON_WINDOW_END),
    )


def seasons_between_dates(date_from, date_to) -> list[int]:
    """Inclusive list of season years spanned by two dates."""
    start = get_season_year_from_date(date_from)
    end = get_season_year_from_date(date_to)
    return list(range(start, end + 1))


def season_start_n_seasons_back(n_seasons: int, reference_date=None) -> pd.Timestamp:
    """Start of the earliest season in an ``n_seasons`` rolling window.

    ``n_seasons=1`` is the current season only.
    """
    if n_seasons < 1:
        raise ValueError("n_seasons must be at least 1.")
    reference = (
        pd.Timestamp.now() if reference_date is None else pd.to_datetime(reference_date)
    )
    target_year = get_season_year_from_date(reference) - (n_seasons - 1)
    return pd.Timestamp(year=target_year, month=SEASON_WINDOW_START[0], day=1)
