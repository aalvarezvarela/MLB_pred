"""Pre-game line-movement features built from the odds tick store.

``closing_lines`` reduces the tick history to a single closing quote per book.
Everything that happened on the way to that close -- where the market opened,
which way it moved, how far, how late, and how much the books disagreed about
the move -- was being discarded. This module reads the same
``data/raw/odds_ticks`` table and keeps that path.

Two rules make the result comparable across seasons:

* Only ``is_pregame`` ticks at least ``safety_margin_minutes`` before first
  pitch are read, the same gate ``closing_lines`` applies. SBR records in-play
  ticks with the same shape as pre-game ones.
* Movement is measured **per book and then aggregated across the books that
  quote in every season**. Pooling ticks first would make tick counts and path
  lengths track the number of books, which grows from four to seven across the
  store, and that drift lands in the middle of any walk-forward split.

Every column carries the ``ODDS_MOVEMENT_`` prefix and the ``_BEFORE`` tag: a
line-movement feature is pre-game by construction, but the tag is what the
pregame contract checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_pred.features.closing_lines import (
    DEFAULT_CLOSE_SAFETY_MARGIN_MINUTES,
    MARKET_RUN_LINE,
    MARKET_TOTALS,
    STABLE_BOOKS,
)
from mlb_pred.odds.encoding import LINE_SCALE

FEATURE_PREFIX = "ODDS_MOVEMENT_"

# A late move is the part of the drift that happened inside this many minutes
# of first pitch. Three hours comfortably contains lineup and weather news for
# a baseball slate while still leaving most of the day's movement outside it.
LATE_WINDOW_MINUTES = 180

_TICK_COLUMNS = {
    "game_pk",
    "market",
    "book_slug",
    "mins_to_tip",
    "is_pregame",
    "left_line",
}

_MARKET_LABELS: dict[str, str] = {
    MARKET_TOTALS: "TOTAL",
    MARKET_RUN_LINE: "RUN_LINE",
}


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"odds ticks is missing required columns: {missing}")


def movement_feature_columns() -> list[str]:
    """Return the stable schema this module emits, in order."""
    columns: list[str] = []
    for label in _MARKET_LABELS.values():
        columns.extend(
            [
                f"{FEATURE_PREFIX}{label}_OPEN_BEFORE",
                f"{FEATURE_PREFIX}{label}_OPEN_TO_CLOSE_BEFORE",
                f"{FEATURE_PREFIX}{label}_ABS_OPEN_TO_CLOSE_BEFORE",
                f"{FEATURE_PREFIX}{label}_MOVED_UP_BEFORE",
                f"{FEATURE_PREFIX}{label}_LATE_{LATE_WINDOW_MINUTES}M_BEFORE",
                f"{FEATURE_PREFIX}{label}_INTRADAY_RANGE_BEFORE",
                f"{FEATURE_PREFIX}{label}_REVERSAL_BEFORE",
                f"{FEATURE_PREFIX}{label}_CROSSED_INTEGER_BEFORE",
                f"{FEATURE_PREFIX}{label}_BOOK_DISAGREE_MOVE_BEFORE",
                f"{FEATURE_PREFIX}{label}_BOOKS_WITH_HISTORY_BEFORE",
            ]
        )
    return columns


def _signed_line(ticks: pd.DataFrame, market: str) -> pd.Series:
    """Decode the doubled SMALLINT line into the market's natural orientation.

    ``left`` is OVER for totals and AWAY for the run line, so the away handicap
    is negated to express the home handicap and keep one sign convention.
    """
    line = pd.to_numeric(ticks["left_line"], errors="coerce") / float(LINE_SCALE)
    return line if market == MARKET_TOTALS else -line


def _per_book_paths(ticks: pd.DataFrame, market: str) -> pd.DataFrame:
    """Reduce each (game, book) tick path to one row of movement statistics."""
    selected = ticks.loc[ticks["market"].eq(market)].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=["game_pk", "open", "close", "high", "low", "late_reference"]
        )
    selected["__line"] = _signed_line(selected, market)
    selected = selected.loc[selected["__line"].notna()]
    # Ascending mins_to_tip runs from the earliest tick to the latest.
    selected = selected.sort_values(["game_pk", "book_slug", "mins_to_tip"])
    grouped = selected.groupby(["game_pk", "book_slug"], sort=False)["__line"]
    paths = pd.DataFrame(
        {
            "open": grouped.first(),
            "close": grouped.last(),
            "high": grouped.max(),
            "low": grouped.min(),
        }
    )
    late = (
        selected.loc[selected["mins_to_tip"].ge(-LATE_WINDOW_MINUTES)]
        .groupby(["game_pk", "book_slug"], sort=False)["__line"]
        .first()
        .rename("late_reference")
    )
    return paths.join(late, how="left").reset_index()


def build_market_movement_features(
    odds_ticks: pd.DataFrame,
    *,
    safety_margin_minutes: int = DEFAULT_CLOSE_SAFETY_MARGIN_MINUTES,
    books: tuple[str, ...] = STABLE_BOOKS,
) -> pd.DataFrame:
    """Build one row of pre-game line-movement features per game."""
    _require_columns(odds_ticks, _TICK_COLUMNS)
    if safety_margin_minutes < 0:
        raise ValueError("safety_margin_minutes cannot be negative.")

    ticks = odds_ticks.loc[
        odds_ticks["is_pregame"].fillna(False).astype(bool)
        & pd.to_numeric(odds_ticks["mins_to_tip"], errors="coerce").le(
            -safety_margin_minutes
        )
        & odds_ticks["book_slug"].isin(books)
    ].copy()
    ticks["game_pk"] = ticks["game_pk"].astype(str)

    frames: list[pd.DataFrame] = []
    for market, label in _MARKET_LABELS.items():
        paths = _per_book_paths(ticks, market)
        if paths.empty:
            continue
        paths["move"] = paths["close"] - paths["open"]
        paths["range"] = paths["high"] - paths["low"]
        paths["late"] = paths["close"] - paths["late_reference"]
        by_game = paths.groupby("game_pk", sort=False)
        prefix = f"{FEATURE_PREFIX}{label}"
        opened = by_game["open"].median()
        closed = by_game["close"].median()
        moved = by_game["move"].median()
        ranged = by_game["range"].median()
        built = pd.DataFrame(
            {
                f"{prefix}_OPEN_BEFORE": opened,
                f"{prefix}_OPEN_TO_CLOSE_BEFORE": moved,
                f"{prefix}_ABS_OPEN_TO_CLOSE_BEFORE": moved.abs(),
                f"{prefix}_MOVED_UP_BEFORE": np.sign(moved),
                f"{prefix}_LATE_{LATE_WINDOW_MINUTES}M_BEFORE": by_game[
                    "late"
                ].median(),
                f"{prefix}_INTRADAY_RANGE_BEFORE": ranged,
                # Movement the market made and then gave back. A line that
                # travelled 1.0 to finish 0.5 higher reversed by 0.5.
                f"{prefix}_REVERSAL_BEFORE": ranged - moved.abs(),
                # Baseball lines cluster on the half run; stepping over a whole
                # number is a different event from drifting within one.
                f"{prefix}_CROSSED_INTEGER_BEFORE": (
                    np.floor(opened) != np.floor(closed)
                ).astype("float64"),
                f"{prefix}_BOOK_DISAGREE_MOVE_BEFORE": by_game["move"].std(ddof=0),
                f"{prefix}_BOOKS_WITH_HISTORY_BEFORE": by_game["move"]
                .size()
                .astype("float64"),
            }
        )
        frames.append(built)

    if not frames:
        output = pd.DataFrame(columns=["GAME_ID", *movement_feature_columns()])
        return output

    output = pd.concat(frames, axis=1)
    output.index.name = "GAME_ID"
    output = output.reset_index()
    # One stable schema: a market absent from a season must not reshape the
    # partition, so every declared column is present even when unquoted.
    for column in movement_feature_columns():
        if column not in output:
            output[column] = np.nan
    return output[["GAME_ID", *movement_feature_columns()]]


__all__ = [
    "FEATURE_PREFIX",
    "LATE_WINDOW_MINUTES",
    "build_market_movement_features",
    "movement_feature_columns",
]
