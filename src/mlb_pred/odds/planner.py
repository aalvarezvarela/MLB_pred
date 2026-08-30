"""What to fetch next -- a query against our own store, not a date range.

"Which dates should I scrape?" has three independent answers, and the planner
is their union. Each exists because of a distinct failure the others miss.

**1. Refresh window (unconditional).** Dates with games in the last few days
are re-fetched whether or not the store already has them. *Presence is not
finality*: a game fetched on the morning it is played is present but its lines
keep moving until first pitch. A presence-based gap check would never bring it
back, so the store would permanently hold a truncated history for every game
it happened to see early.

**2. Gaps.** Games the games table knows about that the odds store does not.
The games table is the reference for what *should* exist. Bounded below by the
store's own earliest game -- before that there is no history to be missing,
only history never collected.

**3. Partial coverage.** Stored games missing a book that most games on their
own date carry. Two guards make this usable rather than a source of infinite
re-fetching:

* **Discontinued books are excluded outright.** A book the source no longer
  carries must never mark a game partial, or those games are re-fetched
  forever waiting for data that does not exist.
* **A book counts as "expected" on a date only if it priced at least half that
  date's games.** Without the threshold a book's launch day marks every other
  game on it partial, permanently. Books also legitimately skip games.

``dry_run`` reports the entire plan and stops before the first request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from mlb_pred.utils.seasons import mlb_slate_date

#: Days of finished slates to re-fetch unconditionally. Wide enough to cover a
#: skipped run plus the gap between a morning fetch and the close.
DEFAULT_REFRESH_DAYS = 3

#: A book must price at least this share of a date's games to be "expected"
#: on that date.
BOOK_EXPECTED_SHARE = 0.5

#: Books SBR no longer carries. A missing one of these is not a gap.
#: Empty for MLB today -- Caesars is still present, unlike the NBA pages --
#: but the mechanism has to exist before it is needed, not after.
DISCONTINUED_BOOKS: frozenset[str] = frozenset()

#: SBR lists MLB slates back to 2015 but carries no prices before 2019.
#: Measured, not assumed: 2015 and 2018 slates return games with zero ticks,
#: 2019 returns ticks from three books. Fetching earlier is pure waste.
FIRST_ODDS_SEASON = 2019


@dataclass
class UpdatePlan:
    """The dates to fetch, and why each one is in the list."""

    refresh_dates: list[date] = field(default_factory=list)
    gap_dates: list[date] = field(default_factory=list)
    partial_dates: list[date] = field(default_factory=list)
    missing_games: int = 0

    @property
    def dates(self) -> list[date]:
        return sorted(
            set(self.refresh_dates) | set(self.gap_dates) | set(self.partial_dates)
        )

    def summary(self) -> str:
        return "\n".join(
            [
                f"plan: {len(self.dates)} date(s) to fetch",
                f"  refresh window : {len(self.refresh_dates)}",
                f"  gaps           : {len(self.gap_dates)} "
                f"({self.missing_games} game(s) absent from the odds store)",
                f"  partial books  : {len(self.partial_dates)}",
            ]
        )


def _as_dates(values) -> set[date]:
    return {pd.Timestamp(v).date() for v in values}


def refresh_window_dates(
    games: pd.DataFrame, *, days: int = DEFAULT_REFRESH_DAYS, today: date | None = None
) -> list[date]:
    """Recent finished slates, re-fetched regardless of what the store holds."""
    if games.empty or days <= 0:
        return []
    today = today or mlb_slate_date()
    # Yesterday backwards: today's games are still moving and are handled by
    # the live path, not the historical store.
    end = today - timedelta(days=1)
    start = end - timedelta(days=max(days - 1, 0))
    game_dates = _as_dates(games["game_date"])
    return sorted(d for d in game_dates if start <= d <= end)


def gap_dates(
    games: pd.DataFrame,
    odds_games: pd.DataFrame,
    *,
    first_season: int = FIRST_ODDS_SEASON,
    odds_fetches: pd.DataFrame | None = None,
    odds_ticks: pd.DataFrame | None = None,
) -> tuple[list[date], int]:
    """Dates holding games the odds store has never seen."""
    if games.empty:
        return [], 0

    eligible = games[games["season_year"] >= first_season]
    if "is_final" in eligible.columns:
        eligible = eligible[eligible["is_final"]]
    if eligible.empty:
        return [], 0

    if odds_fetches is None:
        # Backwards-compatible path for callers/tests without completion data.
        stored = set(odds_games["game_pk"]) if not odds_games.empty else set()
    else:
        completed = odds_fetches[
            odds_fetches.get("ingest_status", pd.Series(index=odds_fetches.index)).isin(
                {"complete", "no_quotes"}
            )
        ]
        stored = set(completed.get("game_pk", pd.Series(dtype=object)))
        # Existing stores predate odds_fetches. A game with ticks is unambiguously
        # complete enough to avoid a needless historical re-fetch.
        if odds_ticks is not None and not odds_ticks.empty:
            stored.update(odds_ticks["game_pk"])
    missing = eligible[~eligible["game_pk"].isin(stored)]

    if not odds_games.empty:
        # Below the store's own earliest game there is no history to be
        # missing -- only history never collected, which is a backfill
        # decision rather than a gap.
        earliest = min(_as_dates(odds_games["game_date"]))
        missing = missing[pd.to_datetime(missing["game_date"]).dt.date >= earliest]

    return sorted(_as_dates(missing["game_date"])), len(missing)


def partial_coverage_dates(
    odds_games: pd.DataFrame,
    odds_ticks: pd.DataFrame,
    *,
    expected_share: float = BOOK_EXPECTED_SHARE,
    discontinued: frozenset[str] = DISCONTINUED_BOOKS,
) -> list[date]:
    """Dates where a stored game is missing a book its slate-mates have."""
    if odds_ticks.empty or odds_games.empty:
        return []

    ticks = odds_ticks[~odds_ticks["book_slug"].isin(discontinued)]
    if ticks.empty:
        return []

    coverage = (
        ticks.groupby(["game_date", "book_slug"])["game_pk"]
        .nunique()
        .reset_index(name="games_with_book")
    )
    per_date = (
        ticks.groupby("game_date")["game_pk"]
        .nunique()
        .reset_index(name="games_on_date")
    )
    coverage = coverage.merge(per_date, on="game_date")

    # Only books that priced most of a date's slate are "expected" there.
    expected = coverage[
        coverage["games_with_book"] >= expected_share * coverage["games_on_date"]
    ]
    if expected.empty:
        return []

    # A date is partial when some expected book did not reach every game.
    partial = expected[expected["games_with_book"] < expected["games_on_date"]]
    return sorted(_as_dates(partial["game_date"]))


def plan_update(
    games: pd.DataFrame,
    odds_games: pd.DataFrame,
    odds_ticks: pd.DataFrame,
    *,
    refresh_days: int = DEFAULT_REFRESH_DAYS,
    first_season: int = FIRST_ODDS_SEASON,
    today: date | None = None,
    odds_fetches: pd.DataFrame | None = None,
) -> UpdatePlan:
    """Union the three reasons to fetch a date."""
    gaps, missing_games = gap_dates(
        games,
        odds_games,
        first_season=first_season,
        odds_fetches=odds_fetches,
        odds_ticks=odds_ticks,
    )
    return UpdatePlan(
        refresh_dates=refresh_window_dates(games, days=refresh_days, today=today),
        gap_dates=gaps,
        partial_dates=partial_coverage_dates(odds_games, odds_ticks),
        missing_games=missing_games,
    )
