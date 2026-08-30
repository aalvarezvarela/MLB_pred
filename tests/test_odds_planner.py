"""The update planner: three reasons to fetch, and the two guards that make
partial-coverage detection usable rather than a source of infinite re-fetching.
"""

from datetime import date

import pandas as pd
import pytest

from mlb_pred.odds.planner import (
    gap_dates,
    partial_coverage_dates,
    plan_update,
    refresh_window_dates,
)


@pytest.fixture()
def games():
    return pd.DataFrame(
        [
            {
                "game_pk": "1",
                "game_date": date(2025, 6, 1),
                "season_year": 2025,
                "is_final": True,
            },
            {
                "game_pk": "2",
                "game_date": date(2025, 6, 2),
                "season_year": 2025,
                "is_final": True,
            },
            {
                "game_pk": "3",
                "game_date": date(2025, 6, 3),
                "season_year": 2025,
                "is_final": True,
            },
            {
                "game_pk": "old",
                "game_date": date(2017, 6, 1),
                "season_year": 2017,
                "is_final": True,
            },
        ]
    )


def _odds_games(pks, day=date(2025, 6, 1)):
    return pd.DataFrame(
        [{"game_pk": p, "game_date": day, "season_year": 2025} for p in pks]
    )


# ---------------------------------------------------------------------------
# 1. Refresh window
# ---------------------------------------------------------------------------
def test_recent_dates_are_refetched_even_when_already_stored(games):
    """Presence is not finality.

    A game fetched on the morning it is played is present but its lines keep
    moving until first pitch, so a presence-based gap check would never bring
    it back and the store would hold a truncated history forever.
    """
    dates = refresh_window_dates(games, days=3, today=date(2025, 6, 4))
    assert dates == [date(2025, 6, 1), date(2025, 6, 2), date(2025, 6, 3)]


def test_todays_slate_is_excluded_from_the_refresh_window(games):
    dates = refresh_window_dates(games, days=3, today=date(2025, 6, 3))
    assert date(2025, 6, 3) not in dates


def test_zero_days_disables_refresh_window(games):
    assert refresh_window_dates(games, days=0, today=date(2025, 6, 4)) == []


# ---------------------------------------------------------------------------
# 2. Gaps
# ---------------------------------------------------------------------------
def test_games_absent_from_the_odds_store_become_gaps(games):
    dates, missing = gap_dates(games, _odds_games(["1"]))
    assert missing == 2
    assert set(dates) == {date(2025, 6, 2), date(2025, 6, 3)}


def test_seasons_before_the_first_odds_season_are_never_gaps(games):
    """SBR lists MLB slates back to 2015 but carries no prices before 2019.

    Measured, not assumed. Fetching earlier is pure waste.
    """
    dates, _ = gap_dates(games, _odds_games(["1", "2", "3"]))
    assert date(2017, 6, 1) not in dates


def test_dates_before_the_stores_own_earliest_game_are_not_gaps(games):
    # Below the store's floor there is no history to be *missing* -- only
    # history never collected, which is a backfill decision, not a gap.
    stored = _odds_games(["3"], day=date(2025, 6, 3))
    dates, _ = gap_dates(games, stored)
    assert date(2025, 6, 1) not in dates
    assert date(2025, 6, 2) not in dates


def test_an_empty_odds_store_makes_everything_a_gap(games):
    dates, missing = gap_dates(games, pd.DataFrame())
    assert missing == 3
    assert len(dates) == 3


def test_dimension_without_ticks_or_explicit_completion_remains_a_gap(games):
    fetches = pd.DataFrame(
        [{"game_pk": "1", "ingest_status": "complete"}]
    )
    dates, missing = gap_dates(
        games,
        _odds_games(["1", "2", "3"]),
        odds_fetches=fetches,
        odds_ticks=pd.DataFrame(),
    )
    assert missing == 2
    assert set(dates) == {date(2025, 6, 2), date(2025, 6, 3)}


# ---------------------------------------------------------------------------
# 3. Partial coverage, and its two guards
# ---------------------------------------------------------------------------
def _ticks(rows):
    return pd.DataFrame(
        [
            {"game_pk": g, "game_date": d, "book_slug": b, "season_year": 2025}
            for g, d, b in rows
        ]
    )


def test_a_book_covering_most_of_a_slate_but_missing_one_game_flags_the_date():
    day = date(2025, 6, 1)
    ticks = _ticks(
        [
            ("1", day, "bet365"),
            ("2", day, "bet365"),
            ("3", day, "bet365"),
            ("1", day, "fanduel"),
            ("2", day, "fanduel"),
        ]
    )
    dates = partial_coverage_dates(_odds_games(["1", "2", "3"]), ticks)
    assert dates == [day]


def test_a_books_launch_day_does_not_mark_the_rest_of_the_slate_partial():
    """Without the share threshold, a book that covered 1 of 11 games on its
    launch day marks the other ten partial permanently."""
    day = date(2025, 6, 1)
    ticks = _ticks(
        [
            ("1", day, "bet365"),
            ("2", day, "bet365"),
            ("3", day, "bet365"),
            ("1", day, "brand_new_book"),
        ]
    )
    dates = partial_coverage_dates(_odds_games(["1", "2", "3"]), ticks)
    assert dates == []


def test_a_discontinued_book_never_marks_a_game_partial():
    """Otherwise those games are re-fetched forever, waiting for data the
    source no longer has."""
    day = date(2025, 6, 1)
    ticks = _ticks(
        [
            ("1", day, "bet365"),
            ("2", day, "bet365"),
            ("3", day, "bet365"),
            ("1", day, "gone_book"),
            ("2", day, "gone_book"),
        ]
    )
    dates = partial_coverage_dates(
        _odds_games(["1", "2", "3"]), ticks, discontinued=frozenset({"gone_book"})
    )
    assert dates == []


def test_full_coverage_flags_nothing():
    day = date(2025, 6, 1)
    ticks = _ticks(
        [(g, day, b) for g in ("1", "2", "3") for b in ("bet365", "fanduel")]
    )
    assert partial_coverage_dates(_odds_games(["1", "2", "3"]), ticks) == []


# ---------------------------------------------------------------------------
# The union
# ---------------------------------------------------------------------------
def test_plan_unions_all_three_reasons_without_duplicates(games):
    plan = plan_update(
        games, _odds_games(["1"]), pd.DataFrame(), today=date(2025, 6, 4)
    )
    assert plan.refresh_dates
    assert plan.gap_dates
    assert plan.dates == sorted(set(plan.dates))
    assert "3 date(s)" in plan.summary() or len(plan.dates) == 3


def test_plan_is_empty_when_everything_is_stored_and_old(games):
    plan = plan_update(
        games, _odds_games(["1", "2", "3"]), pd.DataFrame(), today=date(2026, 1, 1)
    )
    assert plan.dates == []
