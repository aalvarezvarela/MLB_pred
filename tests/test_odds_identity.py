"""Resolving SBR events to ``game_pk`` -- "the hard part".

Doubleheaders are the MLB-specific difficulty: ``(date, home, away)`` is not
unique, and roughly 58 games a season are the second half of one.
"""

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from mlb_pred.odds.identity import (
    FIRST_PITCH_WARN_TOLERANCE,
    ResolutionStats,
    build_game_index,
    resolve_event,
    resolve_events,
)


class FakeScraped:
    """Just the attributes the resolver reads."""

    def __init__(
        self,
        event_id=1,
        game_date=date(2025, 4, 6),
        first_pitch_utc=datetime(2025, 4, 6, 17, 35, tzinfo=UTC),
        team_home="Boston Red Sox",
        team_away="St. Louis Cardinals",
        starter_home="Sean Newcomb",
        starter_away="Andre Pallante",
        season_year=2025,
    ):
        self.event_id = event_id
        self.game_date = game_date
        self.first_pitch_utc = first_pitch_utc
        self.team_home = team_home
        self.team_away = team_away
        self.starter_home = starter_home
        self.starter_away = starter_away
        self.season_year = season_year


@pytest.fixture()
def doubleheader_games():
    """Boston hosted St. Louis twice on 2025-04-06 -- a real slate."""
    return pd.DataFrame(
        [
            {
                "game_pk": "778443",
                "game_date": date(2025, 4, 6),
                "home_team_id": "111",
                "away_team_id": "138",
                "first_pitch_utc": pd.Timestamp("2025-04-06T17:35:00Z"),
                "game_number": 1,
                "doubleheader": "S",
            },
            {
                "game_pk": "778432",
                "game_date": date(2025, 4, 6),
                "home_team_id": "111",
                "away_team_id": "138",
                "first_pitch_utc": pd.Timestamp("2025-04-06T23:10:00Z"),
                "game_number": 2,
                "doubleheader": "S",
            },
        ]
    )


def test_single_game_resolves(doubleheader_games):
    single = doubleheader_games.iloc[:1]
    stats = ResolutionStats()
    row = resolve_event(FakeScraped(), build_game_index(single), stats)

    assert row["game_pk"] == "778443"
    assert row["event_id"] == 1
    assert row["team_home_id"] == "111"
    assert row["team_away_id"] == "138"
    assert stats.matched == 1


def test_doubleheader_first_game_resolves_by_first_pitch(doubleheader_games):
    stats = ResolutionStats()
    row = resolve_event(
        FakeScraped(first_pitch_utc=datetime(2025, 4, 6, 17, 35, tzinfo=UTC)),
        build_game_index(doubleheader_games),
        stats,
    )
    assert row["game_pk"] == "778443"
    assert row["game_number"] == 1
    assert row["is_doubleheader"] is True
    assert stats.matched_doubleheader == 1


def test_doubleheader_second_game_resolves_by_first_pitch(doubleheader_games):
    stats = ResolutionStats()
    row = resolve_event(
        FakeScraped(first_pitch_utc=datetime(2025, 4, 6, 23, 10, tzinfo=UTC)),
        build_game_index(doubleheader_games),
        stats,
    )
    assert row["game_pk"] == "778432"
    assert row["game_number"] == 2


def test_a_doubleheader_start_far_from_both_games_is_refused(doubleheader_games):
    # Better to leave it unmatched with a named reason than to attach a
    # game's odds to the wrong half of a doubleheader.
    stats = ResolutionStats()
    row = resolve_event(
        FakeScraped(first_pitch_utc=datetime(2025, 4, 6, 6, 0, tzinfo=UTC)),
        build_game_index(doubleheader_games),
        stats,
    )
    assert row is None
    assert stats.unmatched["doubleheader_no_close_start"] == 1


def test_unknown_team_name_is_counted_not_silently_dropped(doubleheader_games):
    stats = ResolutionStats()
    row = resolve_event(
        FakeScraped(team_home="Portland Beavers"),
        build_game_index(doubleheader_games),
        stats,
    )
    assert row is None
    assert stats.unmatched["unknown_team_name"] == 1


def test_a_date_with_no_such_game_gets_a_named_reason(doubleheader_games):
    # SBR lists spring training and postponed games the games table does not
    # carry. A miss rate with no named explanation is a mapping bug.
    stats = ResolutionStats()
    row = resolve_event(
        FakeScraped(game_date=date(2025, 3, 1)),
        build_game_index(doubleheader_games),
        stats,
    )
    assert row is None
    assert stats.unmatched["no_game_on_date"] == 1


def test_first_pitch_disagreement_is_reported_but_still_resolves(doubleheader_games):
    single = doubleheader_games.iloc[:1]
    stats = ResolutionStats()
    shifted = datetime(2025, 4, 6, 17, 35, tzinfo=UTC) + FIRST_PITCH_WARN_TOLERANCE
    shifted += timedelta(minutes=5)

    row = resolve_event(
        FakeScraped(first_pitch_utc=shifted), build_game_index(single), stats
    )

    # The match stands -- consistency with the ticks matters more than
    # agreeing with a third party -- but the disagreement is surfaced.
    assert row is not None
    assert len(stats.first_pitch_disagreements) == 1
    assert stats.first_pitch_disagreements[0]["delta_minutes"] == 20


def test_the_scrape_time_first_pitch_is_what_gets_stored(doubleheader_games):
    single = doubleheader_games.iloc[:1]
    sbr_time = datetime(2025, 4, 6, 17, 40, tzinfo=UTC)
    row = resolve_event(FakeScraped(first_pitch_utc=sbr_time), build_game_index(single))

    # Every stored mins_to_tip was computed against SBR's own start time, so
    # storing MLB's instead would desynchronise the ticks from their own key.
    assert row["first_pitch_utc"] == sbr_time


def test_the_provider_id_is_kept(doubleheader_games):
    single = doubleheader_games.iloc[:1]
    row = resolve_event(FakeScraped(event_id=341365), build_game_index(single))
    # Cannot be re-derived later -- a date holds many games -- and it is
    # needed to re-fetch one game.
    assert row["event_id"] == 341365


def test_starters_are_carried_onto_the_dimension(doubleheader_games):
    single = doubleheader_games.iloc[:1]
    row = resolve_event(FakeScraped(), build_game_index(single))
    assert row["starter_home"] == "Sean Newcomb"
    assert row["starter_away"] == "Andre Pallante"


def test_resolve_events_reports_a_match_rate(doubleheader_games):
    events = [
        FakeScraped(event_id=1),
        FakeScraped(
            event_id=2, first_pitch_utc=datetime(2025, 4, 6, 23, 10, tzinfo=UTC)
        ),
        FakeScraped(event_id=3, game_date=date(2025, 3, 1)),
    ]
    rows, stats = resolve_events(events, doubleheader_games)

    assert len(rows) == 2
    assert stats.matched == 2
    assert stats.total == 3
    assert stats.match_rate == pytest.approx(2 / 3)
    assert "no_game_on_date" in stats.summary()
