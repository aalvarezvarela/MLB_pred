"""The odds updater loop: flushing, and per-date failure isolation."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from mlb_pred.fetch_data.sbr.client import SbrFetchError
from mlb_pred.fetch_data.sbr.line_history import LineTick, ScrapedGame
from mlb_pred.local_store import odds_updaters, parquet_store


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(
        type(parquet_store.SETTINGS),
        "raw_data_path",
        property(lambda self: tmp_path),
    )
    yield


@pytest.fixture()
def games():
    return pd.DataFrame(
        [
            {
                "game_pk": "778443",
                "game_date": date(2025, 4, 6),
                "season_year": 2025,
                "home_team_id": "111",
                "away_team_id": "138",
                "first_pitch_utc": pd.Timestamp("2025-04-06T17:35:00Z"),
                "game_number": 1,
                "doubleheader": "S",
                "is_final": True,
            }
        ]
    )


def _scraped(event_id=1):
    tick = LineTick(
        market="totals",
        book_slug="bet365",
        book_name="bet365",
        line_ts=datetime(2025, 4, 6, 12, 0, tzinfo=UTC),
        minutes_to_tip=-335,
        is_opener=True,
        left_line=8.5,
        left_price=-110,
        right_line=8.5,
        right_price=-110,
    )
    return ScrapedGame(
        event_id=event_id,
        game_date=date(2025, 4, 6),
        season_year=2025,
        first_pitch_utc=datetime(2025, 4, 6, 17, 35, tzinfo=UTC),
        team_away="St. Louis Cardinals",
        team_home="Boston Red Sox",
        status_text="Final",
        away_score=3,
        home_score=5,
        starter_away="Andre Pallante",
        starter_home="Sean Newcomb",
        ticks=(tick,),
    )


def test_a_failing_date_is_recorded_and_the_run_continues(monkeypatch, games):
    """A backfill that dies because one page 404'd is worse than one with a
    reported gap -- the planner picks the date up next run anyway."""
    calls = []

    def fake_scrape_dates(days, **kwargs):
        day = list(days)[0]
        calls.append(day)
        if day == date(2025, 4, 7):
            raise SbrFetchError("502 Bad Gateway")
        return [_scraped()]

    monkeypatch.setattr(odds_updaters, "scrape_dates", fake_scrape_dates)
    monkeypatch.setattr(odds_updaters, "new_session", lambda: _NullSession())

    result = odds_updaters.ingest_dates(
        [date(2025, 4, 6), date(2025, 4, 7), date(2025, 4, 6)],
        games=games,
        progress_every=0,
    )

    assert result["failed_dates"] == [date(2025, 4, 7)]
    # The date *after* the failure was still attempted -- that is the point.
    assert calls == [date(2025, 4, 6), date(2025, 4, 7), date(2025, 4, 6)]
    # And the successful dates still landed. (Both scrape the same game here,
    # which the store's primary key correctly collapses to one row.)
    assert result["written"]["odds_ticks"] >= 1
    assert len(parquet_store.read_table("odds_ticks")) == 1


def test_strict_mode_raises(monkeypatch, games):
    def fake_scrape_dates(days, **kwargs):
        raise SbrFetchError("502 Bad Gateway")

    monkeypatch.setattr(odds_updaters, "scrape_dates", fake_scrape_dates)
    monkeypatch.setattr(odds_updaters, "new_session", lambda: _NullSession())

    with pytest.raises(SbrFetchError):
        odds_updaters.ingest_dates(
            [date(2025, 4, 6)], games=games, on_error="raise", progress_every=0
        )


def test_results_flush_before_the_end_so_an_interrupted_run_keeps_them(
    monkeypatch, games
):
    monkeypatch.setattr(odds_updaters, "scrape_dates", lambda days, **kw: [_scraped()])
    monkeypatch.setattr(odds_updaters, "new_session", lambda: _NullSession())

    odds_updaters.ingest_dates(
        [date(2025, 4, 6)] * 5, games=games, flush_every=2, progress_every=0
    )
    assert not parquet_store.read_table("odds_ticks").empty


def test_an_empty_games_table_is_refused_with_a_useful_message():
    with pytest.raises(RuntimeError, match="games table is empty"):
        odds_updaters.ingest_dates([date(2025, 4, 6)], games=pd.DataFrame())


def test_coverage_reports_the_in_play_share(monkeypatch, games):
    monkeypatch.setattr(odds_updaters, "scrape_dates", lambda days, **kw: [_scraped()])
    monkeypatch.setattr(odds_updaters, "new_session", lambda: _NullSession())
    odds_updaters.ingest_dates([date(2025, 4, 6)], games=games, progress_every=0)

    coverage = odds_updaters.odds_coverage()
    assert coverage.loc[0, "season_year"] == 2025
    assert coverage.loc[0, "games_priced"] == 1
    assert coverage.loc[0, "inplay_share"] == 0.0


class _NullSession:
    def close(self):
        pass
