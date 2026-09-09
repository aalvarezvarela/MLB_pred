"""Ingest: encoding, market invariants, bounds, and the leakage discriminator."""

from datetime import UTC, date, datetime

import pytest

from mlb_pred.fetch_data.sbr.line_history import (
    MARKET_MONEYLINE,
    MARKET_RUN_LINE,
    MARKET_TOTALS,
    LineTick,
    ScrapedGame,
)
from mlb_pred.odds.encoding import LineEncodingError
from mlb_pred.odds.ingest import IngestStats, tick_rows_for_game


def _tick(
    market=MARKET_TOTALS,
    minutes_to_tip=-120,
    left_line=8.5,
    left_price=-110,
    right_line=8.5,
    right_price=-110,
    book="bet365",
    minute=0,
):
    return LineTick(
        market=market,
        book_slug=book,
        book_name=book,
        line_ts=datetime(2025, 8, 27, 12, minute, tzinfo=UTC),
        minutes_to_tip=minutes_to_tip,
        is_opener=False,
        left_line=left_line,
        left_price=left_price,
        right_line=right_line,
        right_price=right_price,
    )


def _game(*ticks):
    return ScrapedGame(
        event_id=1,
        game_date=date(2025, 8, 27),
        season_year=2025,
        first_pitch_utc=datetime(2025, 8, 27, 17, 5, tzinfo=UTC),
        team_away="Washington Nationals",
        team_home="New York Yankees",
        status_text="Final",
        away_score=2,
        home_score=11,
        starter_away="Cade Cavalli",
        starter_home="Max Fried",
        ticks=tuple(ticks),
    )


def test_lines_are_stored_doubled():
    stats = IngestStats()
    rows = tick_rows_for_game(_game(_tick(left_line=8.5, right_line=8.5)), "1", stats)
    assert rows[0]["left_line"] == 17
    assert rows[0]["right_line"] == 17


def test_run_line_encodes_signed():
    stats = IngestStats()
    rows = tick_rows_for_game(
        _game(_tick(market=MARKET_RUN_LINE, left_line=1.5, right_line=-1.5)), "1", stats
    )
    assert rows[0]["left_line"] == 3
    assert rows[0]["right_line"] == -3


def test_the_pregame_discriminator_is_materialised_not_filtered():
    """SBR records in-play ticks with the same shape as pre-game ones.

    Nothing else in the row separates a legitimate feature from direct target
    leakage, so this is stored, never left to a runtime filter that one call
    site can forget.
    """
    stats = IngestStats()
    rows = tick_rows_for_game(
        _game(_tick(minutes_to_tip=-90, minute=1), _tick(minutes_to_tip=45, minute=2)),
        "1",
        stats,
    )
    assert [r["is_pregame"] for r in rows] == [True, False]
    assert [r["mins_to_tip"] for r in rows] == [-90, 45]
    assert all(r["mins_to_tip"] is not None for r in rows)


def test_off_the_board_sentinel_becomes_null():
    stats = IngestStats()
    rows = tick_rows_for_game(_game(_tick(left_price=-100_000)), "1", stats)
    assert rows[0]["left_price"] is None
    assert rows[0]["right_price"] == -110


def test_a_run_line_that_is_not_mirrored_is_dropped():
    # A genuine spread is mirrored. A complementary *price* pair landing in the
    # line fields is the known failure mode -- either way this is not a line.
    stats = IngestStats()
    rows = tick_rows_for_game(
        _game(_tick(market=MARKET_RUN_LINE, left_line=-1.5, right_line=-1.5)),
        "1",
        stats,
    )
    assert rows == []
    assert stats.dropped["run_line_not_mirrored"] == 1


def test_totals_whose_sides_disagree_are_dropped():
    stats = IngestStats()
    rows = tick_rows_for_game(_game(_tick(left_line=8.5, right_line=9.5)), "1", stats)
    assert rows == []
    assert stats.dropped["totals_sides_disagree"] == 1


def test_a_pregame_total_with_a_dropped_decimal_is_cleared_but_the_row_survives():
    # Clear the bad field, not the row: the prices are still real quotes.
    stats = IngestStats()
    rows = tick_rows_for_game(
        _game(_tick(left_line=85.0, right_line=85.0, minutes_to_tip=-120)), "1", stats
    )
    assert len(rows) == 1
    assert rows[0]["left_line"] is None
    assert rows[0]["left_price"] == -110
    assert stats.repaired["line_out_of_bounds_cleared"] == 1


def test_in_play_rows_are_exempt_from_the_pregame_bounds():
    # A live run line legitimately blows out during a rout; nulling it would
    # be destroying data, not repairing it.
    stats = IngestStats()
    rows = tick_rows_for_game(
        _game(
            _tick(
                market=MARKET_RUN_LINE,
                left_line=9.5,
                right_line=-9.5,
                minutes_to_tip=120,
            )
        ),
        "1",
        stats,
    )
    assert rows[0]["left_line"] == 19
    assert stats.repaired["line_out_of_bounds_cleared"] == 0


def test_moneyline_never_stores_a_line():
    stats = IngestStats()
    rows = tick_rows_for_game(
        _game(
            _tick(
                market=MARKET_MONEYLINE,
                left_line=None,
                right_line=None,
                left_price=220,
                right_price=-275,
            )
        ),
        "1",
        stats,
    )
    assert rows[0]["left_line"] is None
    assert rows[0]["right_line"] is None
    assert rows[0]["left_price"] == 220


def test_a_quarter_point_quote_stops_the_load_not_rounded():
    stats = IngestStats()
    with pytest.raises(LineEncodingError):
        tick_rows_for_game(_game(_tick(left_line=8.25, right_line=8.25)), "1", stats)
    assert stats.dropped["line_not_half_point"] == 1


def test_a_wholly_empty_quote_is_dropped():
    stats = IngestStats()
    rows = tick_rows_for_game(
        _game(
            _tick(left_line=None, right_line=None, left_price=None, right_price=None)
        ),
        "1",
        stats,
    )
    assert rows == []
    assert stats.dropped["empty_quote"] == 1


def test_stats_account_for_every_source_row():
    stats = IngestStats()
    ticks = [
        _tick(minute=1),
        _tick(
            minute=2, left_line=None, right_line=None, left_price=None, right_price=None
        ),
        _tick(minute=3, market=MARKET_RUN_LINE, left_line=1.5, right_line=1.5),
    ]
    rows = tick_rows_for_game(_game(*ticks), "1", stats)

    assert stats.source_ticks == 3
    assert stats.loaded_ticks == len(rows) == 1
    assert sum(stats.dropped.values()) == 2
    assert stats.as_dict()["dropped"]["empty_quote"] == 1
