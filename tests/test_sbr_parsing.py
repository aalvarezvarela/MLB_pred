"""Parsing the SBR ``__NEXT_DATA__`` payload into ticks.

The fixture is a real, trimmed line-history payload (Nationals @ Yankees,
2025-08-27). Using a captured payload rather than a hand-written one means
these tests fail if SBR changes its shape, which is the point.
"""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from mlb_pred.fetch_data.sbr.client import (
    SbrFetchError,
    extract_next_data,
    parse_utc,
    slugify_bookmaker,
    starter_name,
)
from mlb_pred.fetch_data.sbr.line_history import (
    MARKET_MONEYLINE,
    MARKET_RUN_LINE,
    MARKET_TOTALS,
    parse_line_history_payload,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sbr_line_history_343244.json"


@pytest.fixture(scope="module")
def payload():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def game(payload):
    return parse_line_history_payload(payload)


# ---------------------------------------------------------------------------
# Timestamps -- the reason this module reads the payload and not the DOM
# ---------------------------------------------------------------------------
def test_offset_bearing_timestamps_parse_to_utc():
    assert parse_utc("2025-08-27T14:48:23+00:00") == datetime(
        2025, 8, 27, 14, 48, tzinfo=UTC
    )


def test_a_naive_timestamp_is_refused_not_assumed():
    """The whole reason for reading the embedded JSON.

    The rendered table carries no offset, so a scraper that assumed one
    produced different timestamps per machine with nothing erroring. A naive
    value must return None rather than being localised to anything.
    """
    assert parse_utc("2025-08-27T14:48:23") is None


def test_timestamps_truncate_to_the_minute():
    # Seconds carry no information, and dropping them makes a re-scrape
    # reproduce byte-identical keys -- which is what keeps loads idempotent.
    assert parse_utc("2025-08-27T14:48:23+00:00").second == 0
    assert parse_utc("2025-08-27T14:48:59+00:00") == parse_utc(
        "2025-08-27T14:48:01+00:00"
    )


def test_offsets_are_honoured_not_stripped():
    assert parse_utc("2025-08-27T10:48:00-04:00") == datetime(
        2025, 8, 27, 14, 48, tzinfo=UTC
    )


# ---------------------------------------------------------------------------
# Game-level parse
# ---------------------------------------------------------------------------
def test_game_identity(game):
    assert game.event_id == 343244
    assert game.team_away == "Washington Nationals"
    assert game.team_home == "New York Yankees"
    assert game.first_pitch_utc == datetime(2025, 8, 27, 17, 5, tzinfo=UTC)


def test_game_date_is_the_eastern_date_of_first_pitch(game):
    # 17:05 UTC is 13:05 Eastern, so the slate date is the 27th. This is the
    # convention MLB's own officialDate uses, which is the only reason the odds
    # store and the games table join at all.
    assert game.game_date == date(2025, 8, 27)
    assert game.season_year == 2025


def test_starters_are_captured(game):
    # MLB-specific: a run total is heavily conditioned on the announced
    # starters, and they double as a cross-check on a doubleheader match.
    assert game.starter_away == "Cade Cavalli"
    assert game.starter_home == "Max Fried"


def test_all_three_markets_come_from_one_request(game):
    markets = {tick.market for tick in game.ticks}
    assert markets == {MARKET_TOTALS, MARKET_RUN_LINE, MARKET_MONEYLINE}


def test_every_book_in_the_payload_is_parsed(game):
    assert set(game.books) == {"bet365", "betmgm"}


# ---------------------------------------------------------------------------
# The left/right convention, checked structurally against the raw payload
# ---------------------------------------------------------------------------
def test_run_line_left_is_away_and_right_is_home(game, payload):
    """Getting this backwards silently inverts every run-line feature."""
    raw = payload["props"]["pageProps"]["lineHistoryModel"]["lineHistory"]["oddsViews"][
        0
    ]["spreadHistory"][0]

    ticks = [t for t in game.ticks_for(MARKET_RUN_LINE) if t.left_price is not None]
    match = [t for t in ticks if t.line_ts == parse_utc(raw["oddsDate"])]
    assert match, "fixture tick not found"
    tick = match[0]

    assert tick.left_line == raw["awaySpread"]
    assert tick.left_price == raw["awayOdds"]
    assert tick.right_line == raw["homeSpread"]
    assert tick.right_price == raw["homeOdds"]


def test_run_line_sides_are_mirrored(game):
    for tick in game.ticks_for(MARKET_RUN_LINE):
        if tick.left_line is not None and tick.right_line is not None:
            assert tick.left_line == -tick.right_line


def test_totals_left_is_over_and_both_sides_carry_the_same_number(game, payload):
    raw = payload["props"]["pageProps"]["lineHistoryModel"]["lineHistory"]["oddsViews"][
        0
    ]["totalHistory"][0]

    match = [
        t
        for t in game.ticks_for(MARKET_TOTALS)
        if t.line_ts == parse_utc(raw["oddsDate"]) and t.book_slug == "betmgm"
    ]
    assert match
    tick = match[0]

    assert tick.left_price == raw["overOdds"]
    assert tick.right_price == raw["underOdds"]
    assert tick.left_line == tick.right_line == raw["total"]


def test_moneyline_carries_prices_only(game):
    ticks = game.ticks_for(MARKET_MONEYLINE)
    assert ticks
    for tick in ticks:
        assert tick.left_line is None
        assert tick.right_line is None
        assert tick.left_price is not None or tick.right_price is not None


# ---------------------------------------------------------------------------
# Derived tick fields
# ---------------------------------------------------------------------------
def test_minutes_to_tip_is_negative_pre_game(game):
    for tick in game.ticks:
        assert tick.is_pregame == (tick.minutes_to_tip < 0)
    # Every tick in this fixture was quoted before first pitch.
    assert all(tick.is_pregame for tick in game.ticks)


def test_exactly_one_opener_per_book_and_market(game):
    from collections import Counter

    openers = Counter(
        (tick.book_slug, tick.market) for tick in game.ticks if tick.is_opener
    )
    assert openers
    assert set(openers.values()) == {1}


def test_ticks_are_sorted_deterministically(game):
    keys = [(t.market, t.book_slug, t.line_ts) for t in game.ticks]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------
def test_missing_payload_raises():
    with pytest.raises(SbrFetchError, match="No __NEXT_DATA__"):
        extract_next_data("<html><body>nothing here</body></html>")


def test_malformed_payload_raises():
    html = '<script id="__NEXT_DATA__" type="application/json">{not json}</script>'
    with pytest.raises(SbrFetchError, match="Malformed"):
        extract_next_data(html)


def test_payload_without_a_start_date_raises():
    broken = {
        "props": {
            "pageProps": {
                "lineHistoryModel": {
                    "lineHistory": {"gameView": {"gameId": 1}, "oddsViews": []}
                }
            }
        }
    }
    with pytest.raises(SbrFetchError, match="startDate"):
        parse_line_history_payload(broken)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("Fanatics Sportsbook", "fanatics_sportsbook"),
        ("bet365", "bet365"),
        ("BetRivers", "betrivers"),
        ("Caesars", "caesars"),
    ],
)
def test_book_slugs_match_the_nba_projects_format(name, slug):
    assert slugify_bookmaker(name) == slug


def test_starter_name_formatting():
    assert starter_name({"firstName": "Max", "lastName": "Fried"}) == "Max Fried"
    assert starter_name(None) is None
    assert starter_name({}) is None


def test_a_late_first_pitch_is_filed_under_the_previous_days_slate():
    """The single easiest way to lose the late third of every slate.

    A 02:10 UTC first pitch is a 22:10 Eastern game on the *previous* calendar
    day, and both SBR and MLB's ``officialDate`` file it there. Measured on
    this repo's store: 21.5% of resolved games have a UTC date that differs
    from their slate date. Parsing the UTC date instead would silently fail to
    match every one of them.
    """
    from zoneinfo import ZoneInfo

    late = datetime(2025, 3, 28, 2, 10, tzinfo=UTC)
    eastern_date = late.astimezone(ZoneInfo("America/New_York")).date()

    assert eastern_date == date(2025, 3, 27)
    assert eastern_date != late.date()


def test_game_date_uses_eastern_not_utc(payload):
    """Same rule, exercised through the parser rather than by hand."""
    import copy

    shifted = copy.deepcopy(payload)
    game_view = shifted["props"]["pageProps"]["lineHistoryModel"]["lineHistory"][
        "gameView"
    ]
    game_view["startDate"] = "2025-03-28T02:10:00+00:00"

    game = parse_line_history_payload(shifted)
    assert game.game_date == date(2025, 3, 27)
    assert game.first_pitch_utc.date() == date(2025, 3, 28)
