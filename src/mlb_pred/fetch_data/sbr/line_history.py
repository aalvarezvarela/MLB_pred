"""SBR line-history payload -> tick records.

An odds record is **(game, sportsbook, market, side, line, price, timestamp)**.
Everything else -- books as columns, closing lines, opening lines, consensus --
is a *view* over that record. This module produces the record; the views are
derived downstream.

The ``left`` / ``right`` convention
-----------------------------------
One pair of columns serves all three markets, because a two-way market always
has two sides and naming them positionally lets one table and one set of
feature functions cover every market:

===============  ==========  ==========  ==================================
market           ``left``    ``right``   invariant
===============  ==========  ==========  ==================================
totals           OVER        UNDER       ``left_line == right_line``
run_line         AWAY        HOME        ``left_line == -right_line``
money_line       AWAY        HOME        both line columns NULL by nature
===============  ==========  ==========  ==================================

The cost of the convention is that "left" means different things per market,
and getting it backwards silently inverts every run-line and moneyline
feature. So it is **measured against realised outcomes and pinned by a test**
rather than assumed -- see ``tests/test_odds_orientation.py``, which checks it
against this repo's own final scores.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests

from mlb_pred.fetch_data.sbr.client import (
    SbrFetchError,
    as_float,
    as_int,
    build_daily_odds_url,
    build_line_history_url,
    fetch_next_data,
    new_session,
    parse_utc,
    sleep_politely,
    slugify_bookmaker,
    starter_name,
    team_full_name,
)

MARKET_TOTALS = "totals"
MARKET_RUN_LINE = "run_line"
MARKET_MONEYLINE = "money_line"
ALL_MARKETS: tuple[str, ...] = (MARKET_TOTALS, MARKET_RUN_LINE, MARKET_MONEYLINE)

#: Payload key -> market code. Each holds one ascending list of ticks per book.
#: SBR names the run-line key ``spreadHistory`` because the page template is
#: shared across sports; the market is called ``run_line`` here because that is
#: what it is in baseball.
HISTORY_KEYS: dict[str, str] = {
    "totalHistory": MARKET_TOTALS,
    "spreadHistory": MARKET_RUN_LINE,
    "moneyLineHistory": MARKET_MONEYLINE,
}

#: SBR lists a game under the date its first pitch falls on in *Eastern* time --
#: a 00:30 UTC first pitch belongs to the previous day's slate. MLB's own
#: ``officialDate`` uses the same convention, which is the only reason the two
#: join at all. Get this wrong and the late third of every slate silently
#: fails to match.
EASTERN_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class LineTick:
    """One book's quote for one market at one minute."""

    market: str
    book_slug: str
    book_name: str
    line_ts: datetime
    minutes_to_tip: int
    is_opener: bool
    left_line: float | None
    left_price: int | None
    right_line: float | None
    right_price: int | None

    @property
    def is_pregame(self) -> bool:
        return self.minutes_to_tip < 0


@dataclass(frozen=True)
class ScrapedGame:
    """One game's full line history: every book, every market, one fetch."""

    event_id: int
    game_date: date
    season_year: int
    first_pitch_utc: datetime
    team_away: str
    team_home: str
    status_text: str
    away_score: int | None
    home_score: int | None
    #: SBR's announced starters. MLB-specific: a second, independent handle for
    #: disambiguating a doubleheader, and a cross-check on the game match.
    starter_away: str | None
    starter_home: str | None
    ticks: tuple[LineTick, ...]

    @property
    def books(self) -> tuple[str, ...]:
        return tuple(sorted({tick.book_slug for tick in self.ticks}))

    def ticks_for(self, market: str) -> tuple[LineTick, ...]:
        return tuple(tick for tick in self.ticks if tick.market == market)


@dataclass(frozen=True)
class GameSummary:
    """A game as listed on a day's slate page -- enough to decide whether to fetch it."""

    event_id: int
    game_date: date
    first_pitch_utc: datetime
    team_away: str
    team_home: str
    status_text: str
    starter_away: str | None
    starter_home: str | None


def _sides(
    entry: dict[str, Any], market: str
) -> tuple[float | None, int | None, float | None, int | None]:
    """One payload entry -> ``(left_line, left_price, right_line, right_price)``."""
    if market == MARKET_TOTALS:
        total = as_float(entry.get("total"))
        # Totals quote one number; both sides carry it so the invariant
        # ``left_line == right_line`` holds structurally.
        return (
            total,
            as_int(entry.get("overOdds")),
            total,
            as_int(entry.get("underOdds")),
        )
    if market == MARKET_RUN_LINE:
        return (
            as_float(entry.get("awaySpread")),
            as_int(entry.get("awayOdds")),
            as_float(entry.get("homeSpread")),
            as_int(entry.get("homeOdds")),
        )
    # Moneyline has no line to carry, only prices.
    return None, as_int(entry.get("awayOdds")), None, as_int(entry.get("homeOdds"))


def _ticks_for_book(
    odds_view: dict[str, Any],
    *,
    book_slug: str,
    book_name: str,
    first_pitch_utc: datetime,
    markets: Iterable[str],
) -> list[LineTick]:
    wanted = set(markets)
    ticks: list[LineTick] = []

    for payload_key, market in HISTORY_KEYS.items():
        if market not in wanted:
            continue

        # Collapse to one tick per minute keeping the latest quote, so
        # truncating the timestamp can never produce two rows on one key.
        by_minute: dict[
            datetime, tuple[float | None, int | None, float | None, int | None]
        ] = {}
        for entry in odds_view.get(payload_key) or []:
            if not isinstance(entry, dict):
                continue
            line_ts = parse_utc(entry.get("oddsDate"))
            if line_ts is None:
                continue
            by_minute[line_ts] = _sides(entry, market)

        for index, line_ts in enumerate(sorted(by_minute)):
            left_line, left_price, right_line, right_price = by_minute[line_ts]
            if (
                left_line is None
                and right_line is None
                and left_price is None
                and right_price is None
            ):
                continue
            ticks.append(
                LineTick(
                    market=market,
                    book_slug=book_slug,
                    book_name=book_name,
                    line_ts=line_ts,
                    minutes_to_tip=round(
                        (line_ts - first_pitch_utc).total_seconds() / 60.0
                    ),
                    # The earliest surviving quote is the "Opener" SBR renders
                    # as its own section above the history table.
                    is_opener=index == 0,
                    left_line=left_line,
                    left_price=left_price,
                    right_line=right_line,
                    right_price=right_price,
                )
            )

    return ticks


def parse_line_history_payload(
    payload: dict[str, Any],
    *,
    markets: Iterable[str] = ALL_MARKETS,
) -> ScrapedGame:
    """``__NEXT_DATA__`` from a line-history page -> a :class:`ScrapedGame`."""
    page_props = payload.get("props", {}).get("pageProps", {})
    history_model = page_props.get("lineHistoryModel") or {}
    model = history_model.get("lineHistory") or {}
    game_view = model.get("gameView") or {}

    event_id = as_int(game_view.get("gameId"))
    if event_id is None:
        raise SbrFetchError("Payload carries no gameId")

    first_pitch_utc = parse_utc(game_view.get("startDate"))
    if first_pitch_utc is None:
        raise SbrFetchError(f"Event {event_id}: no usable startDate")

    book_names = {
        str(book.get("machineName") or ""): str(book.get("name") or "")
        for book in history_model.get("sportsbooks") or []
    }

    ticks: list[LineTick] = []
    for odds_view in model.get("oddsViews") or []:
        if not isinstance(odds_view, dict):
            continue
        machine_name = str(odds_view.get("sportsbook") or "").strip()
        if not machine_name:
            continue
        book_name = book_names.get(machine_name) or machine_name
        ticks.extend(
            _ticks_for_book(
                odds_view,
                book_slug=slugify_bookmaker(book_name),
                book_name=book_name,
                first_pitch_utc=first_pitch_utc,
                markets=markets,
            )
        )

    # Some payloads repeat a sportsbook in multiple oddsViews (notably after a
    # postponement). Opener is a property of the complete book/market history,
    # not of each view independently.
    ordered = sorted(ticks, key=lambda t: (t.market, t.book_slug, t.line_ts))
    seen_groups: set[tuple[str, str]] = set()
    normalized: list[LineTick] = []
    for tick in ordered:
        group = (tick.market, tick.book_slug)
        normalized.append(replace(tick, is_opener=group not in seen_groups))
        seen_groups.add(group)

    game_date = first_pitch_utc.astimezone(EASTERN_TZ).date()
    return ScrapedGame(
        event_id=event_id,
        game_date=game_date,
        # An MLB season is contained in one calendar year, so the season is the
        # Eastern-date's year -- no October boundary as in the NBA.
        season_year=game_date.year,
        first_pitch_utc=first_pitch_utc,
        team_away=team_full_name(game_view.get("awayTeam")),
        team_home=team_full_name(game_view.get("homeTeam")),
        status_text=str(game_view.get("gameStatusText") or ""),
        away_score=as_int(game_view.get("awayTeamScore")),
        home_score=as_int(game_view.get("homeTeamScore")),
        starter_away=starter_name(game_view.get("awayStarter")),
        starter_home=starter_name(game_view.get("homeStarter")),
        ticks=tuple(normalized),
    )


def parse_daily_payload(payload: dict[str, Any]) -> list[GameSummary]:
    """``__NEXT_DATA__`` from a day's slate page -> the games it lists."""
    tables = payload.get("props", {}).get("pageProps", {}).get("oddsTables") or []
    summaries: list[GameSummary] = []
    seen: set[int] = set()

    for table in tables:
        for row in (table.get("oddsTableModel") or {}).get("gameRows") or []:
            game_view = (row or {}).get("gameView") or {}
            event_id = as_int(game_view.get("gameId"))
            first_pitch_utc = parse_utc(game_view.get("startDate"))
            if event_id is None or first_pitch_utc is None or event_id in seen:
                continue
            seen.add(event_id)
            summaries.append(
                GameSummary(
                    event_id=event_id,
                    game_date=first_pitch_utc.astimezone(EASTERN_TZ).date(),
                    first_pitch_utc=first_pitch_utc,
                    team_away=team_full_name(game_view.get("awayTeam")),
                    team_home=team_full_name(game_view.get("homeTeam")),
                    status_text=str(game_view.get("gameStatusText") or ""),
                    starter_away=starter_name(game_view.get("awayStarter")),
                    starter_home=starter_name(game_view.get("homeStarter")),
                )
            )

    return summaries


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------
def fetch_game_line_history(
    session: requests.Session,
    event_id: int | str,
    *,
    markets: Iterable[str] = ALL_MARKETS,
) -> ScrapedGame:
    payload = fetch_next_data(session, build_line_history_url(event_id))
    return parse_line_history_payload(payload, markets=markets)


def discover_games_for_date(session: requests.Session, day: date) -> list[GameSummary]:
    """The games SBR lists for ``day``. An empty list means no slate.

    Discovery and detail are separate requests on purpose: a slate listing
    tells you what exists, and detail URLs are never constructed from a
    guessed id range.
    """
    return parse_daily_payload(fetch_next_data(session, build_daily_odds_url(day)))


def scrape_events(
    event_ids: Iterable[int | str],
    *,
    session: requests.Session | None = None,
    markets: Iterable[str] = ALL_MARKETS,
    on_error: str = "warn",
) -> Iterator[ScrapedGame]:
    """Yield a :class:`ScrapedGame` per event id, pausing politely between fetches.

    ``on_error='warn'`` keeps a long backfill going when one game 404s or comes
    back without a payload; ``'raise'`` is the strict mode for tests and small
    targeted runs. A multi-season backfill that dies on hour four because one
    game 404'd is worse than one with a reported gap -- and the update planner
    picks that game up on the next run anyway.
    """
    if on_error not in {"warn", "raise"}:
        raise ValueError("on_error must be 'warn' or 'raise'")

    owned = session is None
    session = session or new_session()
    try:
        for index, event_id in enumerate(event_ids):
            if index:
                sleep_politely()
            try:
                yield fetch_game_line_history(session, event_id, markets=markets)
            except SbrFetchError as exc:
                if on_error == "raise":
                    raise
                print(f"  ! event {event_id}: {exc}")
    finally:
        if owned:
            session.close()


def scrape_dates(
    days: Iterable[date],
    *,
    session: requests.Session | None = None,
    markets: Iterable[str] = ALL_MARKETS,
    on_error: str = "warn",
) -> Iterator[ScrapedGame]:
    """Discover every game listed on each day, then scrape each one's history."""
    owned = session is None
    session = session or new_session()
    try:
        for day in days:
            try:
                summaries = discover_games_for_date(session, day)
            except SbrFetchError as exc:
                if on_error == "raise":
                    raise
                print(f"  ! {day}: {exc}")
                continue

            if not summaries:
                continue
            sleep_politely()
            yield from scrape_events(
                [summary.event_id for summary in summaries],
                session=session,
                markets=markets,
                on_error=on_error,
            )
    finally:
        if owned:
            session.close()
