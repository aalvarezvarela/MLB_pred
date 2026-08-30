"""Season schedule feed -- the backbone of the MLB sports-data layer.

One hydrated request returns an entire season: every game's id, first pitch in
UTC, official slate date, venue, status, final score, announced lineups,
umpire crew, probable starters and first-pitch weather. This is the direct
analogue of the NBA repo preferring the bulk season-schedule endpoint over
~1,400 per-game requests -- here it replaces ~2,430.

Everything this module returns is at **game grain** except the umpire and
lineup frames, which are already at their own natural grain (one row per
official per game, one row per listed player per game).
"""

from __future__ import annotations

import pandas as pd

from mlb_pred.config.constants import (
    HOME_PLATE_OFFICIAL_TYPE,
    INGESTED_GAME_TYPES,
    POSTSEASON_GAME_TYPES,
    is_known_team_id,
    standardize_team_name,
)
from mlb_pred.config.settings import SETTINGS
from mlb_pred.fetch_data.statsapi.client import statsapi_get
from mlb_pred.utils.general_utils import as_id
from mlb_pred.utils.seasons import classify_game_type, season_date_bounds

# Hydrations that make the schedule feed self-sufficient. Each is cheap on the
# server side and saves an entire per-game request family on ours.
SCHEDULE_HYDRATE = ",".join(
    [
        "team",
        "venue(location,fieldInfo,timezone)",
        "probablePitcher",
        "linescore",
        "weather",
        "officials",
        "lineups",
        "decisions",
        "seriesStatus",
    ]
)

GAME_COLUMNS = [
    "game_pk",
    "season_year",
    "game_date",
    "first_pitch_utc",
    "game_type",
    "season_type",
    "series_description",
    "game_number",
    "doubleheader",
    "day_night",
    "scheduled_innings",
    "status_code",
    "status_detailed",
    "status_abstract",
    "is_final",
    "venue_id",
    "venue_name",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
    "home_score",
    "away_score",
    "home_hits",
    "away_hits",
    "home_errors",
    "away_errors",
    "innings_played",
    "extra_innings",
    "total_runs",
    "run_line_margin",
    "home_probable_pitcher_id",
    "away_probable_pitcher_id",
    "weather_condition",
    "weather_temp_f",
    "weather_wind",
    "series_game_number",
    "games_in_series",
]

UMPIRE_COLUMNS = [
    "game_pk",
    "season_year",
    "game_date",
    "official_id",
    "official_name",
    "official_type",
    "is_home_plate",
]

LINEUP_COLUMNS = [
    "game_pk",
    "season_year",
    "game_date",
    "team_id",
    "home",
    "player_id",
    "player_name",
    "lineup_slot",
    "position_code",
    "position_abbrev",
]


def _to_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _linescore_totals(linescore: dict, side: str) -> tuple[int | None, ...]:
    totals = (linescore or {}).get("teams", {}).get(side, {})
    return (
        _to_int(totals.get("runs")),
        _to_int(totals.get("hits")),
        _to_int(totals.get("errors")),
    )


#: ``status.abstractGameState`` for a completed game.
FINAL_STATUS_ABSTRACT = "Final"


class UndeterminedParticipantsError(ValueError):
    """A scheduled game whose participants are still a bracket placeholder."""


def _check_participants(game: dict) -> None:
    """Refuse a game whose teams are placeholders, but only where that is sane.

    The Stats API publishes an unplayed postseason bracket with placeholder
    "teams" ("AL Wild Card #1", "NL Higher Seed"). Those are not games with
    unknown teams -- they are games with *undetermined* teams, and skipping
    them is correct.

    The distinction matters because the alternative reading of an unknown team
    id is an **expansion franchise**, which must never be skipped silently.
    So the escape hatch is deliberately narrow: a placeholder is only accepted
    as such on a postseason game that has not been played. An unknown id on a
    regular-season game, or on any game with a final score, falls through to
    the name map and raises -- which is what should happen the day MLB adds a
    team.
    """
    status = (game.get("status") or {}).get("abstractGameState")
    is_postseason = game.get("gameType") in POSTSEASON_GAME_TYPES
    unplayed = status != FINAL_STATUS_ABSTRACT

    for side in ("home", "away"):
        team = game["teams"][side]["team"]
        if is_known_team_id(as_id(team.get("id"))):
            continue
        if is_postseason and unplayed:
            raise UndeterminedParticipantsError(
                f"game {game.get('gamePk')}: {team.get('name')!r} is a bracket "
                "placeholder"
            )
        # Not explainable as a placeholder -- let the name map decide, loudly.
        standardize_team_name(team.get("name"))


def _parse_game(game: dict, season_year: int) -> dict:
    home = game["teams"]["home"]
    away = game["teams"]["away"]
    linescore = game.get("linescore") or {}
    weather = game.get("weather") or {}
    venue = game.get("venue") or {}

    home_name = standardize_team_name(home["team"]["name"])
    away_name = standardize_team_name(away["team"]["name"])

    home_score, home_hits, home_errors = _linescore_totals(linescore, "home")
    away_score, away_hits, away_errors = _linescore_totals(linescore, "away")

    # The linescore's own team totals are the authority; the schedule's
    # ``teams.*.score`` agrees for finished games but is absent while a game
    # is scheduled, so fall back rather than losing the score.
    if home_score is None:
        home_score = _to_int(home.get("score"))
    if away_score is None:
        away_score = _to_int(away.get("score"))

    scheduled_innings = _to_int(game.get("scheduledInnings")) or 9
    innings_played = _to_int(linescore.get("currentInning"))

    total_runs = (
        home_score + away_score
        if home_score is not None and away_score is not None
        else None
    )
    # Run line is quoted from the home team's perspective, matching the NBA
    # repo's home-minus-away convention for spreads.
    run_line_margin = (
        home_score - away_score
        if home_score is not None and away_score is not None
        else None
    )

    return {
        "game_pk": as_id(game["gamePk"]),
        "season_year": season_year,
        # ``officialDate`` is the local slate date MLB itself assigns. It is
        # the join key across the games table, the odds feed and the daily
        # snapshots, and it already handles both after-midnight finishes and
        # the second game of a doubleheader.
        "game_date": pd.to_datetime(game["officialDate"]).date(),
        # ``gameDate`` is the scheduled UTC start (including official schedule
        # revisions), not guaranteed to be the actual first pitch after a
        # delay. A slate date and an instant cannot do each other's job: the
        # date joins, while the instant orders anything temporal.
        "first_pitch_utc": pd.to_datetime(game["gameDate"], utc=True),
        "game_type": game["gameType"],
        "season_type": classify_game_type(game["gameType"]),
        "series_description": game.get("seriesDescription"),
        "game_number": _to_int(game.get("gameNumber")) or 1,
        "doubleheader": game.get("doubleHeader", "N"),
        "day_night": game.get("dayNight"),
        "scheduled_innings": scheduled_innings,
        "status_code": (game.get("status") or {}).get("statusCode"),
        "status_detailed": (game.get("status") or {}).get("detailedState"),
        "status_abstract": (game.get("status") or {}).get("abstractGameState"),
        "is_final": (game.get("status") or {}).get("abstractGameState")
        == FINAL_STATUS_ABSTRACT
        and total_runs is not None,
        "venue_id": as_id(venue.get("id")),
        "venue_name": venue.get("name"),
        "home_team_id": as_id(home["team"]["id"]),
        "home_team_name": home_name,
        "away_team_id": as_id(away["team"]["id"]),
        "away_team_name": away_name,
        "home_score": home_score,
        "away_score": away_score,
        "home_hits": home_hits,
        "away_hits": away_hits,
        "home_errors": home_errors,
        "away_errors": away_errors,
        "innings_played": innings_played,
        "extra_innings": (
            None if innings_played is None else innings_played > scheduled_innings
        ),
        "total_runs": total_runs,
        "run_line_margin": run_line_margin,
        # Probable starters as published. For a finished game this is the
        # settled announcement; for a scheduled game it is the live signal the
        # prediction path will actually consume, which is why it is captured
        # here and snapshotted daily rather than reconstructed after the fact.
        "home_probable_pitcher_id": as_id(
            (home.get("probablePitcher") or {}).get("id")
        ),
        "away_probable_pitcher_id": as_id(
            (away.get("probablePitcher") or {}).get("id")
        ),
        "weather_condition": weather.get("condition"),
        "weather_temp_f": _to_int(weather.get("temp")),
        "weather_wind": weather.get("wind"),
        "series_game_number": _to_int(game.get("seriesGameNumber")),
        "games_in_series": _to_int(game.get("gamesInSeries")),
    }


def _parse_umpires(game: dict, season_year: int) -> list[dict]:
    game_pk = as_id(game["gamePk"])
    game_date = pd.to_datetime(game["officialDate"]).date()
    rows = []
    for official in game.get("officials") or []:
        person = official.get("official") or {}
        official_type = official.get("officialType")
        rows.append(
            {
                "game_pk": game_pk,
                "season_year": season_year,
                "game_date": game_date,
                "official_id": as_id(person.get("id")),
                "official_name": person.get("fullName"),
                "official_type": official_type,
                "is_home_plate": official_type == HOME_PLATE_OFFICIAL_TYPE,
            }
        )
    return rows


def _parse_lineups(game: dict, season_year: int) -> list[dict]:
    lineups = game.get("lineups") or {}
    if not lineups:
        return []

    game_pk = as_id(game["gamePk"])
    game_date = pd.to_datetime(game["officialDate"]).date()
    rows = []
    for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
        team_id = as_id(game["teams"][side]["team"]["id"])
        for slot, player in enumerate(lineups.get(key) or [], start=1):
            position = player.get("primaryPosition") or {}
            rows.append(
                {
                    "game_pk": game_pk,
                    "season_year": season_year,
                    "game_date": game_date,
                    "team_id": team_id,
                    "home": side == "home",
                    "player_id": as_id(player.get("id")),
                    "player_name": player.get("fullName"),
                    "lineup_slot": slot,
                    "position_code": position.get("code"),
                    "position_abbrev": position.get("abbreviation"),
                }
            )
    return rows


def deduplicate_games(games: pd.DataFrame) -> pd.DataFrame:
    """Collapse the schedule feed's repeated listings of the same ``game_pk``.

    The feed lists a game once per slot it has occupied, which happens in two
    situations and neither is a data error:

    * **Postponed then replayed.** The original slot survives with a
      ``Postponed`` status and no score, alongside the ``Final`` row for the
      day it was actually played.
    * **Suspended then resumed.** Both rows are ``Final`` and carry the same
      score; only ``first_pitch_utc`` differs.

    Left alone this silently doubles a team's game count for the affected
    dates, which is precisely the kind of quiet corruption that never surfaces
    in a rolling mean. The rule: prefer a row that actually finished, and
    among finished rows keep the earliest first pitch -- for a resumed game
    that is the real first pitch, not the continuation.
    """
    if games.empty:
        return games

    ordered = games.sort_values(
        ["game_pk", "is_final", "first_pitch_utc"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return (
        ordered.drop_duplicates(subset="game_pk", keep="first")
        .sort_values(["game_date", "first_pitch_utc", "game_pk"], kind="mergesort")
        .reset_index(drop=True)
    )


def finished_games(games: pd.DataFrame) -> pd.DataFrame:
    """Only games that actually reached a final score.

    Postponed-and-never-replayed and cancelled games are dropped rather than
    stored with NULL scores: they are not games, and a NULL score row would
    reach a rolling mean as a real observation.
    """
    if games.empty:
        return games
    return games[games["is_final"]].reset_index(drop=True)


def fetch_schedule(
    start_date,
    end_date,
    *,
    game_types: frozenset[str] | None = None,
    hydrate: str = SCHEDULE_HYDRATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch the schedule for a date range.

    Returns ``(games, umpires, lineups)``. Games not in ``game_types`` are
    dropped -- by default that means spring training, exhibitions, intrasquad
    games and the All-Star Game never enter the store.
    """
    allowed = INGESTED_GAME_TYPES if game_types is None else game_types

    payload = statsapi_get(
        "schedule",
        {
            "sportId": SETTINGS.statsapi_sport_id,
            "startDate": pd.to_datetime(start_date).strftime("%Y-%m-%d"),
            "endDate": pd.to_datetime(end_date).strftime("%Y-%m-%d"),
            "gameTypes": ",".join(sorted(allowed)),
            "hydrate": hydrate,
        },
    )

    games: list[dict] = []
    umpires: list[dict] = []
    lineups: list[dict] = []

    undetermined = 0
    for slate in payload.get("dates", []):
        for game in slate.get("games", []):
            if game.get("gameType") not in allowed:
                continue
            try:
                _check_participants(game)
            except UndeterminedParticipantsError:
                # A future postseason slot. Not a game yet.
                undetermined += 1
                continue
            season_year = int(
                game.get("season") or pd.to_datetime(game["officialDate"]).year
            )
            games.append(_parse_game(game, season_year))
            umpires.extend(_parse_umpires(game, season_year))
            lineups.extend(_parse_lineups(game, season_year))

    if undetermined:
        print(
            f"  skipped {undetermined} unplayed postseason game(s) whose "
            "participants are still bracket placeholders"
        )

    games_df = deduplicate_games(pd.DataFrame(games, columns=GAME_COLUMNS))

    # The child frames inherit the same double-listing, so resolve them on
    # their own natural keys rather than trusting the feed.
    umpires_df = (
        pd.DataFrame(umpires, columns=UMPIRE_COLUMNS)
        .drop_duplicates(subset=["game_pk", "official_id", "official_type"])
        .reset_index(drop=True)
    )
    lineups_df = (
        pd.DataFrame(lineups, columns=LINEUP_COLUMNS)
        .sort_values(["game_pk", "team_id", "lineup_slot"], kind="mergesort")
        .drop_duplicates(subset=["game_pk", "team_id", "player_id"])
        .reset_index(drop=True)
    )
    return games_df, umpires_df, lineups_df


def fetch_season_schedule(
    season_year: int, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch one whole season in a single request."""
    start, end = season_date_bounds(season_year)
    return fetch_schedule(start, end, **kwargs)
