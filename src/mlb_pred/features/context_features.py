"""Pregame team identity, record, rest, and series-travel features.

All historical state is computed on team-game rows and date-gated. Games on
the same MLB calendar date therefore receive identical prior-result history.
Travel to the current series is included because the venue and trip are known
before first pitch; realised game outcomes are never returned.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime, time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from mlb_pred.config.constants import TEAM_ID_MAP, TEAM_ID_TO_NAME

# Rest and travel are read as a home-versus-away contrast, and season win rate
# is the standard form-gap column. Everything else keeps only its two side
# columns; the difference is recoverable and a tree finds it unaided.
DIFF_FEATURES: tuple[str, ...] = (
    "SCHEDULE_REST_DAYS_BEFORE",
    "SCHEDULE_GAMES_IN_LAST_7_DAYS_BEFORE",
    "TRAVEL_LOG1P_KM_LAST_7_DAYS_BEFORE",
    "TRAVEL_JETLAG_HOURS_FROM_PREVIOUS_SERIES_BEFORE",
    "TEAM_RECORD_WIN_RATIO_SEASON_BEFORE",
    "TEAM_RECORD_WIN_RATIO_LAST_10_GAMES_BEFORE",
)

FIRST_GAME_REST_DAYS = 7
WIN_WINDOWS = (5, 10, 20, 30)
TRAVEL_DAY_WINDOWS = (1, 2, 5, 7, 14)
JETLAG_ADAPTATION_DAYS = 4
EARTH_RADIUS_KM = 6_371.0
NEUTRAL_WIN_RATIO = 0.5

_GAME_COLUMNS = {
    "game_pk",
    "season_year",
    "game_date",
    "first_pitch_utc",
    "game_type",
    "venue_id",
    "home_team_id",
    "away_team_id",
    "day_night",
    "series_game_number",
    "games_in_series",
}
_TEAM_RESULT_COLUMNS = {"game_pk", "team_id", "win"}
_VENUE_COLUMNS = {
    "venue_id",
    "season",
    "city",
    "country",
    "latitude",
    "longitude",
    "timezone_id",
}


def _require_columns(
    frame: pd.DataFrame, required: set[str], *, frame_name: str
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _venue_lookup(venues: pd.DataFrame) -> pd.DataFrame:
    """Create one season-aware venue row, with a city-level coordinate fallback."""
    _require_columns(venues, _VENUE_COLUMNS, frame_name="venues")
    source = venues[list(_VENUE_COLUMNS)].copy()
    source["venue_id"] = source["venue_id"].astype(str)
    source["season"] = pd.to_numeric(source["season"], errors="raise").astype(int)
    source["latitude"] = pd.to_numeric(source["latitude"], errors="coerce")
    source["longitude"] = pd.to_numeric(source["longitude"], errors="coerce")

    city_coordinates = (
        source.dropna(subset=["city", "country", "latitude", "longitude"])
        .groupby(["city", "country"], as_index=False)[["latitude", "longitude"]]
        .median()
        .rename(
            columns={
                "latitude": "__city_latitude",
                "longitude": "__city_longitude",
            }
        )
    )
    source = source.merge(
        city_coordinates,
        on=["city", "country"],
        how="left",
        validate="many_to_one",
    )
    source["latitude"] = source["latitude"].fillna(source["__city_latitude"])
    source["longitude"] = source["longitude"].fillna(source["__city_longitude"])
    source = source.drop(columns=["__city_latitude", "__city_longitude"])
    if source.duplicated(["venue_id", "season"]).any():
        raise ValueError("venues is not unique by (venue_id, season).")
    return source.rename(columns={"season": "season_year"})


def _build_team_log(
    games: pd.DataFrame, team_games: pd.DataFrame, venues: pd.DataFrame
) -> pd.DataFrame:
    _require_columns(games, _GAME_COLUMNS, frame_name="games")
    _require_columns(team_games, _TEAM_RESULT_COLUMNS, frame_name="team games")
    if games["game_pk"].astype(str).duplicated().any():
        raise ValueError("games is not unique by game_pk.")

    game_columns = list(_GAME_COLUMNS)
    schedule = games[game_columns].copy()
    schedule["game_pk"] = schedule["game_pk"].astype(str)
    schedule["venue_id"] = schedule["venue_id"].astype("string")
    schedule["season_year"] = pd.to_numeric(
        schedule["season_year"], errors="raise"
    ).astype(int)
    schedule["game_date"] = pd.to_datetime(
        schedule["game_date"], errors="raise"
    ).dt.normalize()
    schedule["first_pitch_utc"] = pd.to_datetime(
        schedule["first_pitch_utc"], utc=True, errors="raise"
    )
    schedule = schedule.merge(
        _venue_lookup(venues),
        on=["venue_id", "season_year"],
        how="left",
        validate="many_to_one",
    )

    shared = [
        "game_pk",
        "season_year",
        "game_date",
        "first_pitch_utc",
        "game_type",
        "venue_id",
        "day_night",
        "series_game_number",
        "games_in_series",
        "city",
        "country",
        "latitude",
        "longitude",
        "timezone_id",
    ]
    home = schedule[[*shared, "home_team_id", "away_team_id"]].rename(
        columns={"home_team_id": "team_id", "away_team_id": "opponent_team_id"}
    )
    home["home"] = True
    away = schedule[[*shared, "away_team_id", "home_team_id"]].rename(
        columns={"away_team_id": "team_id", "home_team_id": "opponent_team_id"}
    )
    away["home"] = False
    log = pd.concat([home, away], ignore_index=True)
    log["team_id"] = log["team_id"].astype(str)
    log["opponent_team_id"] = log["opponent_team_id"].astype(str)

    results = team_games[["game_pk", "team_id", "win"]].copy()
    results["game_pk"] = results["game_pk"].astype(str)
    results["team_id"] = results["team_id"].astype(str)
    if results.duplicated(["game_pk", "team_id"]).any():
        raise ValueError("team games is not unique by (game_pk, team_id).")
    results["win"] = results["win"].astype("boolean").astype("Float64")
    log = log.merge(
        results, on=["game_pk", "team_id"], how="left", validate="one_to_one"
    )
    return log.sort_values(
        ["team_id", "season_year", "game_date", "first_pitch_utc", "game_pk"],
        kind="mergesort",
    ).reset_index(drop=True)


def _haversine_km(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    values = [
        pd.to_numeric(item, errors="coerce").to_numpy(dtype="float64")
        for item in (lat1, lon1, lat2, lon2)
    ]
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, values)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    )
    return pd.Series(EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(a)), index=lat1.index)


def _timezone_offset_hours(timezone_name: object, game_date: object) -> float:
    if pd.isna(timezone_name) or pd.isna(game_date):
        return np.nan
    try:
        timezone = ZoneInfo(str(timezone_name))
    except ZoneInfoNotFoundError:
        return np.nan
    local = datetime.combine(pd.Timestamp(game_date).date(), time(12), tzinfo=timezone)
    offset = local.utcoffset()
    return np.nan if offset is None else offset.total_seconds() / 3_600.0


def _add_schedule_features(log: pd.DataFrame) -> pd.DataFrame:
    out = log.copy()
    previous_team = out["team_id"].eq(out["team_id"].shift())
    previous_season = out["season_year"].eq(out["season_year"].shift())
    same_team_season = previous_team & previous_season
    previous_date = out["game_date"].shift().where(same_team_season)
    previous_venue = out["venue_id"].shift().where(same_team_season)
    previous_opponent = out["opponent_team_id"].shift().where(same_team_season)
    gap_days = (out["game_date"] - previous_date).dt.days
    venue_changed = (
        out["venue_id"]
        .fillna("__MISSING_VENUE__")
        .ne(previous_venue.fillna("__MISSING_VENUE__"))
    )
    opponent_changed = (
        out["opponent_team_id"]
        .fillna("__MISSING_TEAM__")
        .ne(previous_opponent.fillna("__MISSING_TEAM__"))
    )

    new_series = (
        ~same_team_season
        | pd.to_numeric(out["series_game_number"], errors="coerce").eq(1)
        | venue_changed
        | opponent_changed
        | gap_days.gt(2)
    ).fillna(True)
    out["__series_id"] = new_series.groupby(
        [out["team_id"], out["season_year"]], sort=False
    ).cumsum()
    series_keys = ["team_id", "season_year", "__series_id"]
    out["SCHEDULE_SERIES_GAME_NUMBER_BEFORE"] = pd.to_numeric(
        out["series_game_number"], errors="coerce"
    ).fillna(1.0)
    out["SCHEDULE_IS_SERIES_OPENER_BEFORE"] = new_series.astype("int8")
    out["SCHEDULE_IS_GETAWAY_DAY_BEFORE"] = (
        pd.to_numeric(out["series_game_number"], errors="coerce")
        .eq(pd.to_numeric(out["games_in_series"], errors="coerce"))
        .astype("int8")
    )

    previous_latitude = out["latitude"].shift().where(same_team_season)
    previous_longitude = out["longitude"].shift().where(same_team_season)
    series_distance = _haversine_km(
        previous_latitude, previous_longitude, out["latitude"], out["longitude"]
    ).where(new_series, 0.0)
    series_missing = (
        new_series
        & same_team_season
        & (
            previous_latitude.isna()
            | previous_longitude.isna()
            | out["latitude"].isna()
            | out["longitude"].isna()
        )
    )
    series_distance = series_distance.fillna(0.0)
    out["__travel_event_km"] = series_distance
    out["TRAVEL_KM_FROM_PREVIOUS_SERIES_BEFORE"] = out.groupby(series_keys, sort=False)[
        "__travel_event_km"
    ].transform("first")
    out["TRAVEL_LOCATION_MISSING_BEFORE"] = (
        series_missing.astype("int8")
        .groupby([out[key] for key in series_keys], sort=False)
        .transform("first")
    )

    previous_timezone = out["timezone_id"].shift().where(same_team_season)
    current_offsets = pd.Series(
        [
            _timezone_offset_hours(zone, date)
            for zone, date in zip(out["timezone_id"], out["game_date"], strict=True)
        ],
        index=out.index,
    )
    previous_offsets = pd.Series(
        [
            _timezone_offset_hours(zone, date)
            for zone, date in zip(previous_timezone, out["game_date"], strict=True)
        ],
        index=out.index,
    )
    jetlag = (
        (current_offsets - previous_offsets)
        .abs()
        .where(new_series & gap_days.le(JETLAG_ADAPTATION_DAYS), 0.0)
    )
    jetlag_missing = (
        new_series
        & same_team_season
        & gap_days.le(JETLAG_ADAPTATION_DAYS)
        & (current_offsets.isna() | previous_offsets.isna())
    )
    out["TRAVEL_JETLAG_HOURS_FROM_PREVIOUS_SERIES_BEFORE"] = (
        jetlag.fillna(0.0).groupby([out[key] for key in series_keys]).transform("first")
    )
    out["TRAVEL_TIMEZONE_MISSING_BEFORE"] = (
        jetlag_missing.astype("int8")
        .groupby([out[key] for key in series_keys])
        .transform("first")
    )

    daily = (
        out.groupby(["team_id", "season_year", "game_date"], as_index=False)
        .agg(
            __games_today=("game_pk", "size"),
            __travel_event_km=("__travel_event_km", "sum"),
        )
        .sort_values(["team_id", "season_year", "game_date"], kind="mergesort")
    )
    for column in (
        "SCHEDULE_REST_DAYS_BEFORE",
        "SCHEDULE_CALENDAR_GAP_DAYS_BEFORE",
        "SCHEDULE_CONSECUTIVE_GAME_DAYS_BEFORE",
    ):
        daily[column] = 0.0
    for window in (3, 7, 14):
        daily[f"SCHEDULE_GAMES_IN_LAST_{window}_DAYS_BEFORE"] = 0.0
    for window in TRAVEL_DAY_WINDOWS:
        daily[f"__travel_km_last_{window}_days"] = 0.0

    for _, indices in daily.groupby(
        ["team_id", "season_year"], sort=False
    ).groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        dates = daily.loc[positions, "game_date"]
        day_numbers = dates.to_numpy(dtype="datetime64[D]").astype("int64")
        differences = np.diff(day_numbers, prepend=day_numbers[0])
        calendar_gap = np.where(np.arange(len(positions)) == 0, 8, differences)
        rest_days = np.where(
            np.arange(len(positions)) == 0,
            FIRST_GAME_REST_DAYS,
            np.maximum(differences - 1, 0),
        )
        consecutive = np.zeros(len(positions), dtype="float64")
        run = 0
        for offset in range(1, len(positions)):
            run = run + 1 if differences[offset] == 1 else 0
            consecutive[offset] = run
        daily.loc[positions, "SCHEDULE_REST_DAYS_BEFORE"] = rest_days
        daily.loc[positions, "SCHEDULE_CALENDAR_GAP_DAYS_BEFORE"] = calendar_gap
        daily.loc[positions, "SCHEDULE_CONSECUTIVE_GAME_DAYS_BEFORE"] = consecutive

        games_today = daily.loc[positions, "__games_today"].to_numpy(dtype="float64")
        cumulative_games = np.cumsum(games_today)
        for window in (3, 7, 14):
            left = np.searchsorted(day_numbers, day_numbers - window, side="left")
            before = np.concatenate(([0.0], cumulative_games[:-1]))
            before_left = np.where(left > 0, cumulative_games[left - 1], 0.0)
            daily.loc[positions, f"SCHEDULE_GAMES_IN_LAST_{window}_DAYS_BEFORE"] = (
                before - before_left
            )

        travel = daily.loc[positions, "__travel_event_km"].to_numpy(dtype="float64")
        cumulative_travel = np.cumsum(travel)
        for window in TRAVEL_DAY_WINDOWS:
            left = np.searchsorted(day_numbers, day_numbers - window, side="left")
            before_left = np.where(left > 0, cumulative_travel[left - 1], 0.0)
            daily.loc[positions, f"__travel_km_last_{window}_days"] = (
                cumulative_travel - before_left
            )

    daily["SCHEDULE_IS_NO_REST_BEFORE"] = (
        daily["SCHEDULE_REST_DAYS_BEFORE"].eq(0).astype("int8")
    )
    for window in TRAVEL_DAY_WINDOWS:
        raw = daily[f"__travel_km_last_{window}_days"]
        daily[f"TRAVEL_LOG1P_KM_LAST_{window}_DAYS_BEFORE"] = np.log1p(raw)
    denominator = daily["__travel_km_last_14_days"].replace(0.0, np.nan)
    daily["TRAVEL_RECENCY_RATIO_2D_OVER_14D_BEFORE"] = (
        daily["__travel_km_last_2_days"] / denominator
    ).fillna(0.0)

    daily_features = [
        column
        for column in daily.columns
        if column.startswith(("SCHEDULE_", "TRAVEL_"))
    ]
    out = out.merge(
        daily[["team_id", "season_year", "game_date", *daily_features]],
        on=["team_id", "season_year", "game_date"],
        how="left",
        validate="many_to_one",
    )

    last_game_by_date = (
        out.sort_values(["team_id", "game_date", "first_pitch_utc", "game_pk"])
        .groupby(["team_id", "season_year", "game_date"], as_index=False)
        .tail(1)[["team_id", "season_year", "game_date", "day_night"]]
        .sort_values(["team_id", "season_year", "game_date"])
    )
    last_game_by_date["__previous_day_night"] = last_game_by_date.groupby(
        ["team_id", "season_year"], sort=False
    )["day_night"].shift()
    out = out.merge(
        last_game_by_date[
            ["team_id", "season_year", "game_date", "__previous_day_night"]
        ],
        on=["team_id", "season_year", "game_date"],
        how="left",
        validate="many_to_one",
    )
    out["SCHEDULE_DAY_GAME_AFTER_NIGHT_BEFORE"] = (
        out["day_night"].astype("string").str.lower().eq("day")
        & out["__previous_day_night"].astype("string").str.lower().eq("night")
        & out["SCHEDULE_CALENDAR_GAP_DAYS_BEFORE"].eq(1)
    ).astype("int8")
    return out


def _previous_regular_win_ratios(log: pd.DataFrame) -> dict[tuple[str, int], float]:
    regular = log.loc[log["game_type"].eq("R") & log["win"].notna()].copy()
    ratios = regular.groupby(["team_id", "season_year"])["win"].mean()
    return {
        (str(team_id), int(season_year) + 1): float(value)
        for (team_id, season_year), value in ratios.items()
    }


def _add_win_features(log: pd.DataFrame) -> pd.DataFrame:
    out = log.copy()
    previous_ratios = _previous_regular_win_ratios(out)
    feature_names = [
        "TEAM_RECORD_WINS_SEASON_BEFORE",
        "TEAM_RECORD_GAMES_SEASON_BEFORE",
        "TEAM_RECORD_WIN_RATIO_SEASON_BEFORE",
        "TEAM_RECORD_HAS_CURRENT_SEASON_HISTORY_BEFORE",
        "TEAM_RECORD_CURRENT_WIN_STREAK_BEFORE",
    ]
    # Only the ratio: measured over 17,638 games, WINS_LAST_N and
    # WIN_RATIO_LAST_N correlate at r = 1.000000 because GAMES_LAST_N is
    # constant once a team has N games of history. The season-level pair is
    # kept because games played genuinely varies there.
    for window in WIN_WINDOWS:
        feature_names.append(f"TEAM_RECORD_WIN_RATIO_LAST_{window}_GAMES_BEFORE")
    values = {name: np.zeros(len(out), dtype="float64") for name in feature_names}

    for team_id, team_indices in out.groupby("team_id", sort=False).groups.items():
        history: list[float] = []
        states: dict[tuple[int, str], dict[str, float]] = {}
        team_frame = out.loc[team_indices]
        for _, date_indices in team_frame.groupby(
            "game_date", sort=False
        ).groups.items():
            positions = np.asarray(list(date_indices), dtype=np.int64)
            for position in positions:
                season = int(out.at[position, "season_year"])
                game_type = str(out.at[position, "game_type"])
                state = states.setdefault(
                    (season, game_type), {"wins": 0.0, "games": 0.0, "streak": 0.0}
                )
                values["TEAM_RECORD_WINS_SEASON_BEFORE"][position] = state["wins"]
                values["TEAM_RECORD_GAMES_SEASON_BEFORE"][position] = state["games"]
                values["TEAM_RECORD_HAS_CURRENT_SEASON_HISTORY_BEFORE"][position] = (
                    float(state["games"] > 0)
                )
                values["TEAM_RECORD_WIN_RATIO_SEASON_BEFORE"][position] = (
                    state["wins"] / state["games"]
                    if state["games"]
                    else previous_ratios.get((str(team_id), season), NEUTRAL_WIN_RATIO)
                )
                values["TEAM_RECORD_CURRENT_WIN_STREAK_BEFORE"][position] = state[
                    "streak"
                ]
                for window in WIN_WINDOWS:
                    recent = history[-window:]
                    games = float(len(recent))
                    values[f"TEAM_RECORD_WIN_RATIO_LAST_{window}_GAMES_BEFORE"][
                        position
                    ] = (float(sum(recent)) / games if games else NEUTRAL_WIN_RATIO)

            for position in positions:
                result = out.at[position, "win"]
                if pd.isna(result):
                    continue
                win = float(result)
                history.append(win)
                key = (
                    int(out.at[position, "season_year"]),
                    str(out.at[position, "game_type"]),
                )
                state = states.setdefault(
                    key, {"wins": 0.0, "games": 0.0, "streak": 0.0}
                )
                state["wins"] += win
                state["games"] += 1.0
                state["streak"] = state["streak"] + 1.0 if win else 0.0

    return pd.concat([out, pd.DataFrame(values, index=out.index)], axis=1)


IDENTITY_PREFIX = "TEAM_IDENTITY_"


def _team_slug(team_name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", team_name.upper()).strip("_")


# One slug per franchise, ordered as the id map is, so the column block has a
# stable shape no matter which teams appear in a given slice of games.
TEAM_SLUGS: tuple[tuple[str, str], ...] = tuple(
    (team_id, _team_slug(team_name)) for team_name, team_id in TEAM_ID_MAP.items()
)


def _add_team_identity(wide: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Add the 60-column home/away franchise one-hot block.

    Club identity is not a fact about form, so it carries no history and needs
    no date gate -- which side is batting last is known the moment the schedule
    is published. It is tagged ``_BEFORE`` because the pregame contract requires
    every ``TEAM_*`` column to declare its temporal status, and this one is
    always available.
    """
    sides = games[["game_pk", "home_team_id", "away_team_id"]].copy()
    sides["game_pk"] = sides["game_pk"].astype(str)
    sides["home_team_id"] = sides["home_team_id"].astype(str)
    sides["away_team_id"] = sides["away_team_id"].astype(str)
    source = wide[["GAME_ID"]].merge(
        sides, left_on="GAME_ID", right_on="game_pk", how="left", validate="one_to_one"
    )

    unknown = sorted(
        set(source["home_team_id"])
        .union(source["away_team_id"])
        .difference(TEAM_ID_TO_NAME)
    )
    if unknown:
        raise ValueError(f"Cannot one-hot unknown team ids: {unknown}")

    columns = {}
    for team_id, slug in TEAM_SLUGS:
        columns[f"{IDENTITY_PREFIX}HOME_{slug}_BEFORE"] = (
            source["home_team_id"].eq(team_id).astype("int8")
        )
        columns[f"{IDENTITY_PREFIX}AWAY_{slug}_BEFORE"] = (
            source["away_team_id"].eq(team_id).astype("int8")
        )
    return pd.concat([wide, pd.DataFrame(columns, index=wide.index)], axis=1)


def _wide_team_features(
    team_rows: pd.DataFrame, target_game_ids: set[str]
) -> pd.DataFrame:
    feature_columns = [
        column
        for column in team_rows.columns
        if column.startswith(("SCHEDULE_", "TRAVEL_", "TEAM_RECORD_"))
        and "_BEFORE" in column
    ]
    target = team_rows.loc[team_rows["game_pk"].isin(target_game_ids)]
    counts = target.groupby("game_pk").agg(
        rows=("team_id", "size"), homes=("home", "sum")
    )
    invalid = counts.loc[(counts["rows"] != 2) | (counts["homes"] != 1)]
    if not invalid.empty:
        raise ValueError(
            f"Target games must have exactly one home and one away row: {invalid.head()}"
        )
    home = target.loc[target["home"], ["game_pk", *feature_columns]].rename(
        columns={column: f"{column}_TEAM_HOME" for column in feature_columns}
    )
    away = target.loc[~target["home"], ["game_pk", *feature_columns]].rename(
        columns={column: f"{column}_TEAM_AWAY" for column in feature_columns}
    )
    wide = home.merge(away, on="game_pk", how="inner", validate="one_to_one")
    wide = wide.rename(columns={"game_pk": "GAME_ID"})
    # HOME - AWAY is an exact linear combination of the two side columns, so it
    # is emitted only for the contrasts that are read as a contrast. NBA does
    # the same, hand-picking three.
    missing = sorted(set(DIFF_FEATURES).difference(feature_columns))
    if missing:
        raise ValueError(f"DIFF_FEATURES names unknown context columns: {missing}")
    differences = {
        column.removesuffix("_BEFORE")
        + "_DIFF_BEFORE": wide[f"{column}_TEAM_HOME"]
        - wide[f"{column}_TEAM_AWAY"]
        for column in DIFF_FEATURES
    }
    wide = pd.concat([wide, pd.DataFrame(differences, index=wide.index)], axis=1)
    wide["SCHEDULE_BOTH_TEAMS_NO_REST_BEFORE"] = (
        wide["SCHEDULE_IS_NO_REST_BEFORE_TEAM_HOME"].eq(1)
        & wide["SCHEDULE_IS_NO_REST_BEFORE_TEAM_AWAY"].eq(1)
    ).astype("int8")
    return wide


def build_context_features(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    venues: pd.DataFrame,
    closing_features: pd.DataFrame,
    *,
    target_game_ids: Iterable[str] | None = None,
    include_team_identity: bool = True,
) -> pd.DataFrame:
    """Build identity, schedule, travel and win-form features for the targets.

    The 60-column ``TEAM_IDENTITY_*`` one-hot block matches the NBA project's
    ``team_one_hot_features``. It was dropped once during feature consolidation
    as redundant width -- 60 binary columns over 17,638 rows -- but it was never
    measured to cost anything, and a linear model has no other way to learn that
    Coors Field's tenant scores differently from Oakland's. Pass
    ``include_team_identity=False`` for a run that wants only the metadata
    columns ``GAME_HOME_TEAM_ID`` / ``GAME_AWAY_TEAM_ID``.
    """
    targets = (
        {str(value) for value in target_game_ids}
        if target_game_ids is not None
        else {str(value) for value in closing_features["GAME_ID"]}
    )
    log = _build_team_log(games, team_games, venues)
    log = _add_schedule_features(log)
    log = _add_win_features(log)
    wide = _wide_team_features(log, targets)
    if include_team_identity:
        wide = _add_team_identity(wide, games)
    return wide
