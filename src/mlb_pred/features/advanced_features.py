"""Leakage-safe game-level feature combinations for MLB.

Every column created here contains ``_BEFORE``.  In this module the tag means
that the value is available before first pitch.  Historical outcomes are used
only through strict earlier-calendar-date aggregates, so two games in a
doubleheader cannot enter one another's features.

The leading labels make each family independently selectable:

* ``TEAM_MATCHUP_*`` -- offence-versus-opposing-pitching combinations.
* ``ODDS_MARKET_REGIME_*`` -- league-wide total and spread calibration.
* ``ODDS_INTERACTION_*`` -- combinations of current closing-market inputs.
* ``TEAM_HISTORICAL_MATCHUP_*`` -- prior meetings of the two clubs.
* ``GAME_CALENDAR_*`` / ``GAME_COMPETITION_*`` -- known game context.
* ``TEAM_EXTRA_INNINGS_*`` -- strictly prior extra-inning history.

No realised target-game score, margin, cover, or over/under result is returned.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from mlb_pred.config.constants import (
    POSTSEASON_GAME_TYPES,
    REGULAR_SEASON_GAME_TYPES,
    TEAM_ID_TO_NAME,
    TEAM_NAME_DIVISION_MAP,
    TEAM_NAME_LEAGUE_MAP,
)

PREGAME_TAG = "_BEFORE"

MARKET_GAME_WINDOWS = (15, 30, 75, 150)
MARKET_DAY_WINDOWS = (3, 7, 14)
MARKET_EWM_SPANS = (15, 30, 75)
MARKET_TAIL_THRESHOLDS = (2, 4, 6)
MARKET_REGIME_PAIRS = ((15, 75), (30, 150))

MATCHUP_WINDOWS = (5, 10, 20)
H2H_WINDOWS = (3, 5, 10)
EXTRA_INNINGS_WINDOWS = (5, 10)

ADVANCED_FAMILY_PREFIXES = (
    "TEAM_MATCHUP_",
    "ODDS_MARKET_REGIME_",
    "ODDS_INTERACTION_",
    "TEAM_HISTORICAL_MATCHUP_",
    "GAME_CALENDAR_",
    "GAME_COMPETITION_",
    "TEAM_COMPETITION_",
    "TEAM_EXTRA_INNINGS_",
)

_GAME_HISTORY_COLUMNS = {
    "game_pk",
    "season_year",
    "game_date",
    "game_type",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
    "extra_innings",
}


def _require_columns(
    frame: pd.DataFrame, required: set[str], *, frame_name: str
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    result = pd.to_numeric(numerator, errors="coerce") / den
    return result.replace([np.inf, -np.inf], np.nan)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"Required pregame source column is missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _pair_mean(first: pd.Series, second: pd.Series) -> pd.Series:
    return pd.concat([first, second], axis=1).mean(axis=1, skipna=True)


def _target_ids(
    closing_features: pd.DataFrame, target_game_ids: Iterable[str] | None
) -> set[str]:
    return (
        {str(value) for value in target_game_ids}
        if target_game_ids is not None
        else set(closing_features["GAME_ID"].astype(str))
    )


def _target_template(
    closing_features: pd.DataFrame, target_game_ids: Iterable[str] | None
) -> pd.DataFrame:
    _require_columns(closing_features, {"GAME_ID"}, frame_name="closing features")
    targets = _target_ids(closing_features, target_game_ids)
    template = closing_features.loc[
        closing_features["GAME_ID"].astype(str).isin(targets), ["GAME_ID"]
    ].copy()
    template["GAME_ID"] = template["GAME_ID"].astype(str)
    if template["GAME_ID"].duplicated().any():
        raise ValueError("closing features is not unique by GAME_ID.")
    return template.reset_index(drop=True)


def _rolling_column(metric: str, horizon: int | str, side: str) -> str:
    if horizon == "SEASON":
        suffix = "SEASON_BEFORE_AVG"
    else:
        suffix = f"LAST_ALL_{int(horizon)}_GAMES_BEFORE"
    return f"TEAM_ROLLING_{metric}_{suffix}_TEAM_{side}"


def build_matchup_style_features(
    team_rolling_features: pd.DataFrame,
) -> pd.DataFrame:
    """Cross each offence with the opposing pitching history.

    All inputs are already date-gated ``TEAM_ROLLING_*_BEFORE`` values.  The
    function intentionally cannot accept raw box-score fields.
    """
    _require_columns(
        team_rolling_features, {"GAME_ID"}, frame_name="team rolling features"
    )
    out = team_rolling_features[["GAME_ID"]].copy()
    new: dict[str, pd.Series] = {}

    horizons: tuple[int | str, ...] = (*MATCHUP_WINDOWS, "SEASON")
    for horizon in horizons:
        label = "SEASON" if horizon == "SEASON" else f"LAST_{horizon}_GAMES"
        home_scored = _numeric(
            team_rolling_features, _rolling_column("RUNS_SCORED", horizon, "HOME")
        )
        away_allowed = _numeric(
            team_rolling_features, _rolling_column("RUNS_ALLOWED", horizon, "AWAY")
        )
        away_scored = _numeric(
            team_rolling_features, _rolling_column("RUNS_SCORED", horizon, "AWAY")
        )
        home_allowed = _numeric(
            team_rolling_features, _rolling_column("RUNS_ALLOWED", horizon, "HOME")
        )
        expected_home = _pair_mean(home_scored, away_allowed)
        expected_away = _pair_mean(away_scored, home_allowed)
        new[f"TEAM_MATCHUP_HOME_RUN_EXPECTATION_{label}_BEFORE"] = expected_home
        new[f"TEAM_MATCHUP_AWAY_RUN_EXPECTATION_{label}_BEFORE"] = expected_away
        new[f"TEAM_MATCHUP_TOTAL_RUN_EXPECTATION_{label}_BEFORE"] = (
            expected_home + expected_away
        )

    for side in ("HOME", "AWAY"):
        recent = _numeric(
            team_rolling_features, _rolling_column("RUNS_SCORED", 5, side)
        )
        season = _numeric(
            team_rolling_features, _rolling_column("RUNS_SCORED", "SEASON", side)
        )
        std = _numeric(
            team_rolling_features,
            f"TEAM_ROLLING_RUNS_SCORED_SEASON_BEFORE_STD_TEAM_{side}",
        )
        new[f"TEAM_MATCHUP_RUNS_FORM_Z_{side}_LAST_5_VS_SEASON_BEFORE"] = _safe_ratio(
            recent - season, std
        )

    trend_parts = [
        _numeric(
            team_rolling_features,
            "TEAM_ROLLING_RUNS_SCORED_TREND_SLOPE_LAST_5_GAMES_BEFORE_TEAM_HOME",
        ),
        _numeric(
            team_rolling_features,
            "TEAM_ROLLING_RUNS_ALLOWED_TREND_SLOPE_LAST_5_GAMES_BEFORE_TEAM_AWAY",
        ),
        _numeric(
            team_rolling_features,
            "TEAM_ROLLING_RUNS_SCORED_TREND_SLOPE_LAST_5_GAMES_BEFORE_TEAM_AWAY",
        ),
        _numeric(
            team_rolling_features,
            "TEAM_ROLLING_RUNS_ALLOWED_TREND_SLOPE_LAST_5_GAMES_BEFORE_TEAM_HOME",
        ),
    ]
    new["TEAM_MATCHUP_TOTAL_RUN_EXPECTATION_TREND_LAST_5_GAMES_BEFORE"] = (
        sum(trend_parts) / 2.0
    )

    expected_pa: dict[str, pd.Series] = {}
    for side, opponent in (("HOME", "AWAY"), ("AWAY", "HOME")):
        expected_pa[side] = _pair_mean(
            _numeric(
                team_rolling_features,
                _rolling_column("PLATE_APPEARANCES", "SEASON", side),
            ),
            _numeric(
                team_rolling_features,
                _rolling_column("BATTERS_FACED", "SEASON", opponent),
            ),
        )
        new[f"TEAM_MATCHUP_EXPECTED_PLATE_APPEARANCES_{side}_SEASON_BEFORE"] = (
            expected_pa[side]
        )

    rate_pairs = {
        "K": ("OFFENSE_K_PCT", "PITCHING_K_PCT"),
        "BB": ("OFFENSE_BB_PCT", "PITCHING_BB_PCT"),
        "HR": ("OFFENSE_HR_PER_PA", "PITCHING_HR_PER_BF"),
        "BASERUNNER": (
            "OFFENSE_BASERUNNERS_PER_PA",
            "PITCHING_BASERUNNERS_PER_BF",
        ),
    }
    expected_rates: dict[tuple[str, str], pd.Series] = {}
    for label, (offense_metric, pitching_metric) in rate_pairs.items():
        for side, opponent in (("HOME", "AWAY"), ("AWAY", "HOME")):
            rate = _pair_mean(
                _numeric(
                    team_rolling_features,
                    _rolling_column(offense_metric, "SEASON", side),
                ),
                _numeric(
                    team_rolling_features,
                    _rolling_column(pitching_metric, "SEASON", opponent),
                ),
            )
            expected_rates[(label, side)] = rate
            new[f"TEAM_MATCHUP_EXPECTED_{label}_RATE_{side}_SEASON_BEFORE"] = rate

        new[f"TEAM_MATCHUP_EXPECTED_TOTAL_{label}_EVENTS_SEASON_BEFORE"] = (
            expected_pa["HOME"] * expected_rates[(label, "HOME")]
            + expected_pa["AWAY"] * expected_rates[(label, "AWAY")]
        )

    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def _distribution(
    frame: pd.DataFrame, columns: list[str]
) -> tuple[pd.Series, pd.Series]:
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    return values.mean(axis=1, skipna=True), values.std(axis=1, skipna=True, ddof=0)


def build_odds_interaction_features(
    closing_features: pd.DataFrame,
    *,
    target_game_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Combine current, pregame closing inputs without reading outcomes."""
    template = _target_template(closing_features, target_game_ids)
    source = template.merge(
        closing_features.assign(GAME_ID=closing_features["GAME_ID"].astype(str)),
        on="GAME_ID",
        how="left",
        validate="one_to_one",
    )
    new: dict[str, pd.Series] = {}

    market_prefixes = {
        "TOTAL": "ODDS_TOTAL_",
        "RUN_LINE": "ODDS_RUN_LINE_",
        "MONEY_LINE": "ODDS_MONEY_LINE_",
    }
    for label, prefix in market_prefixes.items():
        overround_columns = sorted(
            column
            for column in source
            if column.startswith(prefix)
            and column.endswith("_OVERROUND")
            and "_CONSENSUS_" not in column
        )
        if overround_columns:
            mean, std = _distribution(source, overround_columns)
            new[f"ODDS_INTERACTION_{label}_OVERROUND_MEAN_BEFORE"] = mean
            new[f"ODDS_INTERACTION_{label}_OVERROUND_STD_BEFORE"] = std

    total_line = _numeric(source, "ODDS_TOTAL_CONSENSUS_LINE_NORMALIZED_MEDIAN")
    total_prob = _numeric(source, "ODDS_TOTAL_CONSENSUS_FAIR_PROB_OVER_MEDIAN")
    total_skew = 2.0 * total_prob - 1.0
    new["ODDS_INTERACTION_TOTAL_PROBABILITY_SKEW_BEFORE"] = total_skew
    new["ODDS_INTERACTION_TOTAL_LINE_X_PROBABILITY_SKEW_BEFORE"] = (
        total_line * total_skew
    )

    total_vig = new.get("ODDS_INTERACTION_TOTAL_OVERROUND_MEAN_BEFORE")
    if total_vig is not None:
        total_std = _numeric(source, "ODDS_TOTAL_CONSENSUS_LINE_NORMALIZED_STD")
        new["ODDS_INTERACTION_TOTAL_DISAGREEMENT_X_OVERROUND_BEFORE"] = (
            total_std * total_vig
        )

    home_handicap = _numeric(
        source, "ODDS_RUN_LINE_CONSENSUS_HOME_HANDICAP_NORMALIZED_MEDIAN"
    )
    run_line_abs = home_handicap.abs()
    home_cover_prob = _numeric(
        source, "ODDS_RUN_LINE_CONSENSUS_FAIR_PROB_HOME_COVER_MEDIAN"
    )
    run_skew = 2.0 * home_cover_prob - 1.0
    new["ODDS_INTERACTION_RUN_LINE_FAVORITE_MAGNITUDE_BEFORE"] = run_line_abs
    new["ODDS_INTERACTION_RUN_LINE_HOME_IS_FAVORITE_BEFORE"] = home_handicap.lt(
        0
    ).astype("int8")
    new["ODDS_INTERACTION_RUN_LINE_COVER_PROBABILITY_SKEW_BEFORE"] = run_skew
    new["ODDS_INTERACTION_RUN_LINE_X_COVER_PROBABILITY_SKEW_BEFORE"] = (
        run_line_abs * run_skew
    )
    new["ODDS_INTERACTION_TOTAL_LINE_X_RUN_LINE_MAGNITUDE_BEFORE"] = (
        total_line * run_line_abs
    )

    run_vig = new.get("ODDS_INTERACTION_RUN_LINE_OVERROUND_MEAN_BEFORE")
    if run_vig is not None:
        run_std = _numeric(
            source, "ODDS_RUN_LINE_CONSENSUS_HOME_HANDICAP_NORMALIZED_STD"
        )
        new["ODDS_INTERACTION_RUN_LINE_DISAGREEMENT_X_OVERROUND_BEFORE"] = (
            run_std * run_vig
        )

    home_win_prob = _numeric(
        source, "ODDS_MONEY_LINE_CONSENSUS_FAIR_PROB_HOME_WIN_MEDIAN"
    )
    new["ODDS_INTERACTION_MONEY_LINE_FAVORITE_PROBABILITY_BEFORE"] = np.maximum(
        home_win_prob, 1.0 - home_win_prob
    )
    new["ODDS_INTERACTION_MONEY_LINE_WIN_PROBABILITY_GAP_BEFORE"] = (
        2.0 * home_win_prob - 1.0
    ).abs()

    implied_home = _numeric(source, "ODDS_DERIVED_IMPLIED_HOME_RUNS_NORMALIZED")
    implied_away = _numeric(source, "ODDS_DERIVED_IMPLIED_AWAY_RUNS_NORMALIZED")
    new["ODDS_INTERACTION_IMPLIED_RUNS_RATIO_HOME_DIV_AWAY_BEFORE"] = _safe_ratio(
        implied_home, implied_away
    )
    new["ODDS_INTERACTION_IMPLIED_RUNS_MAX_BEFORE"] = np.maximum(
        implied_home, implied_away
    )
    new["ODDS_INTERACTION_IMPLIED_RUNS_MIN_BEFORE"] = np.minimum(
        implied_home, implied_away
    )
    new["ODDS_INTERACTION_IMPLIED_RUNS_GAP_BEFORE"] = (
        implied_home - implied_away
    ).abs()
    return pd.concat([template, pd.DataFrame(new, index=template.index)], axis=1)


def _game_history_frame(games: pd.DataFrame) -> pd.DataFrame:
    _require_columns(games, _GAME_HISTORY_COLUMNS, frame_name="games")
    if games["game_pk"].astype(str).duplicated().any():
        raise ValueError("games is not unique by game_pk.")
    out = games.copy()
    out["game_pk"] = out["game_pk"].astype(str)
    out["home_team_id"] = out["home_team_id"].astype(str)
    out["away_team_id"] = out["away_team_id"].astype(str)
    out["season_year"] = pd.to_numeric(out["season_year"], errors="raise").astype(int)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="raise").dt.normalize()
    if "first_pitch_utc" in out:
        out["first_pitch_utc"] = pd.to_datetime(
            out["first_pitch_utc"], utc=True, errors="coerce"
        )
    else:
        out["first_pitch_utc"] = pd.NaT
    out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce")
    out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce")
    complete = out[["home_score", "away_score"]].notna().all(axis=1)
    if "is_final" in out:
        complete &= out["is_final"].fillna(False).astype(bool)
    out["__is_complete"] = complete
    return out.sort_values(
        ["game_date", "first_pitch_utc", "game_pk"], kind="mergesort"
    ).reset_index(drop=True)


def build_historical_matchup_features(
    games: pd.DataFrame,
    closing_features: pd.DataFrame,
    *,
    target_game_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Summarise prior meetings, treating the team pair as unordered."""
    template = _target_template(closing_features, target_game_ids)
    targets = set(template["GAME_ID"])
    history = _game_history_frame(games)
    missing_targets = sorted(targets.difference(history["game_pk"]))
    if missing_targets:
        raise ValueError(
            f"games is missing target GAME_ID values: {missing_targets[:5]}"
        )

    pair_history: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    feature_rows: list[dict[str, object]] = []
    for _, date_games in history.groupby("game_date", sort=False):
        for row in date_games.itertuples(index=False):
            if row.game_pk not in targets:
                continue
            pair = tuple(sorted((row.home_team_id, row.away_team_id)))
            prior = pair_history[pair]
            totals = np.asarray(
                [float(item["total_runs"]) for item in prior], dtype="float64"
            )
            margins = np.asarray(
                [
                    (
                        float(item["team_a_margin"])
                        if row.home_team_id == pair[0]
                        else -float(item["team_a_margin"])
                    )
                    for item in prior
                ],
                dtype="float64",
            )
            values: dict[str, object] = {
                "GAME_ID": row.game_pk,
                "TEAM_HISTORICAL_MATCHUP_GAMES_COUNT_BEFORE": len(prior),
                "TEAM_HISTORICAL_MATCHUP_HAS_HISTORY_BEFORE": int(bool(prior)),
                "TEAM_HISTORICAL_MATCHUP_TOTAL_RUNS_LAST_GAME_BEFORE": (
                    totals[-1] if len(totals) else np.nan
                ),
                "TEAM_HISTORICAL_MATCHUP_DAYS_SINCE_LAST_GAME_BEFORE": (
                    (row.game_date - prior[-1]["game_date"]).days if prior else np.nan
                ),
            }
            for window in H2H_WINDOWS:
                recent_totals = totals[-window:]
                values[
                    f"TEAM_HISTORICAL_MATCHUP_TOTAL_RUNS_LAST_{window}_GAMES_AVG_BEFORE"
                ] = (float(np.mean(recent_totals)) if len(recent_totals) else np.nan)
            recent_five_totals = totals[-5:]
            recent_five_margins = margins[-5:]
            values["TEAM_HISTORICAL_MATCHUP_TOTAL_RUNS_LAST_5_GAMES_STD_BEFORE"] = (
                float(np.std(recent_five_totals, ddof=0))
                if len(recent_five_totals)
                else np.nan
            )
            values[
                "TEAM_HISTORICAL_MATCHUP_CURRENT_HOME_RUN_MARGIN_LAST_5_GAMES_AVG_BEFORE"
            ] = (
                float(np.mean(recent_five_margins))
                if len(recent_five_margins)
                else np.nan
            )
            values[
                "TEAM_HISTORICAL_MATCHUP_CURRENT_HOME_WIN_RATIO_LAST_5_GAMES_BEFORE"
            ] = (
                float(np.mean(recent_five_margins > 0))
                if len(recent_five_margins)
                else np.nan
            )
            feature_rows.append(values)

        # Apply outcomes only after every target on this date has received features.
        for row in date_games.loc[date_games["__is_complete"]].itertuples(index=False):
            pair = tuple(sorted((row.home_team_id, row.away_team_id)))
            home_margin = float(row.home_score - row.away_score)
            pair_history[pair].append(
                {
                    "game_date": row.game_date,
                    "total_runs": float(row.home_score + row.away_score),
                    "team_a_margin": (
                        home_margin if row.home_team_id == pair[0] else -home_margin
                    ),
                }
            )

    values = pd.DataFrame(feature_rows)
    return template.merge(values, on="GAME_ID", how="left", validate="one_to_one")


def build_extra_innings_history_features(
    games: pd.DataFrame,
    closing_features: pd.DataFrame,
    *,
    target_game_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build NBA-overtime-shaped history using completed extra-inning games."""
    template = _target_template(closing_features, target_game_ids)
    targets = set(template["GAME_ID"])
    history = _game_history_frame(games)
    team_history: dict[str, list[float]] = defaultdict(list)
    season_history: dict[tuple[str, int], list[float]] = defaultdict(list)
    rows: list[dict[str, object]] = []

    base_names = (
        "TEAM_EXTRA_INNINGS_LAST_GAME_BEFORE",
        "TEAM_EXTRA_INNINGS_FREQUENCY_LAST_5_GAMES_BEFORE",
        "TEAM_EXTRA_INNINGS_FREQUENCY_LAST_10_GAMES_BEFORE",
        "TEAM_EXTRA_INNINGS_FREQUENCY_SEASON_BEFORE",
        "TEAM_EXTRA_INNINGS_GAMES_SEASON_BEFORE",
    )

    def team_values(team_id: str, season_year: int) -> dict[str, float]:
        prior = team_history[team_id]
        season = season_history[(team_id, season_year)]
        return {
            base_names[0]: prior[-1] if prior else 0.0,
            base_names[1]: float(np.mean(prior[-5:])) if prior else 0.0,
            base_names[2]: float(np.mean(prior[-10:])) if prior else 0.0,
            base_names[3]: float(np.mean(season)) if season else 0.0,
            base_names[4]: float(len(season)),
        }

    for _, date_games in history.groupby("game_date", sort=False):
        for row in date_games.itertuples(index=False):
            if row.game_pk not in targets:
                continue
            home = team_values(row.home_team_id, row.season_year)
            away = team_values(row.away_team_id, row.season_year)
            values: dict[str, object] = {"GAME_ID": row.game_pk}
            for base in base_names:
                values[f"{base}_TEAM_HOME"] = home[base]
                values[f"{base}_TEAM_AWAY"] = away[base]
                diff = f"{base.removesuffix('_BEFORE')}_DIFF_BEFORE"
                values[diff] = home[base] - away[base]
            rows.append(values)

        for row in date_games.loc[date_games["__is_complete"]].itertuples(index=False):
            value = float(bool(row.extra_innings))
            for team_id in (row.home_team_id, row.away_team_id):
                team_history[team_id].append(value)
                season_history[(team_id, row.season_year)].append(value)

    values = pd.DataFrame(rows)
    return template.merge(values, on="GAME_ID", how="left", validate="one_to_one")


def build_calendar_competition_features(
    games: pd.DataFrame,
    closing_features: pd.DataFrame,
    *,
    target_game_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Add calendar, league/division, postseason, and prior-postseason context."""
    template = _target_template(closing_features, target_game_ids)
    history = _game_history_frame(games)
    targets = history.loc[history["game_pk"].isin(template["GAME_ID"])].copy()
    if len(targets) != len(template):
        raise ValueError("games does not contain every requested calendar target.")

    unknown_ids = sorted(
        (set(targets["home_team_id"]) | set(targets["away_team_id"])).difference(
            TEAM_ID_TO_NAME
        )
    )
    if unknown_ids:
        raise ValueError(f"Cannot map competition for unknown team ids: {unknown_ids}")
    home_names = targets["home_team_id"].map(TEAM_ID_TO_NAME)
    away_names = targets["away_team_id"].map(TEAM_ID_TO_NAME)
    home_leagues = home_names.map(TEAM_NAME_LEAGUE_MAP)
    away_leagues = away_names.map(TEAM_NAME_LEAGUE_MAP)
    home_divisions = home_names.map(TEAM_NAME_DIVISION_MAP)
    away_divisions = away_names.map(TEAM_NAME_DIVISION_MAP)
    dates = targets["game_date"]

    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=dates.min(), end=dates.max())
    features: dict[str, pd.Series] = {
        "GAME_CALENDAR_MONTH_BEFORE": dates.dt.month.astype("int8"),
        "GAME_CALENDAR_IS_WEEKEND_BEFORE": dates.dt.weekday.ge(5).astype("int8"),
        "GAME_CALENDAR_IS_US_FEDERAL_HOLIDAY_BEFORE": (
            dates.isin(holidays).astype("int8")
        ),
        "GAME_COMPETITION_SAME_LEAGUE_BEFORE": (
            home_leagues.eq(away_leagues).astype("int8")
        ),
        "GAME_COMPETITION_SAME_DIVISION_BEFORE": (
            home_divisions.eq(away_divisions).astype("int8")
        ),
        "GAME_COMPETITION_IS_INTERLEAGUE_BEFORE": (
            home_leagues.ne(away_leagues).astype("int8")
        ),
        "GAME_COMPETITION_HOME_TEAM_IS_AL_BEFORE": (
            home_leagues.eq("AL").astype("int8")
        ),
        "GAME_COMPETITION_AWAY_TEAM_IS_AL_BEFORE": (
            away_leagues.eq("AL").astype("int8")
        ),
        "GAME_COMPETITION_IS_REGULAR_SEASON_BEFORE": (
            targets["game_type"].isin(REGULAR_SEASON_GAME_TYPES).astype("int8")
        ),
        "GAME_COMPETITION_IS_POSTSEASON_BEFORE": (
            targets["game_type"].isin(POSTSEASON_GAME_TYPES).astype("int8")
        ),
    }
    for game_type, label in (
        ("F", "WILD_CARD"),
        ("D", "DIVISION_SERIES"),
        ("L", "LEAGUE_CHAMPIONSHIP"),
        ("W", "WORLD_SERIES"),
    ):
        features[f"GAME_COMPETITION_IS_{label}_BEFORE"] = (
            targets["game_type"].eq(game_type).astype("int8")
        )

    postseason = history.loc[
        history["game_type"].isin(POSTSEASON_GAME_TYPES),
        ["season_year", "home_team_id", "away_team_id"],
    ]
    appearances = pd.concat(
        [
            postseason[["season_year", "home_team_id"]].rename(
                columns={"home_team_id": "team_id"}
            ),
            postseason[["season_year", "away_team_id"]].rename(
                columns={"away_team_id": "team_id"}
            ),
        ],
        ignore_index=True,
    )
    prior_counts = appearances.groupby(["season_year", "team_id"]).size()

    home_counts = [
        float(prior_counts.get((season - 1, team), 0))
        for season, team in zip(
            targets["season_year"], targets["home_team_id"], strict=True
        )
    ]
    away_counts = [
        float(prior_counts.get((season - 1, team), 0))
        for season, team in zip(
            targets["season_year"], targets["away_team_id"], strict=True
        )
    ]
    features["TEAM_COMPETITION_POSTSEASON_GAMES_LAST_SEASON_BEFORE_TEAM_HOME"] = (
        pd.Series(home_counts, index=targets.index)
    )
    features["TEAM_COMPETITION_POSTSEASON_GAMES_LAST_SEASON_BEFORE_TEAM_AWAY"] = (
        pd.Series(away_counts, index=targets.index)
    )
    features["TEAM_COMPETITION_POSTSEASON_GAMES_LAST_SEASON_DIFF_BEFORE"] = pd.Series(
        home_counts, index=targets.index
    ) - pd.Series(away_counts, index=targets.index)

    result = pd.concat(
        [
            targets[["game_pk"]].rename(columns={"game_pk": "GAME_ID"}),
            pd.DataFrame(features, index=targets.index),
        ],
        axis=1,
    )
    return template.merge(result, on="GAME_ID", how="left", validate="one_to_one")


def _strict_prior_aggregate(
    series: pd.Series,
    dates: pd.Series,
    window: int,
    function: str,
    *,
    calendar_days: bool = False,
) -> pd.Series:
    """Aggregate rows from earlier dates and assign one value to a date group."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    date_days = pd.to_datetime(dates).to_numpy(dtype="datetime64[D]")
    result = np.full(len(values), np.nan, dtype="float64")
    if not len(values):
        return pd.Series(result, index=series.index)
    starts = np.flatnonzero(np.r_[True, date_days[1:] != date_days[:-1]])
    ends = np.r_[starts[1:], len(values)]
    day_numbers = date_days.astype("int64")
    for start, end in zip(starts, ends, strict=True):
        left = (
            int(np.searchsorted(day_numbers, day_numbers[start] - window, side="left"))
            if calendar_days
            else max(0, start - window)
        )
        prior = values[left:start]
        valid = prior[~np.isnan(prior)]
        if function == "count":
            value = float(len(valid))
        elif not len(valid):
            value = np.nan
        elif function == "mean":
            value = float(np.mean(valid))
        elif function == "std":
            value = float(np.std(valid, ddof=0))
        elif function == "median":
            value = float(np.median(valid))
        else:
            raise ValueError(f"Unsupported aggregate function: {function}")
        result[start:end] = value
    return pd.Series(result, index=series.index)


def _strict_prior_ewm(
    series: pd.Series, dates: pd.Series, span: int, *, std: bool = False
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    ewm = numeric.ewm(span=span, min_periods=1)
    full = ewm.std() if std else ewm.mean()
    date_days = pd.to_datetime(dates).to_numpy(dtype="datetime64[D]")
    result = np.full(len(numeric), np.nan, dtype="float64")
    if not len(numeric):
        return pd.Series(result, index=series.index)
    starts = np.flatnonzero(np.r_[True, date_days[1:] != date_days[:-1]])
    ends = np.r_[starts[1:], len(numeric)]
    for start, end in zip(starts, ends, strict=True):
        result[start:end] = np.nan if start == 0 else float(full.iloc[start - 1])
    return pd.Series(result, index=series.index)


def _market_regime_columns(
    *,
    market: str,
    error: pd.Series,
    line_level: pd.Series,
    actual_level: pd.Series,
    positive: pd.Series,
    negative: pd.Series,
    push: pd.Series,
    dates: pd.Series,
) -> dict[str, pd.Series]:
    prefix = f"ODDS_MARKET_REGIME_{market}"
    new: dict[str, pd.Series] = {}

    def key(metric: str, window: str) -> str:
        return f"{prefix}_{metric}_{window}_BEFORE"

    abs_error = error.abs()
    metrics = {
        "BIAS": error,
        "MAE": abs_error,
        "ERROR_STD": error,
        "MEDIAN_ERROR": error,
        "SAMPLE_COUNT": error,
        "LINE_AVG": line_level,
        "ACTUAL_AVG": actual_level,
        ("OVER_RATE" if market == "TOTAL" else "HOME_COVER_RATE"): positive,
        ("UNDER_RATE" if market == "TOTAL" else "AWAY_COVER_RATE"): negative,
        "PUSH_RATE": push,
    }
    functions = {
        "ERROR_STD": "std",
        "MEDIAN_ERROR": "median",
        "SAMPLE_COUNT": "count",
    }
    for window in MARKET_GAME_WINDOWS:
        label = f"{window}G"
        for metric, values in metrics.items():
            new[key(metric, label)] = _strict_prior_aggregate(
                values, dates, window, functions.get(metric, "mean")
            )
    for window in MARKET_DAY_WINDOWS:
        label = f"{window}D"
        for metric, values in metrics.items():
            new[key(metric, label)] = _strict_prior_aggregate(
                values,
                dates,
                window,
                functions.get(metric, "mean"),
                calendar_days=True,
            )

    for span in MARKET_EWM_SPANS:
        new[key("BIAS_EWM", f"{span}G")] = _strict_prior_ewm(error, dates, span)
        new[key("MAE_EWM", f"{span}G")] = _strict_prior_ewm(abs_error, dates, span)
        new[key("ERROR_STD_EWM", f"{span}G")] = _strict_prior_ewm(
            error, dates, span, std=True
        )

    for threshold in MARKET_TAIL_THRESHOLDS:
        valid = error.notna()
        tail = pd.Series(
            np.where(valid, abs_error.gt(threshold).astype(float), np.nan),
            index=error.index,
        )
        for window in (15, 30, 75):
            new[key(f"TAIL_GT_{threshold}", f"{window}G")] = _strict_prior_aggregate(
                tail, dates, window, "mean"
            )

    epsilon = 1e-6
    for short, long in MARKET_REGIME_PAIRS:
        short_label = f"{short}G"
        long_label = f"{long}G"
        bias_short = new[key("BIAS", short_label)]
        bias_long = new[key("BIAS", long_label)]
        std_long = new[key("ERROR_STD", long_label)]
        mae_short = new[key("MAE", short_label)]
        mae_long = new[key("MAE", long_label)]
        std_short = new[key("ERROR_STD", short_label)]
        new[key("BIAS_NORM_DIFF", f"{short}G_VS_{long}G")] = (
            bias_short - bias_long
        ) / (std_long + epsilon)
        new[key("MAE_RATIO", f"{short}G_VS_{long}G")] = mae_short / (mae_long + epsilon)
        new[key("STD_RATIO", f"{short}G_VS_{long}G")] = std_short / (std_long + epsilon)
        for metric in ("BIAS", "MAE", "LINE_AVG", "ACTUAL_AVG"):
            new[key(f"{metric}_DIFF", f"{short}G_VS_{long}G")] = (
                new[key(metric, short_label)] - new[key(metric, long_label)]
            )
    return new


def build_global_market_regime_features(
    games: pd.DataFrame,
    closing_features: pd.DataFrame,
    *,
    target_game_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build total and spread calibration regimes from strictly earlier dates."""
    required_closing = {
        "GAME_ID",
        "GAME_DATE",
        "GAME_FIRST_PITCH_UTC",
        "ODDS_TOTAL_CONSENSUS_LINE_NORMALIZED_MEDIAN",
        "ODDS_RUN_LINE_CONSENSUS_HOME_HANDICAP_NORMALIZED_MEDIAN",
    }
    _require_columns(closing_features, required_closing, frame_name="closing features")
    _require_columns(
        games,
        {"game_pk", "home_score", "away_score", "run_line_margin"},
        frame_name="games",
    )
    if closing_features["GAME_ID"].astype(str).duplicated().any():
        raise ValueError("closing features is not unique by GAME_ID.")
    results = games[["game_pk", "home_score", "away_score", "run_line_margin"]].copy()
    results["game_pk"] = results["game_pk"].astype(str)
    if results["game_pk"].duplicated().any():
        raise ValueError("games is not unique by game_pk.")
    source = closing_features.copy()
    source["GAME_ID"] = source["GAME_ID"].astype(str)
    source = source.merge(
        results.rename(columns={"game_pk": "GAME_ID"}),
        on="GAME_ID",
        how="left",
        validate="one_to_one",
    )
    source["GAME_DATE"] = pd.to_datetime(source["GAME_DATE"], errors="raise")
    source["GAME_FIRST_PITCH_UTC"] = pd.to_datetime(
        source["GAME_FIRST_PITCH_UTC"], utc=True, errors="raise"
    )
    source = source.sort_values(
        ["GAME_DATE", "GAME_FIRST_PITCH_UTC", "GAME_ID"], kind="mergesort"
    ).reset_index(drop=True)
    dates = source["GAME_DATE"]

    total_line = _numeric(source, "ODDS_TOTAL_CONSENSUS_LINE_NORMALIZED_MEDIAN")
    actual_total = pd.to_numeric(source["home_score"], errors="coerce") + pd.to_numeric(
        source["away_score"], errors="coerce"
    )
    total_error = actual_total - total_line
    total_valid = total_error.notna()
    total_over = pd.Series(
        np.where(total_valid, total_error.gt(0).astype(float), np.nan),
        index=source.index,
    )
    total_under = pd.Series(
        np.where(total_valid, total_error.lt(0).astype(float), np.nan),
        index=source.index,
    )
    total_push = pd.Series(
        np.where(total_valid, total_error.eq(0).astype(float), np.nan),
        index=source.index,
    )

    home_handicap = _numeric(
        source, "ODDS_RUN_LINE_CONSENSUS_HOME_HANDICAP_NORMALIZED_MEDIAN"
    )
    actual_margin = pd.to_numeric(source["run_line_margin"], errors="coerce")
    spread_error = actual_margin + home_handicap
    spread_valid = spread_error.notna()
    home_cover = pd.Series(
        np.where(spread_valid, spread_error.gt(0).astype(float), np.nan),
        index=source.index,
    )
    away_cover = pd.Series(
        np.where(spread_valid, spread_error.lt(0).astype(float), np.nan),
        index=source.index,
    )
    spread_push = pd.Series(
        np.where(spread_valid, spread_error.eq(0).astype(float), np.nan),
        index=source.index,
    )

    new = _market_regime_columns(
        market="TOTAL",
        error=total_error,
        line_level=total_line,
        actual_level=actual_total,
        positive=total_over,
        negative=total_under,
        push=total_push,
        dates=dates,
    )
    new.update(
        _market_regime_columns(
            market="SPREAD",
            error=spread_error,
            line_level=home_handicap.abs(),
            actual_level=actual_margin.abs(),
            positive=home_cover,
            negative=away_cover,
            push=spread_push,
            dates=dates,
        )
    )
    all_features = pd.concat(
        [source[["GAME_ID"]], pd.DataFrame(new, index=source.index)], axis=1
    )
    template = _target_template(closing_features, target_game_ids)
    return template.merge(all_features, on="GAME_ID", how="left", validate="one_to_one")


def assert_advanced_feature_contract(features: pd.DataFrame) -> None:
    """Enforce both the family labels and the pregame availability tag."""
    if "GAME_ID" not in features:
        raise ValueError("Advanced features must contain GAME_ID.")
    if features["GAME_ID"].astype(str).duplicated().any():
        raise ValueError("Advanced features is not unique by GAME_ID.")
    invalid_prefix = [
        column
        for column in features
        if column != "GAME_ID" and not column.startswith(ADVANCED_FAMILY_PREFIXES)
    ]
    if invalid_prefix:
        raise ValueError(f"Unlabelled advanced feature columns: {invalid_prefix}")
    invalid_time = [
        column
        for column in features
        if column != "GAME_ID" and PREGAME_TAG not in column
    ]
    if invalid_time:
        raise ValueError(f"Advanced features missing {PREGAME_TAG}: {invalid_time}")
    forbidden = ("HOME_SCORE", "AWAY_SCORE", "TOTAL_RUNS_RESULT", "RUN_MARGIN_RESULT")
    leaking = [
        column
        for column in features
        if any(fragment in column for fragment in forbidden)
    ]
    if leaking:
        raise ValueError(f"Outcome columns reached advanced features: {leaking}")


def build_advanced_pregame_features(
    games: pd.DataFrame,
    closing_features: pd.DataFrame,
    team_rolling_features: pd.DataFrame,
    *,
    target_game_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build and merge every game-level feature family in this module."""
    template = _target_template(closing_features, target_game_ids)
    targets = template["GAME_ID"]
    builders = (
        build_matchup_style_features(team_rolling_features),
        build_global_market_regime_features(
            games, closing_features, target_game_ids=targets
        ),
        build_odds_interaction_features(closing_features, target_game_ids=targets),
        build_historical_matchup_features(
            games, closing_features, target_game_ids=targets
        ),
        build_calendar_competition_features(
            games, closing_features, target_game_ids=targets
        ),
        build_extra_innings_history_features(
            games, closing_features, target_game_ids=targets
        ),
    )
    output = template
    for family in builders:
        overlap = sorted(set(output).intersection(family).difference({"GAME_ID"}))
        if overlap:
            raise ValueError(f"Advanced feature families overlap: {overlap}")
        output = output.merge(family, on="GAME_ID", how="left", validate="one_to_one")
    assert_advanced_feature_contract(output)
    return output
