"""NBA-shaped rolling team and market features adapted to MLB.

The accumulation grain is one team-game row. Every aggregate excludes the
current game before it rolls, season statistics fall back to the previous
regular season, and the final pivot creates one home and one away view per
game. MLB doubleheaders need a stricter rule than NBA: games on the same
calendar date cannot enter one another's history because this store does not
carry a trustworthy result-publication timestamp.

Only columns beginning with ``TEAM_*`` or ``ODDS_*`` leave this module. Raw
box-score outcomes are temporary history sources and are never returned.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mlb_pred.config.settings import PROJECT_ROOT

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "features" / "pregame"

MLB_WINDOWS = (1, 2, 3, 5, 10, 20, 30)
# Preserve NBA's canonical last-5, short 1/2/3, last-10, WMA-5, and 5/10
# trend family. MLB also gets longer 20/30-game context.
BASE_WINDOW = 5
LONG_WINDOWS = (10, 20, 30)
TREND_WINDOWS = (5, 10)
REGULAR_SEASON_GAME_TYPE = "R"
NO_HISTORY_VALUE = 0.0

# NBA rolls a curated set rather than every raw box-score field. These are the
# MLB equivalents: outcomes/volume plus stable offensive and pitching rates.
TEAM_SOURCE_COLUMNS: dict[str, str] = {
    "runs_scored": "RUNS_SCORED",
    "runs_allowed": "RUNS_ALLOWED",
    "run_margin": "RUN_MARGIN",
    "total_runs": "TOTAL_RUNS",
    "win": "WIN_RATE",
    "plate_appearances": "PLATE_APPEARANCES",
    "at_bats": "AT_BATS",
    "outs_recorded": "OUTS_RECORDED",
    "batters_faced": "BATTERS_FACED",
    "pitches_thrown": "PITCHES_THROWN",
    "baserunners_allowed": "BASERUNNERS_ALLOWED",
    "left_on_base": "LEFT_ON_BASE",
    "hits": "HITS",
    "doubles": "DOUBLES",
    "triples": "TRIPLES",
    "home_runs": "HOME_RUNS",
    "total_bases": "TOTAL_BASES",
    "walks": "WALKS",
    "strikeouts": "STRIKEOUTS",
    "hit_by_pitch": "HIT_BY_PITCH",
    "stolen_bases": "STOLEN_BASES",
    "grounded_into_double_play": "GROUNDED_INTO_DOUBLE_PLAY",
    "hits_allowed": "HITS_ALLOWED",
    "home_runs_allowed": "HOME_RUNS_ALLOWED",
    "walks_allowed": "WALKS_ALLOWED",
    "strikeouts_thrown": "STRIKEOUTS_THROWN",
    "hit_batsmen": "HIT_BATSMEN",
    "earned_runs": "EARNED_RUNS",
    "wild_pitches": "WILD_PITCHES",
    "errors": "FIELDING_ERRORS",
    "batters_used": "BATTERS_USED",
    "pitchers_used": "PITCHERS_USED",
    "bullpen_size": "BULLPEN_SIZE",
    "bench_size": "BENCH_SIZE",
    "obp": "OFFENSE_OBP",
    "slg": "OFFENSE_SLG",
    "iso": "OFFENSE_ISO",
    "babip": "OFFENSE_BABIP",
    "runs_per_pa": "OFFENSE_RUNS_PER_PA",
    "k_pct": "OFFENSE_K_PCT",
    "bb_pct": "OFFENSE_BB_PCT",
    "hr_per_pa": "OFFENSE_HR_PER_PA",
    "ground_out_air_out_ratio": "OFFENSE_GROUND_OUT_AIR_OUT_RATIO",
    "k_pct_allowed": "PITCHING_K_PCT",
    "bb_pct_allowed": "PITCHING_BB_PCT",
    "hr_per_bf_allowed": "PITCHING_HR_PER_BF",
    "whip": "PITCHING_WHIP",
    "pitches_per_batter_faced": "PITCHING_PITCHES_PER_BF",
    "strike_pct": "PITCHING_STRIKE_PCT",
}

DERIVED_RATE_COLUMNS: dict[str, str] = {
    "__offense_hits_per_pa": "OFFENSE_HITS_PER_PA",
    "__offense_extra_base_hits_per_pa": "OFFENSE_EXTRA_BASE_HITS_PER_PA",
    "__offense_total_bases_per_pa": "OFFENSE_TOTAL_BASES_PER_PA",
    "__offense_baserunners_per_pa": "OFFENSE_BASERUNNERS_PER_PA",
    "__offense_stolen_base_attempts_per_pa": "OFFENSE_SB_ATTEMPTS_PER_PA",
    "__offense_ground_out_rate": "OFFENSE_GROUND_OUT_RATE",
    "__pitching_era_per_9": "PITCHING_ERA_PER_9",
    "__pitching_baserunners_per_bf": "PITCHING_BASERUNNERS_PER_BF",
    "__pitching_pitches_per_out": "PITCHING_PITCHES_PER_OUT",
    "__pitching_ground_out_rate": "PITCHING_GROUND_OUT_RATE",
    "__fielding_errors_per_out": "FIELDING_ERRORS_PER_OUT",
}

DERIVED_RATE_SOURCE_COLUMNS = {
    "air_outs",
    "caught_stealing",
    "ground_outs",
    "pitching_air_outs",
    "pitching_ground_outs",
}

# Like NBA's short-list, these get the complete 1/2/3/5/10/20/30 family,
# weighted means and trends. Every other source still gets last-10, split,
# season mean/std, observation count, and recent/season ratio.
EXTENDED_TEAM_LABELS = {
    "RUNS_SCORED",
    "RUNS_ALLOWED",
    "RUN_MARGIN",
    "TOTAL_RUNS",
    "PLATE_APPEARANCES",
    "PITCHES_THROWN",
    "OFFENSE_OBP",
    "OFFENSE_SLG",
    "OFFENSE_ISO",
    "OFFENSE_RUNS_PER_PA",
    "OFFENSE_K_PCT",
    "OFFENSE_BB_PCT",
    "OFFENSE_HR_PER_PA",
    "PITCHING_K_PCT",
    "PITCHING_BB_PCT",
    "PITCHING_HR_PER_BF",
    "PITCHING_WHIP",
    "PITCHING_PITCHES_PER_BF",
    "PITCHING_STRIKE_PCT",
    "OFFENSE_HITS_PER_PA",
    "OFFENSE_EXTRA_BASE_HITS_PER_PA",
    "OFFENSE_TOTAL_BASES_PER_PA",
    "PITCHING_ERA_PER_9",
    "PITCHING_BASERUNNERS_PER_BF",
}

_TEAM_REQUIRED = {
    "game_pk",
    "team_id",
    "season_year",
    "game_date",
    "first_pitch_utc",
    "game_type",
    "home",
    "opponent_team_id",
}
_GAME_REQUIRED = {
    "game_pk",
    "season_year",
    "game_date",
    "first_pitch_utc",
    "game_type",
    "home_team_id",
    "away_team_id",
}


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    fill_no_history: bool = True,
) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    result = pd.to_numeric(numerator, errors="coerce") / den
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(NO_HISTORY_VALUE) if fill_no_history else result


def _ratio(frame: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    return _safe_ratio(frame[numerator], frame[denominator], fill_no_history=False)


def _add_derived_rates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["__offense_hits_per_pa"] = _ratio(out, "hits", "plate_appearances")
    out["__offense_extra_base_hits_per_pa"] = _safe_ratio(
        out["doubles"] + out["triples"] + out["home_runs"],
        out["plate_appearances"],
        fill_no_history=False,
    )
    out["__offense_total_bases_per_pa"] = _ratio(
        out, "total_bases", "plate_appearances"
    )
    out["__offense_baserunners_per_pa"] = _safe_ratio(
        out["hits"] + out["walks"] + out["hit_by_pitch"],
        out["plate_appearances"],
        fill_no_history=False,
    )
    out["__offense_stolen_base_attempts_per_pa"] = _safe_ratio(
        out["stolen_bases"] + out["caught_stealing"],
        out["plate_appearances"],
        fill_no_history=False,
    )
    out["__offense_ground_out_rate"] = _safe_ratio(
        out["ground_outs"],
        out["ground_outs"] + out["air_outs"],
        fill_no_history=False,
    )
    out["__pitching_era_per_9"] = _safe_ratio(
        out["earned_runs"] * 27.0,
        out["outs_recorded"],
        fill_no_history=False,
    )
    out["__pitching_baserunners_per_bf"] = _ratio(
        out, "baserunners_allowed", "batters_faced"
    )
    out["__pitching_pitches_per_out"] = _ratio(out, "pitches_thrown", "outs_recorded")
    out["__pitching_ground_out_rate"] = _safe_ratio(
        out["pitching_ground_outs"],
        out["pitching_ground_outs"] + out["pitching_air_outs"],
        fill_no_history=False,
    )
    out["__fielding_errors_per_out"] = _ratio(out, "errors", "outs_recorded")
    return out


def _date_gate(
    frame: pd.DataFrame, values: pd.Series, *, group_keys: list[str]
) -> pd.Series:
    """Give every same-date group the value held before that date began."""
    if frame.empty:
        return pd.Series(index=frame.index, dtype="float64")
    gate_keys = [*group_keys, "game_date"]
    codes = frame.groupby(gate_keys, sort=False, dropna=False).ngroup().to_numpy()
    first_positions = np.flatnonzero(~pd.Series(codes).duplicated().to_numpy())
    first_by_code = np.empty(int(codes.max()) + 1, dtype=np.int64)
    first_by_code[codes[first_positions]] = first_positions
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    return pd.Series(numeric[first_by_code[codes]], index=frame.index)


def _rolling(
    frame: pd.DataFrame,
    source: str,
    *,
    group_keys: list[str],
    window: int,
    function: str,
) -> pd.Series:
    numeric = pd.to_numeric(frame[source], errors="coerce").astype("float64")

    def prior_valid_observations(series: pd.Series) -> pd.Series:
        """Roll completed observations; scheduled null rows never consume a slot."""
        array = series.to_numpy(dtype="float64")
        valid_mask = np.isfinite(array)
        valid_values = array[valid_mask]
        prior_counts = np.cumsum(valid_mask, dtype=np.int64) - valid_mask

        if function == "count":
            after_each = np.minimum(
                np.arange(1, len(valid_values) + 1, dtype="float64"), window
            )
        elif function == "mean":
            after_each = (
                pd.Series(valid_values, dtype="float64")
                .rolling(window, min_periods=1)
                .mean()
                .to_numpy()
            )
        elif function == "wma":

            def weighted(values: np.ndarray) -> float:
                weights = np.arange(1, len(values) + 1, dtype="float64")
                return float(np.dot(values, weights) / weights.sum())

            after_each = (
                pd.Series(valid_values, dtype="float64")
                .rolling(window, min_periods=1)
                .apply(weighted, raw=True)
                .to_numpy()
            )
        elif function == "slope":

            def slope(values: np.ndarray) -> float:
                if len(values) < 2:
                    return np.nan
                x = np.arange(len(values), dtype="float64")
                x -= x.mean()
                return float(np.dot(x, values - values.mean()) / np.dot(x, x))

            after_each = (
                pd.Series(valid_values, dtype="float64")
                .rolling(window, min_periods=2)
                .apply(slope, raw=True)
                .to_numpy()
            )
        else:
            raise ValueError(f"Unsupported rolling function: {function}")

        lookup = np.concatenate(([np.nan], after_each))
        return pd.Series(lookup[prior_counts], index=series.index, dtype="float64")

    values = numeric.groupby(
        [frame[key] for key in group_keys], sort=False, dropna=False
    ).transform(prior_valid_observations)

    return _date_gate(frame, values, group_keys=group_keys)


def _expanding(
    frame: pd.DataFrame,
    source: str,
    *,
    group_keys: list[str],
    function: str,
) -> pd.Series:
    numeric = pd.to_numeric(frame[source], errors="coerce").astype("float64")
    grouping = [frame[key] for key in group_keys]
    if function == "mean":
        values = numeric.groupby(grouping, sort=False, dropna=False).transform(
            lambda series: series.shift(1).expanding(min_periods=1).mean()
        )
    elif function == "std":
        values = numeric.groupby(grouping, sort=False, dropna=False).transform(
            lambda series: series.shift(1).expanding(min_periods=1).std(ddof=0)
        )
    else:
        raise ValueError(f"Unsupported expanding function: {function}")
    return _date_gate(frame, values, group_keys=group_keys)


def _previous_regular_stat(
    frame: pd.DataFrame,
    source: str,
    *,
    extra_keys: list[str],
    function: str,
) -> pd.Series:
    keys = ["team_id", *extra_keys, "season_year"]
    regular = frame.loc[frame["game_type"].eq(REGULAR_SEASON_GAME_TYPE), keys].copy()
    regular["__value"] = pd.to_numeric(
        frame.loc[regular.index, source], errors="coerce"
    ).astype("float64")
    grouped = regular.groupby(keys, as_index=False, dropna=False)["__value"]
    if function == "mean":
        prior = grouped.mean()
    elif function == "std":

        def population_std(values: pd.Series) -> float:
            clean = values.dropna().to_numpy(dtype="float64")
            return float(np.std(clean, ddof=0)) if len(clean) else np.nan

        prior = grouped.agg(population_std)
    else:
        raise ValueError(f"Unsupported previous-season function: {function}")
    prior["season_year"] += 1
    prior = prior.rename(columns={"__value": "__previous"})
    mapped = frame[keys].merge(
        prior, on=keys, how="left", sort=False, validate="many_to_one"
    )
    return pd.Series(mapped["__previous"].to_numpy(), index=frame.index)


def _previous_regular_trend(
    frame: pd.DataFrame,
    source: str,
    *,
    extra_keys: list[str],
    window: int,
) -> pd.Series:
    keys = ["team_id", *extra_keys, "season_year"]
    regular = frame.loc[frame["game_type"].eq(REGULAR_SEASON_GAME_TYPE), keys].copy()
    regular["__value"] = pd.to_numeric(
        frame.loc[regular.index, source], errors="coerce"
    ).astype("float64")

    def ending_slope(values: pd.Series) -> float:
        clean = values.tail(window).dropna().to_numpy(dtype="float64")
        if len(clean) < 2:
            return np.nan
        x = np.arange(len(clean), dtype="float64")
        x -= x.mean()
        return float(np.dot(x, clean - clean.mean()) / np.dot(x, x))

    prior = (
        regular.groupby(keys, as_index=False, dropna=False)["__value"]
        .agg(ending_slope)
        .rename(columns={"__value": "__previous"})
    )
    prior["season_year"] += 1
    mapped = frame[keys].merge(
        prior, on=keys, how="left", sort=False, validate="many_to_one"
    )
    return pd.Series(mapped["__previous"].to_numpy(), index=frame.index)


def _fill_mean_history(
    current: pd.Series, previous: pd.Series, fallback: pd.Series | None = None
) -> pd.Series:
    result = current.fillna(previous)
    if fallback is not None:
        result = result.fillna(fallback)
    return result.fillna(NO_HISTORY_VALUE).astype("float64")


def _source_prefix(label: str) -> str:
    return label if label.startswith("ODDS_") else f"TEAM_ROLLING_{label}"


def _rolling_source_features(
    frame: pd.DataFrame,
    source: str,
    label: str,
    *,
    extended: bool,
) -> dict[str, pd.Series]:
    prefix = _source_prefix(label)
    previous_mean = _previous_regular_stat(
        frame, source, extra_keys=[], function="mean"
    )
    previous_venue_mean = _previous_regular_stat(
        frame, source, extra_keys=["home"], function="mean"
    )
    previous_std = _previous_regular_stat(
        frame, source, extra_keys=["home"], function="std"
    )

    raw_base = _rolling(
        frame,
        source,
        group_keys=["team_id"],
        window=BASE_WINDOW,
        function="mean",
    )
    base = _fill_mean_history(raw_base, previous_mean)
    venue = _fill_mean_history(
        _rolling(
            frame,
            source,
            group_keys=["team_id", "home"],
            window=BASE_WINDOW,
            function="mean",
        ),
        previous_venue_mean,
    )
    season_mean = _fill_mean_history(
        _expanding(
            frame,
            source,
            group_keys=["team_id", "season_year", "home"],
            function="mean",
        ),
        previous_venue_mean,
        base,
    )
    season_std = _fill_mean_history(
        _expanding(
            frame,
            source,
            group_keys=["team_id", "season_year", "home"],
            function="std",
        ),
        previous_std,
    )
    observation_count = _rolling(
        frame,
        source,
        group_keys=["team_id"],
        window=BASE_WINDOW,
        function="count",
    ).fillna(0.0)

    features: dict[str, pd.Series] = {
        f"{prefix}_LAST_ALL_{BASE_WINDOW}_GAMES_BEFORE": base,
        f"{prefix}_LAST_HOME_AWAY_{BASE_WINDOW}_GAMES_BEFORE": venue - base,
        f"{prefix}_LAST_{BASE_WINDOW}_OBSERVATIONS_BEFORE": observation_count,
        f"{prefix}_SEASON_BEFORE_AVG": season_mean,
        f"{prefix}_SEASON_BEFORE_STD": season_std,
        (
            f"TEAM_RATIO_{label}_LAST_{BASE_WINDOW}_DIV_SEASON_AVG_BEFORE"
            if not label.startswith("ODDS_")
            else f"{label}_LAST_{BASE_WINDOW}_DIV_SEASON_AVG_BEFORE"
        ): _safe_ratio(base, season_mean),
    }
    if not extended:
        return features

    rolling_means: dict[int, pd.Series] = {BASE_WINDOW: base}
    for window in MLB_WINDOWS:
        if window == BASE_WINDOW:
            continue
        rolling_means[window] = _fill_mean_history(
            _rolling(
                frame,
                source,
                group_keys=["team_id"],
                window=window,
                function="mean",
            ),
            previous_mean,
        )
        features[f"{prefix}_LAST_ALL_{window}_GAMES_BEFORE"] = rolling_means[window]

    for long_window in LONG_WINDOWS:
        features[
            f"{prefix}_LAST_{BASE_WINDOW}_MINUS_LAST_{long_window}_GAMES_BEFORE"
        ] = (base - rolling_means[long_window])

    weighted = _fill_mean_history(
        _rolling(
            frame,
            source,
            group_keys=["team_id"],
            window=BASE_WINDOW,
            function="wma",
        ),
        previous_mean,
    )
    weighted_venue = _fill_mean_history(
        _rolling(
            frame,
            source,
            group_keys=["team_id", "home"],
            window=BASE_WINDOW,
            function="wma",
        ),
        previous_venue_mean,
    )
    features[f"{prefix}_LAST_{BASE_WINDOW}_WMA_BEFORE"] = weighted
    features[f"{prefix}_LAST_HOME_AWAY_{BASE_WINDOW}_WMA_BEFORE"] = (
        weighted_venue - weighted
    )

    trends: dict[int, pd.Series] = {}
    for window in TREND_WINDOWS:
        trend = (
            _rolling(
                frame,
                source,
                group_keys=["team_id", "season_year"],
                window=window,
                function="slope",
            )
            .fillna(
                _previous_regular_trend(frame, source, extra_keys=[], window=window)
            )
            .fillna(NO_HISTORY_VALUE)
        )
        trends[window] = trend
        features[f"{prefix}_TREND_SLOPE_LAST_{window}_GAMES_BEFORE"] = trend

    venue_trend = (
        _rolling(
            frame,
            source,
            group_keys=["team_id", "season_year", "home"],
            window=BASE_WINDOW,
            function="slope",
        )
        .fillna(
            _previous_regular_trend(
                frame, source, extra_keys=["home"], window=BASE_WINDOW
            )
        )
        .fillna(NO_HISTORY_VALUE)
    )
    features[f"{prefix}_TREND_SLOPE_LAST_{BASE_WINDOW}_HOME_AWAY_GAMES_BEFORE"] = (
        venue_trend - trends[BASE_WINDOW]
    )
    features[f"{prefix}_TREND_SLOPE_LAST_{BASE_WINDOW}_MINUS_LAST_10_GAMES_BEFORE"] = (
        trends[BASE_WINDOW] - trends[10]
    )
    return features


def _prepare_team_context(
    team_games: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    missing_team = sorted(_TEAM_REQUIRED.difference(team_games.columns))
    missing_games = sorted(_GAME_REQUIRED.difference(games.columns))
    if missing_team or missing_games:
        raise ValueError(
            f"Missing team-game columns={missing_team}; game columns={missing_games}"
        )
    required_sources = set(TEAM_SOURCE_COLUMNS).union(DERIVED_RATE_SOURCE_COLUMNS)
    missing_sources = sorted(required_sources.difference(team_games.columns))
    if missing_sources:
        raise ValueError(f"team games is missing rolling sources: {missing_sources}")

    out = team_games.copy()
    out["game_pk"] = out["game_pk"].astype(str)
    out["team_id"] = out["team_id"].astype(str)
    existing_keys = pd.MultiIndex.from_frame(out[["game_pk", "team_id"]])
    skeleton_rows: list[dict[str, object]] = []
    for game in games.itertuples(index=False):
        for home, team_id, opponent_id in (
            (True, str(game.home_team_id), str(game.away_team_id)),
            (False, str(game.away_team_id), str(game.home_team_id)),
        ):
            if (str(game.game_pk), team_id) in existing_keys:
                continue
            skeleton_rows.append(
                {
                    "game_pk": str(game.game_pk),
                    "team_id": team_id,
                    "season_year": game.season_year,
                    "game_date": game.game_date,
                    "first_pitch_utc": game.first_pitch_utc,
                    "game_type": game.game_type,
                    "home": home,
                    "opponent_team_id": opponent_id,
                }
            )
    if skeleton_rows:
        skeleton = pd.DataFrame(skeleton_rows)
        for column in out.columns:
            if column not in skeleton:
                skeleton[column] = np.nan
        out = pd.concat([out, skeleton[out.columns]], ignore_index=True)

    out["game_date"] = pd.to_datetime(out["game_date"], errors="raise").dt.normalize()
    out["first_pitch_utc"] = pd.to_datetime(out["first_pitch_utc"], utc=True)
    out["home"] = out["home"].astype(bool)
    out = out.sort_values(
        ["team_id", "game_date", "first_pitch_utc", "game_pk"], kind="mergesort"
    ).reset_index(drop=True)
    if out.duplicated(["game_pk", "team_id"]).any():
        raise ValueError("team-game context is not unique by (game_pk, team_id).")
    return _add_derived_rates(out)


def _attach_market_sources(
    context: pd.DataFrame, closing_features: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str]]:
    if "GAME_ID" not in closing_features:
        raise ValueError("closing features is missing GAME_ID.")
    if closing_features["GAME_ID"].astype(str).duplicated().any():
        raise ValueError("closing features is not unique by GAME_ID.")
    closing = closing_features.assign(
        GAME_ID=closing_features["GAME_ID"].astype(str)
    ).set_index("GAME_ID")
    game_ids = context["game_pk"]
    home = context["home"]
    sources: dict[str, str] = {}

    # Totals are game-level; normalized lines are the comparable level.
    total_columns = [
        column
        for column in closing.columns
        if column.startswith("ODDS_TOTAL_")
        and column.endswith("_LINE_NORMALIZED")
        and "_CONSENSUS_" not in column
    ]
    for column in total_columns:
        context[f"__{column}"] = game_ids.map(closing[column])
        sources[f"__{column}"] = column

    total_consensus = "ODDS_TOTAL_CONSENSUS_LINE_NORMALIZED_MEDIAN"
    if total_consensus in closing:
        context[f"__{total_consensus}"] = game_ids.map(closing[total_consensus])
        sources[f"__{total_consensus}"] = total_consensus

    # Run-line and moneyline histories must be expressed from this team's side.
    home_run_columns = [
        column
        for column in closing.columns
        if column.startswith("ODDS_RUN_LINE_")
        and column.endswith("_HOME_HANDICAP_NORMALIZED")
        and "_CONSENSUS_" not in column
    ]
    for home_column in home_run_columns:
        away_column = home_column.replace(
            "_HOME_HANDICAP_NORMALIZED", "_AWAY_HANDICAP_NORMALIZED"
        )
        if away_column not in closing:
            continue
        label = home_column.replace(
            "_HOME_HANDICAP_NORMALIZED", "_TEAM_HANDICAP_NORMALIZED"
        )
        home_values = game_ids.map(closing[home_column])
        away_values = game_ids.map(closing[away_column])
        context[f"__{label}"] = home_values.where(home, away_values)
        sources[f"__{label}"] = label

    run_consensus = "ODDS_RUN_LINE_CONSENSUS_HOME_HANDICAP_NORMALIZED_MEDIAN"
    if run_consensus in closing:
        label = "ODDS_RUN_LINE_CONSENSUS_TEAM_HANDICAP_NORMALIZED_MEDIAN"
        home_values = game_ids.map(closing[run_consensus])
        context[f"__{label}"] = home_values.where(home, -home_values)
        sources[f"__{label}"] = label

    run_prob_columns = [
        column
        for column in closing.columns
        if column.startswith("ODDS_RUN_LINE_")
        and column.endswith("_FAIR_PROB_HOME_COVER")
        and "_CONSENSUS_" not in column
    ]
    for home_column in run_prob_columns:
        away_column = home_column.replace(
            "_FAIR_PROB_HOME_COVER", "_FAIR_PROB_AWAY_COVER"
        )
        if away_column not in closing:
            continue
        label = home_column.replace("_FAIR_PROB_HOME_COVER", "_FAIR_PROB_TEAM_COVER")
        context[f"__{label}"] = game_ids.map(closing[home_column]).where(
            home, game_ids.map(closing[away_column])
        )
        sources[f"__{label}"] = label

    money_prob_columns = [
        column
        for column in closing.columns
        if column.startswith("ODDS_MONEY_LINE_")
        and column.endswith("_FAIR_PROB_HOME_WIN")
        and "_CONSENSUS_" not in column
    ]
    for home_column in money_prob_columns:
        away_column = home_column.replace("_FAIR_PROB_HOME_WIN", "_FAIR_PROB_AWAY_WIN")
        if away_column not in closing:
            continue
        label = home_column.replace("_FAIR_PROB_HOME_WIN", "_FAIR_PROB_TEAM_WIN")
        context[f"__{label}"] = game_ids.map(closing[home_column]).where(
            home, game_ids.map(closing[away_column])
        )
        sources[f"__{label}"] = label

    money_consensus = "ODDS_MONEY_LINE_CONSENSUS_FAIR_PROB_HOME_WIN_MEDIAN"
    if money_consensus in closing:
        label = "ODDS_MONEY_LINE_CONSENSUS_FAIR_PROB_TEAM_WIN_MEDIAN"
        home_values = game_ids.map(closing[money_consensus])
        context[f"__{label}"] = home_values.where(home, 1.0 - home_values)
        sources[f"__{label}"] = label

    # Prior-game market misses are legal only after the same shift/date gate as
    # every box-score source. The unshifted columns never leave this function.
    if total_consensus in closing:
        total_line = game_ids.map(closing[total_consensus])
        context["__odds_total_market_error"] = context["total_runs"] - total_line
        context["__odds_total_market_abs_error"] = context[
            "__odds_total_market_error"
        ].abs()
        sources["__odds_total_market_error"] = "ODDS_HISTORY_TOTAL_MARKET_ERROR"
        sources["__odds_total_market_abs_error"] = "ODDS_HISTORY_TOTAL_MARKET_ABS_ERROR"
    if run_consensus in closing:
        home_handicap = game_ids.map(closing[run_consensus])
        team_handicap = home_handicap.where(home, -home_handicap)
        context["__odds_run_line_margin_error"] = context["run_margin"] + team_handicap
        context["__odds_run_line_margin_abs_error"] = context[
            "__odds_run_line_margin_error"
        ].abs()
        sources["__odds_run_line_margin_error"] = "ODDS_HISTORY_RUN_LINE_MARGIN_ERROR"
        sources["__odds_run_line_margin_abs_error"] = (
            "ODDS_HISTORY_RUN_LINE_MARGIN_ABS_ERROR"
        )
    return context, sources


def _wide_team_features(
    team_rows: pd.DataFrame, target_game_ids: set[str]
) -> pd.DataFrame:
    feature_columns = [
        column
        for column in team_rows.columns
        if column.startswith(("TEAM_", "ODDS_")) and "_BEFORE" in column
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

    home = target.loc[target["home"], ["game_pk", *feature_columns]].copy()
    away = target.loc[~target["home"], ["game_pk", *feature_columns]].copy()
    home = home.rename(
        columns={column: f"{column}_TEAM_HOME" for column in feature_columns}
    )
    away = away.rename(
        columns={column: f"{column}_TEAM_AWAY" for column in feature_columns}
    )
    wide = home.merge(away, on="game_pk", how="inner", validate="one_to_one")
    wide = wide.rename(columns={"game_pk": "GAME_ID"})

    differences: dict[str, pd.Series] = {}
    for column in feature_columns:
        home_column = f"{column}_TEAM_HOME"
        away_column = f"{column}_TEAM_AWAY"
        diff_column = (
            column.removesuffix("_BEFORE") + "_DIFF_BEFORE"
            if column.endswith("_BEFORE")
            else f"{column}_DIFF_BEFORE"
        )
        differences[diff_column] = wide[home_column] - wide[away_column]
    if differences:
        wide = pd.concat([wide, pd.DataFrame(differences, index=wide.index)], axis=1)
    return wide


def build_team_rolling_features(
    team_games: pd.DataFrame,
    games: pd.DataFrame,
    closing_features: pd.DataFrame,
    *,
    target_game_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build the complete shifted rolling family and pivot it by game side."""
    context = _prepare_team_context(team_games, games)
    context, odds_sources = _attach_market_sources(context, closing_features)
    sources = {
        **TEAM_SOURCE_COLUMNS,
        **DERIVED_RATE_COLUMNS,
        **odds_sources,
    }
    features: dict[str, pd.Series] = {}
    for source, label in sources.items():
        features.update(
            _rolling_source_features(
                context,
                source,
                label,
                extended=label in EXTENDED_TEAM_LABELS or label.startswith("ODDS_"),
            )
        )
    context = pd.concat([context, pd.DataFrame(features, index=context.index)], axis=1)
    targets = (
        {str(value) for value in target_game_ids}
        if target_game_ids is not None
        else {str(value) for value in closing_features["GAME_ID"]}
    )
    return _wide_team_features(context, targets)


def build_pregame_features(
    closing_features: pd.DataFrame,
    team_rolling_features: pd.DataFrame,
    context_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge current closing markets with strictly historical team features."""
    output = closing_features.merge(
        team_rolling_features, on="GAME_ID", how="left", validate="one_to_one"
    )
    if context_features is not None:
        overlap = sorted(
            set(output.columns)
            .intersection(context_features.columns)
            .difference({"GAME_ID"})
        )
        if overlap:
            raise ValueError(
                f"Context feature columns overlap existing columns: {overlap}"
            )
        output = output.merge(
            context_features, on="GAME_ID", how="left", validate="one_to_one"
        )
    unlabeled = [
        column
        for column in output.columns
        if not column.startswith(("GAME_", "ODDS_", "TEAM_", "SCHEDULE_", "TRAVEL_"))
    ]
    if unlabeled:
        raise ValueError(f"Unlabelled pregame feature columns: {unlabeled}")

    # A family label is not enough: every non-market engineered feature must
    # also declare that it is available before first pitch. Current closing
    # odds and the explicit GAME_* metadata arrive from closing_features and
    # are the only allowed exceptions. This prevents a caller from smuggling a
    # raw result through merely by naming it TEAM_* or ODDS_*.
    closing_columns = set(closing_features.columns)
    missing_temporal_tag = [
        column
        for column in output.columns
        if (
            column.startswith(("TEAM_", "SCHEDULE_", "TRAVEL_"))
            and "_BEFORE" not in column
        )
        or (
            column.startswith(("GAME_", "ODDS_"))
            and "_BEFORE" not in column
            and column not in closing_columns
        )
    ]
    if missing_temporal_tag:
        raise ValueError(
            "Pregame feature columns missing _BEFORE temporal tag: "
            f"{missing_temporal_tag}"
        )
    leaking_fragments = ("HOME_SCORE", "AWAY_SCORE", "RUN_LINE_MARGIN", "LINE_ERROR")
    leaking = [
        column
        for column in output.columns
        if column.startswith("GAME_")
        and any(fragment in column for fragment in leaking_fragments)
    ]
    if leaking:
        raise ValueError(f"Outcome columns reached pregame features: {leaking}")
    return output


def write_pregame_feature_partitions(
    features: pd.DataFrame,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seasons: Iterable[int] | None = None,
) -> list[Path]:
    """Atomically write stable-schema pregame feature partitions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = (
        sorted({int(value) for value in seasons})
        if seasons is not None
        else sorted(int(value) for value in features["GAME_SEASON_YEAR"].unique())
    )
    destinations: list[Path] = []
    for season in selected:
        partition = features.loc[features["GAME_SEASON_YEAR"].eq(season)].copy()
        destination = output_dir / f"pregame_features_{season}.parquet"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.", suffix=".parquet.tmp", dir=output_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            partition.to_parquet(temporary, index=False)
            if pq.ParquetFile(temporary).metadata.num_rows != len(partition):
                raise OSError(f"Row-count validation failed for {temporary}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        destinations.append(destination)
    return destinations
