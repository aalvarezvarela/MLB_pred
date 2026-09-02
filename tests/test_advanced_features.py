from __future__ import annotations

import pandas as pd
import pytest

from mlb_pred.features.advanced_features import (
    ADVANCED_FAMILY_PREFIXES,
    PREGAME_TAG,
    assert_advanced_feature_contract,
    build_advanced_pregame_features,
    build_calendar_competition_features,
    build_extra_innings_history_features,
    build_global_market_regime_features,
    build_historical_matchup_features,
    build_matchup_style_features,
    build_odds_interaction_features,
)


def _games() -> pd.DataFrame:
    rows = [
        # Prior-season postseason context for the Yankees.
        ("0", 2024, "2024-10-20", "W", "147", "111", 4, 2, False),
        ("1", 2025, "2025-04-01", "R", "147", "121", 5, 3, False),
        # Doubleheader: neither row may read either result from this date.
        ("2", 2025, "2025-04-02", "R", "121", "147", 6, 5, True),
        ("3", 2025, "2025-04-02", "R", "121", "147", 2, 1, False),
        # Deliberately large target result to expose an accidental missing shift.
        ("4", 2025, "2025-04-03", "R", "147", "121", 100, 90, True),
    ]
    return pd.DataFrame(
        [
            {
                "game_pk": game_pk,
                "season_year": season,
                "game_date": pd.Timestamp(game_date).date(),
                "first_pitch_utc": pd.Timestamp(f"{game_date}T19:00:00Z"),
                "game_type": game_type,
                "home_team_id": home,
                "away_team_id": away,
                "home_score": home_score,
                "away_score": away_score,
                "run_line_margin": home_score - away_score,
                "extra_innings": extra,
                "is_final": True,
            }
            for (
                game_pk,
                season,
                game_date,
                game_type,
                home,
                away,
                home_score,
                away_score,
                extra,
            ) in rows
        ]
    )


def _closing() -> pd.DataFrame:
    games = _games().set_index("game_pk")
    game_ids = ["1", "2", "3", "4"]
    frame = pd.DataFrame(
        {
            "GAME_ID": game_ids,
            "GAME_DATE": [games.at[value, "game_date"] for value in game_ids],
            "GAME_FIRST_PITCH_UTC": [
                games.at[value, "first_pitch_utc"] for value in game_ids
            ],
            "GAME_SEASON_YEAR": [2025] * 4,
            "ODDS_TOTAL_CONSENSUS_LINE_NORMALIZED_MEDIAN": [7.0, 8.0, 8.5, 9.0],
            "ODDS_TOTAL_CONSENSUS_LINE_NORMALIZED_STD": [0.2, 0.3, 0.4, 0.5],
            "ODDS_TOTAL_CONSENSUS_FAIR_PROB_OVER_MEDIAN": [0.55, 0.5, 0.6, 0.45],
            "ODDS_RUN_LINE_CONSENSUS_HOME_HANDICAP_NORMALIZED_MEDIAN": [
                -1.5,
                1.5,
                1.0,
                -1.5,
            ],
            "ODDS_RUN_LINE_CONSENSUS_HOME_HANDICAP_NORMALIZED_STD": [
                0.1,
                0.2,
                0.2,
                0.1,
            ],
            "ODDS_RUN_LINE_CONSENSUS_FAIR_PROB_HOME_COVER_MEDIAN": [
                0.52,
                0.48,
                0.51,
                0.53,
            ],
            "ODDS_MONEY_LINE_CONSENSUS_FAIR_PROB_HOME_WIN_MEDIAN": [
                0.6,
                0.45,
                0.4,
                0.65,
            ],
            "ODDS_DERIVED_IMPLIED_HOME_RUNS_NORMALIZED": [4.25, 3.25, 3.75, 5.25],
            "ODDS_DERIVED_IMPLIED_AWAY_RUNS_NORMALIZED": [2.75, 4.75, 4.75, 3.75],
        }
    )
    for market in ("TOTAL", "RUN_LINE", "MONEY_LINE"):
        frame[f"ODDS_{market}_BET365_OVERROUND"] = 0.045
        frame[f"ODDS_{market}_FANDUEL_OVERROUND"] = 0.055
    return frame


def _rolling() -> pd.DataFrame:
    rows = pd.DataFrame({"GAME_ID": ["1", "2", "3", "4"]})
    values: dict[str, float] = {}
    for horizon in (5, 10, 20):
        suffix = f"LAST_ALL_{horizon}_GAMES_BEFORE"
        values[f"TEAM_ROLLING_RUNS_SCORED_{suffix}_TEAM_HOME"] = 6.0
        values[f"TEAM_ROLLING_RUNS_ALLOWED_{suffix}_TEAM_HOME"] = 4.0
        values[f"TEAM_ROLLING_RUNS_SCORED_{suffix}_TEAM_AWAY"] = 3.0
        values[f"TEAM_ROLLING_RUNS_ALLOWED_{suffix}_TEAM_AWAY"] = 5.0
    for side, scored, allowed in (("HOME", 6.0, 4.0), ("AWAY", 3.0, 5.0)):
        values[f"TEAM_ROLLING_RUNS_SCORED_SEASON_BEFORE_AVG_TEAM_{side}"] = scored
        values[f"TEAM_ROLLING_RUNS_ALLOWED_SEASON_BEFORE_AVG_TEAM_{side}"] = allowed
        values[f"TEAM_ROLLING_RUNS_SCORED_SEASON_BEFORE_STD_TEAM_{side}"] = 2.0
        values[
            f"TEAM_ROLLING_RUNS_SCORED_TREND_SLOPE_LAST_5_GAMES_BEFORE_TEAM_{side}"
        ] = 0.2
        values[
            f"TEAM_ROLLING_RUNS_ALLOWED_TREND_SLOPE_LAST_5_GAMES_BEFORE_TEAM_{side}"
        ] = 0.1

    for side, pa, bf in (("HOME", 38.0, 36.0), ("AWAY", 37.0, 40.0)):
        values[f"TEAM_ROLLING_PLATE_APPEARANCES_SEASON_BEFORE_AVG_TEAM_{side}"] = pa
        values[f"TEAM_ROLLING_BATTERS_FACED_SEASON_BEFORE_AVG_TEAM_{side}"] = bf
    rates = {
        "OFFENSE_K_PCT": (0.20, 0.25),
        "PITCHING_K_PCT": (0.22, 0.24),
        "OFFENSE_BB_PCT": (0.10, 0.08),
        "PITCHING_BB_PCT": (0.09, 0.11),
        "OFFENSE_HR_PER_PA": (0.04, 0.03),
        "PITCHING_HR_PER_BF": (0.035, 0.045),
        "OFFENSE_BASERUNNERS_PER_PA": (0.32, 0.30),
        "PITCHING_BASERUNNERS_PER_BF": (0.31, 0.34),
    }
    for metric, (home, away) in rates.items():
        values[f"TEAM_ROLLING_{metric}_SEASON_BEFORE_AVG_TEAM_HOME"] = home
        values[f"TEAM_ROLLING_{metric}_SEASON_BEFORE_AVG_TEAM_AWAY"] = away
    for column, value in values.items():
        rows[column] = value
    return rows


def test_matchup_style_crosses_offense_with_opposing_pitching():
    features = build_matchup_style_features(_rolling())
    row = features.iloc[0]

    assert row["TEAM_MATCHUP_HOME_RUN_EXPECTATION_SEASON_BEFORE"] == 5.5
    assert row["TEAM_MATCHUP_AWAY_RUN_EXPECTATION_SEASON_BEFORE"] == 3.5
    assert row["TEAM_MATCHUP_TOTAL_RUN_EXPECTATION_SEASON_BEFORE"] == 9.0
    assert row["TEAM_MATCHUP_EXPECTED_PLATE_APPEARANCES_HOME_SEASON_BEFORE"] == 39
    assert row["TEAM_MATCHUP_EXPECTED_K_RATE_HOME_SEASON_BEFORE"] == pytest.approx(0.22)


def test_odds_interactions_use_only_current_pregame_market_inputs():
    row = build_odds_interaction_features(_closing()).iloc[0]

    assert row["ODDS_INTERACTION_TOTAL_OVERROUND_MEAN_BEFORE"] == pytest.approx(0.05)
    assert row["ODDS_INTERACTION_TOTAL_PROBABILITY_SKEW_BEFORE"] == pytest.approx(0.1)
    assert row[
        "ODDS_INTERACTION_TOTAL_LINE_X_PROBABILITY_SKEW_BEFORE"
    ] == pytest.approx(0.7)
    assert row["ODDS_INTERACTION_IMPLIED_RUNS_GAP_BEFORE"] == pytest.approx(1.5)


def test_market_regime_excludes_every_result_from_the_current_date():
    features = build_global_market_regime_features(_games(), _closing()).set_index(
        "GAME_ID"
    )

    # Game 1: actual total 8 against 7; spread margin 2 against a -1.5 handicap.
    for game_id in ("2", "3"):
        assert features.at[
            game_id, "ODDS_MARKET_REGIME_TOTAL_BIAS_15G_BEFORE"
        ] == pytest.approx(1.0)
        assert features.at[
            game_id, "ODDS_MARKET_REGIME_SPREAD_BIAS_15G_BEFORE"
        ] == pytest.approx(0.5)

    changed = _games()
    changed.loc[changed["game_pk"].isin(["2", "3"]), ["home_score", "away_score"]] = [
        999,
        0,
    ]
    changed["run_line_margin"] = changed["home_score"] - changed["away_score"]
    rebuilt = build_global_market_regime_features(changed, _closing()).set_index(
        "GAME_ID"
    )
    regime_columns = [column for column in features if "MARKET_REGIME" in column]
    pd.testing.assert_series_equal(
        features.loc["2", regime_columns], rebuilt.loc["2", regime_columns]
    )
    pd.testing.assert_series_equal(
        features.loc["3", regime_columns], rebuilt.loc["3", regime_columns]
    )


def test_historical_matchup_and_extra_innings_are_doubleheader_safe():
    closing = _closing()
    h2h = build_historical_matchup_features(_games(), closing).set_index("GAME_ID")
    extra = build_extra_innings_history_features(_games(), closing).set_index("GAME_ID")

    for game_id in ("2", "3"):
        assert h2h.at[game_id, "TEAM_HISTORICAL_MATCHUP_GAMES_COUNT_BEFORE"] == 1
        assert (
            h2h.at[game_id, "TEAM_HISTORICAL_MATCHUP_TOTAL_RUNS_LAST_GAME_BEFORE"] == 8
        )
        assert extra.at[game_id, "TEAM_EXTRA_INNINGS_LAST_GAME_BEFORE_TEAM_HOME"] == 0

    assert h2h.at["4", "TEAM_HISTORICAL_MATCHUP_GAMES_COUNT_BEFORE"] == 3
    assert h2h.at[
        "4", "TEAM_HISTORICAL_MATCHUP_TOTAL_RUNS_LAST_3_GAMES_AVG_BEFORE"
    ] == pytest.approx((8 + 11 + 3) / 3)
    assert extra.at[
        "4", "TEAM_EXTRA_INNINGS_FREQUENCY_LAST_5_GAMES_BEFORE_TEAM_HOME"
    ] == pytest.approx(
        0.25
    )  # Includes the prior-season postseason game.


def test_calendar_competition_and_prior_postseason_context():
    features = build_calendar_competition_features(_games(), _closing()).set_index(
        "GAME_ID"
    )
    row = features.loc["1"]

    assert row["GAME_CALENDAR_MONTH_BEFORE"] == 4
    assert row["GAME_COMPETITION_IS_INTERLEAGUE_BEFORE"] == 1
    assert row["GAME_COMPETITION_SAME_DIVISION_BEFORE"] == 0
    assert row["TEAM_COMPETITION_POSTSEASON_GAMES_LAST_SEASON_BEFORE_TEAM_HOME"] == 1
    assert row["TEAM_COMPETITION_POSTSEASON_GAMES_LAST_SEASON_BEFORE_TEAM_AWAY"] == 0


def test_full_advanced_builder_enforces_family_and_availability_tags():
    features = build_advanced_pregame_features(_games(), _closing(), _rolling())

    assert len(features) == 4
    assert not features.columns.duplicated().any()
    assert all(
        column == "GAME_ID"
        or (column.startswith(ADVANCED_FAMILY_PREFIXES) and PREGAME_TAG in column)
        for column in features
    )
    assert not any(
        "HOME_SCORE" in column or "AWAY_SCORE" in column for column in features
    )

    with pytest.raises(ValueError, match="missing _BEFORE"):
        assert_advanced_feature_contract(
            pd.DataFrame({"GAME_ID": ["1"], "TEAM_MATCHUP_UNSAFE": [1.0]})
        )
