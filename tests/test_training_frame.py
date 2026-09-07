from __future__ import annotations

import pandas as pd
import pytest

from mlb_pred.create_training_data.training_frame import (
    OUTCOME_ONLY_COLUMNS,
    RAW_RUN_LINE_HANDICAP_COLUMN,
    RAW_TOTAL_LINE_COLUMN,
    SPREAD_LINE_HOME_COLUMN,
    assert_no_outcome_features,
    build_training_frame,
    feature_columns,
    load_pregame_features,
)


def _pregame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["1", "2"],
            "GAME_DATE": ["2025-04-01", "2025-04-02"],
            "GAME_SEASON_YEAR": [2025, 2025],
            "GAME_FIRST_PITCH_UTC": [
                "2025-04-01T19:00:00Z",
                "2025-04-02T19:00:00Z",
            ],
            "GAME_HOME_TEAM_ID": ["A", "A"],
            "GAME_AWAY_TEAM_ID": ["B", "B"],
            "TEAM_ROLLING_RUNS_SCORED_LAST_ALL_5_GAMES_BEFORE_TEAM_HOME": [
                4.0,
                4.5,
            ],
            "ODDS_TOTAL_CONSENSUS_LINE_NORMALIZED_MEDIAN": [8.5, 6.0],
            "ODDS_RUN_LINE_CONSENSUS_HOME_HANDICAP_NORMALIZED_MEDIAN": [
                -1.5,
                1.5,
            ],
        }
    )


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": ["1", "2"],
            "game_date": ["2025-04-01", "2025-04-02"],
            "season_year": [2025, 2025],
            "home_team_id": ["A", "A"],
            "away_team_id": ["B", "B"],
            "is_final": [True, True],
            "home_score": [5, 2],
            "away_score": [3, 4],
            "total_runs": [8, 6],
            "run_line_margin": [2, -2],
            "extra_innings": [False, True],
        }
    )


def test_training_frame_derives_total_margin_and_both_market_errors():
    result = build_training_frame(_pregame(), _games()).set_index("GAME_ID")

    assert result["TOTAL_RUNS"].tolist() == [8, 6]
    assert result["RUN_LINE_MARGIN"].tolist() == [2, -2]
    assert result["HOME_MARGIN"].equals(result["RUN_LINE_MARGIN"])
    assert result["LINE_ERROR"].tolist() == pytest.approx([-0.5, 0.0])
    assert result[SPREAD_LINE_HOME_COLUMN].tolist() == pytest.approx([1.5, -1.5])
    assert result["SPREAD_ERROR"].tolist() == pytest.approx([0.5, -0.5])
    assert result["IS_EXTRA_INNINGS"].tolist() == [False, True]


def test_push_is_retained_as_a_zero_regression_target():
    result = build_training_frame(_pregame(), _games())

    assert len(result) == 2
    assert result.loc[result["GAME_ID"].eq("2"), "LINE_ERROR"].iloc[0] == 0.0


def test_feature_allow_list_excludes_metadata_and_every_outcome():
    result = build_training_frame(_pregame(), _games())
    selected = feature_columns(result)

    assert "GAME_ID" not in selected
    assert not set(OUTCOME_ONLY_COLUMNS).intersection(selected)
    # SPREAD_LINE_HOME is the quantity SPREAD_ERROR is defined against. It is
    # still written so the target can be reproduced, but it is not a feature;
    # the closing handicap remains available under its ODDS_* name.
    assert SPREAD_LINE_HOME_COLUMN not in selected
    assert "ODDS_TOTAL_CONSENSUS_LINE_NORMALIZED_MEDIAN" in selected
    assert_no_outcome_features(selected)
    with pytest.raises(ValueError, match="Outcome-derived"):
        assert_no_outcome_features([*selected, "TOTAL_RUNS"])


def test_pregame_source_cannot_already_contain_a_target():
    leaking = _pregame().assign(TOTAL_RUNS=[8, 6])

    with pytest.raises(ValueError, match="already contains scoring outcomes"):
        build_training_frame(leaking, _games())


def test_stored_scores_are_checked_against_recomputed_targets():
    games = _games().assign(total_runs=[999, 6])

    with pytest.raises(ValueError, match="Stored total_runs disagrees"):
        build_training_frame(_pregame(), games)


def test_limit_date_filters_completed_rows_inclusively():
    result = build_training_frame(_pregame(), _games(), limit_date="2025-04-01")

    assert result["GAME_ID"].tolist() == ["1"]


def test_partition_loader_rejects_schema_drift(tmp_path):
    first = _pregame().iloc[[0]]
    second = (
        _pregame()
        .iloc[[1]]
        .drop(columns="TEAM_ROLLING_RUNS_SCORED_LAST_ALL_5_GAMES_BEFORE_TEAM_HOME")
    )
    first.to_parquet(tmp_path / "pregame_features_2024.parquet", index=False)
    second.to_parquet(tmp_path / "pregame_features_2025.parquet", index=False)

    with pytest.raises(ValueError, match="schema changed"):
        load_pregame_features(tmp_path)


def test_partition_loader_selects_requested_seasons(tmp_path):
    _pregame().iloc[[0]].assign(GAME_SEASON_YEAR=2024).to_parquet(
        tmp_path / "pregame_features_2024.parquet", index=False
    )
    _pregame().iloc[[1]].to_parquet(
        tmp_path / "pregame_features_2025.parquet", index=False
    )

    result = load_pregame_features(tmp_path, seasons=[2025])

    assert result["GAME_ID"].tolist() == ["2"]


def test_raw_line_columns_measure_residuals_against_the_executable_close():
    """The raw close is a different price, so it must move the residuals."""
    pregame = _pregame()
    # The raw executable close sits half a run away from the -110/-110
    # restatement on the total, and a full run away on the run line.
    pregame[RAW_TOTAL_LINE_COLUMN] = [9.0, 6.0]
    pregame[RAW_RUN_LINE_HANDICAP_COLUMN] = [-2.5, 1.5]

    result = build_training_frame(
        pregame,
        _games(),
        total_line_column=RAW_TOTAL_LINE_COLUMN,
        run_line_handicap_column=RAW_RUN_LINE_HANDICAP_COLUMN,
    ).set_index("GAME_ID")

    # TOTAL_RUNS 8 against a raw line of 9.0, and 6 against 6.0.
    assert result["LINE_ERROR"].tolist() == pytest.approx([-1.0, 0.0])
    assert result[SPREAD_LINE_HOME_COLUMN].tolist() == pytest.approx([2.5, -1.5])
    # RUN_LINE_MARGIN 2 against an implied 2.5 home margin, then -2 against -1.5.
    assert result["SPREAD_ERROR"].tolist() == pytest.approx([-0.5, -0.5])

    # The outcomes themselves are a property of the game, not of the quote.
    normalized = build_training_frame(pregame, _games()).set_index("GAME_ID")
    assert normalized["TOTAL_RUNS"].equals(result["TOTAL_RUNS"])
    assert normalized["RUN_LINE_MARGIN"].equals(result["RUN_LINE_MARGIN"])
    assert normalized["LINE_ERROR"].tolist() != result["LINE_ERROR"].tolist()


def test_missing_market_quote_yields_a_null_residual_not_a_dropped_row():
    """A game without one market must stay usable for the other target."""
    pregame = _pregame()
    pregame.loc[
        pregame["GAME_ID"].eq("2"),
        "ODDS_RUN_LINE_CONSENSUS_HOME_HANDICAP_NORMALIZED_MEDIAN",
    ] = pd.NA

    result = build_training_frame(pregame, _games()).set_index("GAME_ID")

    assert len(result) == 2
    assert pd.isna(result.loc["2", "SPREAD_ERROR"])
    # The total market was quoted, so its residual survives on the same row.
    assert result.loc["2", "LINE_ERROR"] == 0.0
    assert result.loc["2", "TOTAL_RUNS"] == 6
