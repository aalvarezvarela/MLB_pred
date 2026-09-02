from __future__ import annotations

import pandas as pd
import pytest

from mlb_pred.features.closing_lines import (
    assert_closing_feature_contract,
    build_closing_line_features,
    select_closing_quotes,
)


def _ticks() -> pd.DataFrame:
    base = {
        "game_pk": "1",
        "season_year": 2025,
        "game_date": pd.Timestamp("2025-06-01").date(),
        "book_slug": "bet365",
        "is_opener": False,
    }
    rows = [
        # Valid total close. Stored line is x2: 17 -> 8.5 runs.
        {
            **base,
            "market": "totals",
            "line_ts": pd.Timestamp("2025-06-01T18:40:00Z"),
            "mins_to_tip": -20,
            "is_pregame": True,
            "left_line": 17,
            "right_line": 17,
            "left_price": -130,
            "right_price": 110,
        },
        # Too close to first pitch: must not replace the safe close.
        {
            **base,
            "market": "totals",
            "line_ts": pd.Timestamp("2025-06-01T18:57:00Z"),
            "mins_to_tip": -3,
            "is_pregame": True,
            "left_line": 18,
            "right_line": 18,
            "left_price": -110,
            "right_price": -110,
        },
        # In-play and therefore direct leakage.
        {
            **base,
            "market": "totals",
            "line_ts": pd.Timestamp("2025-06-01T19:05:00Z"),
            "mins_to_tip": 5,
            "is_pregame": False,
            "left_line": 15,
            "right_line": 15,
            "left_price": -110,
            "right_price": -110,
        },
        {
            **base,
            "market": "run_line",
            "line_ts": pd.Timestamp("2025-06-01T18:42:00Z"),
            "mins_to_tip": -18,
            "is_pregame": True,
            "left_line": 3,
            "right_line": -3,
            "left_price": -130,
            "right_price": 110,
        },
        {
            **base,
            "market": "money_line",
            "line_ts": pd.Timestamp("2025-06-01T18:45:00Z"),
            "mins_to_tip": -15,
            "is_pregame": True,
            "left_line": None,
            "right_line": None,
            "left_price": 120,
            "right_price": -140,
        },
    ]
    return pd.DataFrame(rows)


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": ["1"],
            "game_date": [pd.Timestamp("2025-06-01").date()],
            "season_year": [2025],
            "first_pitch_utc": [pd.Timestamp("2025-06-01T19:00:00Z")],
            "team_home_id": ["147"],
            "team_away_id": ["121"],
            "team_home": ["New York Yankees"],
            "team_away": ["New York Mets"],
            "is_doubleheader": [False],
            "game_number": [1],
            # These fields emulate a dangerous frame. The builder must select
            # an explicit pregame allowlist rather than pass them through.
            "home_score": [7],
            "away_score": [4],
            "total_runs": [11],
            "run_line_margin": [3],
        }
    )


def test_close_is_latest_complete_quote_outside_safety_margin():
    close = select_closing_quotes(_ticks(), safety_margin_minutes=5)
    total = close.loc[close["market"].eq("totals")].iloc[0]

    assert total["left_line"] == pytest.approx(8.5)
    assert total["right_line"] == pytest.approx(8.5)
    assert total["minutes_before_start"] == 20
    assert close["is_pregame"].all()


def test_feature_columns_are_labeled_and_outcomes_are_physically_absent():
    features = build_closing_line_features(_games(), _ticks())

    assert all(column.startswith(("GAME_", "ODDS_")) for column in features.columns)
    assert "GAME_TOTAL_RUNS" not in features
    assert "GAME_RUN_LINE_MARGIN" not in features
    assert "home_score" not in features
    assert "total_runs" not in features
    assert features.at[0, "ODDS_TOTAL_BET365_LINE_RAW"] == pytest.approx(8.5)
    assert features.at[0, "ODDS_TOTAL_BET365_LINE_NORMALIZED"] == pytest.approx(9.0)
    assert features.at[0, "ODDS_TOTAL_BET365_PRICE_OVER_NORMALIZED"] == -110
    assert features.at[0, "ODDS_TOTAL_BET365_PRICE_UNDER_NORMALIZED"] == -110


def test_run_line_orientation_and_equal_pay_normalization_are_explicit():
    features = build_closing_line_features(_games(), _ticks())

    assert features.at[0, "ODDS_RUN_LINE_BET365_AWAY_HANDICAP_RAW"] == 1.5
    assert features.at[0, "ODDS_RUN_LINE_BET365_HOME_HANDICAP_RAW"] == -1.5
    assert features.at[
        0, "ODDS_RUN_LINE_BET365_AWAY_HANDICAP_NORMALIZED"
    ] == pytest.approx(1.0)
    assert features.at[
        0, "ODDS_RUN_LINE_BET365_HOME_HANDICAP_NORMALIZED"
    ] == pytest.approx(-1.0)
    assert features.at[0, "ODDS_RUN_LINE_BET365_PRICE_AWAY_NORMALIZED"] == -110
    assert features.at[0, "ODDS_RUN_LINE_BET365_PRICE_HOME_NORMALIZED"] == -110


def test_invalid_run_line_pair_raises_instead_of_silently_flipping_a_side():
    ticks = _ticks()
    ticks.loc[ticks["market"].eq("run_line"), "right_line"] = -4

    with pytest.raises(ValueError, match="orientation invariant"):
        select_closing_quotes(ticks)


def test_null_leakage_discriminator_raises():
    ticks = _ticks()
    ticks["is_pregame"] = ticks["is_pregame"].astype("object")
    ticks.loc[0, "is_pregame"] = None

    with pytest.raises(ValueError, match="in-play leakage"):
        select_closing_quotes(ticks)


def test_contract_rejects_an_unlabeled_or_outcome_column():
    with pytest.raises(ValueError, match="Unlabelled"):
        assert_closing_feature_contract(pd.DataFrame({"weather": [1]}))
    with pytest.raises(ValueError, match="Outcome-derived"):
        assert_closing_feature_contract(pd.DataFrame({"GAME_TOTAL_RUNS": [9]}))


def test_requested_book_schema_is_stable_when_a_season_has_no_quotes():
    present = build_closing_line_features(
        _games(), _ticks(), books=["bet365", "future_book"]
    )
    absent = build_closing_line_features(
        _games(), _ticks().iloc[0:0], books=["bet365", "future_book"]
    )

    assert present.columns.tolist() == absent.columns.tolist()
    assert present.dtypes.astype(str).to_dict() == absent.dtypes.astype(str).to_dict()
    assert present.at[0, "ODDS_TOTAL_FUTURE_BOOK_HAS_QUOTE"] == 0
    assert pd.isna(present.at[0, "ODDS_TOTAL_FUTURE_BOOK_LINE_NORMALIZED"])
