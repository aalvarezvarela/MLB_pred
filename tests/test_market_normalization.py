from __future__ import annotations

import pandas as pd
import pytest

from mlb_pred.features.market_normalization import (
    CENTERED_AMERICAN_PRICE,
    center_run_lines,
    center_total_lines,
    devig_two_way_series,
    estimate_equal_price_line,
)


def test_devig_two_way_uses_american_prices_and_removes_overround():
    fair = devig_two_way_series(pd.Series([-130.0]), pd.Series([110.0]))

    assert fair.at[0, "fair_left"] + fair.at[0, "fair_right"] == pytest.approx(1.0)
    assert fair.at[0, "fair_left"] > 0.5
    assert fair.at[0, "overround"] > 0.0


def test_total_normalization_moves_toward_the_more_expensive_over():
    normalized = center_total_lines(
        pd.Series([8.5]), pd.Series([-130.0]), pd.Series([110.0])
    )

    assert normalized.iloc[0] == pytest.approx(9.0)


def test_run_line_normalization_uses_away_below_home_margin_orientation():
    # Away +1.5 is expensive, so the equal-price expected home margin must move
    # down, not up.  The corresponding normalized home handicap is its negative.
    normalized_away_handicap = center_run_lines(
        pd.Series([1.5]), pd.Series([-130.0]), pd.Series([110.0])
    )

    assert normalized_away_handicap.iloc[0] == pytest.approx(1.0)
    assert -normalized_away_handicap.iloc[0] == pytest.approx(-1.0)


def test_symmetric_quote_keeps_line_and_equal_pay_price_is_explicit():
    normalized = center_total_lines(
        pd.Series([8.5]),
        pd.Series([CENTERED_AMERICAN_PRICE]),
        pd.Series([CENTERED_AMERICAN_PRICE]),
    )
    assert normalized.iloc[0] == pytest.approx(8.5)


def test_integer_line_is_push_aware_and_stays_on_half_run_grid():
    normalized = estimate_equal_price_line(
        8.0,
        -125.0,
        105.0,
        sigma=4.221,
        left_wins_above=True,
    )
    assert normalized * 2 == round(normalized * 2)


def test_incomplete_quote_is_not_normalized():
    normalized = center_total_lines(
        pd.Series([8.5]), pd.Series([-110.0]), pd.Series([None])
    )
    assert pd.isna(normalized.iloc[0])
