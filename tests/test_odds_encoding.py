"""The line encoding must be exact or fail loudly.

Getting a lossy encoding into millions of rows is expensive to undo, so the
guard matters more than the compactness.
"""

import pytest

from mlb_pred.odds.encoding import (
    OFF_THE_BOARD_SENTINEL,
    LineEncodingError,
    american_to_probability,
    decode_line,
    decode_price,
    devig_two_way,
    encode_line,
    encode_price,
)


@pytest.mark.parametrize(
    "value", [8.5, 9.0, 4.5, -1.5, 1.5, -3.5, 2.0, 0.0, -2.5, 10.5, 12.0]
)
def test_every_observed_mlb_quote_round_trips_exactly(value):
    assert decode_line(encode_line(value)) == value


def test_half_points_encode_to_odd_integers():
    assert encode_line(8.5) == 17
    assert encode_line(-1.5) == -3
    assert encode_line(9.0) == 18


@pytest.mark.parametrize("value", [8.25, 8.75, 1.1, -1.25])
def test_a_quarter_point_quote_raises_instead_of_rounding(value):
    # If SBR ever quotes a finer increment this must break the load, not
    # silently store the wrong number.
    with pytest.raises(LineEncodingError):
        encode_line(value)


def test_none_passes_through():
    assert encode_line(None) is None
    assert decode_line(None) is None
    assert encode_price(None) is None
    assert decode_price(None) is None


def test_off_the_board_sentinel_becomes_null_at_the_boundary():
    # Left in, -100000 survives into every mean, std and devig.
    assert encode_price(OFF_THE_BOARD_SENTINEL) is None


@pytest.mark.parametrize("price", [-110, -105, 100, 115, -275, 230, -1500])
def test_real_prices_survive(price):
    assert encode_price(price) == price


@pytest.mark.parametrize("price", [0, 50, -50, 99, -99])
def test_impossible_american_prices_are_nulled(price):
    # No American price lies strictly between -100 and +100.
    assert encode_price(price) is None


def test_american_to_probability():
    assert american_to_probability(-110) == pytest.approx(110 / 210)
    assert american_to_probability(100) == pytest.approx(0.5)
    assert american_to_probability(150) == pytest.approx(100 / 250)


def test_devig_removes_the_overround_and_reports_it():
    fair_left, fair_right, overround = devig_two_way(-110, -110)
    assert fair_left == pytest.approx(0.5)
    assert fair_right == pytest.approx(0.5)
    assert overround == pytest.approx(0.0476, abs=1e-3)


def test_devig_is_asymmetric_when_prices_are():
    fair_left, fair_right, _ = devig_two_way(-275, 220)
    assert fair_left > fair_right
    assert fair_left + fair_right == pytest.approx(1.0)


def test_devig_needs_both_sides():
    assert devig_two_way(-110, None) == (None, None, None)
