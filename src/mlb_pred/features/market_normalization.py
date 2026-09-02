"""Restate two-way MLB markets at an equal-price ``-110/-110`` line.

A sportsbook can express the same market view by moving either the line or the
price.  These helpers remove the two-way vig and shift totals/run lines to the
half-run quote they would carry if both sides paid the same price.  The raw,
executable quote remains separate in the closing-line feature dataset.

Integer lines need special treatment because an exact result can push.  Their
quoted probability is conditional on the game not landing on the line, so the
fair center is solved by monotone bisection rather than a direct quantile.
"""

from __future__ import annotations

from math import floor, isclose
from statistics import NormalDist

import numpy as np
import pandas as pd

# Measured against realised MLB outcomes and closing SBR quotes.  They are
# deliberately market-specific: total runs and home margin have different
# residual distributions.  See docs/mlb-odds-architecture.md.
TOTAL_RUNS_SIGMA = 4.221
RUN_LINE_MARGIN_SIGMA = 4.719

QUOTE_INCREMENT = 0.5
CENTERED_AMERICAN_PRICE = -110.0

_STANDARD_NORMAL = NormalDist()
_BISECTION_ITERATIONS = 100
_BISECTION_HALF_WIDTH_SIGMAS = 10.0


def american_to_implied_probability(prices: pd.Series) -> pd.Series:
    """Convert American prices to their vigged implied probabilities."""
    values = pd.to_numeric(prices, errors="coerce").astype("float64")
    probabilities = pd.Series(np.nan, index=values.index, dtype="float64")
    positive = values >= 100.0
    negative = values <= -100.0
    probabilities.loc[positive] = 100.0 / (values.loc[positive] + 100.0)
    probabilities.loc[negative] = -values.loc[negative] / (
        -values.loc[negative] + 100.0
    )
    return probabilities


def devig_two_way_series(left_price: pd.Series, right_price: pd.Series) -> pd.DataFrame:
    """Return fair side probabilities and the bookmaker overround.

    Both prices are required.  An incomplete quote remains missing rather than
    being converted into a one-sided probability estimate.
    """
    raw_left = american_to_implied_probability(left_price)
    raw_right = american_to_implied_probability(right_price)
    probability_sum = raw_left + raw_right
    valid = probability_sum.gt(0.0)

    fair_left = (raw_left / probability_sum).where(valid)
    fair_right = (raw_right / probability_sum).where(valid)
    overround = (probability_sum - 1.0).where(valid)
    return pd.DataFrame(
        {
            "fair_left": fair_left,
            "fair_right": fair_right,
            "overround": overround,
        },
        index=left_price.index,
    )


def round_to_increment_signed(value: float, increment: float = 0.5) -> float:
    """Round half away from zero to preserve a mirrored signed market."""
    if increment <= 0.0 or not np.isfinite(increment):
        raise ValueError("increment must be finite and positive.")
    if not np.isfinite(value):
        raise ValueError("value must be finite.")
    magnitude = floor(abs(value) / increment + 0.5) * increment
    return float(np.copysign(magnitude, value))


def _integer_line_center(
    line: float, probability_wins_above: float, sigma: float
) -> float:
    """Solve the push-aware center for an integer outcome line."""

    def conditional_win_probability(center: float) -> float:
        wins_above = 1.0 - _STANDARD_NORMAL.cdf((line + 0.5 - center) / sigma)
        wins_below = _STANDARD_NORMAL.cdf((line - 0.5 - center) / sigma)
        return wins_above / (wins_above + wins_below)

    low = line - _BISECTION_HALF_WIDTH_SIGMAS * sigma
    high = line + _BISECTION_HALF_WIDTH_SIGMAS * sigma
    for _ in range(_BISECTION_ITERATIONS):
        midpoint = (low + high) / 2.0
        if conditional_win_probability(midpoint) < probability_wins_above:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def estimate_equal_price_line(
    line: float,
    left_price: float,
    right_price: float,
    *,
    sigma: float,
    left_wins_above: bool,
    quote_increment: float = QUOTE_INCREMENT,
) -> float:
    """Estimate the equal-price line for one complete American-odds quote.

    ``line`` is expressed in canonical outcome space.  For totals, the left
    (OVER) side wins above it.  For a run line, ``line`` is the expected home
    margin while the left (AWAY) side covers below it, so ``left_wins_above``
    must be false.
    """
    numeric_line = float(line)
    if not np.isfinite(numeric_line):
        raise ValueError("line must be finite.")
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive.")

    fair = devig_two_way_series(pd.Series([left_price]), pd.Series([right_price]))
    probability = fair.at[0, "fair_left" if left_wins_above else "fair_right"]
    if pd.isna(probability) or not 0.0 < probability < 1.0:
        raise ValueError("both prices must be valid American odds.")

    if isclose(numeric_line, round(numeric_line), abs_tol=1e-10):
        center = _integer_line_center(numeric_line, probability, sigma)
    else:
        center = numeric_line + sigma * _STANDARD_NORMAL.inv_cdf(probability)
    return round_to_increment_signed(center, quote_increment)


def center_two_way_lines(
    lines: pd.Series,
    left_prices: pd.Series,
    right_prices: pd.Series,
    *,
    sigma: float,
    left_wins_above: bool,
    quote_increment: float = QUOTE_INCREMENT,
) -> pd.Series:
    """Vector-shaped equal-price normalization with exact cached scalars.

    Sportsbook prices and half-run lines repeat heavily.  Normalising each
    distinct ``(line, left price, right price)`` tuple once keeps the exact
    push-aware scalar calculation while avoiding a row-by-row bisection over
    the complete historical store.
    """
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive.")

    quote = pd.DataFrame(
        {
            "line": pd.to_numeric(lines, errors="coerce"),
            "left_price": pd.to_numeric(left_prices, errors="coerce"),
            "right_price": pd.to_numeric(right_prices, errors="coerce"),
        },
        index=lines.index,
    )
    usable = quote.notna().all(axis=1)
    centered = pd.Series(np.nan, index=quote.index, dtype="float64")
    if not usable.any():
        return centered

    distinct = quote.loc[usable].drop_duplicates()
    normalized_values: list[float] = []
    for row in distinct.itertuples(index=False):
        try:
            normalized_values.append(
                estimate_equal_price_line(
                    row.line,
                    row.left_price,
                    row.right_price,
                    sigma=sigma,
                    left_wins_above=left_wins_above,
                    quote_increment=quote_increment,
                )
            )
        except (TypeError, ValueError, ZeroDivisionError):
            normalized_values.append(np.nan)
    distinct = distinct.assign(normalized_line=normalized_values)

    mapped = quote.loc[usable].merge(
        distinct,
        on=["line", "left_price", "right_price"],
        how="left",
        sort=False,
    )
    centered.loc[usable] = mapped["normalized_line"].to_numpy()
    return centered


def center_total_lines(
    lines: pd.Series, over_prices: pd.Series, under_prices: pd.Series
) -> pd.Series:
    """Normalize totals; the left/OVER side wins above the quoted line."""
    return center_two_way_lines(
        lines,
        over_prices,
        under_prices,
        sigma=TOTAL_RUNS_SIGMA,
        left_wins_above=True,
    )


def center_run_lines(
    home_margin_lines: pd.Series,
    away_prices: pd.Series,
    home_prices: pd.Series,
) -> pd.Series:
    """Normalize run lines in canonical home-margin space.

    SBR's left side is AWAY and its left line is the home-margin threshold.  An
    away cover occurs below that threshold, hence ``left_wins_above=False``.
    """
    return center_two_way_lines(
        home_margin_lines,
        away_prices,
        home_prices,
        sigma=RUN_LINE_MARGIN_SIGMA,
        left_wins_above=False,
    )
