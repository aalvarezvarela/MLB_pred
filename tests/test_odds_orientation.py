"""The left/right convention, pinned.

Any sign or side convention that is not enforced by a type must be (a)
measured against realised outcomes and (b) pinned by a test. It is the single
easiest place to lose months of work to a silent sign flip, because a flipped
orientation produces perfectly well-formed numbers.

**Measured on this repo's own data** -- 196 MLB games across 13 slates of the
2025 season, 36,015 ticks, 100% resolved, closing quote per (game, market,
book):

    RUN LINE   right = HOME:  residual std 4.719
               right = AWAY:  residual std 4.908     <- worse, as required
               raw home-margin std 4.588
               corr(-right_line, home_margin) = +0.068

    MONEYLINE  mean devigged fair_right (home) = 0.5409
               realised home win rate          = 0.5495     <- tracks it
               mean overround                  = 0.0438

    TOTALS     both sides carry one number (structural)
               mean devigged fair_left (OVER)  = 0.5012
               corr(fair_left, went_over)      = +0.042

**The moneyline is the decisive test for MLB, not the run line.** A baseball
run line is a near-constant ±1.5 handicap, so it carries almost no information
about margin -- which is why its residual (4.719) is *larger* than the raw
margin std (4.588) and why the two orientations separate only weakly. The NBA
spread genuinely forecasts margin (13.46 against a raw 15.57) and discriminates
sharply; the MLB run line does not, and the information sits in the price
instead. Anyone re-verifying this convention should lean on the moneyline
measurement, where a flip would read 0.459 against a realised 0.550.

The tests below split into two kinds: structural ones that run offline against
a captured payload, and a live one (``-m live``) that re-measures against real
outcomes.
"""

import json
from pathlib import Path

import pytest

from mlb_pred.fetch_data.sbr.line_history import (
    MARKET_MONEYLINE,
    MARKET_RUN_LINE,
    MARKET_TOTALS,
    parse_line_history_payload,
)
from mlb_pred.odds.encoding import devig_two_way

FIXTURE = Path(__file__).parent / "fixtures" / "sbr_line_history_343244.json"

#: Per-market outcome-distribution parameters, measured on the sample above.
#: Any line/price normalisation needs these, and using one market's for
#: another is a real error -- they differ by half a run.
MEASURED_TOTALS_SIGMA = 4.221
MEASURED_RUN_LINE_SIGMA = 4.719


@pytest.fixture(scope="module")
def game():
    return parse_line_history_payload(json.loads(FIXTURE.read_text()))


# ---------------------------------------------------------------------------
# Structural invariants -- these hold on every row, offline
# ---------------------------------------------------------------------------
def test_totals_invariant_both_sides_carry_one_number(game):
    ticks = game.ticks_for(MARKET_TOTALS)
    assert ticks
    for tick in ticks:
        assert tick.left_line == tick.right_line


def test_run_line_invariant_sides_are_mirrored(game):
    ticks = game.ticks_for(MARKET_RUN_LINE)
    assert ticks
    for tick in ticks:
        assert tick.left_line == -tick.right_line


def test_moneyline_invariant_no_line_exists(game):
    ticks = game.ticks_for(MARKET_MONEYLINE)
    assert ticks
    for tick in ticks:
        assert tick.left_line is None and tick.right_line is None


def test_the_home_favourite_is_priced_on_the_right(game):
    """The Yankees were heavy home favourites in the fixture game.

    right = HOME, so the right price must be the shorter one. A flipped
    convention would put the favourite on the left.
    """
    ticks = [
        t
        for t in game.ticks_for(MARKET_MONEYLINE)
        if t.left_price is not None and t.right_price is not None
    ]
    assert ticks
    for tick in ticks:
        fair_left, fair_right, _ = devig_two_way(tick.left_price, tick.right_price)
        assert fair_right > fair_left, "home favourite must price on the right"


def test_the_home_favourite_lays_the_run_line(game):
    """A home favourite is laying -1.5, so the right (home) line is negative."""
    ticks = [t for t in game.ticks_for(MARKET_RUN_LINE) if t.right_line is not None]
    assert ticks
    for tick in ticks:
        assert tick.right_line < 0
        assert tick.left_line > 0


def test_measured_sigmas_are_recorded_per_market():
    # Normalising a total with the run line's sigma (or vice versa) is a real
    # error: they differ by half a run.
    assert MEASURED_TOTALS_SIGMA != MEASURED_RUN_LINE_SIGMA
    assert 3.0 < MEASURED_TOTALS_SIGMA < 6.0
    assert 3.0 < MEASURED_RUN_LINE_SIGMA < 6.0


# ---------------------------------------------------------------------------
# The live re-measurement
# ---------------------------------------------------------------------------
@pytest.mark.live
def test_orientation_still_holds_against_realised_outcomes():
    """Re-measure the convention against this repo's own final scores.

    Deliberately hits the network and the local store. If SBR ever swaps its
    home/away payload keys, this is what catches it -- no column-name check
    downstream ever would.
    """
    from datetime import date, timedelta

    import numpy as np

    from mlb_pred.fetch_data.sbr.client import new_session
    from mlb_pred.fetch_data.sbr.line_history import scrape_dates
    from mlb_pred.local_store.parquet_store import read_table
    from mlb_pred.odds.encoding import decode_line
    from mlb_pred.odds.ingest import build_odds_frames

    games = read_table("games", partitions=[2025])
    if games.empty:
        pytest.skip("games table has no 2025 season")

    home_col = "home_score" if "home_score" in games.columns else "home_runs"
    away_col = "away_score" if "away_score" in games.columns else "away_runs"

    days = [date(2025, 5, 2) + timedelta(days=21 * i) for i in range(4)]
    session = new_session()
    try:
        scraped = list(scrape_dates(days, session=session))
    finally:
        session.close()
    if len(scraped) < 20:
        pytest.skip("too few games scraped to measure")

    _, ticks, _, _ = build_odds_frames(scraped, games)
    pregame = ticks[ticks["is_pregame"]]
    closing = (
        pregame.sort_values("line_ts")
        .groupby(["game_pk", "market", "book_slug"], as_index=False)
        .tail(1)
    )

    outcomes = games[["game_pk", home_col, away_col]].copy()
    outcomes["home_margin"] = outcomes[home_col] - outcomes[away_col]
    merged = closing.merge(outcomes, on="game_pk")

    # --- run line: right = HOME ------------------------------------------
    run_line = merged[
        (merged["market"] == MARKET_RUN_LINE) & merged["right_line"].notna()
    ].copy()
    assert len(run_line) > 100
    right = run_line["right_line"].map(decode_line)
    left = run_line["left_line"].map(decode_line)
    assert np.allclose(left, -right), "run-line sides must stay mirrored"

    assumed = (run_line["home_margin"] + right).std()
    flipped = (run_line["home_margin"] + left).std()
    assert assumed < flipped, (
        f"run-line orientation looks flipped: right=HOME residual {assumed:.3f} "
        f"is not tighter than right=AWAY {flipped:.3f}"
    )

    # --- moneyline: the decisive test for MLB -----------------------------
    moneyline = merged[
        (merged["market"] == MARKET_MONEYLINE)
        & merged["left_price"].notna()
        & merged["right_price"].notna()
    ].copy()
    assert len(moneyline) > 100

    fair_right = moneyline.apply(
        lambda r: devig_two_way(r["left_price"], r["right_price"])[1], axis=1
    )
    realised_home_win_rate = (moneyline["home_margin"] > 0).mean()

    assert fair_right.mean() > 0.5, "home side must carry the majority probability"
    # A flip would put these ~0.09 apart on opposite sides of 0.5.
    assert abs(fair_right.mean() - realised_home_win_rate) < 0.06

    # --- totals: left = OVER ----------------------------------------------
    totals = merged[(merged["market"] == MARKET_TOTALS) & merged["left_line"].notna()]
    assert (totals["left_line"] == totals["right_line"]).all()
