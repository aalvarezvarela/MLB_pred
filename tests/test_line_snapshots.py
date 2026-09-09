"""Acceptance tests for the pre-game snapshot panel.

The load-bearing one is ``test_horizon_zero_reproduces_the_shipped_closing_line``:
``minutes_before_start`` counts backwards, so reading it the wrong way round
returns each series' opener at every horizon.  That failure produces a panel
that is internally consistent, passes every column-name check, and reports a
market which never moves.  Only a comparison against an independently built
closing line catches it, which is why that test exists and why it is pinned
against the shipped feature partition rather than against this module's own
output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlb_pred.features.closing_lines import (
    MARKET_MONEY_LINE,
    MARKET_RUN_LINE,
    MARKET_TOTALS,
    STABLE_BOOKS,
)
from mlb_pred.features.line_snapshots import (
    DEFAULT_SNAPSHOT_GRID,
    build_snapshot_panel,
    eligible_pregame_ticks,
    market_level,
    resolve_line,
    schedule_reliability,
    snapshot_coverage,
)
from mlb_pred.odds.encoding import LINE_SCALE

CLOSING_LINES_DIR = Path("data/features/closing_lines")


def _tick(
    game_pk: str,
    market: str,
    book: str,
    minutes_before_start: float,
    *,
    left_line: float | None,
    left_price: float,
    right_price: float,
) -> dict:
    """One stored tick.  ``mins_to_tip`` is negative before first pitch."""
    right_line = None
    if left_line is not None:
        right_line = left_line if market == MARKET_TOTALS else -left_line
    return {
        "game_pk": game_pk,
        "market": market,
        "book_slug": book,
        "line_ts": pd.Timestamp("2024-06-01", tz="UTC")
        - pd.Timedelta(float(minutes_before_start), unit="m"),
        "mins_to_tip": -minutes_before_start,
        "is_pregame": True,
        "left_line": None if left_line is None else left_line * LINE_SCALE,
        "right_line": None if right_line is None else right_line * LINE_SCALE,
        "left_price": left_price,
        "right_price": right_price,
    }


@pytest.fixture
def three_tick_totals() -> pd.DataFrame:
    """One book's totals path: 8.5 at 800 minutes, 9.0 at 400, 9.5 at 100."""
    return pd.DataFrame(
        [
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                800,
                left_line=8.5,
                left_price=-110,
                right_price=-110,
            ),
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                400,
                left_line=9.0,
                left_price=-110,
                right_price=-110,
            ),
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                100,
                left_line=9.5,
                left_price=-110,
                right_price=-110,
            ),
        ]
    )


def _level_at(panel: pd.DataFrame, snapshot_minutes: int) -> float:
    rows = panel.loc[panel["snapshot_minutes"] == snapshot_minutes]
    assert len(rows) == 1
    return float(rows["level"].iloc[0])


# ---------------------------------------------------------------------------
# T2 -- no tick after the horizon may influence the snapshot
# ---------------------------------------------------------------------------
def test_a_snapshot_reads_the_latest_tick_at_or_before_its_horizon(
    three_tick_totals: pd.DataFrame,
) -> None:
    """The 360-minute snapshot must see the 400-minute tick, not the 800.

    This is the direct test of the sign convention at unit scale: reading
    ``minutes_before_start`` the wrong way round returns 8.5 here, which is the
    opener and looks entirely plausible.
    """
    panel = build_snapshot_panel(three_tick_totals, grid=(0, 360, 720))

    assert _level_at(panel, 360) == pytest.approx(9.0)
    assert _level_at(panel, 720) == pytest.approx(8.5)
    assert _level_at(panel, 0) == pytest.approx(9.5)


def test_ticks_after_the_horizon_cannot_change_it(
    three_tick_totals: pd.DataFrame,
) -> None:
    """Editing a later tick must leave an earlier snapshot untouched.

    Asserts the cutoff by construction rather than by reading the filter: a
    look-ahead would have to change this value.
    """
    before = build_snapshot_panel(three_tick_totals, grid=(360,))

    tampered = three_tick_totals.copy()
    tampered.loc[tampered["mins_to_tip"] == -100, "left_line"] = 20.0 * LINE_SCALE
    tampered.loc[tampered["mins_to_tip"] == -100, "right_line"] = 20.0 * LINE_SCALE
    after = build_snapshot_panel(tampered, grid=(360,))

    assert _level_at(before, 360) == _level_at(after, 360)


# ---------------------------------------------------------------------------
# T3 -- the boundary is inclusive on one side only
# ---------------------------------------------------------------------------
def test_a_tick_exactly_at_the_horizon_is_admissible() -> None:
    """Observable at the horizon; one minute later it was not.

    Both cases are asserted together because a boundary that is inclusive on
    both sides is a one-minute look-ahead, and one that is exclusive on both
    silently discards the tick a horizon most often lands on.
    """
    ticks = pd.DataFrame(
        [
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                700,
                left_line=8.0,
                left_price=-110,
                right_price=-110,
            ),
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                360,
                left_line=9.0,
                left_price=-110,
                right_price=-110,
            ),
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                359,
                left_line=10.0,
                left_price=-110,
                right_price=-110,
            ),
        ]
    )
    panel = build_snapshot_panel(ticks, grid=(360,))

    assert _level_at(panel, 360) == pytest.approx(9.0)


def test_line_age_reports_how_stale_the_carried_quote_is(
    three_tick_totals: pd.DataFrame,
) -> None:
    """A snapshot carries a previous tick forward; its age is information."""
    panel = build_snapshot_panel(three_tick_totals, grid=(360,))

    assert float(panel["tick_minutes_before_start"].iloc[0]) == 400.0
    assert float(panel["line_age_minutes"].iloc[0]) == 40.0
    assert float(panel["n_ticks_so_far"].iloc[0]) == 2.0


def test_a_horizon_before_the_first_tick_produces_no_row(
    three_tick_totals: pd.DataFrame,
) -> None:
    """No quote existed yet, so there is nothing to carry forward."""
    panel = build_snapshot_panel(three_tick_totals, grid=(360, 5000))

    assert set(panel["snapshot_minutes"]) == {360}


def test_a_negative_horizon_is_refused(three_tick_totals: pd.DataFrame) -> None:
    """It would read the market after first pitch and admit in-play ticks."""
    with pytest.raises(ValueError, match="non-negative"):
        build_snapshot_panel(three_tick_totals, grid=(0, -30))


def test_an_empty_grid_is_refused(three_tick_totals: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_snapshot_panel(three_tick_totals, grid=())


# ---------------------------------------------------------------------------
# Market orientation
# ---------------------------------------------------------------------------
def test_the_run_line_level_is_the_home_cover_probability() -> None:
    """The handicap is pinned at +/-1.5, so the level must be the price.

    Using the handicap would give the market a near-constant path and zero every
    move count and dispersion figure derived from it.  The handicap is still
    emitted, as ``home_handicap``.
    """
    ticks = pd.DataFrame(
        [
            _tick(
                "1",
                MARKET_RUN_LINE,
                "bet365",
                400,
                left_line=1.5,
                left_price=-140,
                right_price=120,
            ),
        ]
    )
    panel = build_snapshot_panel(ticks, grid=(360,))
    row = panel.iloc[0]

    # left = AWAY, so the away side is favoured to cover here and the home
    # cover probability sits below one half.
    assert float(row["level"]) == pytest.approx(float(row["fair_right"]))
    assert float(row["level"]) < 0.5
    # raw_line is the home-margin threshold; the handicap is its negation.
    assert float(row["raw_line"]) == pytest.approx(1.5)
    assert float(row["home_handicap"]) == pytest.approx(-1.5)


def test_the_money_line_level_is_the_home_win_probability() -> None:
    """It carries no line at all, so without this it would have no level."""
    ticks = pd.DataFrame(
        [
            _tick(
                "1",
                MARKET_MONEY_LINE,
                "bet365",
                400,
                left_line=None,
                left_price=150,
                right_price=-170,
            ),
        ]
    )
    panel = build_snapshot_panel(ticks, grid=(360,))
    row = panel.iloc[0]

    assert pd.isna(row["raw_line"])
    assert float(row["level"]) == pytest.approx(float(row["fair_right"]))
    assert float(row["level"]) > 0.5
    assert float(row["has_quote"]) == 1.0
    # The raw prices survive: a normalised line is comparable, but only the
    # quoted price is executable.
    assert float(row["price_left"]) == 150.0
    assert float(row["price_right"]) == -170.0


def test_the_totals_level_is_the_raw_line_and_up_means_over() -> None:
    ticks = pd.DataFrame(
        [
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                400,
                left_line=8.5,
                left_price=-105,
                right_price=-115,
            ),
        ]
    )
    panel = build_snapshot_panel(ticks, grid=(360,))
    row = panel.iloc[0]

    assert float(row["level"]) == pytest.approx(8.5)
    assert float(row["fair_up"]) == pytest.approx(float(row["fair_left"]))
    assert pd.isna(row["home_handicap"])


def test_resolve_line_falls_back_to_the_mirrored_side() -> None:
    """A repaired row may carry a valid price with only one line populated."""
    ticks = pd.DataFrame(
        [
            _tick(
                "1",
                MARKET_RUN_LINE,
                "bet365",
                400,
                left_line=1.5,
                left_price=-140,
                right_price=120,
            ),
        ]
    )
    ticks.loc[0, "left_line"] = None

    assert float(resolve_line(ticks).iloc[0]) == pytest.approx(1.5)


def test_the_half_tick_priced_but_not_taken_is_carried_explicitly() -> None:
    """A book re-prices the number before it moves it.

    ``norm_minus_raw`` is the only place that pressure is visible; the line
    itself has not moved.
    """
    ticks = pd.DataFrame(
        [
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                400,
                left_line=8.5,
                left_price=-135,
                right_price=115,
            ),
        ]
    )
    panel = build_snapshot_panel(ticks, grid=(360,))
    row = panel.iloc[0]

    assert float(row["raw_line"]) == pytest.approx(8.5)
    # The over is priced dearly, so the equal-price line sits above the board.
    assert float(row["norm_line"]) > float(row["raw_line"])
    assert float(row["norm_minus_raw"]) == pytest.approx(
        float(row["norm_line"]) - float(row["raw_line"])
    )


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
def test_in_play_and_one_sided_ticks_are_never_eligible() -> None:
    """Two independent hazards, asserted together.

    SBR records in-play ticks with the same shape as pre-game ones, and a quote
    missing a price on either side is not executable and therefore not a quote.
    """
    ticks = pd.DataFrame(
        [
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                400,
                left_line=8.5,
                left_price=-110,
                right_price=-110,
            ),
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                -30,
                left_line=12.0,
                left_price=-110,
                right_price=-110,
            ),
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                200,
                left_line=9.5,
                left_price=-110,
                right_price=np.nan,
            ),
        ]
    )
    ticks.loc[1, "is_pregame"] = False

    eligible = eligible_pregame_ticks(ticks)

    assert len(eligible) == 1
    assert float(eligible["minutes_before_start"].iloc[0]) == 400.0


def test_the_safety_margin_keeps_horizon_zero_off_first_pitch() -> None:
    """Horizon zero is "as late as the store allows", not literally first pitch."""
    ticks = pd.DataFrame(
        [
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                400,
                left_line=8.5,
                left_price=-110,
                right_price=-110,
            ),
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                2,
                left_line=11.0,
                left_price=-110,
                right_price=-110,
            ),
        ]
    )
    panel = build_snapshot_panel(ticks, grid=(0,), safety_margin_minutes=5)

    assert _level_at(panel, 0) == pytest.approx(8.5)


def test_books_may_be_restricted_but_default_to_all() -> None:
    """Which books earn their own columns is the pivot step's decision."""
    ticks = pd.DataFrame(
        [
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                400,
                left_line=8.5,
                left_price=-110,
                right_price=-110,
            ),
            _tick(
                "1",
                MARKET_TOTALS,
                "betmgm",
                400,
                left_line=9.0,
                left_price=-110,
                right_price=-110,
            ),
        ]
    )

    assert len(build_snapshot_panel(ticks, grid=(360,))) == 2
    assert len(build_snapshot_panel(ticks, grid=(360,), books=("bet365",))) == 1


def test_an_empty_tick_frame_keeps_the_panel_schema() -> None:
    """A season with no odds must not reshape the partition."""
    empty = pd.DataFrame(
        columns=[
            "game_pk",
            "market",
            "book_slug",
            "line_ts",
            "mins_to_tip",
            "is_pregame",
            "left_line",
            "left_price",
            "right_line",
            "right_price",
        ]
    )
    panel = build_snapshot_panel(empty, grid=DEFAULT_SNAPSHOT_GRID)

    assert panel.empty
    assert "level" in panel.columns
    assert snapshot_coverage(panel).empty


def test_market_level_is_read_from_one_column_per_market() -> None:
    """Guards the mapping itself, independently of the panel builder."""
    frame = pd.DataFrame(
        {
            "market": [MARKET_TOTALS, MARKET_RUN_LINE, MARKET_MONEY_LINE],
            "raw_line": [8.5, 1.5, np.nan],
            "fair_right": [0.48, 0.42, 0.55],
        }
    )

    assert list(market_level(frame)) == [8.5, 0.42, 0.55]


# ---------------------------------------------------------------------------
# T1 -- the reconstruction test.  Needs the real store.
# ---------------------------------------------------------------------------
@pytest.mark.store
def test_horizon_zero_reproduces_the_shipped_closing_line() -> None:
    """The panel at horizon zero *is* the closing line, so it must match it.

    Built by a different code path -- ``closing_lines`` selects one quote per
    book directly, while this walks a horizon grid -- so agreement is evidence
    the as-of read is right rather than a tautology.

    Restricted to ``STABLE_BOOKS`` because that is what the shipped consensus
    aggregates: per-book closing columns are only emitted for those four, so the
    reference median is defined over them.  The panel deliberately defaults to
    every book, which is why the restriction is explicit here.

    Measured at 99.994% exact agreement over 17,588 games.
    """
    from mlb_pred.local_store.parquet_store import read_table

    shipped = pd.read_parquet(CLOSING_LINES_DIR)
    shipped["GAME_ID"] = shipped["GAME_ID"].astype(str)
    reference = shipped.set_index("GAME_ID")["ODDS_TOTAL_CONSENSUS_LINE_RAW_MEDIAN"]

    panel = build_snapshot_panel(
        read_table("odds_ticks"), grid=(0,), books=STABLE_BOOKS
    )
    totals = panel.loc[panel["market"] == MARKET_TOTALS]
    rebuilt = totals.groupby("game_pk")["raw_line"].median()

    compared = pd.concat(
        [rebuilt.rename("rebuilt"), reference.rename("shipped")], axis=1
    ).dropna()

    assert len(compared) > 17_000
    agreement = np.isclose(compared["rebuilt"], compared["shipped"]).mean()
    assert agreement >= 0.999, f"horizon-zero agreement fell to {agreement:.5f}"


@pytest.mark.store
def test_coverage_does_not_fall_off_a_cliff_across_the_grid() -> None:
    """A horizon that only exists for well-covered games is a biased sample.

    The far end of the grid is where that risk lives, so the floor is asserted
    rather than left to a chart nobody reads.
    """
    from mlb_pred.local_store.parquet_store import read_table

    panel = build_snapshot_panel(read_table("odds_ticks"), books=STABLE_BOOKS)
    coverage = snapshot_coverage(panel)
    totals = coverage.loc[coverage["market"] == MARKET_TOTALS].set_index(
        "snapshot_minutes"
    )["games"]

    assert totals.loc[0] > 17_000
    assert totals.loc[DEFAULT_SNAPSHOT_GRID[-1]] / totals.loc[0] >= 0.85
    # Monotone: a longer horizon can only ever be quoted by fewer games.
    assert totals.sort_index(ascending=False).is_monotonic_increasing


# ---------------------------------------------------------------------------
# T4 -- rescheduled games mislabel their own history
# ---------------------------------------------------------------------------
def test_a_rescheduled_game_is_flagged_rather_than_silently_wrong() -> None:
    """A delayed start leaves earlier ticks pointing at a first pitch that moved.

    Modelled on game 824913: quotes recorded around 17:15 UTC were stored as
    minutes from an 18:00 start, then the game began at 23:15.  Those early
    horizons are meaningless, and nothing in the tick's own shape says so.
    """
    ticks = pd.DataFrame(
        [
            # Labelled against the original start.
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                45,
                left_line=21.0,
                left_price=143,
                right_price=-199,
            ),
            # Labelled against the real one, five hours later on the clock.
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                18,
                left_line=9.0,
                left_price=-110,
                right_price=-110,
            ),
        ]
    )
    ticks.loc[1, "line_ts"] = ticks.loc[0, "line_ts"] + pd.Timedelta(5, unit="h")

    panel = build_snapshot_panel(ticks, grid=(0,))

    assert float(panel["game_start_is_reliable"].iloc[0]) == 0.0
    # Ordering follows the wall clock, so the genuinely last quote wins even
    # though the stale tick claims to sit closer to first pitch.
    assert _level_at(panel, 0) == pytest.approx(9.0)


def test_a_normal_game_is_reported_reliable(
    three_tick_totals: pd.DataFrame,
) -> None:
    """Every tick implies the same first pitch, so nothing moved."""
    panel = build_snapshot_panel(three_tick_totals, grid=(0, 360))

    assert set(panel["game_start_is_reliable"]) == {1.0}


def test_schedule_reliability_is_decided_per_game() -> None:
    """One delayed game must not condemn the rest of the slate."""
    ticks = pd.DataFrame(
        [
            _tick(
                "clean",
                MARKET_TOTALS,
                "bet365",
                400,
                left_line=8.5,
                left_price=-110,
                right_price=-110,
            ),
            _tick(
                "clean",
                MARKET_TOTALS,
                "bet365",
                100,
                left_line=9.0,
                left_price=-110,
                right_price=-110,
            ),
            _tick(
                "delayed",
                MARKET_TOTALS,
                "bet365",
                45,
                left_line=21.0,
                left_price=143,
                right_price=-199,
            ),
            _tick(
                "delayed",
                MARKET_TOTALS,
                "bet365",
                18,
                left_line=9.0,
                left_price=-110,
                right_price=-110,
            ),
        ]
    )
    ticks.loc[3, "line_ts"] = ticks.loc[2, "line_ts"] + pd.Timedelta(5, unit="h")

    reliability = schedule_reliability(eligible_pregame_ticks(ticks))

    assert float(reliability.loc["clean"]) == 1.0
    assert float(reliability.loc["delayed"]) == 0.0


@pytest.mark.store
def test_schedule_drift_stays_rare_in_the_real_store() -> None:
    """Measured at 0.49% of games; a jump means the odds ingest changed.

    Asserted as a ceiling rather than reported, because the failure it guards
    against -- horizons silently referring to a first pitch that never happened
    -- is invisible in every other check.
    """
    from mlb_pred.local_store.parquet_store import read_table

    working = eligible_pregame_ticks(read_table("odds_ticks"))
    reliability = schedule_reliability(working)

    unreliable = 1.0 - reliability.mean()
    assert unreliable < 0.02, f"{unreliable:.4f} of games have drifting horizons"
