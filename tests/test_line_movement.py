"""Acceptance tests for the movement and cross-book layers.

The tests that matter most are the two directional ones.  A trailing window is
answered by an as-of read at ``T + w``, so a sign error there reads the panel at
a *later* moment than the snapshot -- a look-ahead that no column-name check can
see, because the column is still called ``move_last_60``.  ``T5`` pins the
direction against a hand-built path where the right answer is arithmetic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlb_pred.features.closing_lines import (
    MARKET_MONEY_LINE,
    MARKET_RUN_LINE,
    MARKET_TOTALS,
)
from mlb_pred.features.line_cross_book import (
    STEAM_WINDOW_MINUTES,
    add_book_deviation,
    aggregate_across_books,
    steam_move_column,
)
from mlb_pred.features.line_movement import (
    DEFAULT_MOVEMENT_WINDOWS,
    build_movement_panel,
    extended_grid,
    prepare_tick_path,
)
from mlb_pred.features.line_snapshots import prepare_snapshot_ticks
from mlb_pred.odds.encoding import LINE_SCALE


def _tick(
    game_pk: str,
    market: str,
    book: str,
    minutes_before_start: float,
    *,
    left_line: float | None,
    left_price: float = -110,
    right_price: float = -110,
) -> dict:
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
def stepped_path() -> pd.DataFrame:
    """One book, one game: 8.0 -> 8.5 -> 9.0 -> 8.5, at known distances out."""
    return pd.DataFrame(
        [
            _tick("1", MARKET_TOTALS, "bet365", 900, left_line=8.0),
            _tick("1", MARKET_TOTALS, "bet365", 700, left_line=8.5),
            _tick("1", MARKET_TOTALS, "bet365", 400, left_line=9.0),
            _tick("1", MARKET_TOTALS, "bet365", 200, left_line=8.5),
        ]
    )


def _row(panel: pd.DataFrame, snapshot_minutes: int) -> pd.Series:
    rows = panel.loc[panel["snapshot_minutes"] == snapshot_minutes]
    assert len(rows) == 1
    return rows.iloc[0]


# ---------------------------------------------------------------------------
# T5 -- a window looks BACKWARD, and the arithmetic says which way
# ---------------------------------------------------------------------------
def test_a_trailing_window_measures_the_move_into_the_snapshot() -> None:
    """``move_last_w`` must equal ``level(T) - level(T + w)``.

    Built so every window has a different right answer, and so reversing the
    direction flips the sign rather than merely changing the magnitude.
    """
    ticks = pd.DataFrame(
        [
            _tick("1", MARKET_TOTALS, "bet365", 700, left_line=7.0),
            _tick("1", MARKET_TOTALS, "bet365", 500, left_line=8.0),
            _tick("1", MARKET_TOTALS, "bet365", 300, left_line=9.0),
            _tick("1", MARKET_TOTALS, "bet365", 100, left_line=9.5),
        ]
    )
    panel = build_movement_panel(ticks, grid=(120,), windows=(60, 240, 480))
    row = _row(panel, 120)

    # level(120) is the 300-minute tick's 9.0 carried forward.
    assert float(row["level"]) == pytest.approx(9.0)
    # level(180) is also 9.0 -- nothing moved in between.
    assert float(row["move_last_60"]) == pytest.approx(0.0)
    # level(360) is the 500-minute tick's 8.0.
    assert float(row["move_last_240"]) == pytest.approx(1.0)
    # level(600) is the 700-minute tick's 7.0.
    assert float(row["move_last_480"]) == pytest.approx(2.0)


def test_velocity_is_the_window_move_per_hour(stepped_path: pd.DataFrame) -> None:
    panel = build_movement_panel(stepped_path, grid=(300,), windows=(240,))
    row = _row(panel, 300)

    assert float(row["velocity_last_240"]) == pytest.approx(
        float(row["move_last_240"]) / 4.0
    )


def test_a_window_reaching_past_the_opener_is_flagged_not_faked(
    stepped_path: pd.DataFrame,
) -> None:
    """ "No history" and "no movement" are different states and must not merge."""
    panel = build_movement_panel(stepped_path, grid=(600,), windows=(60, 480))
    row = _row(panel, 600)

    assert float(row["has_window_60"]) == 1.0
    # 600 + 480 = 1080 minutes out, before this series had any quote at all.
    assert float(row["has_window_480"]) == 0.0
    assert float(row["move_last_480"]) == 0.0


def test_a_non_positive_window_is_refused(stepped_path: pd.DataFrame) -> None:
    """It would read the panel at a LATER moment than the snapshot."""
    with pytest.raises(ValueError, match="positive minutes"):
        build_movement_panel(stepped_path, windows=(60, -60))
    with pytest.raises(ValueError, match="positive minutes"):
        build_movement_panel(stepped_path, windows=(0,))


def test_an_empty_window_set_is_refused(stepped_path: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_movement_panel(stepped_path, windows=())


def test_the_extended_grid_covers_every_horizon_a_window_needs() -> None:
    assert extended_grid((0, 60), (30, 90)) == (0, 30, 60, 90, 150)


# ---------------------------------------------------------------------------
# The path accumulated so far
# ---------------------------------------------------------------------------
def test_path_counters_read_only_the_prefix(stepped_path: pd.DataFrame) -> None:
    """A snapshot must see the moves before it and none of the moves after.

    The count includes the move made *at* the carried tick: at T=650 the market
    stood at 8.5, having stepped there 700 minutes out, and that step was
    observable.  Only a horizon before the second tick sees an unmoved market.
    """
    panel = build_movement_panel(stepped_path, grid=(150, 350, 650, 800))

    assert float(_row(panel, 800)["n_moves_so_far"]) == 0.0  # opener only
    assert float(_row(panel, 650)["n_moves_so_far"]) == 1.0
    assert float(_row(panel, 350)["n_moves_so_far"]) == 2.0
    assert float(_row(panel, 150)["n_moves_so_far"]) == 3.0


def test_a_reversal_is_counted_when_the_direction_flips(
    stepped_path: pd.DataFrame,
) -> None:
    """8.0 -> 8.5 -> 9.0 is one direction; the drop to 8.5 reverses it."""
    panel = build_movement_panel(stepped_path, grid=(150, 350))

    assert float(_row(panel, 350)["n_reversals_so_far"]) == 0.0
    assert float(_row(panel, 150)["n_reversals_so_far"]) == 1.0


def test_the_opening_direction_is_the_first_move_not_the_latest(
    stepped_path: pd.DataFrame,
) -> None:
    """Forward-filling the most recent move would redefine the feature.

    The path opens upward and ends downward, so a model that confused the two
    would read this game as a market that never changed its mind.
    """
    panel = build_movement_panel(stepped_path, grid=(150,), windows=(60, 240))
    row = _row(panel, 150)

    assert float(row["first_move_direction"]) == 1.0
    assert float(row["move_direction"]) == 1.0  # net 8.0 -> 8.5 is still up
    assert float(row["move_last_240"]) < 0.0  # but the recent window is down
    assert float(row["opposes_opening_direction"]) == 1.0
    assert float(row["net_opposes_opening_direction"]) == 0.0


def test_the_running_range_and_position_describe_the_path(
    stepped_path: pd.DataFrame,
) -> None:
    panel = build_movement_panel(stepped_path, grid=(150,))
    row = _row(panel, 150)

    assert float(row["level_max_so_far"]) == pytest.approx(9.0)
    assert float(row["level_min_so_far"]) == pytest.approx(8.0)
    assert float(row["level_range_so_far"]) == pytest.approx(1.0)
    # Sitting at 8.5 in a range of 8.0-9.0 is the midpoint.
    assert float(row["position_in_range"]) == pytest.approx(0.5)


def test_a_line_that_never_moved_sits_at_the_midpoint() -> None:
    """Neither extreme is honest; the range already says it did not move."""
    ticks = pd.DataFrame(
        [
            _tick("1", MARKET_TOTALS, "bet365", 700, left_line=8.5),
            _tick("1", MARKET_TOTALS, "bet365", 400, left_line=8.5),
        ]
    )
    panel = build_movement_panel(ticks, grid=(300,))
    row = _row(panel, 300)

    assert float(row["level_range_so_far"]) == 0.0
    assert float(row["position_in_range"]) == pytest.approx(0.5)
    assert float(row["level_std_so_far"]) == pytest.approx(0.0)


def test_the_opener_is_the_earliest_tick_and_is_known_at_every_horizon(
    stepped_path: pd.DataFrame,
) -> None:
    panel = build_movement_panel(stepped_path, grid=(150, 350, 650))

    assert set(panel["opener_level"]) == {8.0}
    assert float(_row(panel, 650)["minutes_since_open"]) == 250.0
    assert float(_row(panel, 150)["move_from_open"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Price-only movement, and the run line's two kinds of move
# ---------------------------------------------------------------------------
def test_a_re_price_at_the_same_number_is_not_a_move() -> None:
    """Books price a half-tick before taking it; that is its own category."""
    ticks = pd.DataFrame(
        [
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                700,
                left_line=8.5,
                left_price=-110,
                right_price=-110,
            ),
            _tick(
                "1",
                MARKET_TOTALS,
                "bet365",
                400,
                left_line=8.5,
                left_price=-130,
                right_price=110,
            ),
        ]
    )
    panel = build_movement_panel(ticks, grid=(300,))
    row = _row(panel, 300)

    assert float(row["n_moves_so_far"]) == 0.0
    assert float(row["n_price_only_so_far"]) == 1.0


def test_the_run_line_separates_a_price_move_from_a_handicap_move() -> None:
    """Its level is a probability, so a re-price IS a move -- but the number is not.

    ``n_line_moves_so_far`` is what isolates the rare event: measured on this
    store the handicap differs from its close on 8.1% of games while the cover
    probability differs on 95.0%.
    """
    ticks = pd.DataFrame(
        [
            _tick(
                "1",
                MARKET_RUN_LINE,
                "bet365",
                700,
                left_line=1.5,
                left_price=-140,
                right_price=120,
            ),
            # Same handicap, re-priced.
            _tick(
                "1",
                MARKET_RUN_LINE,
                "bet365",
                500,
                left_line=1.5,
                left_price=-160,
                right_price=135,
            ),
            # The number itself moves.
            _tick(
                "1",
                MARKET_RUN_LINE,
                "bet365",
                300,
                left_line=2.5,
                left_price=-160,
                right_price=135,
            ),
        ]
    )
    panel = build_movement_panel(ticks, grid=(400, 200))

    priced = _row(panel, 400)
    assert float(priced["n_moves_so_far"]) == 1.0  # the probability moved
    assert float(priced["n_line_moves_so_far"]) == 0.0  # the number did not
    # Structurally impossible here: the level IS the price.
    assert float(priced["n_price_only_so_far"]) == 0.0

    moved = _row(panel, 200)
    assert float(moved["n_line_moves_so_far"]) == 1.0


def test_the_moneyline_gets_a_real_path_despite_having_no_line() -> None:
    """Without a probability level every count below would be a silent zero."""
    ticks = pd.DataFrame(
        [
            _tick(
                "1",
                MARKET_MONEY_LINE,
                "bet365",
                700,
                left_line=None,
                left_price=150,
                right_price=-170,
            ),
            _tick(
                "1",
                MARKET_MONEY_LINE,
                "bet365",
                400,
                left_line=None,
                left_price=120,
                right_price=-140,
            ),
        ]
    )
    panel = build_movement_panel(ticks, grid=(300,))
    row = _row(panel, 300)

    assert float(row["n_moves_so_far"]) == 1.0
    assert float(row["move_from_open"]) != 0.0
    assert float(row["n_line_moves_so_far"]) == 0.0


def test_prepare_tick_path_leaves_row_count_and_order_untouched(
    stepped_path: pd.DataFrame,
) -> None:
    """It annotates; it must never filter, reorder or duplicate."""
    prepared = prepare_snapshot_ticks(stepped_path)
    annotated = prepare_tick_path(prepared)

    assert len(annotated) == len(prepared)
    assert list(annotated["minutes_before_start"]) == list(
        prepared["minutes_before_start"]
    )


# ---------------------------------------------------------------------------
# Cross-book
# ---------------------------------------------------------------------------
@pytest.fixture
def four_books() -> pd.DataFrame:
    """Three books drift up into the snapshot; one sits far off and stale."""
    rows = []
    for book, early, late in (
        ("bet365", 8.5, 9.0),
        ("draftkings", 8.5, 9.0),
        ("fanduel", 8.5, 9.0),
    ):
        rows.append(_tick("1", MARKET_TOTALS, book, 400, left_line=early))
        rows.append(_tick("1", MARKET_TOTALS, book, 100, left_line=late))
    rows.append(_tick("1", MARKET_TOTALS, "caesars", 400, left_line=11.0))
    return pd.DataFrame(rows)


def test_the_consensus_is_a_median_so_a_stale_book_cannot_drag_it(
    four_books: pd.DataFrame,
) -> None:
    """A mean would sit above every book that is actually trading."""
    panel = build_movement_panel(four_books, grid=(60,), windows=(60,))
    consensus = aggregate_across_books(panel)
    row = consensus.iloc[0]

    assert float(row["n_books_quoting"]) == 4.0
    assert float(row["consensus_level"]) == pytest.approx(9.0)
    assert float(row["crossbook_range"]) == pytest.approx(2.0)
    assert float(row["crossbook_std"]) > 0.0


def test_steam_counts_books_moving_the_same_way(four_books: pd.DataFrame) -> None:
    """Three of four books moved up together; the fourth did not move at all."""
    panel = build_movement_panel(four_books, grid=(60,), windows=(60,))
    row = aggregate_across_books(panel).iloc[0]

    assert float(row["steam_books_up"]) == 3.0
    assert float(row["steam_books_down"]) == 0.0
    assert float(row["steam_movers"]) == 3.0
    assert float(row["steam_fraction"]) == pytest.approx(0.75)
    # Everyone who moved, agreed.
    assert float(row["steam_agreement"]) == pytest.approx(1.0)


def test_a_quiet_market_reports_zero_steam_not_missing_steam() -> None:
    ticks = pd.DataFrame(
        [
            _tick("1", MARKET_TOTALS, "bet365", 400, left_line=8.5),
            _tick("1", MARKET_TOTALS, "fanduel", 400, left_line=8.5),
        ]
    )
    panel = build_movement_panel(ticks, grid=(60,), windows=(60,))
    row = aggregate_across_books(panel).iloc[0]

    assert float(row["steam_movers"]) == 0.0
    assert float(row["steam_fraction"]) == 0.0
    assert float(row["steam_agreement"]) == 0.0


def test_the_steam_window_is_pinned_not_inherited(
    four_books: pd.DataFrame,
) -> None:
    """Adding a SHORTER window must not silently redefine steam.

    The NBA repo hit exactly this: steam was "the shortest configured window",
    so introducing a 15-minute window changed an existing feature's meaning
    without any code in the steam path being touched.

    The shorter window has to be genuinely shorter than
    ``STEAM_WINDOW_MINUTES`` for this to test anything.  A first version of this
    test configured ``(60, 120, 480)``, where the pinned window and the shortest
    one are the same column -- so unpinning it passed.
    """
    assert min(DEFAULT_MOVEMENT_WINDOWS) == STEAM_WINDOW_MINUTES

    panel = build_movement_panel(four_books, grid=(60,), windows=(30, 60, 480))

    assert steam_move_column(panel) == f"move_last_{STEAM_WINDOW_MINUTES}"
    assert "move_last_30" in panel.columns

    # Steam must read the same column, and report the same figures, whether or
    # not a shorter window happens to be configured alongside it.
    pinned = aggregate_across_books(panel).iloc[0]
    without_short = aggregate_across_books(
        build_movement_panel(four_books, grid=(60,), windows=(60, 480))
    ).iloc[0]
    assert float(pinned["steam_books_up"]) == float(without_short["steam_books_up"])
    assert float(pinned["steam_fraction"]) == float(without_short["steam_fraction"])


def test_the_shortest_window_is_a_documented_fallback(
    four_books: pd.DataFrame,
) -> None:
    """With the pinned window absent, fall back rather than raise a KeyError."""
    narrowed = build_movement_panel(four_books, grid=(60,), windows=(120, 480))

    assert steam_move_column(narrowed) == "move_last_120"


def test_book_deviation_is_scale_free_and_flags_the_outlier(
    four_books: pd.DataFrame,
) -> None:
    """An outlying book is either the stale one or the sharp one.

    Which of the two is the model's call, but it can only make it if the
    deviation is handed over explicitly.
    """
    panel = build_movement_panel(four_books, grid=(60,), windows=(60,))
    consensus = aggregate_across_books(panel)
    with_deviation = add_book_deviation(panel, consensus)

    caesars = with_deviation.loc[with_deviation["book_slug"] == "caesars"].iloc[0]
    bet365 = with_deviation.loc[with_deviation["book_slug"] == "bet365"].iloc[0]

    assert float(caesars["deviation_from_consensus"]) == pytest.approx(2.0)
    assert float(caesars["is_outlier_book"]) == 1.0
    assert float(bet365["deviation_from_consensus"]) == pytest.approx(0.0)
    assert float(bet365["is_outlier_book"]) == 0.0
    # The panel is not duplicated by the join.
    assert len(with_deviation) == len(panel)


def test_a_lone_quoting_book_disagrees_with_nobody() -> None:
    """Zero dispersion is the reading, not missing data."""
    ticks = pd.DataFrame([_tick("1", MARKET_TOTALS, "bet365", 400, left_line=8.5)])
    panel = build_movement_panel(ticks, grid=(300,), windows=(60,))
    consensus = aggregate_across_books(panel)

    assert float(consensus["crossbook_std"].iloc[0]) == 0.0
    assert float(consensus["crossbook_range"].iloc[0]) == 0.0
    deviation = add_book_deviation(panel, consensus)
    assert float(deviation["deviation_z"].iloc[0]) == 0.0


def test_empty_inputs_keep_the_schema() -> None:
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
    panel = build_movement_panel(empty)
    assert panel.empty
    consensus = aggregate_across_books(panel)
    assert consensus.empty
    assert "steam_net" in consensus.columns


def test_aggregating_a_panel_without_windows_fails_loudly() -> None:
    """Steam has no meaning without a move column; say so rather than KeyError."""
    with pytest.raises(ValueError, match="move_last_"):
        aggregate_across_books(pd.DataFrame({"game_pk": ["1"], "level": [8.5]}))


def test_markets_are_aggregated_separately() -> None:
    """Levels are runs on one market and probabilities on the others.

    Pooling them would produce a consensus in no units at all.
    """
    ticks = pd.DataFrame(
        [
            _tick("1", MARKET_TOTALS, "bet365", 400, left_line=8.5),
            _tick(
                "1",
                MARKET_RUN_LINE,
                "bet365",
                400,
                left_line=1.5,
                left_price=-140,
                right_price=120,
            ),
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
    panel = build_movement_panel(ticks, grid=(300,), windows=(60,))
    consensus = aggregate_across_books(panel).set_index("market")

    assert len(consensus) == 3
    assert float(consensus.loc[MARKET_TOTALS, "consensus_level"]) == pytest.approx(8.5)
    assert 0.0 < float(consensus.loc[MARKET_RUN_LINE, "consensus_level"]) < 1.0
    assert 0.0 < float(consensus.loc[MARKET_MONEY_LINE, "consensus_level"]) < 1.0


def test_default_windows_are_documented_in_the_schema() -> None:
    """The emitted schema must follow the configured windows, not a literal."""
    from mlb_pred.features.line_movement import movement_panel_columns

    columns = movement_panel_columns(DEFAULT_MOVEMENT_WINDOWS)

    for window in DEFAULT_MOVEMENT_WINDOWS:
        assert f"move_last_{window}" in columns
    assert "move_acceleration" in columns
    assert "move_last_15" not in columns


def test_the_declared_schema_matches_what_is_actually_built(
    stepped_path: pd.DataFrame,
) -> None:
    """``movement_panel_columns`` is what a pivot step will read.

    If it drifts from the builder, the pivot silently drops a family or asks for
    a column that does not exist -- so the two are pinned to each other rather
    than maintained in parallel.
    """
    from mlb_pred.features.line_movement import movement_panel_columns

    windows = (60, 240)
    panel = build_movement_panel(stepped_path, grid=(120, 300), windows=windows)

    declared = movement_panel_columns(windows)
    missing = [column for column in declared if column not in panel.columns]

    assert not missing, f"declared but not built: {missing}"


def test_the_consensus_schema_matches_what_is_actually_built(
    four_books: pd.DataFrame,
) -> None:
    from mlb_pred.features.line_cross_book import CONSENSUS_COLUMNS, DEVIATION_COLUMNS

    panel = build_movement_panel(four_books, grid=(60,), windows=(60,))
    consensus = aggregate_across_books(panel)
    with_deviation = add_book_deviation(panel, consensus)

    assert not [c for c in CONSENSUS_COLUMNS if c not in consensus.columns]
    assert not [c for c in DEVIATION_COLUMNS if c not in with_deviation.columns]
    # The join must not leave its scratch columns behind.
    assert "crossbook_std" not in with_deviation.columns
