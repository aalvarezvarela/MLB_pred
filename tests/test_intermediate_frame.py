"""Acceptance tests for the (game, snapshot) training frame.

The three that carry the phase:

* **T7** -- the anchor column holds the *snapshot* quote, and on a game whose
  line moved it must differ from the close.  A build that quietly anchored on
  the close would still produce a plausible frame with plausible targets; only
  comparing the two catches it.
* **T8** -- the target is reproducible from the frame alone.  ``LINE_ERROR``
  plus the anchor must equal ``TOTAL_RUNS`` exactly, which is the same
  invariant ``training_frame`` enforces for the closing dataset.
* **T9** -- no closing quote, realised score or raw timestamp reaches the
  training file.  Checked against the file's own columns, never against a
  helper's output: screening the helper would make the test a no-op.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_pred.create_training_data.intermediate_frame import (
    ANCHOR_RUN_LINE_NORMALIZED,
    ANCHOR_RUN_LINE_RAW,
    ANCHOR_TOTAL_LINE_NORMALIZED,
    ANCHOR_TOTAL_LINE_RAW,
    EXCLUDED_PREFIXES,
    HORIZON_COLUMN,
    LINE_ERROR_COLUMN,
    RUN_LINE_MARGIN_COLUMN,
    SPREAD_ERROR_COLUMN,
    SPREAD_LINE_HOME_COLUMN,
    TOTAL_RUNS_COLUMN,
    build_intermediate_frame,
    feature_columns,
    select_horizon,
    split_pregame_columns,
)
from mlb_pred.features.closing_lines import (
    MARKET_MONEY_LINE,
    MARKET_RUN_LINE,
    MARKET_TOTALS,
)
from mlb_pred.odds.encoding import LINE_SCALE

BOOKS = ("bet365", "caesars", "draftkings", "fanduel")
GRID = (0, 60, 240, 720)
WINDOWS = (60, 240)


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
        "line_ts": pd.Timestamp("2024-06-01T23:00", tz="UTC")
        - pd.Timedelta(float(minutes_before_start), unit="m"),
        "mins_to_tip": -minutes_before_start,
        "is_pregame": True,
        "left_line": None if left_line is None else left_line * LINE_SCALE,
        "right_line": None if right_line is None else right_line * LINE_SCALE,
        "left_price": left_price,
        "right_price": right_price,
    }


@pytest.fixture
def ticks() -> pd.DataFrame:
    """Two games.  Game 1's total climbs 8.0 -> 9.0; game 2's never moves."""
    rows: list[dict] = []
    for book in BOOKS:
        rows += [
            _tick("1", MARKET_TOTALS, book, 900, left_line=8.0),
            _tick("1", MARKET_TOTALS, book, 500, left_line=8.5),
            _tick("1", MARKET_TOTALS, book, 100, left_line=9.0),
            _tick("2", MARKET_TOTALS, book, 900, left_line=7.5),
            _tick("2", MARKET_TOTALS, book, 100, left_line=7.5),
        ]
        for game in ("1", "2"):
            rows += [
                _tick(
                    game,
                    MARKET_RUN_LINE,
                    book,
                    900,
                    left_line=1.5,
                    left_price=-140,
                    right_price=120,
                ),
                _tick(
                    game,
                    MARKET_RUN_LINE,
                    book,
                    100,
                    left_line=1.5,
                    left_price=-160,
                    right_price=135,
                ),
                _tick(
                    game,
                    MARKET_MONEY_LINE,
                    book,
                    900,
                    left_line=None,
                    left_price=150,
                    right_price=-170,
                ),
                _tick(
                    game,
                    MARKET_MONEY_LINE,
                    book,
                    100,
                    left_line=None,
                    left_price=135,
                    right_price=-155,
                ),
            ]
    return pd.DataFrame(rows)


@pytest.fixture
def pregame() -> pd.DataFrame:
    """A pregame partition carrying one column of each family that matters."""
    return pd.DataFrame(
        {
            "GAME_ID": ["1", "2"],
            "GAME_DATE": pd.to_datetime(["2024-06-01", "2024-06-01"]),
            "GAME_SEASON_YEAR": [2024, 2024],
            # A timestamp matching no prefix rule.  It reached a built training
            # file before rule 3 existed.
            "GAME_FIRST_PITCH_UTC": pd.to_datetime(
                ["2024-06-01T23:00Z", "2024-06-01T23:00Z"]
            ),
            "TEAM_ROLLING_RUNS_LAST_5_BEFORE_TEAM_HOME": [4.2, 5.1],
            "ODDS_HISTORY_TOTAL_MARKET_ERROR_SEASON_BEFORE_AVG_TEAM_HOME": [0.1, -0.2],
            "ODDS_MARKET_REGIME_TOTAL_BIAS_30G_BEFORE": [0.05, 0.05],
            "STATCAST_XWOBA_L10_BEFORE_TEAM_HOME": [0.31, 0.33],
            "PARK_RUN_FACTOR_EXPANDING_BEFORE": [1.02, 0.97],
            # Excluded families.
            "TEAM_AVAILABILITY_HITTER_N_INJURED_BEFORE_TEAM_HOME": [1.0, 0.0],
            "UMPIRE_RUN_FACTOR_EXPANDING_BEFORE": [1.01, 1.00],
            "ODDS_MOVEMENT_TOTAL_OPEN_TO_CLOSE_BEFORE": [0.5, 0.0],
            # This game's closing quote.
            "ODDS_TOTAL_CONSENSUS_LINE_RAW_MEDIAN": [9.0, 7.5],
            "ODDS_RUN_LINE_CONSENSUS_HOME_HANDICAP_RAW_MEDIAN": [-1.5, -1.5],
            "ODDS_MONEY_LINE_CONSENSUS_FAIR_PROB_HOME_MEDIAN": [0.55, 0.52],
            "ODDS_INTERACTION_TOTAL_PROBABILITY_SKEW_BEFORE": [0.01, 0.02],
        }
    )


@pytest.fixture
def games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": ["1", "2"],
            "game_date": pd.to_datetime(["2024-06-01", "2024-06-01"]),
            "season_year": [2024, 2024],
            "is_final": [True, True],
            "home_score": [6, 3],
            "away_score": [4, 2],
            "first_pitch_utc": pd.to_datetime(
                ["2024-06-01T23:00Z", "2024-06-01T23:00Z"]
            ),
        }
    )


@pytest.fixture
def built(ticks, pregame, games):
    return build_intermediate_frame(
        ticks, pregame, games, grid=GRID, windows=WINDOWS, books=BOOKS
    )


# ---------------------------------------------------------------------------
# T7 -- the anchor is the snapshot, not the close
# ---------------------------------------------------------------------------
def test_the_anchor_holds_the_snapshot_quote_not_the_closing_one(built) -> None:
    """Game 1's total climbed, so its anchor must differ across horizons.

    A build that anchored on the close would produce the same number on every
    row -- internally consistent, plausible, and wrong.
    """
    training, _ = built
    game = training.loc[training["GAME_ID"] == "1"].set_index(HORIZON_COLUMN)

    assert float(game.loc[720, ANCHOR_TOTAL_LINE_RAW]) == pytest.approx(8.0)
    assert float(game.loc[240, ANCHOR_TOTAL_LINE_RAW]) == pytest.approx(8.5)
    assert float(game.loc[60, ANCHOR_TOTAL_LINE_RAW]) == pytest.approx(9.0)
    assert float(game.loc[0, ANCHOR_TOTAL_LINE_RAW]) == pytest.approx(9.0)


def test_a_game_whose_line_never_moved_anchors_identically_at_every_horizon(
    built,
) -> None:
    """The control for the test above: no movement means no variation."""
    training, _ = built
    game = training.loc[training["GAME_ID"] == "2"]

    assert set(game[ANCHOR_TOTAL_LINE_RAW]) == {7.5}


def test_the_horizon_column_carries_every_grid_point(built) -> None:
    training, _ = built

    assert sorted(training[HORIZON_COLUMN].unique()) == sorted(GRID)
    assert len(training) == len(GRID) * 2


# ---------------------------------------------------------------------------
# T8 -- targets are reproducible from the frame alone
# ---------------------------------------------------------------------------
def test_line_error_plus_the_anchor_reproduces_the_total(built) -> None:
    """The invariant ``training_frame`` enforces, restated per snapshot."""
    training, _ = built
    rebuilt = training[LINE_ERROR_COLUMN] + training[ANCHOR_TOTAL_LINE_NORMALIZED]

    assert np.allclose(rebuilt, training[TOTAL_RUNS_COLUMN], atol=1e-10)


def test_the_target_moves_with_the_anchor_across_horizons(built) -> None:
    """Same game, same result, different reference line: different residual.

    This is the whole reason the dataset exists, so it is asserted rather than
    assumed.
    """
    training, _ = built
    game = training.loc[training["GAME_ID"] == "1"].set_index(HORIZON_COLUMN)

    # Ten runs scored; the line was 8.0 at twelve hours and 9.0 at the close.
    assert float(game.loc[720, LINE_ERROR_COLUMN]) == pytest.approx(2.0)
    assert float(game.loc[0, LINE_ERROR_COLUMN]) == pytest.approx(1.0)


def test_the_run_line_anchor_uses_the_home_margin_convention(
    ticks, pregame, games
) -> None:
    """A home handicap is negative when the home side is favoured.

    ``SPREAD_ERROR`` is positive when the home team beat the line, matching
    ``training_frame``; routing both through named columns is what stops that
    sign becoming a silent inversion.

    Asserted against the **raw** anchor, because that is the number on the
    board and the sign convention is about that number.  The normalised default
    is checked separately below.
    """
    training, _ = build_intermediate_frame(
        ticks,
        pregame,
        games,
        market=MARKET_RUN_LINE,
        grid=GRID,
        windows=WINDOWS,
        books=BOOKS,
        use_normalized_anchor=False,
    )
    game = training.loc[training["GAME_ID"] == "1"].iloc[0]

    # left = AWAY at +1.5, so the home side lays 1.5.
    assert float(game[ANCHOR_RUN_LINE_RAW]) == pytest.approx(-1.5)
    assert float(game[SPREAD_LINE_HOME_COLUMN]) == pytest.approx(1.5)
    assert float(game[RUN_LINE_MARGIN_COLUMN]) == pytest.approx(2.0)
    # Won by two, laying one and a half: covered.
    assert float(game[SPREAD_ERROR_COLUMN]) == pytest.approx(0.5)
    assert float(game[SPREAD_ERROR_COLUMN]) > 0


def test_the_normalized_run_line_restates_the_price_as_a_handicap(
    ticks, pregame, games
) -> None:
    """With the handicap pinned at +/-1.5, the price carries the information.

    Normalising converts that price into the equal-price handicap it implies, so
    the anchor moves well away from the number on the board -- here a -1.5 home
    side priced at +120 restates to roughly half a run.  That is the intended
    behaviour and matches ``training_frame``'s default for the closing dataset,
    but it means a return measured against it is comparable rather than
    executable, which is why the raw anchor is carried too.
    """
    training, _ = build_intermediate_frame(
        ticks,
        pregame,
        games,
        market=MARKET_RUN_LINE,
        grid=GRID,
        windows=WINDOWS,
        books=BOOKS,
    )
    game = training.loc[training["GAME_ID"] == "1"].iloc[0]

    assert float(game[ANCHOR_RUN_LINE_RAW]) == pytest.approx(-1.5)
    assert float(game[ANCHOR_RUN_LINE_NORMALIZED]) != pytest.approx(-1.5)
    # The residual still reproduces from the frame alone.
    assert float(
        game[SPREAD_ERROR_COLUMN] + game[SPREAD_LINE_HOME_COLUMN]
    ) == pytest.approx(float(game[RUN_LINE_MARGIN_COLUMN]))


def test_the_moneyline_cannot_anchor_a_residual(ticks, pregame, games) -> None:
    """It has no line, so there is nothing to measure a residual against."""
    with pytest.raises(ValueError, match="no.*line to anchor"):
        build_intermediate_frame(
            ticks, pregame, games, market=MARKET_MONEY_LINE, grid=GRID
        )


def test_the_raw_and_normalized_anchors_are_both_carried(built) -> None:
    """One is executable, the other comparable; the gap must stay visible."""
    training, _ = built

    assert ANCHOR_TOTAL_LINE_RAW in training.columns
    assert ANCHOR_TOTAL_LINE_NORMALIZED in training.columns


def test_the_anchor_choice_changes_the_target(ticks, pregame, games) -> None:
    """A return against the normalised line is comparable, not executable."""
    priced = pd.DataFrame(ticks)
    # Skew game 1's closing prices so the two anchors genuinely disagree.
    late = (priced["game_pk"] == "1") & (priced["mins_to_tip"] == -100)
    priced.loc[late & priced["market"].eq(MARKET_TOTALS), "left_price"] = -140
    priced.loc[late & priced["market"].eq(MARKET_TOTALS), "right_price"] = 120

    normalized, _ = build_intermediate_frame(
        priced, pregame, games, grid=GRID, windows=WINDOWS, books=BOOKS
    )
    raw, _ = build_intermediate_frame(
        priced,
        pregame,
        games,
        grid=GRID,
        windows=WINDOWS,
        books=BOOKS,
        use_normalized_anchor=False,
    )
    close_norm = normalized.loc[
        (normalized["GAME_ID"] == "1") & (normalized[HORIZON_COLUMN] == 0),
        LINE_ERROR_COLUMN,
    ].iloc[0]
    close_raw = raw.loc[
        (raw["GAME_ID"] == "1") & (raw[HORIZON_COLUMN] == 0), LINE_ERROR_COLUMN
    ].iloc[0]

    assert close_norm != close_raw


# ---------------------------------------------------------------------------
# T9 -- physical separation
# ---------------------------------------------------------------------------
def test_no_closing_quote_reaches_the_training_frame(built) -> None:
    """Scanned on the frame's own columns, not on ``feature_columns``.

    ``feature_columns`` already excludes what this looks for, so screening it
    would make the check pass no matter what was in the file.
    """
    training, _ = built
    offenders = [
        column
        for column in training.columns
        if column.startswith(("ODDS_CLOSING_",))
        or column.startswith(("ODDS_TOTAL_", "ODDS_RUN_LINE_", "ODDS_MONEY_LINE_"))
        or column.startswith(("ODDS_INTERACTION_", "ODDS_DERIVED_"))
    ]

    assert not offenders, offenders


def test_no_excluded_family_reaches_the_training_frame(built) -> None:
    """Availability, umpire and open-to-close movement, all absent."""
    training, _ = built
    offenders = [
        column for column in training.columns if column.startswith(EXCLUDED_PREFIXES)
    ]

    assert not offenders, offenders


def test_raw_timestamps_stay_in_the_scoring_frame(built) -> None:
    """They would let a model pin individual games, and nothing needs them."""
    training, scoring = built

    assert "FIRST_PITCH_UTC" not in training.columns
    assert "SNAPSHOT_TS_UTC" not in training.columns
    assert "FIRST_PITCH_UTC" in scoring.columns
    assert "SNAPSHOT_TS_UTC" in scoring.columns


def test_the_scoring_frame_carries_the_close_and_the_moneyline_price(built) -> None:
    """CLV needs the close; a return simulation needs an executable price."""
    training, scoring = built
    closing = [c for c in scoring.columns if c.startswith("ODDS_CLOSING_")]
    money = [c for c in scoring.columns if "MONEY_LINE" in c]

    assert closing
    assert any("PRICE_LEFT" in c for c in money)
    assert any("PRICE_RIGHT" in c for c in money)
    assert not [c for c in training.columns if "MONEY_LINE" in c]


def test_the_two_frames_line_up_row_for_row(built) -> None:
    """They are joined back after scoring, so the keys must match exactly."""
    training, scoring = built
    keys = ["GAME_ID", HORIZON_COLUMN]

    assert len(scoring) == len(training)
    assert (
        training[keys]
        .sort_values(keys)
        .reset_index(drop=True)
        .equals(scoring[keys].sort_values(keys).reset_index(drop=True))
    )


def test_outcomes_are_kept_but_excluded_from_the_feature_list(built) -> None:
    """Reproducibility needs them in the file; a model must never see them."""
    training, _ = built
    features = feature_columns(training)

    assert TOTAL_RUNS_COLUMN in training.columns
    for column in (
        TOTAL_RUNS_COLUMN,
        RUN_LINE_MARGIN_COLUMN,
        LINE_ERROR_COLUMN,
        "RUNS_TEAM_HOME",
        "RUNS_TEAM_AWAY",
        HORIZON_COLUMN,
    ):
        assert column not in features


# ---------------------------------------------------------------------------
# Column routing and horizon selection
# ---------------------------------------------------------------------------
def test_prior_game_market_history_survives_the_split(pregame) -> None:
    """Those games had closed, so their closing lines were genuinely known.

    The prefixes are disjoint from the current game's quote by construction --
    ``ODDS_HISTORY_TOTAL_*`` does not start with ``ODDS_TOTAL_`` -- which is what
    lets a prefix rule separate them at all.
    """
    kept, closing, excluded = split_pregame_columns(pregame)

    assert "ODDS_HISTORY_TOTAL_MARKET_ERROR_SEASON_BEFORE_AVG_TEAM_HOME" in kept
    assert "ODDS_MARKET_REGIME_TOTAL_BIAS_30G_BEFORE" in kept
    assert "ODDS_TOTAL_CONSENSUS_LINE_RAW_MEDIAN" in closing
    assert "ODDS_INTERACTION_TOTAL_PROBABILITY_SKEW_BEFORE" in closing
    assert set(excluded) == {
        "TEAM_AVAILABILITY_HITTER_N_INJURED_BEFORE_TEAM_HOME",
        "UMPIRE_RUN_FACTOR_EXPANDING_BEFORE",
        "ODDS_MOVEMENT_TOTAL_OPEN_TO_CLOSE_BEFORE",
    }


def test_a_timestamp_matching_no_prefix_is_still_routed_away(pregame) -> None:
    """Rule 3 is by dtype, because the name-based rules had already missed one.

    ``GAME_FIRST_PITCH_UTC`` matches neither the excluded nor the closing
    prefixes, and reached a built training file. The twelve per-book
    ``*_CLOSE_TIMESTAMP_UTC`` columns were only caught because they happen to
    start with an odds prefix.
    """
    kept, scoring, _ = split_pregame_columns(pregame)

    assert "GAME_FIRST_PITCH_UTC" in scoring
    assert "GAME_FIRST_PITCH_UTC" not in kept
    # The date survives: walk-forward splitting is defined on it.
    assert "GAME_DATE" in kept


def test_no_timestamp_column_survives_into_the_training_frame(built) -> None:
    """Asserted by dtype over the whole frame, not against a list of names."""
    training, scoring = built
    timestamps = [
        column
        for column in training.columns
        if pd.api.types.is_datetime64_any_dtype(training[column])
        and column != "GAME_DATE"
    ]

    assert not timestamps, timestamps
    assert "GAME_FIRST_PITCH_UTC" in scoring.columns
    # Routed, not relabelled: a timestamp is not a closing quote.
    assert "ODDS_CLOSING_GAME_FIRST_PITCH_UTC" not in scoring.columns


def test_every_pregame_column_is_routed_exactly_once(pregame) -> None:
    """A column that fell through all three lists would vanish silently."""
    kept, closing, excluded = split_pregame_columns(pregame)
    routed = [*kept, *closing, *excluded]

    assert sorted(routed) == sorted(pregame.columns)
    assert len(routed) == len(set(routed))


def test_reused_baseball_families_reach_the_training_frame(built) -> None:
    training, _ = built

    for column in (
        "TEAM_ROLLING_RUNS_LAST_5_BEFORE_TEAM_HOME",
        "STATCAST_XWOBA_L10_BEFORE_TEAM_HOME",
        "PARK_RUN_FACTOR_EXPANDING_BEFORE",
        "ODDS_HISTORY_TOTAL_MARKET_ERROR_SEASON_BEFORE_AVG_TEAM_HOME",
    ):
        assert column in training.columns
        # A reused family is constant across horizons: it describes the game,
        # not the moment.
        assert training.groupby("GAME_ID")[column].nunique().max() == 1


def test_selecting_a_horizon_leaves_one_row_per_game(built) -> None:
    """The property the whole grain exists to guarantee."""
    training, _ = built

    for minutes in GRID:
        rows = select_horizon(training, minutes)
        assert len(rows) == rows["GAME_ID"].nunique() == 2


def test_selecting_a_horizon_that_is_not_on_the_grid_fails_loudly(built) -> None:
    training, _ = built

    with pytest.raises(ValueError, match="No rows at"):
        select_horizon(training, 999)


def test_a_duplicated_game_at_one_horizon_is_refused(built) -> None:
    """Guards the assertion itself, not just the happy path."""
    training, _ = built
    doubled = pd.concat([training, training.head(1)], ignore_index=True)

    with pytest.raises(ValueError, match="exactly one row per game"):
        select_horizon(doubled, int(doubled[HORIZON_COLUMN].iloc[0]))


def test_the_snapshot_block_carries_path_and_cross_book_columns(built) -> None:
    """The anchor, the consensus and a non-anchor book's deviation."""
    training, _ = built
    columns = set(training.columns)

    assert "ODDS_SNAP_TOTAL_ANCHOR_NORM_MINUS_RAW_BEFORE" in columns
    assert "ODDS_SNAP_TOTAL_ANCHOR_MOVE_LAST_240_BEFORE" in columns
    assert "ODDS_SNAP_TOTAL_CONSENSUS_STEAM_AGREEMENT_BEFORE" in columns
    assert "ODDS_SNAP_TOTAL_CAESARS_DEVIATION_Z_BEFORE" in columns
    # The non-anchor market is present, but only compactly.
    assert "ODDS_SNAP_RUN_LINE_CONSENSUS_LEVEL_BEFORE" in columns
    assert "ODDS_SNAP_RUN_LINE_ANCHOR_LEVEL_BEFORE" not in columns


def test_an_unknown_anchor_book_is_refused(ticks, pregame, games) -> None:
    with pytest.raises(ValueError, match="among the quoted books"):
        build_intermediate_frame(
            ticks, pregame, games, grid=GRID, anchor_book="nosuchbook", books=BOOKS
        )
