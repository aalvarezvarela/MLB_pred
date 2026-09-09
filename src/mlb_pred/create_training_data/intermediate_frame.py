"""Create the one-row-per-(game, snapshot) MLB training dataframe.

The closing-line dataset asks "given the market's final word, what happened?".
This one asks the same question at an arbitrary number of minutes before first
pitch: the quote **as of that moment** replaces the close as the anchor, and the
target is the same residual measured against it.  ``TIME_TO_MATCH_MIN`` carries
the horizon so a training run can filter to the one it is modelling.

Three rules make that safe, and each is enforced rather than documented:

* **The anchor is the snapshot, never the close.**  The target is derived
  against the anchor column and a bet settles into the same number, so the two
  must be the same quantity.  The genuine close is renamed into a physically
  separate scoring frame, where it measures closing-line value and cannot reach
  the feature matrix.  A prefix convention was not enough: it relies on every
  downstream consumer remembering to filter.
* **Families whose publication time cannot be reconstructed are excluded.**  We
  hold the final starting lineup and the final IL state, not the moment either
  became public, so availability cannot be restated as of a twelve-hour
  horizon.  The same argument excludes the home-plate umpire.  See
  ``EXCLUDED_PREFIXES``.
* **Prior games' market history stays.**  ``ODDS_HISTORY_*`` and
  ``ODDS_MARKET_REGIME_*`` roll up games that had already closed, so their
  closing lines were genuinely known at the snapshot.  That is the one place a
  closing line is legitimately readable, and the prefixes are disjoint from the
  current game's quote by construction.

One naming divergence from the NBA source is deliberate.  There, the anchor keeps
the closing column's own name (``ODDS_TOTAL_LINE_bet365``) so its established
training pipeline needs no change, at the cost of a column whose name says
"closing" and whose contents do not.  MLB has no such pipeline yet, so the anchor
is named for what it is.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from mlb_pred.features.closing_lines import (
    MARKET_MONEY_LINE,
    MARKET_RUN_LINE,
    MARKET_TOTALS,
    STABLE_BOOKS,
)
from mlb_pred.features.line_cross_book import add_book_deviation, aggregate_across_books
from mlb_pred.features.line_movement import (
    DEFAULT_MOVEMENT_WINDOWS,
    build_movement_panel,
    window_feature_columns,
)
from mlb_pred.features.line_snapshots import DEFAULT_SNAPSHOT_GRID

#: The book whose quote defines the target.
#:
#: Must be a real book: a cross-book consensus is steadier and better sampled,
#: but it is not a price anyone can take, so a return measured against it is
#: hypothetical.  bet365 is the only book quoting essentially every game in
#: every season of this store (2019: 2,445/2,445 ... 2026: 2,077/2,079), which
#: is what stops the anchor's availability from being a season indicator.
ANCHOR_BOOK = "bet365"

ANCHORABLE_MARKETS = (MARKET_TOTALS, MARKET_RUN_LINE)

MARKET_LABELS: dict[str, str] = {
    MARKET_TOTALS: "TOTAL",
    MARKET_RUN_LINE: "RUN_LINE",
    MARKET_MONEY_LINE: "MONEY_LINE",
}

SNAPSHOT_PREFIX = "ODDS_SNAP_"
HORIZON_COLUMN = "TIME_TO_MATCH_MIN"

#: Panel columns given their own column for the anchor book on the anchor
#: market.  Everything a bettor could read off one screen, plus the path that
#: led to it.
ANCHOR_FEATURES: tuple[str, ...] = (
    # --- the quote, exactly as the book published it -----------------------
    "raw_line",
    "home_handicap",
    "price_left",
    "price_right",
    "has_quote",
    "line_age_minutes",
    # --- derived pricing ---------------------------------------------------
    "norm_line",
    "norm_minus_raw",
    "level",
    "fair_left",
    "fair_right",
    "fair_up",
    "overround",
    # --- the path so far ---------------------------------------------------
    "n_ticks_so_far",
    "n_moves_so_far",
    "n_line_moves_so_far",
    "n_price_only_so_far",
    "n_reversals_so_far",
    "level_max_so_far",
    "level_min_so_far",
    "level_std_so_far",
    "level_range_so_far",
    "position_in_range",
    "opener_level",
    "opener_fair_up",
    "first_move_direction",
    "move_from_open",
    "abs_move_from_open",
    "pct_move_from_open",
    "move_direction",
    "prob_move_from_open",
    "minutes_since_open",
    "n_moves_per_hour",
    "opposes_opening_direction",
    "net_opposes_opening_direction",
    # --- position against the rest of the market ---------------------------
    "deviation_from_consensus",
    "abs_deviation_from_consensus",
    "deviation_z",
    "is_outlier_book",
)

#: Cross-book columns kept for the anchor market.
CONSENSUS_FEATURES: tuple[str, ...] = (
    "consensus_level",
    "consensus_raw_line",
    "consensus_norm_line",
    "consensus_norm_minus_raw",
    "consensus_home_handicap",
    "consensus_fair_up",
    "consensus_overround",
    "crossbook_std",
    "crossbook_range",
    "n_books_quoting",
    "median_line_age",
    "max_line_age",
    "consensus_move_from_open",
    "consensus_move_recent",
    "consensus_n_moves",
    "consensus_n_line_moves",
    "consensus_opener_level",
    "steam_books_up",
    "steam_books_down",
    "steam_net",
    "steam_movers",
    "steam_fraction",
    "steam_agreement",
)

#: What a non-anchor book contributes.  Its level already moved the consensus
#: and its dispersion; what is left worth naming is where it stands relative to
#: everyone else, which is the stale-or-sharp question.
BOOK_DEVIATION_FEATURES: tuple[str, ...] = (
    "deviation_from_consensus",
    "deviation_z",
    "is_outlier_book",
)

#: What a *non-anchor market* contributes.  It prices the same game, so it
#: carries real information about it, but it does not need the anchor's full
#: treatment and would double the market block if it got one.
COMPACT_MARKET_FEATURES: tuple[str, ...] = (
    "consensus_level",
    "consensus_norm_minus_raw",
    "consensus_home_handicap",
    "consensus_fair_up",
    "crossbook_std",
    "n_books_quoting",
    "consensus_move_from_open",
    "consensus_move_recent",
    "consensus_n_moves",
    "steam_net",
    "steam_agreement",
)

#: The moneyline stays out of the feature matrix for now and rides in the
#: scoring frame instead, where it can be evaluated without being trainable.
#: Promote it only once a ``line only`` versus ``line + moneyline`` walk-forward
#: comparison shows incremental value -- the same rule the feature
#: consolidation used.  Its raw prices are carried so that comparison, and any
#: return simulation, has an executable number to work with.
MONEY_LINE_SIDECAR_FEATURES: tuple[str, ...] = (
    "consensus_level",
    "consensus_fair_up",
    "consensus_move_from_open",
    "n_books_quoting",
)
MONEY_LINE_SIDECAR_ANCHOR_FEATURES: tuple[str, ...] = (
    "price_left",
    "price_right",
    "level",
    "has_quote",
)

#: Pregame families that cannot be restated as of a snapshot, and are therefore
#: absent from this dataset entirely rather than gated.
#:
#: ``TEAM_AVAILABILITY_*`` reads the final starting lineup and the final IL
#: state.  We hold the outcome of those processes, not their publication times,
#: so their state twelve hours out is not reconstructable, and
#: ``snapshots/archive.py`` cannot be backfilled.  ``UMPIRE_*`` fails the same
#: test: the store holds the final recorded crew with no publication time.
#: ``ODDS_MOVEMENT_*`` measures open-to-**close**, so every one of its columns
#: is a look-ahead at any horizon before the close; the snapshot path block
#: replaces it.
EXCLUDED_PREFIXES: tuple[str, ...] = (
    "TEAM_AVAILABILITY_",
    "UMPIRE_",
    "ODDS_MOVEMENT_",
)

#: Pregame columns holding *this* game's closing quote.  Renamed under
#: ``ODDS_CLOSING_`` and moved to the scoring frame, where they measure closing
#: line value.  ``ODDS_HISTORY_`` and ``ODDS_MARKET_REGIME_`` are deliberately
#: not listed: they summarise games that had already finished.
CLOSING_PREFIXES: tuple[str, ...] = (
    "ODDS_TOTAL_",
    "ODDS_RUN_LINE_",
    "ODDS_MONEY_LINE_",
    "ODDS_DERIVED_",
    "ODDS_INTERACTION_",
)

ANCHOR_TOTAL_LINE_RAW = f"{SNAPSHOT_PREFIX}ANCHOR_TOTAL_LINE_RAW_BEFORE"
ANCHOR_TOTAL_LINE_NORMALIZED = f"{SNAPSHOT_PREFIX}ANCHOR_TOTAL_LINE_NORMALIZED_BEFORE"
ANCHOR_RUN_LINE_RAW = f"{SNAPSHOT_PREFIX}ANCHOR_RUN_LINE_HOME_HANDICAP_RAW_BEFORE"
ANCHOR_RUN_LINE_NORMALIZED = (
    f"{SNAPSHOT_PREFIX}ANCHOR_RUN_LINE_HOME_HANDICAP_NORMALIZED_BEFORE"
)

TOTAL_RUNS_COLUMN = "TOTAL_RUNS"
RUN_LINE_MARGIN_COLUMN = "RUN_LINE_MARGIN"
LINE_ERROR_COLUMN = "LINE_ERROR"
SPREAD_LINE_HOME_COLUMN = "SPREAD_LINE_HOME"
SPREAD_ERROR_COLUMN = "SPREAD_ERROR"

METADATA_COLUMNS: tuple[str, ...] = (
    "GAME_ID",
    "GAME_DATE",
    "GAME_SEASON_YEAR",
    HORIZON_COLUMN,
)

_GAMES_REQUIRED = {
    "game_pk",
    "game_date",
    "season_year",
    "is_final",
    "home_score",
    "away_score",
    "first_pitch_utc",
}


def _require(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _snapshot_column(market: str, scope: str, feature: str) -> str:
    """Compose one snapshot column name.

    The cross-book features already carry a ``consensus_`` stem, so the scope
    prefix is stripped from the feature rather than repeated.  Otherwise the
    block emits ``..._CONSENSUS_CONSENSUS_LEVEL_...`` for the aggregated columns
    and plain ``..._CONSENSUS_STEAM_...`` for the rest -- one family with two
    naming rules, which every downstream selector would then have to know about.
    """
    name = feature.upper()
    if scope == "CONSENSUS":
        name = name.removeprefix("CONSENSUS_")
    return f"{SNAPSHOT_PREFIX}{MARKET_LABELS[market]}_{scope}_{name}_BEFORE"


def _pivot(
    rows: pd.DataFrame,
    features: Iterable[str],
    *,
    market: str,
    scope: str,
) -> pd.DataFrame:
    """One row per (game, snapshot), one column per named feature."""
    present = [feature for feature in features if feature in rows.columns]
    keyed = rows.set_index(["game_pk", "snapshot_minutes"])[present]
    return keyed.rename(
        columns={
            feature: _snapshot_column(market, scope, feature) for feature in present
        }
    )


def anchor_feature_names(
    market: str,
    *,
    windows: tuple[int, ...] = DEFAULT_MOVEMENT_WINDOWS,
) -> list[str]:
    """The anchor book's schema for one market, in order."""
    features = [*ANCHOR_FEATURES, *window_feature_columns(windows)]
    if 60 in windows and 240 in windows:
        features.append("move_acceleration")
    return [_snapshot_column(market, "ANCHOR", feature) for feature in features]


def _market_block(
    panel: pd.DataFrame,
    consensus: pd.DataFrame,
    *,
    market: str,
    is_anchor_market: bool,
    anchor_book: str,
    books: tuple[str, ...],
    windows: tuple[int, ...],
) -> pd.DataFrame:
    """Wide snapshot columns for one market."""
    market_panel = panel.loc[panel["market"].eq(market)]
    market_consensus = consensus.loc[consensus["market"].eq(market)]
    if not is_anchor_market:
        return _pivot(
            market_consensus,
            COMPACT_MARKET_FEATURES,
            market=market,
            scope="CONSENSUS",
        )

    features = [*ANCHOR_FEATURES, *window_feature_columns(windows), "move_acceleration"]
    parts = [
        _pivot(
            market_panel.loc[market_panel["book_slug"].eq(anchor_book)],
            features,
            market=market,
            scope="ANCHOR",
        ),
        _pivot(market_consensus, CONSENSUS_FEATURES, market=market, scope="CONSENSUS"),
    ]
    for book in books:
        if book == anchor_book:
            continue
        parts.append(
            _pivot(
                market_panel.loc[market_panel["book_slug"].eq(book)],
                BOOK_DEVIATION_FEATURES,
                market=market,
                scope=book.upper(),
            )
        )
    return pd.concat(parts, axis=1)


#: The one date column the training frame keeps.  Walk-forward splitting needs
#: it; a *timestamp* is a different thing and is routed away below.
KEPT_DATE_COLUMNS: tuple[str, ...] = ("GAME_DATE",)


def split_pregame_columns(
    features: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """Partition pregame columns into kept, scoring-only and excluded.

    Three rules, in order:

    1. ``EXCLUDED_PREFIXES`` -- families whose publication time cannot be
       reconstructed.  Gone entirely.
    2. ``CLOSING_PREFIXES`` -- this game's closing quote.  Scoring only.
    3. **Any remaining timestamp column.**  A raw instant lets a model pin an
       individual game, and nothing in training needs one; ``GAME_DATE`` is kept
       because walk-forward splitting is defined on it.

    Rule 3 is by dtype rather than by name on purpose.  Naming the columns would
    have to be kept correct by hand forever, and it already failed once:
    ``GAME_FIRST_PITCH_UTC`` reached a built training file because it matched no
    prefix, while the twelve per-book ``*_CLOSE_TIMESTAMP_UTC`` columns happened
    to be caught by rule 2 for an unrelated reason.

    The fail-*closed* allow-list is Phase 4's job; this split is what makes the
    scoring frame possible at all.
    """
    excluded = [
        column for column in features.columns if column.startswith(EXCLUDED_PREFIXES)
    ]
    excluded_set = set(excluded)
    scoring = [
        column
        for column in features.columns
        if column not in excluded_set
        and (
            column.startswith(CLOSING_PREFIXES)
            or (
                column not in KEPT_DATE_COLUMNS
                and is_datetime64_any_dtype(features[column])
            )
        )
    ]
    routed = excluded_set.union(scoring)
    kept = [column for column in features.columns if column not in routed]
    return kept, scoring, excluded


def _scores(games: pd.DataFrame) -> pd.DataFrame:
    _require(games, _GAMES_REQUIRED, "games")
    scoring = games.copy()
    scoring["game_pk"] = scoring["game_pk"].astype(str)
    if scoring["game_pk"].duplicated().any():
        raise ValueError("Games are not unique by game_pk.")
    home = pd.to_numeric(scoring["home_score"], errors="coerce")
    away = pd.to_numeric(scoring["away_score"], errors="coerce")
    scoring["RUNS_TEAM_HOME"] = home
    scoring["RUNS_TEAM_AWAY"] = away
    scoring[TOTAL_RUNS_COLUMN] = home + away
    scoring[RUN_LINE_MARGIN_COLUMN] = home - away
    scoring["FIRST_PITCH_UTC"] = pd.to_datetime(
        scoring["first_pitch_utc"], utc=True, errors="coerce"
    )
    keep = scoring["is_final"].fillna(False).astype(bool) & home.notna() & away.notna()
    return scoring.loc[
        keep,
        [
            "game_pk",
            "RUNS_TEAM_HOME",
            "RUNS_TEAM_AWAY",
            TOTAL_RUNS_COLUMN,
            RUN_LINE_MARGIN_COLUMN,
            "FIRST_PITCH_UTC",
        ],
    ].rename(columns={"game_pk": "GAME_ID"})


def build_intermediate_frame(
    odds_ticks: pd.DataFrame,
    pregame_features: pd.DataFrame,
    games: pd.DataFrame,
    *,
    market: str = MARKET_TOTALS,
    grid: tuple[int, ...] = DEFAULT_SNAPSHOT_GRID,
    windows: tuple[int, ...] = DEFAULT_MOVEMENT_WINDOWS,
    anchor_book: str = ANCHOR_BOOK,
    books: tuple[str, ...] = STABLE_BOOKS,
    use_normalized_anchor: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the (game, snapshot) training frame and its scoring sidecar.

    ``use_normalized_anchor`` selects which stored quote the target is measured
    against.  The equal-price restatement is the right modelling target -- it is
    comparable across books and across horizons rather than confounded with how
    each book happened to be pricing -- but it is not literally the number on the
    board, so a return computed against it is comparable rather than executable.
    Both are carried, so the difference stays visible.
    """
    if market not in ANCHORABLE_MARKETS:
        raise ValueError(
            f"market must be one of {ANCHORABLE_MARKETS}; the moneyline has no "
            f"line to anchor a residual against. Got {market!r}."
        )
    if anchor_book not in books:
        raise ValueError(
            f"anchor book {anchor_book!r} must be among the quoted books {books}."
        )

    panel = build_movement_panel(odds_ticks, grid=grid, windows=windows, books=books)
    if panel.empty:
        raise ValueError("No pre-game ticks produced a snapshot panel.")
    consensus = aggregate_across_books(panel)
    panel = add_book_deviation(panel, consensus)

    blocks = [
        _market_block(
            panel,
            consensus,
            market=quoted,
            is_anchor_market=quoted == market,
            anchor_book=anchor_book,
            books=books,
            windows=windows,
        )
        for quoted in (MARKET_TOTALS, MARKET_RUN_LINE)
    ]
    wide = pd.concat(blocks, axis=1)

    # Rows are the anchor book's own snapshots.  A row without an anchor quote
    # has no line to measure a residual against, so it is not a row.
    anchor_rows = panel.loc[
        panel["market"].eq(market) & panel["book_slug"].eq(anchor_book)
    ].set_index(["game_pk", "snapshot_minutes"])
    wide = wide.loc[wide.index.isin(anchor_rows.index)].copy()
    wide["GAME_START_IS_RELIABLE_BEFORE"] = anchor_rows.loc[
        wide.index, "game_start_is_reliable"
    ]
    # Absence of a book is a fact about the market, not missing data.
    for column in wide.columns:
        if column.endswith("_HAS_QUOTE_BEFORE"):
            wide[column] = wide[column].fillna(0.0)

    wide = wide.reset_index().rename(
        columns={"game_pk": "GAME_ID", "snapshot_minutes": HORIZON_COLUMN}
    )

    kept, routed_to_scoring, _ = split_pregame_columns(pregame_features)
    base = pregame_features[kept].copy()
    base["GAME_ID"] = base["GAME_ID"].astype(str)
    closing_frame = pregame_features[["GAME_ID", *routed_to_scoring]].copy()
    closing_frame["GAME_ID"] = closing_frame["GAME_ID"].astype(str)
    # Only market columns are restated under ``ODDS_CLOSING_``.  A routed
    # timestamp is not a closing quote and keeps its own name, so the sidecar
    # does not end up carrying ``ODDS_CLOSING_GAME_FIRST_PITCH_UTC``.
    closing_frame = closing_frame.rename(
        columns={
            column: f"ODDS_CLOSING_{column.removeprefix('ODDS_')}"
            for column in routed_to_scoring
            if column.startswith("ODDS_")
        }
    )

    frame = wide.merge(base, on="GAME_ID", how="inner", validate="many_to_one")
    frame = frame.merge(
        _scores(games), on="GAME_ID", how="inner", validate="many_to_one"
    )

    anchor_raw, anchor_norm = (
        (ANCHOR_TOTAL_LINE_RAW, ANCHOR_TOTAL_LINE_NORMALIZED)
        if market == MARKET_TOTALS
        else (ANCHOR_RUN_LINE_RAW, ANCHOR_RUN_LINE_NORMALIZED)
    )
    if market == MARKET_TOTALS:
        frame[anchor_raw] = frame[_snapshot_column(market, "ANCHOR", "raw_line")]
        frame[anchor_norm] = frame[_snapshot_column(market, "ANCHOR", "norm_line")]
        chosen = frame[anchor_norm] if use_normalized_anchor else frame[anchor_raw]
        frame[LINE_ERROR_COLUMN] = frame[TOTAL_RUNS_COLUMN] - chosen
    else:
        # ``home_handicap`` is negative when the home team is favoured, so the
        # margin the home side must beat is its negation -- the same convention
        # ``training_frame`` uses for the closing dataset.  Routing both through
        # a named column is what stops that sign becoming a silent error.
        frame[anchor_raw] = -frame[_snapshot_column(market, "ANCHOR", "raw_line")]
        frame[anchor_norm] = -frame[_snapshot_column(market, "ANCHOR", "norm_line")]
        chosen = frame[anchor_norm] if use_normalized_anchor else frame[anchor_raw]
        frame[SPREAD_LINE_HOME_COLUMN] = -chosen
        frame[SPREAD_ERROR_COLUMN] = (
            frame[RUN_LINE_MARGIN_COLUMN] - frame[SPREAD_LINE_HOME_COLUMN]
        )

    frame["SNAPSHOT_TS_UTC"] = frame["FIRST_PITCH_UTC"] - pd.to_timedelta(
        frame[HORIZON_COLUMN], unit="m"
    )

    scoring = _build_scoring_frame(frame, closing_frame, panel, consensus, anchor_book)
    training = frame.drop(
        columns=[
            column
            for column in ("FIRST_PITCH_UTC", "SNAPSHOT_TS_UTC")
            if column in frame.columns
        ]
    )
    training = training.sort_values(["GAME_DATE", "GAME_ID", HORIZON_COLUMN])
    training = training.reset_index(drop=True)
    scoring = scoring.sort_values(["GAME_ID", HORIZON_COLUMN]).reset_index(drop=True)
    return training, scoring


def _build_scoring_frame(
    frame: pd.DataFrame,
    closing_frame: pd.DataFrame,
    panel: pd.DataFrame,
    consensus: pd.DataFrame,
    anchor_book: str,
) -> pd.DataFrame:
    """Everything the feature matrix must never see, keyed to the same rows.

    Physically separate rather than prefixed, because a prefix relies on every
    consumer remembering to filter and one that forgets gets a closing line in
    ``X``.
    """
    keys = [c for c in ("GAME_ID", HORIZON_COLUMN) if c in frame.columns]
    scoring = frame[[*keys, "FIRST_PITCH_UTC", "SNAPSHOT_TS_UTC"]].copy()
    scoring = scoring.merge(closing_frame, on="GAME_ID", how="left")

    money = panel.loc[
        panel["market"].eq(MARKET_MONEY_LINE) & panel["book_slug"].eq(anchor_book)
    ]
    money_consensus = consensus.loc[consensus["market"].eq(MARKET_MONEY_LINE)]
    money_wide = pd.concat(
        [
            _pivot(
                money,
                MONEY_LINE_SIDECAR_ANCHOR_FEATURES,
                market=MARKET_MONEY_LINE,
                scope="ANCHOR",
            ),
            _pivot(
                money_consensus,
                MONEY_LINE_SIDECAR_FEATURES,
                market=MARKET_MONEY_LINE,
                scope="CONSENSUS",
            ),
        ],
        axis=1,
    ).reset_index()
    money_wide = money_wide.rename(
        columns={"game_pk": "GAME_ID", "snapshot_minutes": HORIZON_COLUMN}
    )
    return scoring.merge(money_wide, on=keys, how="left")


def feature_columns(training_frame: pd.DataFrame) -> list[str]:
    """Numeric model features: everything kept, minus metadata and targets."""
    from pandas.api.types import is_numeric_dtype

    excluded = set(METADATA_COLUMNS).union(
        {
            TOTAL_RUNS_COLUMN,
            RUN_LINE_MARGIN_COLUMN,
            LINE_ERROR_COLUMN,
            SPREAD_ERROR_COLUMN,
            SPREAD_LINE_HOME_COLUMN,
            "RUNS_TEAM_HOME",
            "RUNS_TEAM_AWAY",
        }
    )
    return [
        column
        for column in training_frame.columns
        if column not in excluded and is_numeric_dtype(training_frame[column])
    ]


def select_horizon(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Keep one horizon, and prove it left one temporal observation per game.

    Filtering first is what makes each model's rows independent.  Pooling
    horizons overstates the evidence by roughly their number -- adjacent
    snapshots resolve to an identical totals line in 92.4% of games -- and the
    NBA repo's answer was a per-row weight that nothing read.  The duplicate
    check is cheap enough to run always and is the whole point of the column.
    """
    rows = frame.loc[frame[HORIZON_COLUMN] == minutes]
    if rows.empty:
        available = sorted(frame[HORIZON_COLUMN].unique())
        raise ValueError(
            f"No rows at {HORIZON_COLUMN}={minutes}; available: {available}"
        )
    if rows["GAME_ID"].duplicated().any():
        raise ValueError(
            "Horizon filter must leave exactly one row per game; "
            f"{int(rows['GAME_ID'].duplicated().sum())} duplicates at {minutes}."
        )
    return rows.reset_index(drop=True)


__all__ = [
    "ANCHOR_BOOK",
    "ANCHOR_RUN_LINE_NORMALIZED",
    "ANCHOR_RUN_LINE_RAW",
    "ANCHOR_TOTAL_LINE_NORMALIZED",
    "ANCHOR_TOTAL_LINE_RAW",
    "CLOSING_PREFIXES",
    "EXCLUDED_PREFIXES",
    "HORIZON_COLUMN",
    "MARKET_LABELS",
    "SNAPSHOT_PREFIX",
    "anchor_feature_names",
    "build_intermediate_frame",
    "feature_columns",
    "select_horizon",
    "split_pregame_columns",
]
