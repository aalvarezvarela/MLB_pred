"""Build one leakage-safe closing-market feature row per MLB game.

The source remains the long-form tick store.  A close is the latest *complete*
quote no nearer than five minutes before first pitch; later pregame ticks and
all in-play ticks are excluded.  Raw executable quotes and their equal-price
``-110/-110`` equivalents are both retained.

Column families are intentionally machine-readable:

* ``GAME_*`` -- identifiers and schedule context known before first pitch.
* ``ODDS_TOTAL_*`` -- total-runs market.
* ``ODDS_RUN_LINE_*`` -- run-line market.
* ``ODDS_MONEY_LINE_*`` -- moneyline market (devigged, no line to shift).
* ``ODDS_DERIVED_*`` -- cross-market quantities such as implied team runs.

No realised score, result, cover, over/under outcome, or line error is read by
this module.  Such targets belong in a physically separate scoring sidecar.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mlb_pred.config.settings import PROJECT_ROOT
from mlb_pred.features.market_normalization import (
    center_run_lines,
    center_total_lines,
    devig_two_way_series,
)
from mlb_pred.odds.encoding import LINE_SCALE

DEFAULT_CLOSE_SAFETY_MARGIN_MINUTES = 5
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "features" / "closing_lines"

# Only these books quote every season in the store.  betmgm and betrivers
# appear from 2022 and fanatics_sportsbook from 2025, so per-book columns for
# them switch on in the middle of the history -- exactly across the split a
# walk-forward model trains over.  They still contribute to the consensus
# aggregates, where an absent book simply lowers BOOK_COUNT.
STABLE_BOOKS: tuple[str, ...] = ("bet365", "caesars", "draftkings", "fanduel")

MARKET_TOTALS = "totals"
MARKET_RUN_LINE = "run_line"
MARKET_MONEY_LINE = "money_line"
SUPPORTED_MARKETS = (MARKET_TOTALS, MARKET_RUN_LINE, MARKET_MONEY_LINE)

_TICK_COLUMNS = {
    "game_pk",
    "season_year",
    "market",
    "book_slug",
    "line_ts",
    "mins_to_tip",
    "is_pregame",
    "left_line",
    "left_price",
    "right_line",
    "right_price",
}
_GAME_COLUMNS = {
    "game_pk",
    "game_date",
    "season_year",
    "first_pitch_utc",
    "team_home_id",
    "team_away_id",
    "team_home",
    "team_away",
    "is_doubleheader",
    "game_number",
}

_FORBIDDEN_OUTCOME_COLUMNS = {
    "GAME_HOME_SCORE",
    "GAME_AWAY_SCORE",
    "GAME_TOTAL_RUNS",
    "GAME_RUN_LINE_MARGIN",
    "TOTAL_RUNS",
    "RUN_LINE_MARGIN",
    "LINE_ERROR",
    "IS_OVER",
    "IS_UNDER",
    "HOME_COVERED",
    "AWAY_COVERED",
}


def _require_columns(
    frame: pd.DataFrame, required: set[str], *, frame_name: str
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _book_label(book_slug: object) -> str:
    label = str(book_slug).strip().upper()
    if not label:
        raise ValueError("book_slug cannot be blank.")
    return "".join(character if character.isalnum() else "_" for character in label)


def decode_lines(values: pd.Series) -> pd.Series:
    encoded = pd.to_numeric(values, errors="coerce").astype("float64")
    invalid = encoded.notna() & ~np.isclose(encoded, np.round(encoded), atol=1e-10)
    if invalid.any():
        examples = encoded.loc[invalid].head(5).tolist()
        raise ValueError(
            "Stored odds lines must use the exact integer x2 encoding; "
            f"found {examples}."
        )
    return encoded / LINE_SCALE


def assert_structural_market_invariants(ticks: pd.DataFrame) -> None:
    left = decode_lines(ticks["left_line"])
    right = decode_lines(ticks["right_line"])

    totals = ticks["market"].eq(MARKET_TOTALS) & left.notna() & right.notna()
    bad_totals = totals & ~np.isclose(left, right, rtol=0.0, atol=1e-10)

    run_lines = ticks["market"].eq(MARKET_RUN_LINE) & left.notna() & right.notna()
    bad_run_lines = run_lines & ~np.isclose(left + right, 0.0, rtol=0.0, atol=1e-10)

    money_lines = ticks["market"].eq(MARKET_MONEY_LINE)
    bad_money_lines = money_lines & (left.notna() | right.notna())
    if bad_totals.any() or bad_run_lines.any() or bad_money_lines.any():
        raise ValueError(
            "Odds side-orientation invariant failed before closing selection: "
            f"totals={int(bad_totals.sum())}, "
            f"run_line={int(bad_run_lines.sum())}, "
            f"money_line={int(bad_money_lines.sum())}."
        )


def select_closing_quotes(
    ticks: pd.DataFrame,
    *,
    safety_margin_minutes: int = DEFAULT_CLOSE_SAFETY_MARGIN_MINUTES,
) -> pd.DataFrame:
    """Select the latest complete, executable quote before first pitch.

    Stored ``mins_to_tip`` is negative before the game.  The returned
    ``minutes_before_start`` is positive so that sign convention never leaks
    into feature code.
    """
    _require_columns(ticks, _TICK_COLUMNS, frame_name="odds ticks")
    if safety_margin_minutes < 0:
        raise ValueError("safety_margin_minutes must be non-negative.")
    if ticks.empty:
        return ticks.copy()
    if ticks["is_pregame"].isna().any() or ticks["mins_to_tip"].isna().any():
        raise ValueError(
            "is_pregame and mins_to_tip must be non-null; they are the stored "
            "boundary between pre-game features and in-play leakage."
        )

    unknown_markets = sorted(set(ticks["market"].dropna()) - set(SUPPORTED_MARKETS))
    if unknown_markets:
        raise ValueError(f"Unsupported market(s): {unknown_markets}")
    assert_structural_market_invariants(ticks)

    fair_prices = devig_two_way_series(ticks["left_price"], ticks["right_price"])
    prices_complete = fair_prices[["fair_left", "fair_right"]].notna().all(axis=1)
    lines_complete = ticks[["left_line", "right_line"]].notna().all(axis=1)
    market_complete = ticks["market"].eq(MARKET_MONEY_LINE) | lines_complete
    eligible = (
        ticks["is_pregame"].fillna(False).astype(bool)
        & pd.to_numeric(ticks["mins_to_tip"], errors="coerce").le(
            -safety_margin_minutes
        )
        & prices_complete
        & market_complete
    )

    closing = (
        ticks.loc[eligible]
        .sort_values("line_ts", kind="mergesort")
        .groupby(["game_pk", "market", "book_slug"], sort=False, as_index=False)
        .tail(1)
        .copy()
    )
    closing["left_line"] = decode_lines(closing["left_line"])
    closing["right_line"] = decode_lines(closing["right_line"])
    closing["left_price"] = pd.to_numeric(
        closing["left_price"], errors="coerce"
    ).astype("float64")
    closing["right_price"] = pd.to_numeric(
        closing["right_price"], errors="coerce"
    ).astype("float64")
    closing["minutes_before_start"] = -pd.to_numeric(
        closing["mins_to_tip"], errors="raise"
    ).astype("int64")

    if not closing.empty:
        if not closing["is_pregame"].all():
            raise AssertionError("Closing selection admitted an in-play quote.")
        if closing["minutes_before_start"].lt(safety_margin_minutes).any():
            raise AssertionError("Closing selection violated its safety margin.")
        if closing.duplicated(["game_pk", "market", "book_slug"]).any():
            raise AssertionError("Closing selection is not unique at its grain.")
    return closing.reset_index(drop=True)


def _with_normalized_market_values(closing: pd.DataFrame) -> pd.DataFrame:
    quotes = closing.copy()
    fair = devig_two_way_series(quotes["left_price"], quotes["right_price"])
    quotes[["fair_left", "fair_right", "overround"]] = fair
    quotes["normalized_line"] = np.nan

    totals = quotes["market"].eq(MARKET_TOTALS)
    quotes.loc[totals, "normalized_line"] = center_total_lines(
        quotes.loc[totals, "left_line"],
        quotes.loc[totals, "left_price"],
        quotes.loc[totals, "right_price"],
    )

    run_lines = quotes["market"].eq(MARKET_RUN_LINE)
    # SBR left = AWAY.  Its line is also the canonical home-margin threshold.
    quotes.loc[run_lines, "normalized_line"] = center_run_lines(
        quotes.loc[run_lines, "left_line"],
        quotes.loc[run_lines, "left_price"],
        quotes.loc[run_lines, "right_price"],
    )
    return quotes


def _game_frame(odds_games: pd.DataFrame) -> pd.DataFrame:
    _require_columns(odds_games, _GAME_COLUMNS, frame_name="odds games")
    if odds_games["game_pk"].duplicated().any():
        raise ValueError("odds games must contain one row per game_pk.")
    required_values = [
        "game_pk",
        "game_date",
        "season_year",
        "first_pitch_utc",
        "team_home_id",
        "team_away_id",
    ]
    missing_values = [
        column for column in required_values if odds_games[column].isna().any()
    ]
    if missing_values:
        raise ValueError(f"odds games has null required values in: {missing_values}")

    renamed = odds_games[
        [
            "game_pk",
            "game_date",
            "season_year",
            "first_pitch_utc",
            "team_home_id",
            "team_away_id",
            "team_home",
            "team_away",
            "is_doubleheader",
            "game_number",
        ]
    ].rename(
        columns={
            "game_pk": "GAME_ID",
            "game_date": "GAME_DATE",
            "season_year": "GAME_SEASON_YEAR",
            "first_pitch_utc": "GAME_FIRST_PITCH_UTC",
            "team_home_id": "GAME_HOME_TEAM_ID",
            "team_away_id": "GAME_AWAY_TEAM_ID",
            "team_home": "GAME_HOME_TEAM_NAME",
            "team_away": "GAME_AWAY_TEAM_NAME",
            "is_doubleheader": "GAME_IS_DOUBLEHEADER",
            "game_number": "GAME_NUMBER",
        }
    )
    return renamed.copy()


def _add_book_features(
    output: pd.DataFrame, quotes: pd.DataFrame, book_slug: str
) -> dict[str, pd.Series]:
    book = _book_label(book_slug)
    columns: dict[str, pd.Series] = {}
    game_ids = output["GAME_ID"]
    selected = quotes.loc[quotes["book_slug"].eq(book_slug)].set_index(
        ["game_pk", "market"]
    )

    def market_frame(market: str) -> pd.DataFrame:
        if market not in selected.index.get_level_values("market"):
            return pd.DataFrame(index=pd.Index([], name="game_pk"))
        return selected.xs(market, level="market")

    total = market_frame(MARKET_TOTALS)
    total_prefix = f"ODDS_TOTAL_{book}"
    columns[f"{total_prefix}_HAS_QUOTE"] = game_ids.isin(total.index).astype("int8")
    if not total.empty:
        total_values = {
            f"{total_prefix}_LINE_RAW": total["left_line"],
            f"{total_prefix}_PRICE_OVER_RAW": total["left_price"],
            f"{total_prefix}_PRICE_UNDER_RAW": total["right_price"],
            f"{total_prefix}_FAIR_PROB_OVER": total["fair_left"],
            f"{total_prefix}_FAIR_PROB_UNDER": total["fair_right"],
            f"{total_prefix}_OVERROUND": total["overround"],
            f"{total_prefix}_LINE_NORMALIZED": total["normalized_line"],
            f"{total_prefix}_LINE_NORMALIZED_MINUS_RAW": (
                total["normalized_line"] - total["left_line"]
            ),
            f"{total_prefix}_CLOSE_MINUTES_BEFORE_START": total["minutes_before_start"],
            f"{total_prefix}_CLOSE_TIMESTAMP_UTC": total["line_ts"],
        }
        for column, values in total_values.items():
            columns[column] = game_ids.map(values)

    run_line = market_frame(MARKET_RUN_LINE)
    run_prefix = f"ODDS_RUN_LINE_{book}"
    columns[f"{run_prefix}_HAS_QUOTE"] = game_ids.isin(run_line.index).astype("int8")
    if not run_line.empty:
        run_values = {
            f"{run_prefix}_AWAY_HANDICAP_RAW": run_line["left_line"],
            f"{run_prefix}_HOME_HANDICAP_RAW": run_line["right_line"],
            f"{run_prefix}_PRICE_AWAY_RAW": run_line["left_price"],
            f"{run_prefix}_PRICE_HOME_RAW": run_line["right_price"],
            f"{run_prefix}_FAIR_PROB_AWAY_COVER": run_line["fair_left"],
            f"{run_prefix}_FAIR_PROB_HOME_COVER": run_line["fair_right"],
            f"{run_prefix}_OVERROUND": run_line["overround"],
            f"{run_prefix}_AWAY_HANDICAP_NORMALIZED": run_line["normalized_line"],
            f"{run_prefix}_HOME_HANDICAP_NORMALIZED": -run_line["normalized_line"],
            f"{run_prefix}_HOME_HANDICAP_NORMALIZED_MINUS_RAW": (
                -run_line["normalized_line"] - run_line["right_line"]
            ),
            f"{run_prefix}_CLOSE_MINUTES_BEFORE_START": run_line[
                "minutes_before_start"
            ],
            f"{run_prefix}_CLOSE_TIMESTAMP_UTC": run_line["line_ts"],
        }
        for column, values in run_values.items():
            columns[column] = game_ids.map(values)

    money_line = market_frame(MARKET_MONEY_LINE)
    money_prefix = f"ODDS_MONEY_LINE_{book}"
    columns[f"{money_prefix}_HAS_QUOTE"] = game_ids.isin(money_line.index).astype(
        "int8"
    )
    if not money_line.empty:
        money_values = {
            f"{money_prefix}_PRICE_AWAY_RAW": money_line["left_price"],
            f"{money_prefix}_PRICE_HOME_RAW": money_line["right_price"],
            f"{money_prefix}_FAIR_PROB_AWAY_WIN": money_line["fair_left"],
            f"{money_prefix}_FAIR_PROB_HOME_WIN": money_line["fair_right"],
            f"{money_prefix}_OVERROUND": money_line["overround"],
            f"{money_prefix}_CLOSE_MINUTES_BEFORE_START": money_line[
                "minutes_before_start"
            ],
            f"{money_prefix}_CLOSE_TIMESTAMP_UTC": money_line["line_ts"],
        }
        for column, values in money_values.items():
            columns[column] = game_ids.map(values)

    _ensure_book_schema(columns, output.index, book)
    return columns


def _ensure_book_schema(
    output: dict[str, pd.Series], index: pd.Index, book: str
) -> None:
    """Keep every season partition on one stable per-book schema."""
    total_prefix = f"ODDS_TOTAL_{book}"
    run_prefix = f"ODDS_RUN_LINE_{book}"
    money_prefix = f"ODDS_MONEY_LINE_{book}"
    timestamps = {
        f"{total_prefix}_CLOSE_TIMESTAMP_UTC",
        f"{run_prefix}_CLOSE_TIMESTAMP_UTC",
        f"{money_prefix}_CLOSE_TIMESTAMP_UTC",
    }
    flags = {
        f"{total_prefix}_HAS_QUOTE",
        f"{run_prefix}_HAS_QUOTE",
        f"{money_prefix}_HAS_QUOTE",
    }
    numeric = {
        f"{total_prefix}_LINE_RAW",
        f"{total_prefix}_PRICE_OVER_RAW",
        f"{total_prefix}_PRICE_UNDER_RAW",
        f"{total_prefix}_FAIR_PROB_OVER",
        f"{total_prefix}_FAIR_PROB_UNDER",
        f"{total_prefix}_OVERROUND",
        f"{total_prefix}_LINE_NORMALIZED",
        f"{total_prefix}_LINE_NORMALIZED_MINUS_RAW",
        f"{total_prefix}_CLOSE_MINUTES_BEFORE_START",
        f"{run_prefix}_AWAY_HANDICAP_RAW",
        f"{run_prefix}_HOME_HANDICAP_RAW",
        f"{run_prefix}_PRICE_AWAY_RAW",
        f"{run_prefix}_PRICE_HOME_RAW",
        f"{run_prefix}_FAIR_PROB_AWAY_COVER",
        f"{run_prefix}_FAIR_PROB_HOME_COVER",
        f"{run_prefix}_OVERROUND",
        f"{run_prefix}_AWAY_HANDICAP_NORMALIZED",
        f"{run_prefix}_HOME_HANDICAP_NORMALIZED",
        f"{run_prefix}_HOME_HANDICAP_NORMALIZED_MINUS_RAW",
        f"{run_prefix}_CLOSE_MINUTES_BEFORE_START",
        f"{money_prefix}_PRICE_AWAY_RAW",
        f"{money_prefix}_PRICE_HOME_RAW",
        f"{money_prefix}_FAIR_PROB_AWAY_WIN",
        f"{money_prefix}_FAIR_PROB_HOME_WIN",
        f"{money_prefix}_OVERROUND",
        f"{money_prefix}_CLOSE_MINUTES_BEFORE_START",
    }
    for column in sorted(flags):
        if column not in output:
            output[column] = pd.Series(0, index=index, dtype="int8")
        else:
            output[column] = output[column].astype("int8")
    for column in sorted(numeric):
        if column not in output:
            output[column] = pd.Series(np.nan, index=index, dtype="float64")
        else:
            output[column] = pd.to_numeric(output[column], errors="coerce").astype(
                "float64"
            )
    for column in sorted(timestamps):
        if column not in output:
            output[column] = pd.Series(pd.NaT, index=index, dtype="datetime64[ns, UTC]")
        else:
            output[column] = pd.to_datetime(output[column], utc=True)


def _add_distribution(output: pd.DataFrame, columns: list[str], *, prefix: str) -> None:
    if not columns:
        return
    # Three statistics, not seven: measured across 17,638 games, STD and RANGE
    # correlate 0.99, MEAN and MEDIAN 0.99, MAD is zero in 92.4% of rows and
    # IQR in 71.0%.  Level, disagreement and how many books were quoting are
    # what the remaining four were spelling out.
    values = output[columns].apply(pd.to_numeric, errors="coerce")
    output[f"{prefix}_BOOK_COUNT"] = values.notna().sum(axis=1).astype("int16")
    output[f"{prefix}_MEDIAN"] = values.median(axis=1, skipna=True)
    output[f"{prefix}_STD"] = values.std(axis=1, skipna=True, ddof=0)


def _add_cross_book_features(output: pd.DataFrame) -> None:
    _add_distribution(
        output,
        [
            column
            for column in output.columns
            if column.startswith("ODDS_TOTAL_") and column.endswith("_LINE_RAW")
        ],
        prefix="ODDS_TOTAL_CONSENSUS_LINE_RAW",
    )
    _add_distribution(
        output,
        [
            column
            for column in output.columns
            if column.startswith("ODDS_TOTAL_") and column.endswith("_LINE_NORMALIZED")
        ],
        prefix="ODDS_TOTAL_CONSENSUS_LINE_NORMALIZED",
    )
    _add_distribution(
        output,
        [
            column
            for column in output.columns
            if column.startswith("ODDS_TOTAL_") and column.endswith("_FAIR_PROB_OVER")
        ],
        prefix="ODDS_TOTAL_CONSENSUS_FAIR_PROB_OVER",
    )
    _add_distribution(
        output,
        [
            column
            for column in output.columns
            if column.startswith("ODDS_RUN_LINE_")
            and column.endswith("_HOME_HANDICAP_RAW")
        ],
        prefix="ODDS_RUN_LINE_CONSENSUS_HOME_HANDICAP_RAW",
    )
    _add_distribution(
        output,
        [
            column
            for column in output.columns
            if column.startswith("ODDS_RUN_LINE_")
            and column.endswith("_HOME_HANDICAP_NORMALIZED")
        ],
        prefix="ODDS_RUN_LINE_CONSENSUS_HOME_HANDICAP_NORMALIZED",
    )
    _add_distribution(
        output,
        [
            column
            for column in output.columns
            if column.startswith("ODDS_RUN_LINE_")
            and column.endswith("_FAIR_PROB_HOME_COVER")
        ],
        prefix="ODDS_RUN_LINE_CONSENSUS_FAIR_PROB_HOME_COVER",
    )
    _add_distribution(
        output,
        [
            column
            for column in output.columns
            if column.startswith("ODDS_MONEY_LINE_")
            and column.endswith("_FAIR_PROB_HOME_WIN")
        ],
        prefix="ODDS_MONEY_LINE_CONSENSUS_FAIR_PROB_HOME_WIN",
    )

    total = "ODDS_TOTAL_CONSENSUS_LINE_NORMALIZED_MEDIAN"
    home_handicap = "ODDS_RUN_LINE_CONSENSUS_HOME_HANDICAP_NORMALIZED_MEDIAN"
    if total in output.columns and home_handicap in output.columns:
        # home margin = -home handicap; solve total=home+away, margin=home-away.
        home_margin = -output[home_handicap]
        output["ODDS_DERIVED_IMPLIED_HOME_RUNS_NORMALIZED"] = (
            output[total] + home_margin
        ) / 2.0
        output["ODDS_DERIVED_IMPLIED_AWAY_RUNS_NORMALIZED"] = (
            output[total] - home_margin
        ) / 2.0


def assert_closing_feature_contract(features: pd.DataFrame) -> None:
    """Raise if a feature is unlabeled or a realised outcome survived."""
    unlabeled = [
        column
        for column in features.columns
        if not (column.startswith("GAME_") or column.startswith("ODDS_"))
    ]
    if unlabeled:
        raise ValueError(f"Unlabelled closing feature columns: {unlabeled}")
    forbidden = sorted(_FORBIDDEN_OUTCOME_COLUMNS.intersection(features.columns))
    if forbidden:
        raise ValueError(
            f"Outcome-derived columns reached pregame features: {forbidden}"
        )


def build_closing_line_features(
    odds_games: pd.DataFrame,
    odds_ticks: pd.DataFrame,
    *,
    safety_margin_minutes: int = DEFAULT_CLOSE_SAFETY_MARGIN_MINUTES,
    books: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Create the labeled, one-row-per-game closing-market feature frame."""
    output = _game_frame(odds_games)
    closing = select_closing_quotes(
        odds_ticks, safety_margin_minutes=safety_margin_minutes
    )
    quotes = _with_normalized_market_values(closing)

    quoted_books = {str(value) for value in quotes["book_slug"].dropna().unique()}
    selected_books = sorted(
        quoted_books.intersection(STABLE_BOOKS)
        if books is None
        else {str(value) for value in books}
    )
    labels = [_book_label(book) for book in selected_books]
    if len(labels) != len(set(labels)):
        raise ValueError(
            "Distinct book slugs collide after column-label normalization."
        )
    book_columns: dict[str, pd.Series] = {}
    for book_slug in selected_books:
        book_columns.update(_add_book_features(output, quotes, book_slug))
    if book_columns:
        output = pd.concat(
            [output, pd.DataFrame(book_columns, index=output.index)], axis=1
        )
    _add_cross_book_features(output)

    # Stable order matters when season files are read as one parquet dataset.
    # A book absent in an old season must not reshuffle its columns.
    game_columns = [column for column in output.columns if column.startswith("GAME_")]
    odds_columns = sorted(
        column for column in output.columns if column.startswith("ODDS_")
    )
    output = output[game_columns + odds_columns]

    output = output.sort_values(
        ["GAME_DATE", "GAME_FIRST_PITCH_UTC", "GAME_ID"], kind="mergesort"
    ).reset_index(drop=True)
    assert_closing_feature_contract(output)
    return output


def write_closing_feature_partition(
    features: pd.DataFrame,
    season_year: int,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Atomically write one season of generated closing-line features."""
    if not features.empty and not features["GAME_SEASON_YEAR"].eq(season_year).all():
        raise ValueError("Feature frame contains rows outside the requested season.")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"closing_lines_{season_year}.parquet"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".parquet.tmp", dir=output_dir
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        features.to_parquet(temporary, index=False)
        if pq.ParquetFile(temporary).metadata.num_rows != len(features):
            raise OSError(f"Row-count validation failed for {temporary}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
