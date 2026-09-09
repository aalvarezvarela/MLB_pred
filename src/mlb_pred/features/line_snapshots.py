"""The as-of view of every market at each pre-game snapshot.

``closing_lines`` reduces the tick history to one quote per book, taken as late
as the store allows.  This module answers the same question at an arbitrary
number of minutes before first pitch, so a model can be trained to bet when it
is actually betting rather than only at the close.

The store records line *changes*, not samples: there is no row for "the line at
360 minutes out", only a row each time a book moved.  A snapshot is therefore a
last-observation-carried-forward read, never an equality match, and the age of
the carried quote is itself information -- which is why ``line_age_minutes`` is
emitted beside every quote rather than treated as bookkeeping.

Sign convention
---------------
The store holds ``mins_to_tip`` **negative** before first pitch.  This module
converts once, at the boundary, into ``minutes_before_start`` (positive), the
same restatement ``select_closing_quotes`` performs and for the same reason: the
negative convention must never reach feature code.

That conversion is load-bearing.  ``minutes_before_start`` counts *backwards*,
so within the ticks eligible at a horizon the **largest** value is the earliest
tick and the **smallest** is the most recent.  Reading it the wrong way round
returns the opener at every horizon, which does not look like a bug -- it looks
like a perfectly stable market.  ``_as_of`` sorts chronologically and takes the
last row for exactly this reason, and ``tests/test_line_snapshots.py`` pins the
horizon-zero panel against the shipped closing lines so the mistake cannot
survive a test run.

What each market *moves in*
---------------------------
One canonical ``level`` per market, because a movement or dispersion feature is
only comparable within a market:

* **totals** -- the raw line.  The number on the board is what moves.
* **run_line** -- the devigged HOME cover probability.  MLB run lines are pinned
  at +/-1.5 on 88% of quotes, so the handicap is nearly constant while the book
  re-prices underneath it: measured over 16,000 games the handicap differs from
  its close on 8.1% of games at twelve hours out while the cover probability
  differs on 95.0%.  Using the handicap as the level would zero every move
  count, reversal and dispersion figure for the market.  The handicap is kept
  alongside as ``home_handicap``, because on the ~12% of quotes away from +/-1.5
  it is highly informative about the matchup.
* **money_line** -- the devigged HOME win probability.  There is no line at all.

Run-line and moneyline levels are probabilities while the totals level is runs.
Nothing here compares levels across markets, but that is the reason no
cross-market difference feature should ever be built on ``level``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_pred.features.closing_lines import (
    DEFAULT_CLOSE_SAFETY_MARGIN_MINUTES,
    MARKET_MONEY_LINE,
    MARKET_RUN_LINE,
    MARKET_TOTALS,
    SUPPORTED_MARKETS,
    assert_structural_market_invariants,
    decode_lines,
)
from mlb_pred.features.market_normalization import (
    center_run_lines,
    center_total_lines,
    devig_two_way_series,
)

#: Minutes before first pitch at which the market is sampled.
#:
#: Deliberately denser than any single model needs.  Snapshots are rows, so an
#: unwanted horizon is removed with a filter on ``snapshot_minutes`` -- no
#: rebuild -- whereas adding one back means regenerating the dataset.
#: Over-sampling is the cheap direction.
#:
#: The far end stops at 1080 rather than 1440.  Measured totals coverage over
#: 17,588 games: 99.9% at 120 minutes, 97.6% at 360, 92.1% at 720, 88.2% at
#: 1080, 82.6% at 1260 and 70.9% at 1440.  Beyond ~18 hours the missing games
#: are the ones whose books opened late, which is itself correlated with how a
#: market behaves, so a deeper horizon buys a biased sample rather than lead
#: time.
#:
#: ``0`` is the closing snapshot.  It is not literally first pitch:
#: ``safety_margin_minutes`` still applies, so horizon zero resolves to the last
#: complete quote at least five minutes out.  That makes it this store's closing
#: line and puts the snapshot dataset on the same footing as the closing-line
#: dataset -- same bet, same moment, different feature construction.
DEFAULT_SNAPSHOT_GRID: tuple[int, ...] = (
    0,
    15,
    30,
    45,
    60,
    90,
    120,
    180,
    240,
    300,
    360,
    480,
    600,
    720,
    900,
    1080,
)

GROUP_KEYS = ["game_pk", "market", "book_slug"]

PANEL_COLUMNS: tuple[str, ...] = (
    "game_pk",
    "snapshot_minutes",
    "market",
    "book_slug",
    "raw_line",
    "home_handicap",
    "norm_line",
    "norm_minus_raw",
    "level",
    "price_left",
    "price_right",
    "fair_left",
    "fair_right",
    "fair_up",
    "overround",
    "line_age_minutes",
    "tick_minutes_before_start",
    "n_ticks_so_far",
    "has_quote",
    "game_start_is_reliable",
)

#: How far the first pitch a tick was labelled against may differ, within one
#: game, before that game's timing is called unreliable.
#:
#: ``mins_to_tip`` is computed at ingest against the first pitch known *then*.
#: When a game is postponed or delayed the earlier ticks keep the old
#: reference, so a quote posted six hours out can be stored as "forty minutes
#: before first pitch".  Measured over the store: 13,934 of 3,432,529 pre-game
#: ticks (0.406%) are labelled against a first pitch more than two minutes from
#: the stored one, spread over 86 of 17,588 games (0.49%), with a median worst
#: drift of 175 minutes.  2020 is the worst season at 3.30%, which fits its
#: rescheduling.
#:
#: These games are flagged rather than dropped: the flag is the honest encoding,
#: and the affected horizons are recoverable from ``line_ts`` if a later stage
#: wants them.
SCHEDULE_DRIFT_TOLERANCE_MINUTES = 2.0

_TICK_COLUMNS = {
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
}


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"odds ticks is missing required columns: {missing}")


def resolve_line(ticks: pd.DataFrame) -> pd.Series:
    """Collapse the two stored line columns into one canonical line per row.

    Canonical means *outcome space*, matching ``market_normalization``:

    * **totals** -- the total itself.  Both sides quote the same number, so
      either is sufficient and the second is a fallback for a row where only one
      survived the load-time repairs.
    * **run_line** -- the home-margin threshold the home side must exceed, which
      is SBR's ``left`` (away) line.  ``center_run_lines`` expects exactly this,
      and ``closing_lines`` derives ``HOME_HANDICAP_RAW`` as its negation.
    * **money_line** -- masked.  It carries no line, and making that explicit
      beats relying on both stored columns happening to be null.
    """
    left = decode_lines(ticks["left_line"])
    right = decode_lines(ticks["right_line"])

    # A run line is mirrored, so the away line is the negated home line; a total
    # is quoted identically on both sides.
    mirrored = ticks["market"].eq(MARKET_RUN_LINE)
    right_as_left = right.where(~mirrored, -right)

    resolved = left.where(left.notna(), right_as_left)
    return resolved.mask(ticks["market"].eq(MARKET_MONEY_LINE))


def market_level(frame: pd.DataFrame, line_column: str = "raw_line") -> pd.Series:
    """The canonical quantity each market *moves in*.

    Every movement, dispersion and path feature is computed from this one
    series.  See the module docstring for why the run line is a probability
    rather than its handicap.
    """
    level = pd.to_numeric(frame[line_column], errors="coerce")
    probability_markets = frame["market"].isin([MARKET_RUN_LINE, MARKET_MONEY_LINE])
    if probability_markets.any():
        level = level.mask(
            probability_markets, pd.to_numeric(frame["fair_right"], errors="coerce")
        )
    return level


def market_up_probability(frame: pd.DataFrame) -> pd.Series:
    """Devigged probability of the side that wins when ``level`` goes UP.

    ``left`` does not mean the same thing across markets, so pairing every
    market with ``fair_left`` would invert the sign against ``level`` on two of
    the three:

    * **totals** -- level is the total; up means OVER wins, which is the left
      side.
    * **run_line** -- level *is* the home cover probability, so up means HOME
      covers: the right side.
    * **money_line** -- level *is* the home win probability: the right side
      again.
    """
    up = pd.to_numeric(frame["fair_left"], errors="coerce")
    right_is_up = frame["market"].isin([MARKET_RUN_LINE, MARKET_MONEY_LINE])
    if right_is_up.any():
        up = up.mask(right_is_up, pd.to_numeric(frame["fair_right"], errors="coerce"))
    return up


def eligible_pregame_ticks(
    ticks: pd.DataFrame,
    *,
    safety_margin_minutes: int = DEFAULT_CLOSE_SAFETY_MARGIN_MINUTES,
    books: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Restate the tick store as complete, pre-game, positively-timed quotes.

    The eligibility rules are those of ``select_closing_quotes``, deliberately
    and not by coincidence: horizon zero must reproduce the shipped closing
    line, and it can only do that if "which ticks count" means the same thing in
    both places.  A quote missing a price on either side is not executable and
    is therefore not a quote.
    """
    _require_columns(ticks, _TICK_COLUMNS)
    if safety_margin_minutes < 0:
        raise ValueError("safety_margin_minutes must be non-negative.")
    if ticks.empty:
        return pd.DataFrame(columns=[*ticks.columns, "minutes_before_start"])
    if ticks["is_pregame"].isna().any() or ticks["mins_to_tip"].isna().any():
        raise ValueError(
            "is_pregame and mins_to_tip must be non-null; they are the stored "
            "boundary between pre-game features and in-play leakage."
        )
    unknown_markets = sorted(set(ticks["market"].dropna()) - set(SUPPORTED_MARKETS))
    if unknown_markets:
        raise ValueError(f"Unsupported market(s): {unknown_markets}")
    assert_structural_market_invariants(ticks)

    fair = devig_two_way_series(ticks["left_price"], ticks["right_price"])
    prices_complete = fair[["fair_left", "fair_right"]].notna().all(axis=1)
    lines_complete = ticks[["left_line", "right_line"]].notna().all(axis=1)
    market_complete = ticks["market"].eq(MARKET_MONEY_LINE) | lines_complete

    keep = (
        ticks["is_pregame"].fillna(False).astype(bool)
        & pd.to_numeric(ticks["mins_to_tip"], errors="coerce").le(
            -safety_margin_minutes
        )
        & prices_complete
        & market_complete
    )
    if books is not None:
        keep &= ticks["book_slug"].isin(books)

    working = ticks.loc[keep].copy()
    working["game_pk"] = working["game_pk"].astype(str)
    working["book_slug"] = working["book_slug"].astype(str)
    # The one place the stored negative convention is read.  Everything
    # downstream counts minutes *before* first pitch, positively.
    working["minutes_before_start"] = -pd.to_numeric(
        working["mins_to_tip"], errors="raise"
    ).astype("float64")
    working["raw_line"] = resolve_line(working)
    working["price_left"] = pd.to_numeric(
        working["left_price"], errors="coerce"
    ).astype("float64")
    working["price_right"] = pd.to_numeric(
        working["right_price"], errors="coerce"
    ).astype("float64")
    working["fair_left"] = fair.loc[keep, "fair_left"].to_numpy()
    working["fair_right"] = fair.loc[keep, "fair_right"].to_numpy()
    working["overround"] = fair.loc[keep, "overround"].to_numpy()
    return working


def schedule_reliability(ticks: pd.DataFrame) -> pd.Series:
    """Whether a game's ticks all agree on when first pitch was.

    Each tick implies a first pitch of ``line_ts + minutes_before_start``.  For a
    game whose start never moved, every tick implies the same instant.  A spread
    wider than ``SCHEDULE_DRIFT_TOLERANCE_MINUTES`` means the game was postponed
    or delayed after some ticks were recorded, so those ticks' horizons refer to
    a first pitch that never happened.

    Derived from the ticks alone rather than from ``games``: the panel is built
    from the tick store, and a check that needed a second table would be one a
    caller could forget to supply.
    """
    if ticks.empty:
        return pd.Series(dtype="float64")
    implied = pd.to_datetime(ticks["line_ts"], utc=True) + pd.to_timedelta(
        ticks["minutes_before_start"], unit="m"
    )
    spread = implied.groupby(ticks["game_pk"]).agg(lambda s: s.max() - s.min())
    drift_minutes = spread.dt.total_seconds() / 60.0
    return (drift_minutes <= SCHEDULE_DRIFT_TOLERANCE_MINUTES).astype("float64")


def as_of_ticks(chronological: pd.DataFrame, snapshot_minutes: int) -> pd.DataFrame:
    """Latest eligible tick at least ``snapshot_minutes`` before first pitch.

    ``>=`` is the leakage filter, inclusive of the boundary only: a tick exactly
    at the horizon was observable, a tick one minute later was not.

    ``chronological`` must be sorted by ``line_ts`` ASCENDING within each group,
    so the last row of a group is the most recent tick.

    Ordering is by wall clock rather than by ``minutes_before_start`` even though
    the filter uses the latter.  The two disagree on rescheduled games, where
    early ticks keep a reference to a first pitch that moved -- see
    ``SCHEDULE_DRIFT_TOLERANCE_MINUTES``.  ``line_ts`` is the observed truth and
    is what ``select_closing_quotes`` orders by, so following it keeps horizon
    zero equal to the shipped close.
    """
    eligible = chronological.loc[
        chronological["minutes_before_start"] >= snapshot_minutes
    ]
    if eligible.empty:
        return eligible

    grouped = eligible.groupby(GROUP_KEYS, sort=False)
    latest = grouped.tail(1).copy()
    latest["n_ticks_so_far"] = (
        grouped.size()
        .reindex(pd.MultiIndex.from_frame(latest[GROUP_KEYS]))
        .to_numpy()
        .astype("float64")
    )
    return latest


def _add_normalized_lines(
    ticks: pd.DataFrame, *, normalize_totals: bool, normalize_run_lines: bool
) -> pd.DataFrame:
    """Restate each quote at its equal-price equivalent line.

    The raw line is kept regardless: it is the one that could actually be bet,
    while the centered one is the one comparable across books and horizons.  The
    difference between them -- the half-tick a book has priced but not yet taken
    -- is carried explicitly as ``norm_minus_raw``, because it is invisible in
    the line itself and measured here it is the strongest single market-internal
    predictor of where the line goes next.
    """
    ticks["norm_line"] = np.nan
    totals = ticks["market"].eq(MARKET_TOTALS)
    run_lines = ticks["market"].eq(MARKET_RUN_LINE)

    if normalize_totals and totals.any():
        ticks.loc[totals, "norm_line"] = center_total_lines(
            ticks.loc[totals, "raw_line"],
            ticks.loc[totals, "price_left"],
            ticks.loc[totals, "price_right"],
        )
    elif totals.any():
        ticks.loc[totals, "norm_line"] = ticks.loc[totals, "raw_line"]

    if normalize_run_lines and run_lines.any():
        # SBR left = AWAY, and its line is the canonical home-margin threshold.
        ticks.loc[run_lines, "norm_line"] = center_run_lines(
            ticks.loc[run_lines, "raw_line"],
            ticks.loc[run_lines, "price_left"],
            ticks.loc[run_lines, "price_right"],
        )
    elif run_lines.any():
        ticks.loc[run_lines, "norm_line"] = ticks.loc[run_lines, "raw_line"]

    # A one-sided or unrepresentable quote leaves the centering undefined.
    # Falling back to the raw line keeps the column populated and loses only the
    # small pricing correction, which beats dropping the row.
    ticks["norm_line"] = ticks["norm_line"].fillna(ticks["raw_line"])
    # Zero rather than NaN on the moneyline, which has no line to correct.
    ticks["norm_minus_raw"] = (ticks["norm_line"] - ticks["raw_line"]).fillna(0.0)
    return ticks


def prepare_snapshot_ticks(
    ticks: pd.DataFrame,
    *,
    books: tuple[str, ...] | None = None,
    safety_margin_minutes: int = DEFAULT_CLOSE_SAFETY_MARGIN_MINUTES,
    normalize_totals: bool = True,
    normalize_run_lines: bool = True,
) -> pd.DataFrame:
    """Eligible ticks, in chronological order, with every derived quantity.

    Deliberately does all the per-tick arithmetic -- decoded lines, devigged
    probabilities, centered lines, the canonical ``level`` -- *before* any
    horizon is considered.  That leaves the as-of read as the single
    horizon-dependent step in the pipeline, so there is exactly one place a
    look-ahead can enter and exactly one place to test.

    It also means the movement layer reads the same numbers the panel does.
    Computing ``level`` twice, once per consumer, is how a windowed feature ends
    up measured against a different definition of the line than the snapshot it
    is attached to.
    """
    working = eligible_pregame_ticks(
        ticks, safety_margin_minutes=safety_margin_minutes, books=books
    )
    if working.empty:
        return working

    working = _add_normalized_lines(
        working,
        normalize_totals=normalize_totals,
        normalize_run_lines=normalize_run_lines,
    )
    working["level"] = market_level(working)
    working["fair_up"] = market_up_probability(working)
    # The home side's handicap, the shape ``closing_lines`` reports.  Kept even
    # though the run-line level is a probability: away from +/-1.5 the number
    # itself carries the matchup.
    working["home_handicap"] = -working["raw_line"]
    working.loc[working["market"].eq(MARKET_TOTALS), "home_handicap"] = np.nan
    # Explicit availability, so "this book had no quote at T" is a value a model
    # can read rather than a NaN that row-level cleaning may act on.
    working["has_quote"] = working["level"].notna().astype("float64")

    # Chronological within each series: earliest first, so ``tail(1)`` is "most
    # recent".
    return working.sort_values([*GROUP_KEYS, "line_ts"], kind="mergesort")


def build_snapshot_panel(
    ticks: pd.DataFrame,
    *,
    grid: tuple[int, ...] = DEFAULT_SNAPSHOT_GRID,
    books: tuple[str, ...] | None = None,
    safety_margin_minutes: int = DEFAULT_CLOSE_SAFETY_MARGIN_MINUTES,
    normalize_totals: bool = True,
    normalize_run_lines: bool = True,
) -> pd.DataFrame:
    """Long panel: one row per (game, market, book, snapshot).

    ``books`` defaults to every book in the store.  Restricting it here would
    silently change the consensus, and an absent book should lower a book count
    rather than vanish -- the choice of which books earn their own columns
    belongs to the pivot step, not to the panel.
    """
    validate_grid(grid)
    chronological = prepare_snapshot_ticks(
        ticks,
        books=books,
        safety_margin_minutes=safety_margin_minutes,
        normalize_totals=normalize_totals,
        normalize_run_lines=normalize_run_lines,
    )
    if chronological.empty:
        return pd.DataFrame(columns=list(PANEL_COLUMNS))

    panel = stack_horizons(chronological, grid)
    if panel.empty:
        return pd.DataFrame(columns=list(PANEL_COLUMNS))
    # Per game, not per row: a delayed start mislabels the whole game's history.
    panel["game_start_is_reliable"] = (
        panel["game_pk"].map(schedule_reliability(chronological)).astype("float64")
    )
    return panel[list(PANEL_COLUMNS)].reset_index(drop=True)


def validate_grid(grid: tuple[int, ...]) -> None:
    """Refuse a grid that cannot describe a pre-game moment."""
    if not grid:
        raise ValueError("grid must not be empty.")
    # A negative horizon would place the snapshot *after* first pitch and admit
    # in-play ticks: a direct look-ahead that every downstream column-name check
    # would pass.
    if any(minutes < 0 for minutes in grid):
        raise ValueError(
            "snapshot horizons must be non-negative minutes before first pitch; "
            f"a negative horizon reads the market after it starts. Got {grid}."
        )


def stack_horizons(chronological: pd.DataFrame, grid: tuple[int, ...]) -> pd.DataFrame:
    """Run the as-of read once per horizon and stack the results.

    Shared with the movement layer, which stacks the same prepared ticks over an
    extended grid to answer its trailing windows.  One implementation means one
    leakage surface.
    """
    frames = []
    for snapshot_minutes in sorted(set(grid)):
        latest = as_of_ticks(chronological, snapshot_minutes)
        if latest.empty:
            continue
        frames.append(latest.assign(snapshot_minutes=int(snapshot_minutes)))
    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.rename(columns={"minutes_before_start": "tick_minutes_before_start"})
    # How stale the carried quote already is at the snapshot instant.
    panel["line_age_minutes"] = (
        panel["tick_minutes_before_start"] - panel["snapshot_minutes"]
    )
    return panel


def snapshot_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows and distinct games per (market, snapshot).

    An acceptance check rather than a feature: coverage must not fall off a
    cliff at the long horizons, because a snapshot that only exists for
    well-covered games is a biased sample rather than a longer lead time.
    """
    if panel.empty:
        return pd.DataFrame(
            columns=["market", "snapshot_minutes", "rows", "games", "books"]
        )
    return (
        panel.groupby(["market", "snapshot_minutes"], sort=True)
        .agg(
            rows=("game_pk", "size"),
            games=("game_pk", "nunique"),
            books=("book_slug", "nunique"),
        )
        .reset_index()
    )


__all__ = [
    "DEFAULT_SNAPSHOT_GRID",
    "as_of_ticks",
    "SCHEDULE_DRIFT_TOLERANCE_MINUTES",
    "GROUP_KEYS",
    "PANEL_COLUMNS",
    "build_snapshot_panel",
    "eligible_pregame_ticks",
    "market_level",
    "market_up_probability",
    "prepare_snapshot_ticks",
    "resolve_line",
    "schedule_reliability",
    "stack_horizons",
    "validate_grid",
    "snapshot_coverage",
]
