"""Cross-book consensus, dispersion and steam, as of each snapshot.

Aggregates the per-book movement panel down to one row per
``(game, market, snapshot)``.  Everything is computed from the books' states at
the horizon, so nothing here can see a price that had not been posted.

The consensus is a **median**, not a mean.  A single stale book sitting a run
off the market is common in this store -- ``line_age_minutes`` routinely runs to
hours -- and a mean would drag the consensus toward it.  Dispersion is reported
separately, precisely so disagreement stays visible instead of being averaged
away.

"Steam" is the count and share of books that moved the same way inside a fixed
recent window.  Cross-book agreement over a short window is the classic
sharp-money signature, and unlike most such signals it is fully observable at
the horizon.

Aggregation is on ``level``, the canonical quantity each market moves in, rather
than on the raw line.  Using the raw line would leave the moneyline with nothing
to aggregate -- it has no line -- and would make every run-line dispersion
figure a near-constant, since 88% of run-line quotes are pinned at +/-1.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONSENSUS_KEYS = ["game_pk", "market", "snapshot_minutes"]

#: The window steam is measured over.
#:
#: Pinned rather than derived from "the shortest window configured".  Steam is
#: *cross-book agreement*, which needs enough books to have moved for agreement
#: to mean anything, and letting the shortest configured window define it means
#: that merely adding a shorter window to the movement config silently redefines
#: an existing feature.  That happened in the NBA repo when its 15- and
#: 30-minute windows were introduced.
STEAM_WINDOW_MINUTES = 60

#: A book this many population standard deviations from the consensus is either
#: the stale one or the sharp one.  The model is better placed than we are to
#: decide which, but it can only do that if the deviation is given to it.
OUTLIER_Z_THRESHOLD = 1.5

CONSENSUS_COLUMNS: tuple[str, ...] = (
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

DEVIATION_COLUMNS: tuple[str, ...] = (
    "deviation_from_consensus",
    "abs_deviation_from_consensus",
    "deviation_z",
    "is_outlier_book",
)


def steam_move_column(panel: pd.DataFrame) -> str:
    """The ``move_last_<w>`` column steam is measured over.

    Prefers ``STEAM_WINDOW_MINUTES`` and falls back to the shortest window on
    offer.  The fallback is why this is derived rather than hardcoded: a caller
    configuring ``windows=(120, 480)`` would otherwise hit a bare ``KeyError``
    deep inside the aggregation.
    """
    candidates: list[tuple[int, str]] = []
    for column in panel.columns:
        if column.startswith("move_last_"):
            suffix = column.removeprefix("move_last_")
            if suffix.isdigit():
                candidates.append((int(suffix), column))
    if not candidates:
        raise ValueError(
            "No move_last_<minutes> column found; build the movement panel "
            "before aggregating across books."
        )
    preferred = f"move_last_{STEAM_WINDOW_MINUTES}"
    if any(column == preferred for _, column in candidates):
        return preferred
    return min(candidates)[1]


def _steam_counts(panel: pd.DataFrame, move_column: str) -> pd.DataFrame:
    """Directional agreement across the books quoting at each snapshot.

    ``steam_fraction`` is the share of *quoting* books moving in the dominant
    direction: 0.0 when nobody moved, 0.5 on an even two-up/two-down split of
    four, 1.0 when every quoting book agreed.  It is deliberately not normalised
    by the number of movers, so "two books moved and agreed" scores below "five
    books moved and agreed" -- but that also means it cannot by itself separate
    a quiet market from a split one, which is why ``steam_movers`` and
    ``steam_agreement`` sit beside it.

    Written as grouped sums rather than a ``groupby.apply``.  The panel has
    hundreds of thousands of (game, market, snapshot) groups, and constructing a
    Series per group cost more than every other aggregation here combined.
    """
    directions = np.sign(panel[move_column].fillna(0.0))
    keys = [panel[key] for key in CONSENSUS_KEYS]
    counts = (
        pd.DataFrame(
            {
                "steam_books_up": directions.gt(0).astype("float64"),
                "steam_books_down": directions.lt(0).astype("float64"),
                "__books": 1.0,
            }
        )
        .groupby(keys, sort=False)
        .sum()
    )

    dominant = counts[["steam_books_up", "steam_books_down"]].max(axis=1)
    movers = counts["steam_books_up"] + counts["steam_books_down"]
    counts["steam_net"] = counts["steam_books_up"] - counts["steam_books_down"]
    counts["steam_movers"] = movers
    counts["steam_fraction"] = dominant / counts["__books"].replace(0.0, np.nan)
    counts["steam_agreement"] = (dominant / movers.replace(0.0, np.nan)).fillna(0.0)
    return counts.drop(columns="__books")


def aggregate_across_books(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, market, snapshot) summarising every book at T."""
    if panel.empty:
        return pd.DataFrame(columns=[*CONSENSUS_KEYS, *CONSENSUS_COLUMNS])

    move_column = steam_move_column(panel)
    grouped = panel.groupby(CONSENSUS_KEYS, sort=False)

    consensus = grouped.agg(
        consensus_level=("level", "median"),
        consensus_raw_line=("raw_line", "median"),
        consensus_norm_line=("norm_line", "median"),
        consensus_norm_minus_raw=("norm_minus_raw", "median"),
        consensus_home_handicap=("home_handicap", "median"),
        consensus_fair_up=("fair_up", "median"),
        consensus_overround=("overround", "median"),
        __level_sum=("level", "sum"),
        __level_max=("level", "max"),
        __level_min=("level", "min"),
        n_books_quoting=("level", "count"),
        median_line_age=("line_age_minutes", "median"),
        max_line_age=("line_age_minutes", "max"),
        consensus_move_from_open=("move_from_open", "median"),
        consensus_move_recent=(move_column, "median"),
        consensus_n_moves=("n_moves_so_far", "median"),
        consensus_n_line_moves=("n_line_moves_so_far", "median"),
        consensus_opener_level=("opener_level", "median"),
    )
    # Population std, from grouped sums rather than a lambda: with two books
    # quoting, a sample std would report the gap between them inflated by
    # sqrt(2) for no reason, and a per-group lambda over this many groups is the
    # single slowest thing in the module.
    squares = (
        (panel["level"] ** 2)
        .groupby([panel[key] for key in CONSENSUS_KEYS], sort=False)
        .sum()
    )
    count = consensus["n_books_quoting"].replace(0, np.nan)
    variance = (squares / count) - (consensus["__level_sum"] / count) ** 2
    consensus["crossbook_std"] = np.sqrt(variance.clip(lower=0.0))
    consensus["crossbook_range"] = consensus["__level_max"] - consensus["__level_min"]
    consensus = consensus.drop(columns=["__level_sum", "__level_max", "__level_min"])
    consensus = consensus.join(_steam_counts(panel, move_column))

    # A single quoting book disagrees with nobody; that is zero dispersion, not
    # missing data.
    consensus["crossbook_std"] = consensus["crossbook_std"].fillna(0.0)
    consensus["crossbook_range"] = consensus["crossbook_range"].fillna(0.0)
    return consensus.reset_index()[[*CONSENSUS_KEYS, *CONSENSUS_COLUMNS]]


def add_book_deviation(panel: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    """Each book's distance from the consensus at the same instant."""
    if panel.empty:
        return panel

    merged = panel.merge(
        consensus[[*CONSENSUS_KEYS, "consensus_level", "crossbook_std"]],
        on=CONSENSUS_KEYS,
        how="left",
        validate="many_to_one",
    )
    merged["deviation_from_consensus"] = merged["level"] - merged["consensus_level"]
    merged["abs_deviation_from_consensus"] = merged["deviation_from_consensus"].abs()
    # Scale-free, because "half a run off" means something different on a tight
    # market than on a scattered one -- and the raw gap is not comparable across
    # markets at all, now that two of the three are measured in probability.
    spread = merged["crossbook_std"].replace(0.0, np.nan)
    merged["deviation_z"] = (merged["deviation_from_consensus"] / spread).fillna(0.0)
    merged["is_outlier_book"] = (
        merged["deviation_z"].abs() > OUTLIER_Z_THRESHOLD
    ).astype("float64")
    return merged.drop(columns=["consensus_level", "crossbook_std"])


__all__ = [
    "CONSENSUS_COLUMNS",
    "CONSENSUS_KEYS",
    "DEVIATION_COLUMNS",
    "OUTLIER_Z_THRESHOLD",
    "STEAM_WINDOW_MINUTES",
    "add_book_deviation",
    "aggregate_across_books",
    "steam_move_column",
]
