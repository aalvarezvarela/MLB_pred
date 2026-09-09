"""Line-movement features, each computable from ticks at or before the snapshot.

Everything here is derived only from ticks with ``minutes_before_start >=
snapshot_minutes``, so nothing can encode a price the bettor could not have
seen.  Two structural choices keep that guarantee cheap to verify.

**The path is accumulated per tick, not recomputed per horizon.**  Every "so
far" quantity -- move counts, running extremes, reversals -- is a cumulative
statistic over the series in chronological order, so the value carried by a tick
*is* the state of the market at that tick.  The as-of read then picks it up for
free.  The NBA implementation instead re-aggregated the eligible prefix once per
horizon, which is O(horizons x ticks) and, more importantly, is a second place
the eligibility filter has to be written correctly.  Here the filter exists
once, in ``line_snapshots.as_of_ticks``.

**Trailing windows are a second as-of read, not a second code path.**  "How far
has the line moved in the last hour, as of T" is exactly ``level(T) -
level(T + 60)``, so it is answered by stacking the same prepared ticks over an
extended grid.  A window is therefore incapable of reaching forward: it is built
from a horizon that is strictly further from first pitch.

Two move counts are kept because they answer different questions, and in
baseball they are not the same question:

* ``n_moves_so_far`` counts changes in ``level`` -- the quantity the market
  actually moves in, which for the run line and the moneyline is a devigged
  probability.
* ``n_line_moves_so_far`` counts changes in the *number on the board*.  For
  totals the two coincide by construction.  For the run line they emphatically
  do not: the handicap is pinned at +/-1.5 on 88% of quotes and differs from its
  close on 8.1% of games, while the cover probability differs on 95.0%.  A
  handicap move is the rare, structurally different event, and collapsing it
  into the price would hide it.

``n_distinct_levels`` is deliberately not emitted.  It is ``n_moves_so_far + 1``
on any path that does not revisit a level, and the two are near-duplicates of
the kind the feature consolidation removed; ``level_range_so_far`` already
describes how far the path spread.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_pred.features.closing_lines import (
    DEFAULT_CLOSE_SAFETY_MARGIN_MINUTES,
    MARKET_MONEY_LINE,
    MARKET_RUN_LINE,
)
from mlb_pred.features.line_snapshots import (
    DEFAULT_SNAPSHOT_GRID,
    GROUP_KEYS,
    prepare_snapshot_ticks,
    schedule_reliability,
    stack_horizons,
    validate_grid,
)

#: Trailing look-back windows, in minutes before the snapshot.
#:
#: NBA's 15- and 30-minute windows are dropped: measured on this store only 3.1%
#: of MLB line moves land inside 30 minutes of first pitch, so those windows
#: would be a near-constant zero that costs five columns each.
#:
#: Cost scales with ``len(grid) * len(windows)``, not with either alone -- every
#: pair needs its own as-of read at ``T + w``.
DEFAULT_MOVEMENT_WINDOWS: tuple[int, ...] = (60, 120, 240, 480)

#: Value used where a window's history does not reach back far enough.  Paired
#: with a ``has_window_`` flag so a model can tell "the market did not move" from
#: "the market did not yet exist", and so the row does not accumulate NaNs: the
#: long horizons are systematically the ones lacking history, and bare NaNs
#: would let a NaN-count cleaner delete precisely the rows a lead-time
#: comparison exists to study.
MISSING_WINDOW_VALUE = 0.0

#: Quantities carried per tick and read at the snapshot.
PATH_STATE_COLUMNS: tuple[str, ...] = (
    "n_moves_so_far",
    "n_line_moves_so_far",
    "n_price_only_so_far",
    "n_reversals_so_far",
    "level_max_so_far",
    "level_min_so_far",
    "level_std_so_far",
    "first_move_direction",
    "opener_level",
    "opener_fair_up",
    "opener_minutes_before_start",
)

#: Quantities derived once the snapshot horizon is known.
PATH_DERIVED_COLUMNS: tuple[str, ...] = (
    "move_from_open",
    "abs_move_from_open",
    "pct_move_from_open",
    "move_direction",
    "prob_move_from_open",
    "minutes_since_open",
    "level_range_so_far",
    "position_in_range",
    "n_moves_per_hour",
    "opposes_opening_direction",
    "net_opposes_opening_direction",
)


def window_feature_columns(
    windows: tuple[int, ...] = DEFAULT_MOVEMENT_WINDOWS,
) -> list[str]:
    """The per-window schema, in order."""
    names: list[str] = []
    for window in windows:
        names.extend(
            [
                f"has_window_{window}",
                f"move_last_{window}",
                f"abs_move_last_{window}",
                f"velocity_last_{window}",
                f"prob_move_last_{window}",
                f"norm_minus_raw_move_last_{window}",
            ]
        )
    return names


def movement_panel_columns(
    windows: tuple[int, ...] = DEFAULT_MOVEMENT_WINDOWS,
) -> list[str]:
    """Everything this module adds to a snapshot panel, in order."""
    columns = [
        *PATH_STATE_COLUMNS,
        *PATH_DERIVED_COLUMNS,
        *window_feature_columns(windows),
    ]
    if 60 in windows and 240 in windows:
        columns.append("move_acceleration")
    return columns


def extended_grid(
    grid: tuple[int, ...] = DEFAULT_SNAPSHOT_GRID,
    windows: tuple[int, ...] = DEFAULT_MOVEMENT_WINDOWS,
) -> tuple[int, ...]:
    """Horizons needed to answer every windowed question asked on ``grid``."""
    return tuple(sorted(set(grid) | {t + w for t in grid for w in windows}))


def _cumulative_std(values: pd.Series, keys: list[pd.Series]) -> pd.Series:
    """Population standard deviation of the path so far.

    ``ddof=0`` rather than pandas' default, because it is defined at n=1 -- a
    single observation has zero realised dispersion, which is the honest reading
    and avoids a fill that would otherwise have to guess.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    present = numeric.notna()
    grouped_present = present.astype("float64").groupby(keys, sort=False)
    count = grouped_present.cumsum()
    total = numeric.fillna(0.0).groupby(keys, sort=False).cumsum()
    square = (numeric.fillna(0.0) ** 2).groupby(keys, sort=False).cumsum()
    safe = count.replace(0.0, np.nan)
    variance = (square / safe) - (total / safe) ** 2
    return np.sqrt(variance.clip(lower=0.0))


def prepare_tick_path(chronological: pd.DataFrame) -> pd.DataFrame:
    """Annotate each tick with the state of its series up to and including it.

    Computed once over the full prepared history.  Filtering to a horizon later
    keeps a chronological *prefix*, and a prefix never changes the predecessor of
    any row it retains, so these columns stay valid under every horizon -- which
    is what makes the per-tick accumulation equivalent to a per-horizon
    aggregation, and considerably cheaper.
    """
    if chronological.empty:
        return chronological

    working = chronological.copy()
    keys = [working[key] for key in GROUP_KEYS]
    grouped_level = working["level"].groupby(keys, sort=False)

    previous_level = grouped_level.shift()
    previous_fair = working["fair_up"].groupby(keys, sort=False).shift()
    previous_line = working["raw_line"].groupby(keys, sort=False).shift()

    working["level_delta"] = working["level"] - previous_level
    working["fair_delta"] = working["fair_up"] - previous_fair
    working["is_move"] = working["level_delta"].notna() & working["level_delta"].ne(0.0)
    working["is_line_move"] = (
        previous_line.notna()
        & working["raw_line"].notna()
        & working["raw_line"].ne(previous_line)
    )
    # A tick that only re-prices the same number.  Books routinely price a
    # half-tick before taking it, and that pressure is invisible in the line.
    working["is_price_only"] = (
        working["level_delta"].notna()
        & working["level_delta"].eq(0.0)
        & working["fair_delta"].notna()
        & working["fair_delta"].ne(0.0)
    )
    # On the run line and the moneyline the level *is* the price, so the two
    # categories collapse: every re-price is a move and "price-only" cannot occur
    # by construction.  Left as a structural zero rather than dropped, so the
    # column stays comparable across markets.
    working.loc[
        working["market"].isin([MARKET_RUN_LINE, MARKET_MONEY_LINE]), "is_price_only"
    ] = False
    working["move_sign"] = np.sign(working["level_delta"].fillna(0.0))

    for source, target in (
        ("is_move", "n_moves_so_far"),
        ("is_line_move", "n_line_moves_so_far"),
        ("is_price_only", "n_price_only_so_far"),
    ):
        working[target] = (
            working[source].astype("float64").groupby(keys, sort=False).cumsum()
        )

    # A reversal is a move whose direction differs from the previous move's.
    # Counted over the moves alone, then carried back onto every tick, so a
    # quiet stretch does not reset it.
    moves = working.loc[working["is_move"]]
    if moves.empty:
        working["n_reversals_so_far"] = 0.0
    else:
        previous_sign = (
            moves["move_sign"]
            .groupby([moves[key] for key in GROUP_KEYS], sort=False)
            .shift()
        )
        flipped = previous_sign.notna() & moves["move_sign"].ne(previous_sign)
        # Assigned positionally rather than reindexed-and-filled: reindexing a
        # boolean Series onto the full frame yields object dtype, and filling
        # that back to bool is the deprecated silent downcast.
        reversal = pd.Series(False, index=working.index, dtype=bool)
        reversal.loc[flipped.index] = flipped.to_numpy()
        working["__reversal"] = reversal
        working["n_reversals_so_far"] = (
            working["__reversal"].astype("float64").groupby(keys, sort=False).cumsum()
        )
        working = working.drop(columns="__reversal")

    working["level_max_so_far"] = grouped_level.cummax()
    working["level_min_so_far"] = grouped_level.cummin()
    working["level_std_so_far"] = _cumulative_std(working["level"], keys)

    # The direction of the FIRST move, not the most recent one.  Only the first
    # move's sign is allowed to propagate: forward-filling the latest instead
    # would silently redefine "opening direction" as "current direction".
    first_signs = working["move_sign"].where(working["is_move"])
    seen = first_signs.notna().astype("float64").groupby(keys, sort=False).cumsum()
    working["first_move_direction"] = (
        first_signs.where(seen.eq(1.0)).groupby(keys, sort=False).ffill().fillna(0.0)
    )

    # The opener is the earliest tick of the series, so it is known at every
    # horizon and a whole-group ``first`` reads nothing a snapshot could not see.
    for source, target in (
        ("level", "opener_level"),
        ("fair_up", "opener_fair_up"),
        ("minutes_before_start", "opener_minutes_before_start"),
    ):
        working[target] = working[source].groupby(keys, sort=False).transform("first")
    return working


def _add_derived_path_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Quantities that need the snapshot horizon as well as the carried state."""
    panel["move_from_open"] = panel["level"] - panel["opener_level"]
    panel["abs_move_from_open"] = panel["move_from_open"].abs()
    # Only meaningful against a non-zero base; a pick'em run line sits at zero,
    # so guard rather than emit an infinity.
    base = panel["opener_level"].replace(0.0, np.nan)
    panel["pct_move_from_open"] = (panel["move_from_open"] / base.abs()).fillna(
        MISSING_WINDOW_VALUE
    )
    panel["move_direction"] = np.sign(panel["move_from_open"].fillna(0.0))
    panel["prob_move_from_open"] = panel["fair_up"] - panel["opener_fair_up"]
    panel["minutes_since_open"] = (
        panel["opener_minutes_before_start"] - panel["snapshot_minutes"]
    )

    panel["level_range_so_far"] = panel["level_max_so_far"] - panel["level_min_so_far"]
    span = panel["level_range_so_far"].replace(0.0, np.nan)
    # A line that never moved is at neither extreme; the midpoint is the honest
    # encoding, and the range itself already records that it did not move.
    panel["position_in_range"] = (
        (panel["level"] - panel["level_min_so_far"]) / span
    ).fillna(0.5)

    # ``n_moves_so_far`` grows as the snapshot approaches first pitch, so it is
    # partly a proxy for elapsed time.  The rate separates a busy market from a
    # longer one.
    elapsed_hours = (panel["minutes_since_open"] / 60.0).replace(0.0, np.nan)
    panel["n_moves_per_hour"] = (panel["n_moves_so_far"] / elapsed_hours).fillna(0.0)
    return panel


def _add_windowed_features(
    panel: pd.DataFrame,
    chronological: pd.DataFrame,
    *,
    grid: tuple[int, ...],
    windows: tuple[int, ...],
) -> pd.DataFrame:
    """Trailing moves, answered by a second as-of read at ``T + w``."""
    lookup = stack_horizons(chronological, extended_grid(grid, windows))
    keys = [*GROUP_KEYS, "snapshot_minutes"]
    if lookup.empty:
        for window in windows:
            for name in window_feature_columns((window,)):
                panel[name] = MISSING_WINDOW_VALUE
        return panel

    lookup = lookup[[*keys, "level", "fair_up", "norm_minus_raw"]].rename(
        columns={
            "level": "level_then",
            "fair_up": "fair_then",
            "norm_minus_raw": "norm_minus_raw_then",
        }
    )

    for window in windows:
        shifted = lookup.copy()
        # The state at (snapshot + window) is the state one window earlier.
        shifted["snapshot_minutes"] = shifted["snapshot_minutes"] - window
        merged = panel[keys].merge(shifted, on=keys, how="left")

        has_history = merged["level_then"].notna().to_numpy()
        panel[f"has_window_{window}"] = has_history.astype("float64")
        move = panel["level"].to_numpy() - merged["level_then"].to_numpy()
        panel[f"move_last_{window}"] = np.where(has_history, move, MISSING_WINDOW_VALUE)
        panel[f"abs_move_last_{window}"] = np.abs(panel[f"move_last_{window}"])
        panel[f"velocity_last_{window}"] = panel[f"move_last_{window}"] / (
            window / 60.0
        )
        panel[f"prob_move_last_{window}"] = np.where(
            has_history,
            panel["fair_up"].to_numpy() - merged["fair_then"].to_numpy(),
            MISSING_WINDOW_VALUE,
        )
        # How the half-tick priced but not taken has itself drifted: pressure
        # building or releasing, which the level alone cannot show.
        panel[f"norm_minus_raw_move_last_{window}"] = np.where(
            has_history,
            panel["norm_minus_raw"].to_numpy()
            - merged["norm_minus_raw_then"].to_numpy(),
            MISSING_WINDOW_VALUE,
        )

    # Recent pace against the longer trend it sits inside.
    if 60 in windows and 240 in windows:
        panel["move_acceleration"] = (
            panel["move_last_60"] - panel["move_last_240"] / 4.0
        )
    return panel


def _add_shape_features(panel: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    """Direction of travel now, against the direction the market first took.

    A reversal after a strong opening move is a different state from a steady
    drift of the same size.  Comparing the recent window against the *net*
    opener-to-now direction would not be the opening direction at all: a line
    that moved up and then further up would score the same as one that moved
    down and then back up past its open.  Both readings are emitted.
    """
    recent = f"move_last_{min(windows)}"
    recent_direction = (
        np.sign(panel[recent]) if recent in panel else panel["move_direction"]
    )
    panel["opposes_opening_direction"] = (
        (recent_direction * panel["first_move_direction"]) < 0
    ).astype("float64")
    panel["net_opposes_opening_direction"] = (
        (panel["move_direction"] * panel["first_move_direction"]) < 0
    ).astype("float64")
    return panel


def build_movement_panel(
    ticks: pd.DataFrame,
    *,
    grid: tuple[int, ...] = DEFAULT_SNAPSHOT_GRID,
    windows: tuple[int, ...] = DEFAULT_MOVEMENT_WINDOWS,
    books: tuple[str, ...] | None = None,
    safety_margin_minutes: int = DEFAULT_CLOSE_SAFETY_MARGIN_MINUTES,
) -> pd.DataFrame:
    """Snapshot panel with the path that led to each snapshot attached."""
    validate_grid(grid)
    if not windows:
        raise ValueError("windows must not be empty.")
    if any(window <= 0 for window in windows):
        # A non-positive window would read the panel at (snapshot - |w|), a
        # LATER moment than the snapshot itself: a direct look-ahead that would
        # pass every column-name check downstream.
        raise ValueError(f"look-back windows must be positive minutes; got {windows}.")

    chronological = prepare_snapshot_ticks(
        ticks, books=books, safety_margin_minutes=safety_margin_minutes
    )
    if chronological.empty:
        return pd.DataFrame()
    chronological = prepare_tick_path(chronological)

    panel = stack_horizons(chronological, grid)
    if panel.empty:
        return pd.DataFrame()
    panel["game_start_is_reliable"] = (
        panel["game_pk"].map(schedule_reliability(chronological)).astype("float64")
    )
    panel = _add_derived_path_features(panel)
    panel = _add_windowed_features(panel, chronological, grid=grid, windows=windows)
    panel = _add_shape_features(panel, windows)
    return panel.reset_index(drop=True)


__all__ = [
    "DEFAULT_MOVEMENT_WINDOWS",
    "MISSING_WINDOW_VALUE",
    "PATH_DERIVED_COLUMNS",
    "PATH_STATE_COLUMNS",
    "build_movement_panel",
    "extended_grid",
    "movement_panel_columns",
    "prepare_tick_path",
    "window_feature_columns",
]
