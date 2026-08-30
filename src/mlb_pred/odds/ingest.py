"""Scraped games -> tick rows, with every drop and repair counted by reason.

Three things happen here and nowhere else:

1. **Encoding.** Lines are doubled to ``SMALLINT``; the "off the board"
   sentinel becomes NULL at this boundary rather than downstream where each
   consumer would have to remember it.
2. **The pre-game discriminator is materialised.** SBR records in-play ticks
   with exactly the same shape as pre-game ones -- nothing else in the row
   separates a legitimate feature from direct target leakage. So
   ``mins_to_tip`` and ``is_pregame`` are computed at load and stored NOT NULL.
   A runtime filter can be forgotten at one call site; a NOT NULL column
   cannot.
3. **Repairs justified by market invariants**, not by magnitude heuristics
   where an invariant exists. A genuine run line is mirrored
   (``left == -right``); a genuine total is identical on both sides. A row
   violating its market's invariant is not a line, whatever the number says.

Every drop is counted into :class:`IngestStats` and persisted, so "we lost 4%
of season 2023" is answerable rather than tribal knowledge.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from mlb_pred.fetch_data.sbr.line_history import (
    MARKET_MONEYLINE,
    MARKET_RUN_LINE,
    MARKET_TOTALS,
)
from mlb_pred.odds.encoding import LineEncodingError, encode_line, encode_price

#: Plausibility bands for *pre-game* quotes. A dropped decimal turns 8.5 into
#: 85, which is what these catch. Deliberately wide: the point is to catch the
#: impossible, not to second-guess the market.
TOTALS_BOUNDS = (3.0, 25.0)
RUN_LINE_BOUNDS = (-10.0, 10.0)

#: In-play rows are exempt from the bands above. A live run line legitimately
#: blows out during a rout, and nulling it would be destroying data, not
#: repairing it. Only the impossible value is ever cleared -- and the field is
#: cleared, never the row.

TICK_COLUMNS = [
    "game_pk",
    "season_year",
    "game_date",
    "market",
    "book_slug",
    "line_ts",
    "mins_to_tip",
    "is_pregame",
    "is_opener",
    "left_line",
    "left_price",
    "right_line",
    "right_price",
]


@dataclass
class IngestStats:
    """Row counts in and out, and the reason for every difference."""

    source_ticks: int = 0
    loaded_ticks: int = 0
    dropped: Counter = field(default_factory=Counter)
    repaired: Counter = field(default_factory=Counter)
    games_seen: int = 0
    games_loaded: int = 0
    books: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        lines = [
            f"games {self.games_loaded}/{self.games_seen}, "
            f"ticks {self.loaded_ticks}/{self.source_ticks}"
        ]
        for reason, count in self.dropped.most_common():
            lines.append(f"  dropped [{reason}]: {count}")
        for reason, count in self.repaired.most_common():
            lines.append(f"  repaired [{reason}]: {count}")
        if self.books:
            lines.append(f"  books: {dict(self.books.most_common())}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "source_ticks": self.source_ticks,
            "loaded_ticks": self.loaded_ticks,
            "dropped": dict(self.dropped),
            "repaired": dict(self.repaired),
            "games_seen": self.games_seen,
            "games_loaded": self.games_loaded,
            "books": dict(self.books),
        }


def _within(value: float | None, bounds: tuple[float, float]) -> bool:
    return value is None or bounds[0] <= value <= bounds[1]


def _validate_market_invariant(tick, stats: IngestStats) -> bool:
    """Reject a row whose numbers cannot be that market's line.

    Structural, not magnitude-based: whatever the value is, a run line that is
    not mirrored and a total that differs between its two sides are not lines.
    """
    if tick.market == MARKET_TOTALS:
        if (
            tick.left_line is not None
            and tick.right_line is not None
            and abs(tick.left_line - tick.right_line) > 1e-9
        ):
            stats.dropped["totals_sides_disagree"] += 1
            return False
    elif tick.market == MARKET_RUN_LINE:
        if (
            tick.left_line is not None
            and tick.right_line is not None
            and abs(tick.left_line + tick.right_line) > 1e-9
        ):
            # A mirrored pair is the definition of a spread. A complementary
            # *price* pair (-110/-110) landing in the line fields is the known
            # failure mode; either way this is not a run line.
            stats.dropped["run_line_not_mirrored"] += 1
            return False
    elif tick.market == MARKET_MONEYLINE:
        if tick.left_line is not None or tick.right_line is not None:
            stats.repaired["moneyline_line_cleared"] += 1
    return True


def _bounds_for(market: str) -> tuple[float, float] | None:
    if market == MARKET_TOTALS:
        return TOTALS_BOUNDS
    if market == MARKET_RUN_LINE:
        return RUN_LINE_BOUNDS
    return None


def tick_rows_for_game(scraped, game_pk: str, stats: IngestStats) -> list[dict]:
    """One :class:`ScrapedGame` -> encoded tick rows."""
    rows: list[dict] = []

    for tick in scraped.ticks:
        stats.source_ticks += 1

        if not _validate_market_invariant(tick, stats):
            continue

        left_line = tick.left_line
        right_line = tick.right_line

        if tick.market == MARKET_MONEYLINE:
            # No line exists on a moneyline by nature.
            left_line = right_line = None
        else:
            bounds = _bounds_for(tick.market)
            # In-play rows are exempt: the band describes a pre-game market.
            if bounds is not None and tick.is_pregame:
                if not _within(left_line, bounds) or not _within(right_line, bounds):
                    stats.repaired["line_out_of_bounds_cleared"] += 1
                    left_line = right_line = None

        try:
            encoded_left = encode_line(left_line)
            encoded_right = encode_line(right_line)
        except LineEncodingError:
            # A finer quote increment invalidates the storage encoding. Stop the
            # load loudly so the game cannot be marked complete with missing rows.
            stats.dropped["line_not_half_point"] += 1
            raise

        left_price = encode_price(tick.left_price)
        right_price = encode_price(tick.right_price)

        if (
            encoded_left is None
            and encoded_right is None
            and left_price is None
            and right_price is None
        ):
            stats.dropped["empty_quote"] += 1
            continue

        stats.books[tick.book_slug] += 1
        rows.append(
            {
                "game_pk": game_pk,
                "season_year": scraped.season_year,
                "game_date": scraped.game_date,
                "market": tick.market,
                "book_slug": tick.book_slug,
                "line_ts": tick.line_ts,
                # NOT NULL by construction: these are the only thing separating
                # a legitimate feature row from target leakage.
                "mins_to_tip": tick.minutes_to_tip,
                "is_pregame": tick.is_pregame,
                "is_opener": tick.is_opener,
                "left_line": encoded_left,
                "left_price": left_price,
                "right_line": encoded_right,
                "right_price": right_price,
            }
        )

    stats.loaded_ticks += len(rows)
    return rows


def build_odds_frames(
    scraped_games, games: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, IngestStats, object]:
    """Scraped games + the games table -> ``(game dim, tick fact, stats, resolution)``."""
    from mlb_pred.odds.identity import ResolutionStats, build_game_index, resolve_event

    index = build_game_index(games)
    resolution = ResolutionStats()
    stats = IngestStats()

    dimension_rows: list[dict] = []
    tick_rows: list[dict] = []

    for scraped in scraped_games:
        stats.games_seen += 1
        resolved = resolve_event(scraped, index, resolution)
        if resolved is None:
            continue

        source_tick_count = len(scraped.ticks)
        rows = tick_rows_for_game(scraped, resolved["game_pk"], stats)
        if not rows:
            # A resolved game with no usable tick is still worth recording in
            # the dimension: it is the difference between "no odds existed" and
            # "we never looked".
            stats.dropped["game_without_usable_ticks"] += 1

        resolved.update(
            {
                "ingest_status": (
                    "complete"
                    if rows
                    else ("no_quotes" if source_tick_count == 0 else "rejected")
                ),
                "source_tick_count": source_tick_count,
                "usable_tick_count": len(rows),
                "dropped_tick_count": source_tick_count - len(rows),
                "fetched_at_utc": datetime.now(UTC),
            }
        )
        dimension_rows.append(resolved)
        tick_rows.extend(rows)
        stats.games_loaded += 1

    dimension = pd.DataFrame(dimension_rows)
    ticks = normalize_opener_flags(pd.DataFrame(tick_rows, columns=TICK_COLUMNS))
    return dimension, ticks, stats, resolution


def normalize_opener_flags(ticks: pd.DataFrame) -> pd.DataFrame:
    """Derive exactly one opener from the complete stored tick history."""
    if ticks.empty:
        return ticks
    result = ticks.copy()
    keys = ["game_pk", "market", "book_slug"]
    result["is_opener"] = False
    first_index = result.groupby(keys, sort=False)["line_ts"].idxmin()
    result.loc[first_index, "is_opener"] = True
    return result
