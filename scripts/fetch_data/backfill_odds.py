#!/usr/bin/env python
"""Backfill SportsbookReview odds into the local store, season by season.

Order matters and is not negotiable: the games table is the reference for what
should exist, so it must be backfilled first. Odds have no useful identity
until they can be joined to a ``game_pk``.

Measured, not assumed: SBR lists MLB slates back to 2015 but carries **no
prices before 2019** -- 2015 and 2018 slates return games with zero ticks.
The default start is therefore 2019.

Examples::

    python scripts/fetch_data/backfill_odds.py                    # 2019..current
    python scripts/fetch_data/backfill_odds.py --seasons 2024 2025
    python scripts/fetch_data/backfill_odds.py --seasons 2025 --limit-dates 5
    python scripts/fetch_data/backfill_odds.py --dry-run
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from mlb_pred.local_store.odds_updaters import ingest_dates, odds_coverage
from mlb_pred.local_store.parquet_store import read_table
from mlb_pred.odds.planner import FIRST_ODDS_SEASON, gap_dates
from mlb_pred.utils.seasons import mlb_slate_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    parser.add_argument(
        "--limit-dates",
        type=int,
        default=None,
        help="Fetch at most this many slate dates per season (smoke runs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the dates that would be fetched and stop.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on the first scrape failure instead of warning and "
        "continuing. A long backfill should not use this.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    games = read_table("games")
    if games.empty:
        print(
            "The games table is empty. Run scripts/fetch_data/backfill_mlb_data.py "
            "first -- odds cannot be resolved without it."
        )
        return 1

    seasons = args.seasons or list(
        range(FIRST_ODDS_SEASON, mlb_slate_date().year + 1)
    )
    fetches = read_table("odds_fetches")
    ticks = read_table("odds_ticks")

    for season_year in seasons:
        season_games = games[games["season_year"] == season_year]
        if "is_final" in season_games.columns:
            season_games = season_games[season_games["is_final"]]
        if season_games.empty:
            print(f"\n=== season {season_year}: no games in the games table, skipping")
            continue

        # An lh_game dimension row is not a completion marker: a rejected or
        # interrupted scrape can create it without any usable ticks. Explicit
        # backfills also intentionally ignore the live planner's earliest-store
        # lower bound so requesting 2019 still works when collection began later.
        days, missing_games = gap_dates(
            season_games,
            pd.DataFrame(),
            first_season=FIRST_ODDS_SEASON,
            odds_fetches=fetches,
            odds_ticks=ticks,
        )
        if args.limit_dates is not None:
            days = days[: args.limit_dates]

        print(
            f"\n=== season {season_year}: {missing_games} incomplete game(s) "
            f"in the odds store across {len(days)} slate date(s) ==="
        )
        if not days:
            continue
        if args.dry_run:
            print(f"  would fetch {days[0]} .. {days[-1]}")
            continue

        result = ingest_dates(
            days, games=games, on_error="raise" if args.strict else "warn"
        )
        print(f"  written: {result['written']}")
        for stats in result["stats"]:
            print("  " + stats.summary().replace("\n", "\n  "))
        for resolution in result["resolution"]:
            print("  " + resolution.summary().replace("\n", "\n  "))
        fetches = read_table("odds_fetches")
        ticks = read_table("odds_ticks")

    if not args.dry_run:
        coverage = odds_coverage()
        if not coverage.empty:
            print("\n=== odds coverage ===")
            print(coverage.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
