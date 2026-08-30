#!/usr/bin/env python
"""Backfill the local MLB store for a range of seasons.

Gap-driven: safe to interrupt and re-run. Each run asks the store what is
missing and fetches only that, so a run that stops halfway costs nothing.

Examples::

    python scripts/fetch_data/backfill_mlb_data.py                  # 2015..current
    python scripts/fetch_data/backfill_mlb_data.py --seasons 2024 2025
    python scripts/fetch_data/backfill_mlb_data.py --games-only     # schedule pass
    python scripts/fetch_data/backfill_mlb_data.py --boxscore-limit 50
"""

from __future__ import annotations

import argparse
import sys

from mlb_pred.config.settings import SETTINGS
from mlb_pred.fetch_data.statsapi.client import RateLimitedError
from mlb_pred.local_store.parquet_store import table_summary
from mlb_pred.local_store.updaters import (
    update_boxscores,
    update_games,
    update_transactions,
    update_venues,
)
from mlb_pred.utils.seasons import mlb_slate_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        help="Season years to backfill. Defaults to FIRST_SEASON..current year.",
    )
    parser.add_argument(
        "--games-only",
        action="store_true",
        help="Only refresh games/umpires/lineups/venues; skip box scores.",
    )
    parser.add_argument(
        "--boxscore-limit",
        type=int,
        default=None,
        help="Fetch at most this many box scores per season (smoke runs).",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable the progress bar."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seasons = args.seasons or list(
        range(SETTINGS.first_season, mlb_slate_date().year + 1)
    )

    print(f"Backfilling seasons: {seasons[0]}..{seasons[-1]}")
    SETTINGS.ensure_directories_exist()

    for season_year in seasons:
        print(f"\n=== season {season_year} ===")
        try:
            print(" games/umpires/lineups:", update_games(season_year))
            print(" venues:", update_venues(season_year))
            print(" transactions:", update_transactions(season_year))

            if not args.games_only:
                print(
                    " boxscores:",
                    update_boxscores(
                        season_year,
                        limit=args.boxscore_limit,
                        show_progress=not args.no_progress,
                    ),
                )
        except RateLimitedError as exc:
            # Throttled, not broken. Everything written so far is durable and
            # the gap query resumes from here.
            print(f"\nStopping: {exc}")
            print("Re-run this script later to continue.")
            return 2

    print("\n=== store summary ===")
    print(table_summary().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
