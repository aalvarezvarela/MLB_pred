#!/usr/bin/env python
"""Daily odds update, driven by the planner.

The planner unions three independent reasons to fetch a date -- a rolling
refresh window, gaps against the games table, and partial book coverage. Run
``--dry-run`` to see the whole plan without making a single request.

The refresh window is unconditional on purpose: a game fetched on the morning
it is played is *present* but not *final*, and a presence-based gap check would
never bring it back.

Examples::

    python scripts/update_databases/update_odds.py --dry-run
    python scripts/update_databases/update_odds.py
    python scripts/update_databases/update_odds.py --refresh-days 7
    python scripts/update_databases/update_odds.py --sync-postgres
"""

from __future__ import annotations

import argparse
import sys

from mlb_pred.local_store.odds_updaters import odds_coverage, update_odds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-days", type=int, default=3)
    parser.add_argument("--limit-dates", type=int, default=None)
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help="Fetch only the recent refresh window; ignore historical gaps and "
        "partial-coverage dates.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--sync-postgres",
        action="store_true",
        help="Also push the refreshed odds store into Postgres.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    result = update_odds(
        refresh_days=args.refresh_days,
        dry_run=args.dry_run,
        limit_dates=args.limit_dates,
        refresh_only=args.refresh_only,
    )

    if args.dry_run:
        return 0

    if result.get("written"):
        print(f"\nwritten: {result['written']}")
    for stats in result.get("stats", []):
        print(stats.summary())
    for resolution in result.get("resolution", []):
        print(resolution.summary())

    coverage = odds_coverage()
    if not coverage.empty:
        print("\n=== odds coverage ===")
        print(coverage.to_string(index=False))

    if args.sync_postgres:
        from mlb_pred.postgre_db.odds_load import sync_odds_to_postgres

        print("\nSyncing odds to Postgres...")
        sync_odds_to_postgres()

    return 0


if __name__ == "__main__":
    sys.exit(main())
