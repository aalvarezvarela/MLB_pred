#!/usr/bin/env python
"""Daily data update: refresh the last few finished slates.

The daily lifecycle is three *distinct* jobs, and keeping them separate is
what lets realised performance be measured without touching the prediction
path:

1. update data   <- this script
2. predict       (not yet built)
3. settle results (not yet built)

Today's slate is excluded on purpose: a box score fetched mid-game is a
partial box score that looks complete, and written into an insert-only table
it poisons that game permanently.

Examples::

    python scripts/update_databases/update_mlb_data.py
    python scripts/update_databases/update_mlb_data.py --days-back 7
    python scripts/update_databases/update_mlb_data.py --sync-postgres
"""

from __future__ import annotations

import argparse
import sys

from mlb_pred.fetch_data.statsapi.client import RateLimitedError
from mlb_pred.local_store.updaters import update_recent_days


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days-back",
        type=int,
        default=3,
        help="How many finished slates to refresh. More than one on purpose: "
        "games get suspended and resumed and scores get corrected.",
    )
    parser.add_argument(
        "--sync-postgres",
        action="store_true",
        help="Also push the refreshed local store into Postgres.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        results = update_recent_days(args.days_back)
    except RateLimitedError as exc:
        print(f"Stopping: {exc}")
        return 2

    for table, count in sorted(results.items()):
        print(f"{table}: {count} rows")

    if args.sync_postgres:
        from mlb_pred.postgre_db.load import sync_local_store_to_postgres

        print("\nSyncing to Postgres...")
        sync_local_store_to_postgres()

    return 0


if __name__ == "__main__":
    sys.exit(main())
