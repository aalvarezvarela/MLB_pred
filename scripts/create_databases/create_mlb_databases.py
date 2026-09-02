#!/usr/bin/env python
"""Create every MLB Postgres schema and table.

Targets whichever environment ``[Database] DB_ENV`` (or the ``DB_ENV``
environment variable) selects. The shipped default is ``local``; select a
hosted environment only once its credentials and TLS settings are ready.

Examples::

    python scripts/create_databases/create_mlb_databases.py
    DB_ENV=aiven python scripts/create_databases/create_mlb_databases.py
    python scripts/create_databases/create_mlb_databases.py --drop-existing
    python scripts/create_databases/create_mlb_databases.py --sync
"""

from __future__ import annotations

import argparse
import sys

from mlb_pred.postgre_db.config.db_config import connect_mlb_db, get_db_env
from mlb_pred.postgre_db.odds_schema import (
    create_schema as create_odds_schema,
)
from mlb_pred.postgre_db.odds_schema import (
    drop_schema as drop_odds_schema,
)
from mlb_pred.postgre_db.schema import create_all_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="DROP each table before recreating it. Destroys data.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="After creating, push the local Parquet store into Postgres.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        metavar="YEAR",
        help="Restrict both sports and odds synchronization to these MLB seasons.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare remote row counts with local Parquet for --seasons.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = get_db_env()

    seasons = sorted(set(args.seasons or [])) or None
    if seasons and any(year < 1876 or year > 2100 for year in seasons):
        print(f"Invalid MLB season list: {seasons}")
        return 2

    if args.drop_existing:
        answer = input(
            f"--drop-existing will DROP every MLB table in the '{env}' "
            "database. Type the environment name to confirm: "
        )
        if answer.strip() != env:
            print("Aborted.")
            return 1

    print(f"Creating MLB schemas in the '{env}' database...")
    created = create_all_tables(drop_existing=args.drop_existing)
    conn = connect_mlb_db()
    try:
        if args.drop_existing:
            drop_odds_schema(conn)
        create_odds_schema(conn)
        created.append("odds_line_history")
    finally:
        conn.close()
    print(f"\n{len(created)} table(s) ready.")

    if args.sync:
        from mlb_pred.postgre_db.load import sync_local_store_to_postgres

        print("\nSyncing local store...")
        sync_local_store_to_postgres(seasons=seasons)
        from mlb_pred.postgre_db.odds_load import sync_odds_to_postgres

        sync_odds_to_postgres(seasons=seasons)

    if args.verify or (args.sync and seasons):
        if not seasons:
            print("--verify requires --seasons.")
            return 2
        from mlb_pred.postgre_db.verify import verify_remote_counts

        print("\nVerifying remote row counts...")
        checks = verify_remote_counts(seasons)
        failed = [check for check in checks if not check.matches]
        for check in checks:
            marker = "OK" if check.matches else "MISMATCH"
            print(
                f"  {marker:8} {check.target}: "
                f"remote={check.actual:,} local={check.expected:,}"
            )
        if failed:
            print(f"\n{len(failed)} table(s) failed row-count verification.")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
