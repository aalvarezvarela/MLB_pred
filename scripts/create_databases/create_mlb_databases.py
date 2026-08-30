#!/usr/bin/env python
"""Create every MLB Postgres schema and table.

Targets whichever environment ``[Database] DB_ENV`` (or the ``DB_ENV``
environment variable) selects. The shipped default is ``local``; point it at
``aiven`` only once the credentials are in ``config.secrets.ini`` or the
environment.

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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = get_db_env()

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
        sync_local_store_to_postgres()
        from mlb_pred.postgre_db.odds_load import sync_odds_to_postgres

        sync_odds_to_postgres()

    return 0


if __name__ == "__main__":
    sys.exit(main())
