"""Write the local store's tables into Postgres.

The fetchers produce plain DataFrames and know nothing about storage. This
module is the only place that turns one into rows, so the local Parquet store
and Postgres stay two backends for the same schemas rather than two schemas.

Writes are ``INSERT ... ON CONFLICT DO UPDATE`` on each table's declared
primary key -- the same insert-only-with-upsert semantics as the Parquet
store, which is what lets every updater be re-run at any cadence.
"""

from __future__ import annotations

import math

import pandas as pd
import psycopg
from psycopg import sql

from mlb_pred.config.settings import SETTINGS
from mlb_pred.local_store.parquet_store import read_table
from mlb_pred.local_store.tables import TABLES, get_spec
from mlb_pred.postgre_db.config.db_config import connect_mlb_db
from mlb_pred.postgre_db.schema import TABLE_DEFINITIONS

# Local table name -> the settings key naming its Postgres schema.
_SCHEMA_SETTING_FOR_TABLE = {
    table: setting for setting, (table, _, _) in TABLE_DEFINITIONS.items()
}

# Columns that are reserved words in SQL and quoted in the DDL.
_QUOTED_COLUMNS = {"left", "right"}

BATCH_SIZE = 5_000


def schema_for(table: str) -> str:
    try:
        setting = _SCHEMA_SETTING_FOR_TABLE[table]
    except KeyError:
        raise KeyError(f"No Postgres schema registered for table {table!r}.") from None
    return getattr(SETTINGS, setting)


def _clean(value):
    """Coerce a pandas value into something psycopg can bind."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    # numpy scalars -> python scalars
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, str):
        try:
            return item()
        except (ValueError, AttributeError):
            return value
    return value


def _target_columns(conn: psycopg.Connection, schema: str, table: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """,
            (schema, table),
        )
        return [row[0] for row in cur.fetchall()]


def upsert_dataframe(
    df: pd.DataFrame,
    table: str,
    *,
    conn: psycopg.Connection | None = None,
) -> int:
    """Upsert ``df`` into its Postgres table. Returns rows written."""
    if df is None or df.empty:
        return 0

    spec = get_spec(table)
    schema = schema_for(table)
    owns_connection = conn is None
    conn = conn or connect_mlb_db()

    try:
        available = set(_target_columns(conn, schema, table))
        if not available:
            raise RuntimeError(
                f"Table {schema}.{table} does not exist. "
                "Run scripts/create_databases/create_mlb_databases.py first."
            )

        # Only send columns the table actually has: the fetchers may gain a
        # field before the DDL does, and dropping it here is better than
        # failing the whole load.
        columns = [c for c in df.columns if c in available]
        missing_key = [c for c in spec.primary_key if c not in columns]
        if missing_key:
            raise ValueError(
                f"Cannot upsert into {schema}.{table}: primary-key column(s) "
                f"{missing_key} absent from the frame."
            )

        identifiers = [sql.Identifier(c) for c in columns]
        update_columns = [c for c in columns if c not in spec.primary_key]

        if update_columns:
            conflict_action: sql.Composable = sql.SQL("DO UPDATE SET {}").format(
                sql.SQL(", ").join(
                    sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(c))
                    for c in update_columns
                )
            )
        else:
            conflict_action = sql.SQL("DO NOTHING")

        statement = sql.SQL(
            "INSERT INTO {}.{} ({}) VALUES ({}) ON CONFLICT ({}) {}"
        ).format(
            sql.Identifier(schema),
            sql.Identifier(table),
            sql.SQL(", ").join(identifiers),
            sql.SQL(", ").join(sql.Placeholder() * len(columns)),
            sql.SQL(", ").join(sql.Identifier(c) for c in spec.primary_key),
            conflict_action,
        )

        payload = df[columns].astype(object).where(pd.notna(df[columns]), None)
        rows = [tuple(_clean(v) for v in record) for record in payload.to_numpy()]

        with conn.cursor() as cur:
            for start in range(0, len(rows), BATCH_SIZE):
                cur.executemany(statement, rows[start : start + BATCH_SIZE])
        conn.commit()
        return len(rows)
    except Exception:
        conn.rollback()
        raise
    finally:
        if owns_connection:
            conn.close()


def sync_local_store_to_postgres(
    tables: list[str] | None = None,
    seasons: list[int] | None = None,
) -> dict[str, int]:
    """Push the local Parquet store into Postgres.

    The local store is the working copy during development; this is the one
    step that promotes it. Safe to re-run -- every write is an upsert.
    """
    selected = tables or [t for t in TABLES if t in _SCHEMA_SETTING_FOR_TABLE]
    results: dict[str, int] = {}

    conn = connect_mlb_db()
    try:
        for table in selected:
            spec = get_spec(table)
            partitions = (
                seasons if seasons and spec.partition_column == "season_year" else None
            )
            frame = read_table(table, partitions=partitions)
            written = upsert_dataframe(frame, table, conn=conn)
            results[table] = written
            print(f"{table}: {written} rows -> {schema_for(table)}.{table}")
    finally:
        conn.close()

    return results
