"""Compare season-scoped local Parquet row counts with the remote mirror."""

from __future__ import annotations

from dataclasses import dataclass

import psycopg
from psycopg import sql

from mlb_pred.local_store.parquet_store import read_table
from mlb_pred.local_store.tables import get_spec
from mlb_pred.postgre_db.config.db_config import connect_mlb_db
from mlb_pred.postgre_db.load import schema_for
from mlb_pred.postgre_db.odds_schema import SCHEMA as ODDS_SCHEMA
from mlb_pred.postgre_db.schema import TABLE_DEFINITIONS


@dataclass(frozen=True)
class CountCheck:
    source: str
    target: str
    expected: int
    actual: int

    @property
    def matches(self) -> bool:
        return self.expected == self.actual


def _remote_count(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    season_column: str,
    seasons: list[int],
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT count(*) FROM {}.{} WHERE {} = ANY(%s)").format(
                sql.Identifier(schema),
                sql.Identifier(table),
                sql.Identifier(season_column),
            ),
            (seasons,),
        )
        return int(cur.fetchone()[0])


def verify_remote_counts(
    seasons: list[int],
    *,
    conn: psycopg.Connection | None = None,
) -> list[CountCheck]:
    """Return local-vs-remote count checks for every mirrored fact table."""
    if not seasons:
        raise ValueError("At least one season is required for remote verification.")
    owns_connection = conn is None
    conn = conn or connect_mlb_db()
    checks: list[CountCheck] = []
    try:
        for _, (table, _, _) in TABLE_DEFINITIONS.items():
            spec = get_spec(table)
            local = read_table(table, partitions=seasons)
            season_column = spec.partition_column or "season_year"
            checks.append(
                CountCheck(
                    source=table,
                    target=f"{schema_for(table)}.{table}",
                    expected=len(local),
                    actual=_remote_count(
                        conn,
                        schema_for(table),
                        table,
                        season_column,
                        seasons,
                    ),
                )
            )

        odds_tables = (
            ("odds_games", "lh_game"),
            ("odds_ticks", "lh_line"),
            ("odds_fetches", "lh_fetch"),
        )
        for local_table, remote_table in odds_tables:
            local = read_table(local_table, partitions=seasons)
            checks.append(
                CountCheck(
                    source=local_table,
                    target=f"{ODDS_SCHEMA}.{remote_table}",
                    expected=len(local),
                    actual=_remote_count(
                        conn,
                        ODDS_SCHEMA,
                        remote_table,
                        "season_year",
                        seasons,
                    ),
                )
            )
        return checks
    finally:
        if owns_connection:
            conn.close()
