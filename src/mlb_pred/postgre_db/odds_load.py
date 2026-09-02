"""Load the local odds store into the partitioned Postgres star schema.

Insert-only throughout (``ON CONFLICT DO NOTHING``). The local Parquet store
keeps ``book_slug`` as a readable string; this module resolves it to the
``lh_book`` surrogate id at the boundary, so the narrow fact table stays narrow
without making the working copy cryptic.

Bulk path: ``COPY`` into an ``UNLOGGED`` staging table, then one
``INSERT … SELECT … ON CONFLICT DO NOTHING`` into the partitioned target,
staged one season at a time. ``executemany`` is a round trip per row, which is
untenable over a WAN at this volume, and ``UNLOGGED`` keeps the copy itself out
of the WAL.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from types import SimpleNamespace

import pandas as pd
import psycopg
from psycopg import sql

from mlb_pred.local_store.parquet_store import read_table
from mlb_pred.postgre_db.config.db_config import connect_mlb_db
from mlb_pred.postgre_db.odds_schema import (
    SCHEMA,
    create_schema,
    create_season_partition,
    ensure_books,
    market_ids,
)

_LINE_COLUMNS = (
    "game_pk",
    "season_year",
    "market_id",
    "book_id",
    "line_ts",
    "mins_to_tip",
    "is_pregame",
    "is_opener",
    "left_line",
    "left_price",
    "right_line",
    "right_price",
)

_GAME_COLUMNS = (
    "game_pk",
    "event_id",
    "game_date",
    "season_year",
    "first_pitch_utc",
    "team_home_id",
    "team_away_id",
    "team_home",
    "team_away",
    "starter_home",
    "starter_away",
    "is_doubleheader",
    "game_number",
)

_FETCH_COLUMNS = (
    "game_pk",
    "event_id",
    "season_year",
    "game_date",
    "ingest_status",
    "source_tick_count",
    "usable_tick_count",
    "dropped_tick_count",
    "fetched_at_utc",
)


def _clean(value):
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, str):
        try:
            return item()
        except (ValueError, AttributeError):
            return value
    return value


def upsert_games(conn: psycopg.Connection, dimension: pd.DataFrame) -> int:
    """Insert game-dimension rows. Insert-only, never an upsert.

    The dimension is deliberately not upserted: every stored tick's
    ``mins_to_tip`` was computed against the ``first_pitch_utc`` on this row,
    so silently moving it would desynchronise rows this run is not touching.
    """
    if dimension.empty:
        return 0

    columns = [c for c in _GAME_COLUMNS if c in dimension.columns]
    statement = sql.SQL(
        "INSERT INTO {}.lh_game ({}) VALUES ({}) ON CONFLICT (game_pk) DO NOTHING"
    ).format(
        sql.Identifier(SCHEMA),
        sql.SQL(", ").join(sql.Identifier(c) for c in columns),
        sql.SQL(", ").join(sql.Placeholder() * len(columns)),
    )

    frame = dimension[columns].astype(object).where(pd.notna(dimension[columns]), None)
    rows = [tuple(_clean(v) for v in record) for record in frame.to_numpy()]
    with conn.cursor() as cur:
        cur.executemany(statement, rows)
    conn.commit()
    return len(rows)


def copy_lines_for_season(
    conn: psycopg.Connection,
    ticks: pd.DataFrame,
    season_year: int,
    book_ids: dict[str, int],
) -> int:
    """COPY one season's ticks through an UNLOGGED stage into the partition."""
    season_ticks = ticks[ticks["season_year"] == season_year]
    if season_ticks.empty:
        return 0

    create_season_partition(conn, season_year)

    markets = market_ids()
    staged = pd.DataFrame(
        {
            "game_pk": season_ticks["game_pk"].astype(str),
            "season_year": season_ticks["season_year"].astype(int),
            "market_id": season_ticks["market"].map(markets).astype("Int16"),
            "book_id": season_ticks["book_slug"].map(book_ids).astype("Int16"),
            "line_ts": season_ticks["line_ts"],
            "mins_to_tip": season_ticks["mins_to_tip"].astype(int),
            "is_pregame": season_ticks["is_pregame"].astype(bool),
            "is_opener": season_ticks["is_opener"].astype(bool),
            "left_line": season_ticks["left_line"].astype("Int16"),
            "left_price": season_ticks["left_price"].astype("Int16"),
            "right_line": season_ticks["right_line"].astype("Int16"),
            "right_price": season_ticks["right_price"].astype("Int16"),
        }
    )
    # An unmapped market or book would violate NOT NULL; fail here with a
    # readable message instead of at COPY time.
    if staged["market_id"].isna().any() or staged["book_id"].isna().any():
        raise ValueError("Unmapped market or book slug in ticks; cannot load.")

    stage_table = f"lh_line_stage_{season_year}"
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "CREATE UNLOGGED TABLE IF NOT EXISTS {}.{} "
                "(LIKE {}.lh_line INCLUDING DEFAULTS)"
            ).format(
                sql.Identifier(SCHEMA),
                sql.Identifier(stage_table),
                sql.Identifier(SCHEMA),
            )
        )
        cur.execute(
            sql.SQL("TRUNCATE {}.{}").format(
                sql.Identifier(SCHEMA), sql.Identifier(stage_table)
            )
        )

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        for record in staged[list(_LINE_COLUMNS)].itertuples(index=False, name=None):
            writer.writerow(["" if v is None or pd.isna(v) else v for v in record])
        buffer.seek(0)

        copy_statement = sql.SQL("COPY {}.{} ({}) FROM STDIN WITH (FORMAT csv)").format(
            sql.Identifier(SCHEMA),
            sql.Identifier(stage_table),
            sql.SQL(", ").join(sql.Identifier(c) for c in _LINE_COLUMNS),
        )
        with cur.copy(copy_statement) as copy:
            copy.write(buffer.read())

        cur.execute(
            sql.SQL(
                "INSERT INTO {schema}.lh_line ({cols}) "
                "SELECT {cols} FROM {schema}.{stage} "
                "ON CONFLICT DO NOTHING"
            ).format(
                schema=sql.Identifier(SCHEMA),
                stage=sql.Identifier(stage_table),
                cols=sql.SQL(", ").join(sql.Identifier(c) for c in _LINE_COLUMNS),
            )
        )
        inserted = cur.rowcount
        cur.execute(
            sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                sql.Identifier(SCHEMA), sql.Identifier(stage_table)
            )
        )
    conn.commit()
    return inserted


def upsert_fetches(conn: psycopg.Connection, fetches: pd.DataFrame) -> int:
    if fetches.empty:
        return 0
    columns = [column for column in _FETCH_COLUMNS if column in fetches.columns]
    updates = [column for column in columns if column != "game_pk"]
    statement = sql.SQL(
        "INSERT INTO {}.lh_fetch ({}) VALUES ({}) ON CONFLICT (game_pk) "
        "DO UPDATE SET {}"
    ).format(
        sql.Identifier(SCHEMA),
        sql.SQL(", ").join(sql.Identifier(c) for c in columns),
        sql.SQL(", ").join(sql.Placeholder() * len(columns)),
        sql.SQL(", ").join(
            sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c))
            for c in updates
        ),
    )
    frame = fetches[columns].astype(object).where(pd.notna(fetches[columns]), None)
    rows = [tuple(_clean(v) for v in record) for record in frame.to_numpy()]
    with conn.cursor() as cur:
        cur.executemany(statement, rows)
    conn.commit()
    return len(rows)


def record_load_meta(
    conn: psycopg.Connection,
    season_year: int,
    stats,
    resolution,
    *,
    confidence: str = "high",
) -> None:
    """Persist how a season was loaded.

    "We are not sure about this season" is a fact the modelling layer needs,
    and ``available_seasons()`` reads this rather than a hardcoded range.
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "INSERT INTO {}.lh_load_meta "
                "(season_year, confidence, source_ticks, loaded_ticks, "
                " games_seen, games_loaded, dropped_rows, resolution) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (season_year) DO UPDATE SET "
                "  confidence = EXCLUDED.confidence, "
                "  source_ticks = EXCLUDED.source_ticks, "
                "  loaded_ticks = EXCLUDED.loaded_ticks, "
                "  games_seen = EXCLUDED.games_seen, "
                "  games_loaded = EXCLUDED.games_loaded, "
                "  dropped_rows = EXCLUDED.dropped_rows, "
                "  resolution = EXCLUDED.resolution, "
                "  loaded_at = now()"
            ).format(sql.Identifier(SCHEMA)),
            (
                season_year,
                confidence,
                stats.source_ticks,
                stats.loaded_ticks,
                stats.games_seen,
                stats.games_loaded,
                json.dumps(dict(stats.dropped)),
                json.dumps(
                    {
                        "matched": resolution.matched,
                        "matched_doubleheader": resolution.matched_doubleheader,
                        "unmatched": dict(resolution.unmatched),
                        "first_pitch_disagreements": len(
                            resolution.first_pitch_disagreements
                        ),
                    }
                ),
            ),
        )
    conn.commit()


def sync_odds_to_postgres(seasons: list[int] | None = None) -> dict[str, int]:
    """Push the local odds store into Postgres. Safe to re-run."""
    dimension = read_table("odds_games", partitions=seasons)
    ticks = read_table("odds_ticks", partitions=seasons)
    fetches = read_table("odds_fetches", partitions=seasons)
    if dimension.empty and ticks.empty:
        print("Local odds store is empty; nothing to sync.")
        return {}

    conn = connect_mlb_db()
    try:
        create_schema(conn)
        book_ids = ensure_books(conn, sorted(ticks["book_slug"].dropna().unique()))

        games_written = upsert_games(conn, dimension)
        fetches_written = upsert_fetches(conn, fetches)
        target_seasons = seasons or sorted(ticks["season_year"].unique())

        results = {"lh_game": games_written, "lh_fetch": fetches_written}
        for season_year in target_seasons:
            written = copy_lines_for_season(conn, ticks, int(season_year), book_ids)
            results[f"lh_line_{season_year}"] = written
            season_ticks = ticks[ticks["season_year"] == season_year]
            season_fetches = fetches[fetches["season_year"] == season_year]
            stats = SimpleNamespace(
                source_ticks=int(
                    season_fetches.get("source_tick_count", pd.Series(dtype=int)).sum()
                ),
                loaded_ticks=len(season_ticks),
                games_seen=len(season_fetches),
                games_loaded=int(
                    season_fetches.get("ingest_status", pd.Series(dtype=object))
                    .isin({"complete", "no_quotes"})
                    .sum()
                ),
                dropped=Counter(
                    season_fetches.get("ingest_status", pd.Series(dtype=object))
                    .value_counts()
                    .to_dict()
                ),
            )
            resolution = SimpleNamespace(
                matched=stats.games_loaded,
                matched_doubleheader=0,
                unmatched=Counter(),
                first_pitch_disagreements=[],
            )
            record_load_meta(conn, int(season_year), stats, resolution)
            print(f"  season {season_year}: {written} tick(s) inserted")
        return results
    finally:
        conn.close()


def available_seasons(conn: psycopg.Connection | None = None) -> pd.DataFrame:
    """What the store actually holds, read from provenance rather than assumed."""
    owned = conn is None
    conn = conn or connect_mlb_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT * FROM {}.lh_load_meta ORDER BY season_year").format(
                    sql.Identifier(SCHEMA)
                )
            )
            assert cur.description is not None
            columns = [c.name for c in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=columns)
    finally:
        if owned:
            conn.close()
