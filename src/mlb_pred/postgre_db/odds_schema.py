"""Star schema for the SBR line-history store.

Deliberately narrow -- every column is a join key, a filter, or a price:

* ``line`` and ``price`` are ``SMALLINT``. Lines are half-points and stored
  doubled (``8.5 -> 17``), which is exact and costs 2 bytes against ~12 for
  ``NUMERIC``; the increment was measured on real MLB quotes before the
  encoding was committed to. Prices are American odds and fit natively once
  the "off the board" sentinel is nulled.
* Anything derivable from ``game_pk`` (team names, matchup URLs) or from the
  parsed values is dropped.
* ``lh_line`` is LIST-partitioned by season, so a season can be dropped
  instantly under storage pressure and season-filtered reads prune to one
  partition without a secondary index.

``mins_to_tip`` and ``is_pregame`` are **not conveniences**. SBR records
in-play ticks with exactly the same shape as pre-game ones, so they are the
only thing separating a legitimate feature row from target leakage. Both are
NOT NULL.

Everything here is **insert-only** (``ON CONFLICT DO NOTHING``). Never upsert a
tick and never replace a game's rows wholesale: the source can lose data --
SBR dropped Caesars from its NBA pages and those historical rows became
unrefetchable -- and a "delete then reload" refresh would destroy them
permanently. Insert-only makes re-fetching cost one HTTP request and nothing
else, which is what makes it safe to re-fetch aggressively.
"""

from __future__ import annotations

import psycopg
from psycopg import sql

SCHEMA = "odds_line_history"

#: Market dimension. ``run_line`` rather than ``point_spread``: SBR reuses a
#: cross-sport page template that calls it a spread, but in baseball it is the
#: run line and the rest of this project names it accordingly.
MARKETS: tuple[tuple[int, str], ...] = (
    (1, "totals"),
    (2, "run_line"),
    (3, "money_line"),
)

DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS {schema}.lh_book (
        book_id  SMALLINT PRIMARY KEY,
        slug     TEXT NOT NULL UNIQUE,
        name     TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {schema}.lh_market (
        market_id SMALLINT PRIMARY KEY,
        code      TEXT NOT NULL UNIQUE
    )
    """,
    # The game dimension. `event_id` is the provider's id, stored because it
    # cannot be re-derived later (a date holds many games) and is needed to
    # re-fetch one game. The starter columns are MLB-specific: a run total is
    # heavily conditioned on the announced starters, and they are a second,
    # independent handle for confirming a doubleheader match.
    """
    CREATE TABLE IF NOT EXISTS {schema}.lh_game (
        game_pk         TEXT PRIMARY KEY,
        event_id        INTEGER NOT NULL,
        game_date       DATE        NOT NULL,
        season_year     SMALLINT    NOT NULL,
        first_pitch_utc TIMESTAMPTZ NOT NULL,
        team_home_id    TEXT,
        team_away_id    TEXT,
        team_home       TEXT,
        team_away       TEXT,
        starter_home    TEXT,
        starter_away    TEXT,
        is_doubleheader BOOLEAN NOT NULL DEFAULT FALSE,
        game_number     SMALLINT NOT NULL DEFAULT 1,
        UNIQUE (event_id)
    )
    """,
)

LINE_TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS {schema}.lh_line (
        game_pk     TEXT        NOT NULL,
        season_year SMALLINT    NOT NULL,
        market_id   SMALLINT    NOT NULL,
        book_id     SMALLINT    NOT NULL,
        line_ts     TIMESTAMPTZ NOT NULL,
        mins_to_tip INTEGER     NOT NULL,
        is_pregame  BOOLEAN     NOT NULL,
        is_opener   BOOLEAN     NOT NULL DEFAULT FALSE,
        left_line   SMALLINT,
        left_price  SMALLINT,
        right_line  SMALLINT,
        right_price SMALLINT,
        -- season_year is last because Postgres requires the partition key
        -- inside any unique constraint; game_pk stays leading so "all lines
        -- for game X" still uses the index prefix.
        PRIMARY KEY (game_pk, market_id, book_id, line_ts, season_year)
    ) PARTITION BY LIST (season_year)
"""

METADATA_DDL_STATEMENTS: tuple[str, ...] = (
    # Provenance. "We are not sure about these seasons" is a fact the modelling
    # layer needs, and it belongs next to the data rather than in tribal
    # knowledge. `available_seasons()` reads this instead of a hardcoded range.
    """
    CREATE TABLE IF NOT EXISTS {schema}.lh_load_meta (
        season_year   SMALLINT PRIMARY KEY,
        confidence    TEXT NOT NULL,
        source_ticks  INTEGER,
        loaded_ticks  INTEGER,
        games_seen    INTEGER,
        games_loaded  INTEGER,
        dropped_rows  JSONB,
        resolution    JSONB,
        loaded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {schema}.lh_fetch (
        game_pk            TEXT PRIMARY KEY,
        event_id           INTEGER,
        season_year        SMALLINT NOT NULL,
        game_date          DATE NOT NULL,
        ingest_status      TEXT NOT NULL,
        source_tick_count  INTEGER NOT NULL,
        usable_tick_count  INTEGER NOT NULL,
        dropped_tick_count INTEGER NOT NULL,
        fetched_at_utc     TIMESTAMPTZ NOT NULL
    )
    """,
)


def create_schema(conn: psycopg.Connection) -> None:
    """Create the schema, dimensions and the partitioned fact table."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(SCHEMA))
        )
        for statement in DDL_STATEMENTS:
            cur.execute(sql.SQL(statement.format(schema=SCHEMA)))
        cur.execute(sql.SQL(LINE_TABLE_DDL.format(schema=SCHEMA)))
        for statement in METADATA_DDL_STATEMENTS:
            cur.execute(sql.SQL(statement.format(schema=SCHEMA)))

        for market_id, code in MARKETS:
            cur.execute(
                sql.SQL(
                    "INSERT INTO {}.lh_market (market_id, code) VALUES (%s, %s) "
                    "ON CONFLICT (market_id) DO NOTHING"
                ).format(sql.Identifier(SCHEMA)),
                (market_id, code),
            )
    conn.commit()


def partition_name(season_year: int) -> str:
    return f"lh_line_{season_year}"


def create_season_partition(conn: psycopg.Connection, season_year: int) -> None:
    """Partition bounds are DDL and cannot be parameterised, so embed the int."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {}.{} "
                "PARTITION OF {}.lh_line FOR VALUES IN ({})"
            ).format(
                sql.Identifier(SCHEMA),
                sql.Identifier(partition_name(season_year)),
                sql.Identifier(SCHEMA),
                sql.Literal(int(season_year)),
            )
        )
    conn.commit()


def ensure_books(conn: psycopg.Connection, slugs: list[str]) -> dict[str, int]:
    """Register unseen bookmakers; return the full slug -> id mapping.

    Ids are assigned in first-seen sorted order rather than by SERIAL so a
    reload from the same source is reproducible.
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT slug, book_id FROM {}.lh_book").format(
                sql.Identifier(SCHEMA)
            )
        )
        mapping = {slug: book_id for slug, book_id in cur.fetchall()}

        next_id = max(mapping.values(), default=0) + 1
        for slug in sorted(set(slugs)):
            if slug in mapping:
                continue
            cur.execute(
                sql.SQL(
                    "INSERT INTO {}.lh_book (book_id, slug, name) "
                    "VALUES (%s, %s, %s) ON CONFLICT (slug) DO NOTHING"
                ).format(sql.Identifier(SCHEMA)),
                (next_id, slug, slug.replace("_", " ").title()),
            )
            mapping[slug] = next_id
            next_id += 1
    conn.commit()
    return mapping


def market_ids() -> dict[str, int]:
    return {code: market_id for market_id, code in MARKETS}


def drop_schema(conn: psycopg.Connection) -> None:
    """Tear everything down. Used by ``--reset``."""
    conn.rollback()  # callers may arrive from a failed transaction
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(SCHEMA))
        )
    conn.commit()
