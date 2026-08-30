"""DDL for every MLB domain, one schema per domain.

Mirrors the NBA repo's layout: a single Postgres database with one schema per
subject area, so a domain can be dropped and rebuilt without touching the
others.

Two conventions carry over deliberately:

* **All provider ids are ``TEXT``.** They are opaque identifiers, never
  arithmetic operands, and keeping them textual stops a missing value from
  coercing an id column to float and silently reformatting every id in it.
* **Plausibility CHECKs at the schema boundary.** The NBA games table guards
  itself with ``pts > 40`` -- a floor, because a basketball team cannot score
  40. Baseball needs the mirror image: zero runs is legal, so the guard is a
  *ceiling* on runs plus a plausibility band on outs recorded. What both catch
  is the same failure: a box score fetched while the game was still in
  progress, which looks complete and, written into an insert-only table,
  poisons that game permanently.
"""

from __future__ import annotations

import psycopg
from psycopg import sql

from mlb_pred.config.constants import MAX_PLAUSIBLE_GAME_OUTS, MAX_PLAUSIBLE_TEAM_RUNS
from mlb_pred.config.settings import SETTINGS
from mlb_pred.postgre_db.config.db_config import (
    connect_mlb_db,
    create_schema_if_not_exists,
)

# ---------------------------------------------------------------------------
# Table definitions: schema setting -> (table name, DDL body, index columns)
# ---------------------------------------------------------------------------

GAMES_DDL = f"""
    game_pk                   TEXT    NOT NULL,
    season_year               INTEGER NOT NULL,
    -- The local slate date MLB itself assigns. Join key across the games
    -- table, the odds feed and the daily snapshots. Already correct for
    -- after-midnight finishes and for the second game of a doubleheader.
    game_date                 DATE    NOT NULL,
    -- MLB's scheduled UTC start, including official schedule revisions. It
    -- is not guaranteed to be the actual first pitch after a delay.
    first_pitch_utc           TIMESTAMPTZ,
    game_type                 VARCHAR(2) NOT NULL,
    season_type               VARCHAR(40),
    series_description        VARCHAR(60),
    -- gamePk is unique per game, so a doubleheader needs no compound key.
    -- game_number and doubleheader are kept as context, not as identity.
    game_number               SMALLINT,
    doubleheader              VARCHAR(1),
    day_night                 VARCHAR(10),
    scheduled_innings         SMALLINT,
    status_code               VARCHAR(10),
    status_detailed           VARCHAR(40),
    status_abstract           VARCHAR(20),
    is_final                  BOOLEAN NOT NULL,
    venue_id                  TEXT,
    venue_name                VARCHAR(80),
    home_team_id              TEXT    NOT NULL,
    home_team_name            VARCHAR(60),
    away_team_id              TEXT    NOT NULL,
    away_team_name            VARCHAR(60),
    home_score                 SMALLINT CHECK (home_score BETWEEN 0 AND {MAX_PLAUSIBLE_TEAM_RUNS}),
    away_score                 SMALLINT CHECK (away_score BETWEEN 0 AND {MAX_PLAUSIBLE_TEAM_RUNS}),
    home_hits                 SMALLINT,
    away_hits                 SMALLINT,
    home_errors               SMALLINT,
    away_errors               SMALLINT,
    innings_played            SMALLINT,
    extra_innings             BOOLEAN,
    total_runs                SMALLINT,
    -- Home minus away, matching the run line's own convention.
    run_line_margin           SMALLINT,
    home_probable_pitcher_id  TEXT,
    away_probable_pitcher_id  TEXT,
    weather_condition         VARCHAR(40),
    weather_temp_f            SMALLINT,
    weather_wind              VARCHAR(60),
    series_game_number        SMALLINT,
    games_in_series           SMALLINT,
    PRIMARY KEY (game_pk)
"""

TEAM_GAMES_DDL = f"""
    game_pk                    TEXT    NOT NULL,
    team_id                    TEXT    NOT NULL,
    season_year                INTEGER NOT NULL,
    game_date                  DATE    NOT NULL,
    first_pitch_utc            TIMESTAMPTZ,
    game_type                  VARCHAR(2),
    venue_id                   TEXT,
    -- Home/away is a boolean column, never a column suffix. Two rows per
    -- game; the wide one-row-per-game form is a final merge step.
    home                       BOOLEAN NOT NULL,
    opponent_team_id           TEXT    NOT NULL,

    -- Outcome
    runs_scored                SMALLINT CHECK (runs_scored BETWEEN 0 AND {MAX_PLAUSIBLE_TEAM_RUNS}),
    runs_allowed               SMALLINT CHECK (runs_allowed BETWEEN 0 AND {MAX_PLAUSIBLE_TEAM_RUNS}),
    run_margin                 SMALLINT,
    total_runs                 SMALLINT,
    win                        BOOLEAN,

    -- Volume / tempo. Stored separately from the efficiency rates below so
    -- that expected_events and expected_value_per_event can be recombined
    -- against a specific opponent rather than only their product.
    plate_appearances          SMALLINT,
    at_bats                    SMALLINT,
    outs_recorded              SMALLINT CHECK (outs_recorded IS NULL OR outs_recorded <= {MAX_PLAUSIBLE_GAME_OUTS}),
    innings_pitched            NUMERIC(5, 3),
    batters_faced              SMALLINT,
    pitches_thrown             SMALLINT,
    baserunners_allowed        SMALLINT,
    left_on_base               SMALLINT,

    -- Offensive component counts
    hits                       SMALLINT,
    singles                    SMALLINT,
    doubles                    SMALLINT,
    triples                    SMALLINT,
    home_runs                   SMALLINT,
    total_bases                SMALLINT,
    walks                      SMALLINT,
    intentional_walks          SMALLINT,
    strikeouts                 SMALLINT,
    hit_by_pitch               SMALLINT,
    sac_flies                  SMALLINT,
    sac_bunts                  SMALLINT,
    stolen_bases               SMALLINT,
    caught_stealing            SMALLINT,
    grounded_into_double_play  SMALLINT,
    rbi                        SMALLINT,
    ground_outs                SMALLINT,
    air_outs                   SMALLINT,
    fly_outs                   SMALLINT,
    line_outs                  SMALLINT,
    pop_outs                   SMALLINT,

    -- Pitching component counts
    hits_allowed               SMALLINT,
    home_runs_allowed          SMALLINT,
    walks_allowed              SMALLINT,
    intentional_walks_allowed  SMALLINT,
    strikeouts_thrown          SMALLINT,
    hit_batsmen                SMALLINT,
    earned_runs                SMALLINT,
    wild_pitches               SMALLINT,
    balks                      SMALLINT,
    inherited_runners          SMALLINT,
    inherited_runners_scored   SMALLINT,
    pitching_ground_outs       SMALLINT,
    pitching_air_outs          SMALLINT,
    strikes                    SMALLINT,
    balls                      SMALLINT,
    errors                     SMALLINT,

    -- Efficiency rates: scale-free, so comparable across opponents and eras.
    obp                        NUMERIC(8, 6),
    slg                        NUMERIC(8, 6),
    iso                        NUMERIC(8, 6),
    babip                      NUMERIC(8, 6),
    runs_per_pa                NUMERIC(8, 6),
    k_pct                      NUMERIC(8, 6),
    bb_pct                     NUMERIC(8, 6),
    hr_per_pa                  NUMERIC(8, 6),
    ground_out_air_out_ratio   NUMERIC(10, 6),
    k_pct_allowed              NUMERIC(8, 6),
    bb_pct_allowed             NUMERIC(8, 6),
    hr_per_bf_allowed          NUMERIC(8, 6),
    whip                       NUMERIC(10, 6),
    pitches_per_batter_faced   NUMERIC(8, 6),
    strike_pct                 NUMERIC(8, 6),

    -- Roster context
    batters_used               SMALLINT,
    pitchers_used              SMALLINT,
    bullpen_size               SMALLINT,
    bench_size                 SMALLINT,
    PRIMARY KEY (game_pk, team_id)
"""

BATTER_GAMES_DDL = """
    game_pk                    TEXT    NOT NULL,
    team_id                    TEXT    NOT NULL,
    player_id                  TEXT    NOT NULL,
    season_year                INTEGER NOT NULL,
    game_date                  DATE    NOT NULL,
    opponent_team_id           TEXT,
    home                       BOOLEAN,
    player_name                VARCHAR(80),
    position_code              VARCHAR(4),
    position_abbrev            VARCHAR(4),
    -- Lineup slot 1-9; substitution_index 0 is the player who started there.
    lineup_slot                SMALLINT,
    substitution_index         SMALLINT,
    is_starter                 BOOLEAN,
    had_plate_appearance       BOOLEAN,
    plate_appearances          SMALLINT,
    at_bats                    SMALLINT,
    runs                       SMALLINT,
    hits                       SMALLINT,
    singles                    SMALLINT,
    doubles                    SMALLINT,
    triples                    SMALLINT,
    home_runs                   SMALLINT,
    rbi                        SMALLINT,
    walks                      SMALLINT,
    intentional_walks          SMALLINT,
    strikeouts                 SMALLINT,
    hit_by_pitch               SMALLINT,
    sac_flies                  SMALLINT,
    sac_bunts                  SMALLINT,
    total_bases                SMALLINT,
    stolen_bases               SMALLINT,
    caught_stealing            SMALLINT,
    grounded_into_double_play  SMALLINT,
    left_on_base               SMALLINT,
    ground_outs                SMALLINT,
    air_outs                   SMALLINT,
    PRIMARY KEY (game_pk, team_id, player_id)
"""

PITCHER_APPEARANCES_DDL = f"""
    game_pk                    TEXT    NOT NULL,
    team_id                    TEXT    NOT NULL,
    player_id                  TEXT    NOT NULL,
    season_year                INTEGER NOT NULL,
    game_date                  DATE    NOT NULL,
    first_pitch_utc            TIMESTAMPTZ,
    opponent_team_id           TEXT,
    home                       BOOLEAN,
    venue_id                   TEXT,
    player_name                VARCHAR(80),
    -- The grain's whole point: a starter's history is counted in starts, a
    -- reliever's workload in days. Both need this flag to be first-class.
    is_starter                 BOOLEAN,
    appearance_number          SMALLINT,
    outs_recorded              SMALLINT CHECK (outs_recorded IS NULL OR outs_recorded <= {MAX_PLAUSIBLE_GAME_OUTS}),
    innings_pitched            NUMERIC(5, 3),
    batters_faced              SMALLINT,
    pitches_thrown             SMALLINT,
    strikes                    SMALLINT,
    balls                      SMALLINT,
    runs_allowed               SMALLINT,
    earned_runs                SMALLINT,
    hits_allowed               SMALLINT,
    doubles_allowed            SMALLINT,
    triples_allowed            SMALLINT,
    home_runs_allowed          SMALLINT,
    walks_allowed              SMALLINT,
    intentional_walks_allowed  SMALLINT,
    strikeouts                 SMALLINT,
    hit_batsmen                SMALLINT,
    wild_pitches               SMALLINT,
    balks                      SMALLINT,
    ground_outs                SMALLINT,
    air_outs                   SMALLINT,
    fly_outs                   SMALLINT,
    line_outs                  SMALLINT,
    pop_outs                   SMALLINT,
    inherited_runners          SMALLINT,
    inherited_runners_scored   SMALLINT,
    wins                       SMALLINT,
    losses                     SMALLINT,
    saves                      SMALLINT,
    holds                      SMALLINT,
    blown_saves                SMALLINT,
    complete_games             SMALLINT,
    k_pct                      NUMERIC(8, 6),
    bb_pct                     NUMERIC(8, 6),
    hr_per_bf                  NUMERIC(8, 6),
    whip                       NUMERIC(10, 6),
    pitches_per_batter_faced   NUMERIC(8, 6),
    strike_pct                 NUMERIC(8, 6),
    ground_out_air_out_ratio   NUMERIC(10, 6),
    PRIMARY KEY (game_pk, team_id, player_id)
"""

UMPIRES_DDL = """
    game_pk         TEXT    NOT NULL,
    season_year     INTEGER NOT NULL,
    game_date       DATE    NOT NULL,
    official_id     TEXT    NOT NULL,
    official_name   VARCHAR(80),
    official_type   VARCHAR(20) NOT NULL,
    -- The home-plate umpire is the crew position that moves run scoring:
    -- strike-zone size drives K% and BB%. The rest of the crew is stored so a
    -- crew-level feature stays possible.
    is_home_plate   BOOLEAN NOT NULL,
    PRIMARY KEY (game_pk, official_id, official_type)
"""

LINEUPS_DDL = """
    game_pk          TEXT    NOT NULL,
    season_year      INTEGER NOT NULL,
    game_date        DATE    NOT NULL,
    team_id          TEXT    NOT NULL,
    home             BOOLEAN,
    player_id        TEXT    NOT NULL,
    player_name      VARCHAR(80),
    lineup_slot      SMALLINT,
    position_code    VARCHAR(4),
    position_abbrev  VARCHAR(4),
    PRIMARY KEY (game_pk, team_id, player_id)
"""

VENUES_DDL = """
    venue_id         TEXT    NOT NULL,
    season           INTEGER NOT NULL,
    venue_name       VARCHAR(80),
    city             VARCHAR(60),
    state            VARCHAR(10),
    country          VARCHAR(40),
    latitude         NUMERIC(10, 6),
    longitude        NUMERIC(10, 6),
    elevation_ft     NUMERIC(8, 2),
    -- Compass bearing from home plate to centre field. Without it a wind
    -- direction is uninterpretable across parks.
    azimuth_angle    NUMERIC(6, 2),
    timezone_id      VARCHAR(60),
    timezone_offset  SMALLINT,
    capacity         INTEGER,
    turf_type        VARCHAR(20),
    roof_type        VARCHAR(20),
    left_line        SMALLINT,
    "left"           SMALLINT,
    left_center      SMALLINT,
    center           SMALLINT,
    right_center     SMALLINT,
    "right"          SMALLINT,
    right_line       SMALLINT,
    active           BOOLEAN,
    PRIMARY KEY (venue_id, season)
"""

TRANSACTIONS_DDL = """
    transaction_id     TEXT    NOT NULL,
    player_id          TEXT,
    player_name        VARCHAR(80),
    season_year        INTEGER NOT NULL,
    -- The announcement date. The ONLY date a point-in-time availability
    -- feature may filter on.
    known_date         DATE    NOT NULL,
    -- When the move took effect. Frequently EARLIER than known_date: an IL
    -- placement is routinely backdated. Filtering on this is genuine
    -- look-ahead leakage, not merely optimism.
    effective_date     DATE,
    resolution_date    DATE,
    is_backdated       BOOLEAN,
    backdated_days     SMALLINT,
    from_team_id       TEXT,
    to_team_id         TEXT,
    type_code          VARCHAR(10),
    type_desc          VARCHAR(60),
    description        TEXT,
    is_injury_related  BOOLEAN,
    PRIMARY KEY (transaction_id)
"""

STATCAST_GAMES_DDL = """
    game_pk         TEXT PRIMARY KEY,
    season_year     INTEGER NOT NULL,
    game_date       DATE NOT NULL,
    fetch_status    VARCHAR(20) NOT NULL,
    pitch_rows      INTEGER NOT NULL,
    fetched_at_utc  TIMESTAMPTZ NOT NULL
"""

STATCAST_PITCHES_DDL = """
    game_pk TEXT NOT NULL, game_date DATE, game_year SMALLINT, game_type VARCHAR(4),
    home_team VARCHAR(4), away_team VARCHAR(4), inning SMALLINT,
    inning_topbot VARCHAR(10), at_bat_number SMALLINT NOT NULL,
    pitch_number SMALLINT NOT NULL, pitch_type VARCHAR(4), pitch_name VARCHAR(40),
    description VARCHAR(60), events VARCHAR(40), batter_id TEXT, pitcher_id TEXT,
    stand VARCHAR(1), p_throws VARCHAR(1), balls SMALLINT, strikes SMALLINT,
    outs_when_up SMALLINT, on_1b TEXT, on_2b TEXT, on_3b TEXT,
    release_speed DOUBLE PRECISION, effective_speed DOUBLE PRECISION,
    release_spin_rate DOUBLE PRECISION, spin_axis DOUBLE PRECISION,
    release_extension DOUBLE PRECISION, release_pos_x DOUBLE PRECISION,
    release_pos_y DOUBLE PRECISION, release_pos_z DOUBLE PRECISION,
    pfx_x DOUBLE PRECISION, pfx_z DOUBLE PRECISION, plate_x DOUBLE PRECISION,
    plate_z DOUBLE PRECISION, zone SMALLINT, sz_top DOUBLE PRECISION,
    sz_bot DOUBLE PRECISION, launch_speed DOUBLE PRECISION,
    launch_angle DOUBLE PRECISION, hit_distance_sc DOUBLE PRECISION,
    launch_speed_angle SMALLINT, bb_type VARCHAR(20),
    estimated_ba_using_speedangle DOUBLE PRECISION,
    estimated_slg_using_speedangle DOUBLE PRECISION,
    estimated_woba_using_speedangle DOUBLE PRECISION,
    woba_value DOUBLE PRECISION, woba_denom DOUBLE PRECISION,
    babip_value DOUBLE PRECISION, iso_value DOUBLE PRECISION,
    hc_x DOUBLE PRECISION, hc_y DOUBLE PRECISION, hit_location SMALLINT,
    if_fielding_alignment VARCHAR(30), of_fielding_alignment VARCHAR(30),
    home_score SMALLINT, away_score SMALLINT, bat_score SMALLINT, fld_score SMALLINT,
    post_home_score SMALLINT, post_away_score SMALLINT,
    delta_home_win_exp DOUBLE PRECISION, delta_run_exp DOUBLE PRECISION,
    n_thruorder_pitcher SMALLINT,
    n_priorpa_thisgame_player_at_bat SMALLINT,
    arm_angle DOUBLE PRECISION,
    bat_speed DOUBLE PRECISION, swing_length DOUBLE PRECISION,
    season_year INTEGER NOT NULL,
    PRIMARY KEY (game_pk, at_bat_number, pitch_number)
"""


TABLE_DEFINITIONS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    # setting name -> (table name, DDL, indexed columns)
    "schema_name_games": (
        "games",
        GAMES_DDL,
        ("game_date", "season_year", "home_team_id", "away_team_id", "venue_id"),
    ),
    "schema_name_team_games": (
        "team_games",
        TEAM_GAMES_DDL,
        ("game_date", "team_id", "season_year", "opponent_team_id", "venue_id"),
    ),
    "schema_name_batters": (
        "batter_games",
        BATTER_GAMES_DDL,
        ("game_date", "player_id", "team_id", "season_year"),
    ),
    "schema_name_pitchers": (
        "pitcher_appearances",
        PITCHER_APPEARANCES_DDL,
        ("game_date", "player_id", "team_id", "season_year", "is_starter"),
    ),
    "schema_name_umpires": (
        "umpires",
        UMPIRES_DDL,
        ("game_date", "official_id", "season_year"),
    ),
    "schema_name_lineups": (
        "lineups",
        LINEUPS_DDL,
        ("game_date", "player_id", "team_id"),
    ),
    "schema_name_venues": ("venues", VENUES_DDL, ("season",)),
    "schema_name_transactions": (
        "transactions",
        TRANSACTIONS_DDL,
        ("known_date", "player_id", "season_year", "is_injury_related"),
    ),
    "schema_name_statcast_games": (
        "statcast_games",
        STATCAST_GAMES_DDL,
        ("game_date", "season_year", "fetch_status"),
    ),
    "schema_name_statcast_pitches": (
        "statcast_pitches",
        STATCAST_PITCHES_DDL,
        ("game_date", "season_year", "pitcher_id", "batter_id"),
    ),
}


def create_table(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    ddl: str,
    index_columns: tuple[str, ...] = (),
    *,
    drop_existing: bool = False,
) -> None:
    create_schema_if_not_exists(conn, schema)

    with conn.cursor() as cur:
        if drop_existing:
            cur.execute(
                sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
            )

        cur.execute(
            sql.SQL("CREATE TABLE IF NOT EXISTS {}.{} (" + ddl + ")").format(
                sql.Identifier(schema), sql.Identifier(table)
            )
        )

        for column in index_columns:
            bare = column.strip('"')
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}({})").format(
                    sql.Identifier(f"idx_{table}_{bare}"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                    sql.Identifier(bare),
                )
            )
    conn.commit()


def create_all_tables(*, drop_existing: bool = False) -> list[str]:
    """Create every domain schema and table. Returns the qualified names."""
    created = []
    conn = connect_mlb_db()
    try:
        for setting, (table, ddl, indexes) in TABLE_DEFINITIONS.items():
            schema = getattr(SETTINGS, setting)
            create_table(conn, schema, table, ddl, indexes, drop_existing=drop_existing)
            created.append(f"{schema}.{table}")
            print(f"Created {schema}.{table}")
    finally:
        conn.close()
    return created
