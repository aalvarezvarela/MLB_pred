"""The table registry: one place that knows every dataset's natural key.

Every updater in this repo is **gap-driven and insert-only** -- it asks the
store what is missing rather than being told a date range. That only works if
"already present" has a single, unambiguous definition per table, which is
what this registry provides.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TableSpec:
    """How one dataset is keyed, partitioned and ordered."""

    name: str
    #: Columns that uniquely identify a row. Re-inserting the same key is a
    #: no-op, which is what makes every updater safe to re-run at any cadence.
    primary_key: tuple[str, ...]
    #: Column the store partitions on. ``None`` means a single file.
    partition_column: str | None = "season_year"
    #: Column carrying the game id, where one exists. This is what the
    #: gap query compares against the games table.
    game_column: str | None = "game_pk"
    sort_columns: tuple[str, ...] = field(default_factory=tuple)


TABLES: dict[str, TableSpec] = {
    "games": TableSpec(
        name="games",
        primary_key=("game_pk",),
        sort_columns=("game_date", "first_pitch_utc", "game_pk"),
    ),
    "team_games": TableSpec(
        name="team_games",
        primary_key=("game_pk", "team_id"),
        sort_columns=("game_date", "game_pk", "team_id"),
    ),
    "batter_games": TableSpec(
        name="batter_games",
        primary_key=("game_pk", "team_id", "player_id"),
        sort_columns=("game_date", "game_pk", "team_id", "lineup_slot"),
    ),
    "pitcher_appearances": TableSpec(
        name="pitcher_appearances",
        primary_key=("game_pk", "team_id", "player_id"),
        sort_columns=("game_date", "game_pk", "team_id", "appearance_number"),
    ),
    "umpires": TableSpec(
        name="umpires",
        primary_key=("game_pk", "official_id", "official_type"),
        sort_columns=("game_date", "game_pk", "official_type"),
    ),
    "lineups": TableSpec(
        name="lineups",
        primary_key=("game_pk", "team_id", "player_id"),
        sort_columns=("game_date", "game_pk", "team_id", "lineup_slot"),
    ),
    "venues": TableSpec(
        name="venues",
        primary_key=("venue_id", "season"),
        partition_column="season",
        game_column=None,
        sort_columns=("season", "venue_id"),
    ),
    "transactions": TableSpec(
        name="transactions",
        primary_key=("transaction_id",),
        game_column=None,
        sort_columns=("known_date", "transaction_id"),
    ),
    # --- odds -------------------------------------------------------------
    # The game dimension carries the provider id alongside game_pk. That
    # mapping cannot be re-derived later (a date holds many games) and is
    # needed to re-fetch one game, so it is a stored row, never a join to redo.
    "odds_games": TableSpec(
        name="odds_games",
        primary_key=("game_pk",),
        sort_columns=("game_date", "first_pitch_utc", "game_pk"),
    ),
    # The tick fact. One row per (game, market, book, minute) -- the atomic
    # odds record. Closing lines, opening lines and consensus are all views
    # over this, never stored shapes of their own.
    "odds_ticks": TableSpec(
        name="odds_ticks",
        primary_key=("game_pk", "market", "book_slug", "line_ts"),
        sort_columns=("game_date", "game_pk", "market", "book_slug", "line_ts"),
    ),
    "odds_fetches": TableSpec(
        name="odds_fetches",
        primary_key=("game_pk",),
        sort_columns=("game_date", "game_pk"),
    ),
    # Baseball Savant pitch-level details. Completion is kept separately so a
    # failed/empty response can never be mistaken for an ingested game.
    "statcast_games": TableSpec(
        name="statcast_games",
        primary_key=("game_pk",),
        sort_columns=("game_date", "game_pk"),
    ),
    "statcast_pitches": TableSpec(
        name="statcast_pitches",
        primary_key=("game_pk", "at_bat_number", "pitch_number"),
        sort_columns=("game_date", "game_pk", "at_bat_number", "pitch_number"),
    ),
}

#: Tables built from a per-game box-score fetch. The gap query drives these
#: three (plus umpires) off the games table.
BOXSCORE_TABLES: tuple[str, ...] = (
    "team_games",
    "batter_games",
    "pitcher_appearances",
    "umpires",
)


def get_spec(table: str) -> TableSpec:
    try:
        return TABLES[table]
    except KeyError:
        raise KeyError(
            f"Unknown table {table!r}. Known tables: {sorted(TABLES)}."
        ) from None
