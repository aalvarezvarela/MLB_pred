"""Gap-driven updaters.

Every updater here asks the store what is missing rather than being told a
date range:

1. Load the games that *should* exist for a season (from the schedule feed,
   which is one cheap request).
2. Ask the target table which of those ids it already has.
3. Fetch only the gap.
4. Insert.

The reference for "what should exist" is always the games table. That makes
each updater idempotent, safe to run on any cadence, and nearly free when
there is nothing to do -- and it means an aborted run costs nothing, because
the next run resumes exactly where it stopped.

Two rules are enforced rather than documented:

* **Never ingest an in-progress game.** A box score fetched mid-game is a
  *partial* box score that looks complete. Written into an insert-only table
  it poisons that game permanently. Only games the schedule feed reports as
  final are eligible, and the current date is excluded by default.
* **Stop on throttling, retry on transient failure.** The client raises
  ``RateLimitedError`` rather than retrying into a ban; the updater lets it
  propagate.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from tqdm import tqdm

from mlb_pred.fetch_data.statsapi.boxscore import (
    fetch_boxscore,
    game_meta_from_schedule,
    parse_boxscore,
)
from mlb_pred.fetch_data.statsapi.client import RateLimitedError, StatsApiError
from mlb_pred.fetch_data.statsapi.schedule import (
    fetch_schedule,
    fetch_season_schedule,
    finished_games,
)
from mlb_pred.fetch_data.statsapi.transactions import fetch_transactions
from mlb_pred.fetch_data.statsapi.venues import fetch_venues
from mlb_pred.local_store.parquet_store import (
    existing_game_pks,
    read_table,
    replace_game_rows,
    write_table,
)
from mlb_pred.local_store.tables import BOXSCORE_TABLES
from mlb_pred.utils.seasons import mlb_slate_date, season_date_bounds


def update_games(
    season_year: int,
    *,
    exclude_dates: list[date] | None = None,
    include_unfinished: bool = False,
) -> dict[str, int]:
    """Refresh the games, umpires and lineups tables for one season.

    The whole season comes back in one request, so this is re-run in full
    rather than incrementally -- the upsert makes that cheap and it lets a
    corrected score or a rescheduled game propagate.
    """
    games, umpires, lineups = fetch_season_schedule(season_year)

    if not include_unfinished:
        games = finished_games(games)

    excluded = set(exclude_dates or [])
    if excluded:
        games = games[~games["game_date"].isin(excluded)].reset_index(drop=True)

    eligible = set(games["game_pk"])
    umpires = umpires[umpires["game_pk"].isin(eligible)]
    lineups = lineups[lineups["game_pk"].isin(eligible)]

    game_pks = {str(v) for v in games["game_pk"]}
    return {
        "games": write_table("games", games),
        "umpires": replace_game_rows(
            "umpires", umpires, game_pks=game_pks, season_year=season_year
        ),
        "lineups": replace_game_rows(
            "lineups", lineups, game_pks=game_pks, season_year=season_year
        ),
    }


def missing_boxscore_games(season_year: int) -> pd.DataFrame:
    """Games present in the games table but absent from the box-score tables.

    A game counts as done only when it is present in *every* box-score-derived
    table. Requiring all of them means a run that died between writing
    ``team_games`` and writing ``pitcher_appearances`` is repaired on the next
    pass instead of leaving a game permanently half-ingested.
    """
    games = read_table("games", partitions=[season_year])
    if games.empty:
        return games

    games = games[games["is_final"]] if "is_final" in games.columns else games

    done: set[str] | None = None
    for table in BOXSCORE_TABLES:
        present = existing_game_pks(table, season_year)
        done = present if done is None else (done & present)

    return games[~games["game_pk"].isin(done or set())].reset_index(drop=True)


#: How many games to buffer before writing. Small enough that an interrupted
#: run loses at most this many fetches, large enough that the Parquet rewrite
#: cost stays negligible. Buffering a whole season would contradict the
#: "stopping early is free" property the gap-driven design depends on.
BOXSCORE_FLUSH_EVERY = 250


def update_boxscores(
    season_year: int,
    *,
    limit: int | None = None,
    show_progress: bool = True,
    flush_every: int = BOXSCORE_FLUSH_EVERY,
) -> dict[str, int]:
    """Fetch box scores for whatever the store is missing.

    Results are flushed to the store every ``flush_every`` games rather than
    once at the end, so an interrupted run keeps everything it had already
    fetched and the next run's gap query resumes from there.

    Args:
        season_year: Season to fill.
        limit: Fetch at most this many games (useful for a smoke run).
        show_progress: Draw a tqdm bar.
        flush_every: Games to buffer between writes.
    """
    missing = missing_boxscore_games(season_year)
    if missing.empty:
        print(f"[{season_year}] box scores already complete.")
        return dict.fromkeys(BOXSCORE_TABLES, 0)

    if limit is not None:
        missing = missing.head(limit)

    print(f"[{season_year}] fetching {len(missing)} missing box scores.")
    metas = game_meta_from_schedule(missing)

    collected: dict[str, list[dict]] = {table: [] for table in BOXSCORE_TABLES}
    collected_game_pks: set[str] = set()
    written: dict[str, int] = dict.fromkeys(BOXSCORE_TABLES, 0)
    failures: list[str] = []

    def flush() -> None:
        if not collected_game_pks:
            return
        for table, rows in collected.items():
            written[table] += replace_game_rows(
                table,
                pd.DataFrame(rows),
                game_pks=collected_game_pks,
                season_year=season_year,
            )
            rows.clear()
        collected_game_pks.clear()

    iterator = tqdm(metas, disable=not show_progress, desc=f"boxscores {season_year}")
    for index, meta in enumerate(iterator, start=1):
        try:
            parsed = parse_boxscore(fetch_boxscore(meta["game_pk"]), meta)
        except RateLimitedError:
            # Throttled. Persist what we have and stop -- the gap query will
            # resume from here on the next run.
            print("Rate limited. Writing partial results and stopping.")
            break
        except StatsApiError as exc:
            failures.append(f"{meta['game_pk']}: {exc}")
            continue

        for table in BOXSCORE_TABLES:
            collected[table].extend(parsed[table])
        collected_game_pks.add(str(meta["game_pk"]))

        if flush_every and index % flush_every == 0:
            flush()

    flush()

    if failures:
        print(f"{len(failures)} game(s) failed and were skipped:")
        for failure in failures[:10]:
            print(f"  - {failure}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    return written


def update_venues(season_year: int) -> dict[str, int]:
    """Refresh park records for a season.

    Re-fetched per season because dimensions and roof state are versioned.
    """
    return {"venues": write_table("venues", fetch_venues(season_year))}


def update_transactions(season_year: int) -> dict[str, int]:
    """Refresh the transaction feed for a season.

    Re-run in full rather than incrementally: a transaction can be corrected
    or superseded after it is first published, and the upsert absorbs that.
    """
    start, end = season_date_bounds(season_year)
    return {"transactions": write_table("transactions", fetch_transactions(start, end))}


def update_season(
    season_year: int,
    *,
    exclude_today: bool = True,
    boxscore_limit: int | None = None,
) -> dict[str, int]:
    """Run every updater for one season, in dependency order.

    Games first: it is the reference for what should exist, so nothing
    downstream can run before it.
    """
    exclude_dates = [mlb_slate_date()] if exclude_today else []

    results: dict[str, int] = {}
    results.update(update_games(season_year, exclude_dates=exclude_dates))
    results.update(update_venues(season_year))
    results.update(update_boxscores(season_year, limit=boxscore_limit))
    results.update(update_transactions(season_year))
    return results


def update_recent_days(days_back: int = 3) -> dict[str, int]:
    """Daily-orchestrator path: refresh the last few slates only.

    Deliberately looks back several days rather than one. Games get suspended
    and resumed, scores get corrected, and a run that failed yesterday should
    be repaired without a manual backfill.

    Today's slate is excluded: a box score fetched mid-game is a partial box
    score that looks complete.
    """
    end = mlb_slate_date() - timedelta(days=1)
    start = end - timedelta(days=max(days_back - 1, 0))

    games, umpires, lineups = fetch_schedule(start, end)
    games = finished_games(games)
    if games.empty:
        print(f"No finished games between {start} and {end}.")
        return {}

    results = {"games": write_table("games", games)}
    for season_year, season_games in games.groupby("season_year"):
        season_pks = {str(v) for v in season_games["game_pk"]}
        for table, frame in (("umpires", umpires), ("lineups", lineups)):
            count = replace_game_rows(
                table,
                frame[frame["game_pk"].astype(str).isin(season_pks)],
                game_pks=season_pks,
                season_year=int(season_year),
            )
            results[table] = results.get(table, 0) + count

    collected: dict[str, list[dict]] = {table: [] for table in BOXSCORE_TABLES}
    for meta in game_meta_from_schedule(games):
        parsed = parse_boxscore(fetch_boxscore(meta["game_pk"]), meta)
        for table in BOXSCORE_TABLES:
            collected[table].extend(parsed[table])

    for season_year, season_games in games.groupby("season_year"):
        season_pks = {str(v) for v in season_games["game_pk"]}
        for table, rows in collected.items():
            frame = pd.DataFrame(rows)
            if not frame.empty:
                frame = frame[frame["game_pk"].astype(str).isin(season_pks)]
            count = replace_game_rows(
                table,
                frame,
                game_pks=season_pks,
                season_year=int(season_year),
            )
            results[table] = results.get(table, 0) + count

    seasons = sorted(games["season_year"].unique())
    for season_year in seasons:
        results.update(update_transactions(int(season_year)))

    return results
