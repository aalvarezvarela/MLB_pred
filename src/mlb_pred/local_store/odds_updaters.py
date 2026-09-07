"""Gap-driven odds updaters for the local store.

Same discipline as the sports-data updaters: ask the store what is missing,
fetch only that, write insert-only. What differs is the planner -- odds have
three independent reasons to fetch a date, not one, because *presence is not
finality* for a market that keeps moving until first pitch.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from mlb_pred.fetch_data.sbr.client import SbrFetchError, new_session
from mlb_pred.fetch_data.sbr.line_history import scrape_dates
from mlb_pred.local_store.parquet_store import (
    read_table,
    rewrite_partition,
    write_table,
)
from mlb_pred.odds.ingest import build_odds_frames, normalize_opener_flags
from mlb_pred.odds.planner import FIRST_ODDS_SEASON, UpdatePlan, plan_update


def _write(dimension: pd.DataFrame, ticks: pd.DataFrame) -> dict[str, int]:
    counts = {"odds_ticks": write_table("odds_ticks", ticks)}
    if not ticks.empty:
        _repair_affected_openers(ticks)
    counts["odds_games"] = write_table("odds_games", dimension)
    fetch_columns = [
        "game_pk",
        "season_year",
        "game_date",
        "event_id",
        "ingest_status",
        "source_tick_count",
        "usable_tick_count",
        "dropped_tick_count",
        "fetched_at_utc",
    ]
    available = [column for column in fetch_columns if column in dimension.columns]
    counts["odds_fetches"] = write_table("odds_fetches", dimension[available])
    return counts


def _repair_affected_openers(incoming: pd.DataFrame) -> None:
    """Normalize opener flags across old and incoming ticks for touched games."""
    for season_year, season_incoming in incoming.groupby("season_year"):
        stored = read_table("odds_ticks", partitions=[int(season_year)])
        affected = set(season_incoming["game_pk"].astype(str))
        mask = stored["game_pk"].astype(str).isin(affected)
        repaired = normalize_opener_flags(stored.loc[mask])
        write_table("odds_ticks", repaired)


def repair_all_opener_flags(seasons: list[int] | None = None) -> int:
    """One-time idempotent migration for stores created before opener repair."""
    ticks = read_table("odds_ticks")
    if ticks.empty:
        return 0
    targets = seasons or sorted(int(v) for v in ticks["season_year"].unique())
    changed = 0
    for season_year in targets:
        frame = read_table("odds_ticks", partitions=[season_year])
        repaired = normalize_opener_flags(frame)
        changed += int(
            (frame["is_opener"].to_numpy() != repaired["is_opener"].to_numpy()).sum()
        )
        rewrite_partition("odds_ticks", season_year, repaired)
    return changed


def ingest_dates(
    days: list[date],
    *,
    games: pd.DataFrame | None = None,
    on_error: str = "warn",
    flush_every: int = 100,
    progress_every: int = 20,
) -> dict:
    """Scrape and store the odds for a list of slate dates.

    The loop is per *date* rather than over one flat generator so that a date
    that fails outright is recorded on ``failed_dates`` and the run continues.
    A multi-season backfill that dies on hour four because one page 404'd is
    worse than one with a reported gap -- and the planner picks those dates up
    on the next run anyway. Partial success is a first-class outcome the caller
    reads, not an exception.

    Results flush every ``flush_every`` games, so an interrupted run keeps
    everything it had already fetched.
    """
    if games is None:
        games = read_table("games")
    if games.empty:
        raise RuntimeError(
            "The games table is empty. Odds have no useful identity until they "
            "can be joined to it -- run the sports-data backfill first."
        )

    session = new_session()
    written = {"odds_games": 0, "odds_ticks": 0, "odds_fetches": 0}
    all_stats: list = []
    all_resolution: list = []
    buffer: list = []
    failed_dates: list[date] = []

    def flush() -> None:
        if not buffer:
            return
        dimension, ticks, stats, resolution = build_odds_frames(buffer, games)
        counts = _write(dimension, ticks)
        for key, value in counts.items():
            written[key] += value
        all_stats.append(stats)
        all_resolution.append(resolution)
        buffer.clear()

    try:
        for index, day in enumerate(days, start=1):
            try:
                # Strict inside one date so failures reach this boundary and are
                # recorded. The outer policy still decides whether to continue.
                buffer.extend(scrape_dates([day], session=session, on_error="raise"))
            except SbrFetchError as exc:
                if on_error == "raise":
                    raise
                print(f"  ! {day}: {exc}")
                failed_dates.append(day)
                continue

            if len(buffer) >= flush_every:
                flush()
            if progress_every and index % progress_every == 0:
                print(
                    f"  ... {index}/{len(days)} dates, "
                    f"{written['odds_ticks']} tick(s) written"
                )
        flush()
    finally:
        session.close()

    if failed_dates:
        print(f"  {len(failed_dates)} date(s) failed and were skipped:")
        for day in failed_dates[:10]:
            print(f"    - {day}")
        if len(failed_dates) > 10:
            print(f"    ... and {len(failed_dates) - 10} more")

    return {
        "written": written,
        "stats": all_stats,
        "resolution": all_resolution,
        "failed_dates": failed_dates,
    }


def current_plan(
    *,
    refresh_days: int = 3,
    first_season: int = FIRST_ODDS_SEASON,
    today: date | None = None,
) -> UpdatePlan:
    """What the planner would fetch right now, without fetching anything."""
    return plan_update(
        read_table("games"),
        read_table("odds_games"),
        read_table("odds_ticks"),
        odds_fetches=read_table("odds_fetches"),
        refresh_days=refresh_days,
        first_season=first_season,
        today=today,
    )


def update_odds(
    *,
    refresh_days: int = 3,
    first_season: int = FIRST_ODDS_SEASON,
    dry_run: bool = False,
    limit_dates: int | None = None,
    refresh_only: bool = False,
    today: date | None = None,
) -> dict:
    """Run the planner and fetch what it asks for."""
    plan = current_plan(
        refresh_days=refresh_days, first_season=first_season, today=today
    )
    print(plan.summary())

    days = plan.refresh_dates if refresh_only else plan.dates
    if refresh_only:
        print(f"  selected       : {len(days)} recent refresh date(s) only")
    if limit_dates is not None:
        days = days[:limit_dates]

    if dry_run:
        print("\n(dry run -- stopping before the first request)")
        return {"plan": plan, "dates": days}

    if not days:
        print("Nothing to fetch.")
        return {"plan": plan, "dates": []}

    result = ingest_dates(days)
    result["plan"] = plan
    result["dates"] = days
    return result


def odds_coverage() -> pd.DataFrame:
    """Per-season coverage. The honest answer to "what can we train on?".

    ``games_priced`` counts games that actually carry ticks; ``games_scraped``
    counts games in the dimension. The difference is games SBR served a payload
    for that held no usable quote -- recording those is deliberate, because
    "no odds existed" and "we never looked" are different facts and only the
    dimension row distinguishes them.

    ``inplay_share`` is the fraction of ticks quoted at or after first pitch.
    Those rows are stored, not dropped, but every one of them is target leakage
    if a feature reads it -- which is what ``is_pregame`` exists to prevent.
    """
    ticks = read_table("odds_ticks")
    dimension = read_table("odds_games")
    if ticks.empty:
        return pd.DataFrame()

    per_season = (
        ticks.groupby("season_year")
        .agg(
            ticks=("game_pk", "size"),
            games_priced=("game_pk", "nunique"),
            books=("book_slug", "nunique"),
            pregame_ticks=("is_pregame", "sum"),
        )
        .reset_index()
    )
    per_season["inplay_share"] = (
        1.0 - per_season["pregame_ticks"] / per_season["ticks"]
    ).round(4)
    per_season = per_season.drop(columns=["pregame_ticks"])

    if not dimension.empty:
        scraped = dimension.groupby("season_year")["game_pk"].nunique()
        per_season["games_scraped"] = per_season["season_year"].map(scraped)
        per_season["games_without_quotes"] = (
            per_season["games_scraped"] - per_season["games_priced"]
        )
    return per_season
