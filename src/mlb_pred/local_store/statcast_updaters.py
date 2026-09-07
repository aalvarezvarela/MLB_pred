"""Gap-driven Statcast ingestion with explicit per-game completion records."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from tqdm import tqdm

from mlb_pred.fetch_data.statcast.client import StatcastError, fetch_statcast_game
from mlb_pred.local_store.parquet_store import (
    read_table,
    replace_game_rows,
    write_table,
)

STATCAST_COMPLETE_STATUSES = frozenset({"complete", "no_data"})


def missing_statcast_games(season_year: int) -> pd.DataFrame:
    games = read_table("games", partitions=[season_year])
    if games.empty:
        return games
    if "is_final" in games:
        games = games[games["is_final"]]
    fetched = read_table("statcast_games", partitions=[season_year])
    complete: set[str] = set()
    if not fetched.empty:
        complete = set(
            fetched[fetched["fetch_status"].isin(STATCAST_COMPLETE_STATUSES)][
                "game_pk"
            ].astype(str)
        )
    return games[~games["game_pk"].astype(str).isin(complete)].reset_index(drop=True)


def update_statcast(
    season_year: int,
    *,
    limit: int | None = None,
    flush_every: int = 25,
    show_progress: bool = True,
) -> dict[str, int]:
    missing = missing_statcast_games(season_year)
    if limit is not None:
        missing = missing.head(limit)
    if missing.empty:
        return {"statcast_games": 0, "statcast_pitches": 0, "failures": 0}

    pitches: list[pd.DataFrame] = []
    statuses: list[dict] = []
    written = {"statcast_games": 0, "statcast_pitches": 0, "failures": 0}

    def flush() -> None:
        if not statuses:
            return
        status_frame = pd.DataFrame(statuses)
        game_pks = set(status_frame["game_pk"].astype(str))
        pitch_frame = (
            pd.concat(pitches, ignore_index=True) if pitches else pd.DataFrame()
        )
        # Facts first, completion marker last. A crash can cause a harmless
        # re-fetch, never a false "complete" game.
        written["statcast_pitches"] += replace_game_rows(
            "statcast_pitches",
            pitch_frame,
            game_pks=game_pks,
            season_year=season_year,
        )
        written["statcast_games"] += write_table("statcast_games", status_frame)
        pitches.clear()
        statuses.clear()

    iterator = tqdm(
        missing.to_dict("records"),
        disable=not show_progress,
        desc=f"statcast {season_year}",
    )
    for game in iterator:
        game_pk = str(game["game_pk"])
        try:
            frame = fetch_statcast_game(game_pk, season_year)
        except StatcastError as exc:
            written["failures"] += 1
            print(f"  ! Statcast game {game_pk}: {exc}")
            continue

        if not frame.empty:
            pitches.append(frame)
        statuses.append(
            {
                "game_pk": game_pk,
                "season_year": season_year,
                "game_date": game["game_date"],
                "fetch_status": "complete" if len(frame) else "no_data",
                "pitch_rows": len(frame),
                "fetched_at_utc": datetime.now(UTC),
            }
        )
        if flush_every and len(statuses) >= flush_every:
            flush()
    flush()
    return written


def statcast_coverage() -> pd.DataFrame:
    games = read_table("statcast_games")
    pitches = read_table("statcast_pitches")
    if games.empty:
        return pd.DataFrame()
    result = (
        games.groupby(["season_year", "fetch_status"])["game_pk"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )
    pitch_counts = pitches.groupby("season_year").size() if not pitches.empty else {}
    result["pitches"] = result["season_year"].map(pitch_counts).fillna(0).astype(int)
    return result
