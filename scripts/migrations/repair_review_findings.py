#!/usr/bin/env python
"""Idempotently repair local partitions affected by the initial review findings."""

from __future__ import annotations

import sys

import pandas as pd

from mlb_pred.fetch_data.statcast.client import STATCAST_COLUMNS
from mlb_pred.local_store.odds_updaters import repair_all_opener_flags
from mlb_pred.local_store.parquet_store import read_table, rewrite_partition


def repair_boxscore_columns() -> dict[str, int]:
    changed = {"team_games": 0, "batter_games": 0}
    team_games = read_table("team_games")
    if not team_games.empty:
        for season in sorted(int(v) for v in team_games["season_year"].unique()):
            frame = read_table("team_games", partitions=[season])
            touched = False
            for column in ("inherited_runners", "inherited_runners_scored"):
                if column not in frame:
                    continue
                values = pd.to_numeric(frame[column], errors="coerce")
                if values.notna().any() and values.dropna().eq(0).all():
                    frame[column] = pd.NA
                    touched = True
            if touched:
                rewrite_partition("team_games", season, frame)
                changed["team_games"] += len(frame)

    batters = read_table("batter_games")
    if not batters.empty:
        for season in sorted(int(v) for v in batters["season_year"].unique()):
            frame = read_table("batter_games", partitions=[season])
            touched = False
            if "pitches_seen" in frame and frame["pitches_seen"].isna().all():
                frame = frame.drop(columns="pitches_seen")
                touched = True
            expected = frame["plate_appearances"].fillna(0).gt(0)
            if "had_plate_appearance" not in frame or not frame[
                "had_plate_appearance"
            ].equals(expected):
                frame["had_plate_appearance"] = expected
                touched = True
            if touched:
                rewrite_partition("batter_games", season, frame)
                changed["batter_games"] += len(frame)
    return changed


def repair_statcast_schema() -> int:
    """Bring existing pitch partitions onto the frozen enrichment schema."""
    pitches = read_table("statcast_pitches")
    if pitches.empty:
        return 0
    rewritten = 0
    for season in sorted(int(v) for v in pitches["season_year"].unique()):
        frame = read_table("statcast_pitches", partitions=[season])
        missing = [column for column in STATCAST_COLUMNS if column not in frame]
        if not missing:
            continue
        for column in missing:
            frame[column] = None
        rewrite_partition("statcast_pitches", season, frame[list(STATCAST_COLUMNS)])
        rewritten += len(frame)
    return rewritten


def main() -> int:
    boxscore = repair_boxscore_columns()
    openers = repair_all_opener_flags()
    statcast = repair_statcast_schema()
    print(f"box-score rows rewritten: {boxscore}")
    print(f"odds opener flags changed: {openers}")
    print(f"Statcast rows brought onto current schema: {statcast}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
