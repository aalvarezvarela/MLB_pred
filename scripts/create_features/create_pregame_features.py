"""Build closing-market plus NBA-shaped rolling MLB feature partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlb_pred.features.advanced_features import build_advanced_pregame_features
from mlb_pred.features.context_features import build_context_features
from mlb_pred.features.rolling_features import (
    DEFAULT_OUTPUT_DIR,
    build_pregame_features,
    build_team_rolling_features,
    write_pregame_feature_partitions,
)
from mlb_pred.local_store.parquet_store import read_table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLOSING_DIR = PROJECT_ROOT / "data" / "features" / "closing_lines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        help="Output seasons (default: every generated closing partition).",
    )
    parser.add_argument("--closing-dir", type=Path, default=CLOSING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    closing_paths = sorted(args.closing_dir.glob("closing_lines_*.parquet"))
    if not closing_paths:
        raise SystemExit(
            "No closing feature partitions found. Run create_closing_lines.py first."
        )
    closing = pd.concat(
        [pd.read_parquet(path) for path in closing_paths], ignore_index=True
    )
    seasons = sorted(
        set(args.seasons or closing["GAME_SEASON_YEAR"].astype(int).unique())
    )
    unknown_seasons = sorted(set(seasons).difference(closing["GAME_SEASON_YEAR"]))
    if unknown_seasons:
        raise SystemExit(f"No closing partitions found for seasons: {unknown_seasons}")
    target_closing = closing.loc[closing["GAME_SEASON_YEAR"].isin(seasons)].copy()

    # Load all earlier team seasons as context. Restricting this read to output
    # seasons would erase cross-offseason team and odds history. Later games are
    # harmless because every rolling operation is shifted and ordered by date.
    team_games = read_table("team_games")
    games = read_table("games")
    venues = read_table("venues")

    team_rolling = build_team_rolling_features(
        team_games,
        games.loc[games["game_pk"].astype(str).isin(set(closing["GAME_ID"]))],
        closing,
        target_game_ids=target_closing["GAME_ID"],
    )
    context = build_context_features(
        games,
        team_games,
        venues,
        closing,
        target_game_ids=target_closing["GAME_ID"],
    )
    advanced = build_advanced_pregame_features(
        games,
        closing,
        team_rolling,
        target_game_ids=target_closing["GAME_ID"],
    )
    context = context.merge(advanced, on="GAME_ID", how="left", validate="one_to_one")
    features = build_pregame_features(target_closing, team_rolling, context)
    destinations = write_pregame_feature_partitions(
        features, output_dir=args.output_dir, seasons=seasons
    )
    print(
        f"Built {len(features):,} games x {len(features.columns):,} labeled "
        f"pregame columns across {len(destinations)} partition(s)."
    )
    for destination in destinations:
        print(destination)


if __name__ == "__main__":
    main()
