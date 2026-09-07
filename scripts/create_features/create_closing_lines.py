"""Create leakage-safe closing-line feature parquet partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlb_pred.features.closing_lines import (
    DEFAULT_CLOSE_SAFETY_MARGIN_MINUTES,
    DEFAULT_OUTPUT_DIR,
    STABLE_BOOKS,
    build_closing_line_features,
    write_closing_feature_partition,
)
from mlb_pred.local_store.parquet_store import partition_paths, read_table


def _available_seasons() -> list[int]:
    prefix = "odds_ticks_"
    seasons = []
    for path in partition_paths("odds_ticks"):
        suffix = path.stem.removeprefix(prefix)
        if suffix.isdigit():
            seasons.append(int(suffix))
    return sorted(seasons)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        help="Season years to build (default: every local odds partition).",
    )
    parser.add_argument(
        "--safety-margin-minutes",
        type=int,
        default=DEFAULT_CLOSE_SAFETY_MARGIN_MINUTES,
        help="Ignore quotes closer than this to first pitch (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seasons = sorted(set(args.seasons or _available_seasons()))
    if not seasons:
        raise SystemExit("No local odds-tick seasons were found.")
    for season_year in seasons:
        odds_games = read_table("odds_games", partitions=[season_year])
        odds_ticks = read_table("odds_ticks", partitions=[season_year])
        if odds_games.empty or odds_ticks.empty:
            print(f"{season_year}: skipped (odds games or ticks are empty)")
            continue
        features = build_closing_line_features(
            odds_games,
            odds_ticks,
            safety_margin_minutes=args.safety_margin_minutes,
            # A fixed book list keeps every partition on one schema without
            # scanning the whole store, and confines per-book columns to the
            # books that quote in every season. The others still reach the
            # consensus aggregates, where an absent book lowers BOOK_COUNT.
            books=STABLE_BOOKS,
        )
        destination = write_closing_feature_partition(
            features, season_year, output_dir=args.output_dir
        )
        odds_columns = sum(column.startswith("ODDS_") for column in features.columns)
        print(
            f"{season_year}: wrote {len(features):,} games x "
            f"{odds_columns:,} odds features to {destination}"
        )


if __name__ == "__main__":
    main()
