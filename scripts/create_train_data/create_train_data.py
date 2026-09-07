#!/usr/bin/env python3
"""Create the MLB historical training CSV from pregame feature partitions."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import pandas as pd

from mlb_pred.config.dataset_versions import TRAINING_DATA_SCHEMA_VERSION
from mlb_pred.config.settings import PROJECT_ROOT
from mlb_pred.create_training_data.training_frame import (
    DEFAULT_PREGAME_DIR,
    LINE_ERROR_COLUMN,
    RAW_RUN_LINE_HANDICAP_COLUMN,
    RAW_TOTAL_LINE_COLUMN,
    SPREAD_ERROR_COLUMN,
    build_training_frame,
    feature_columns,
    load_pregame_features,
)
from mlb_pred.local_store.parquet_store import read_table

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "train_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=str,
        help="Latest completed game date to include (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        help="Calendar seasons to include (default: every pregame partition).",
    )
    parser.add_argument("--pregame-dir", type=Path, default=DEFAULT_PREGAME_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV destination (default: versioned file under data/train_data).",
    )
    # The NBA script normalizes at this step and can be told not to.  MLB
    # normalizes upstream in market_normalization and carries both shapes into
    # the pregame partition, so the equivalent control selects which stored
    # quote the residual targets are measured against.
    parser.add_argument(
        "--raw-lines",
        action="store_true",
        help=(
            "Measure LINE_ERROR and SPREAD_ERROR against the raw executable "
            "close instead of the -110/-110 normalized close."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".csv.tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        # A truncated/empty file should never replace a valid training dataset.
        if temporary.stat().st_size == 0:
            raise OSError(f"CSV write produced an empty file: {temporary}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    pregame = load_pregame_features(args.pregame_dir, seasons=args.seasons)
    games = read_table("games", partitions=args.seasons)
    line_columns = (
        {
            "total_line_column": RAW_TOTAL_LINE_COLUMN,
            "run_line_handicap_column": RAW_RUN_LINE_HANDICAP_COLUMN,
        }
        if args.raw_lines
        else {}
    )
    training = build_training_frame(
        pregame,
        games,
        seasons=args.seasons,
        limit_date=args.limit,
        **line_columns,
    )
    # An inner join silently discards a scheduled game that never completed.
    # Say how many, so a half-ingested season cannot look like a full one.
    # With --limit the same count also absorbs rows cut by the cutoff, so the
    # message must not attribute all of them to missing results.
    dropped = len(pregame) - len(training)
    if dropped:
        reason = (
            "no completed final score or after the --limit cutoff"
            if args.limit
            else "no completed final score"
        )
        print(f"Excluded {dropped:,} of {len(pregame):,} pregame row(s): {reason}.")

    maximum_date = pd.to_datetime(training["GAME_DATE"], errors="raise").max()
    destination = args.output
    if destination is None:
        destination = DEFAULT_OUTPUT_DIR / (
            f"training_data_{TRAINING_DATA_SCHEMA_VERSION}_{maximum_date:%Y%m%d}.csv"
        )
    _atomic_write_csv(training, destination)

    print(
        f"Training data: {len(training):,} completed games x "
        f"{len(training.columns):,} columns ({len(feature_columns(training)):,} "
        "model feature columns)."
    )
    print(
        f"Target coverage: {LINE_ERROR_COLUMN}="
        f"{int(training[LINE_ERROR_COLUMN].notna().sum()):,}, "
        f"{SPREAD_ERROR_COLUMN}="
        f"{int(training[SPREAD_ERROR_COLUMN].notna().sum()):,}."
    )
    print(destination)
    print(f'sha256: "{_sha256(destination)}"')


if __name__ == "__main__":
    main()
