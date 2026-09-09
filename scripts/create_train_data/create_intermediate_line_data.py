#!/usr/bin/env python3
"""Create the MLB intermediate-line training CSV, one row per (game, snapshot).

One builder, one ``--market`` argument, one pair of files per market.  The
market selects the anchor, the target and the level definition; every other
module is market-agnostic and takes it as a parameter.

Two files are always written.  The scoring sidecar holds the closing quotes, the
realised scores' timestamps and the moneyline block -- everything the feature
matrix must never see.  They are physically separate rather than distinguished
by a prefix, because a prefix relies on every downstream consumer remembering to
filter and one that forgets gets a closing line in ``X``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import pandas as pd

from mlb_pred.config.dataset_versions import INTERMEDIATE_LINE_SCHEMA_VERSION
from mlb_pred.config.settings import PROJECT_ROOT
from mlb_pred.create_training_data.intermediate_frame import (
    ANCHOR_BOOK,
    ANCHORABLE_MARKETS,
    HORIZON_COLUMN,
    build_intermediate_frame,
    feature_columns,
)
from mlb_pred.create_training_data.training_frame import (
    DEFAULT_PREGAME_DIR,
    load_pregame_features,
)
from mlb_pred.features.closing_lines import MARKET_TOTALS, STABLE_BOOKS
from mlb_pred.features.line_movement import DEFAULT_MOVEMENT_WINDOWS
from mlb_pred.features.line_snapshots import DEFAULT_SNAPSHOT_GRID
from mlb_pred.local_store.parquet_store import read_table

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "train_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market",
        choices=ANCHORABLE_MARKETS,
        default=MARKET_TOTALS,
        help="Which market anchors the target (default: totals).",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        help="Calendar seasons to include (default: every pregame partition).",
    )
    parser.add_argument("--pregame-dir", type=Path, default=DEFAULT_PREGAME_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--grid",
        nargs="+",
        type=int,
        default=list(DEFAULT_SNAPSHOT_GRID),
        help="Snapshot horizons, in minutes before first pitch.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=list(DEFAULT_MOVEMENT_WINDOWS),
        help="Trailing look-back windows, in minutes before each snapshot.",
    )
    parser.add_argument("--anchor-book", default=ANCHOR_BOOK)
    parser.add_argument(
        "--raw-anchor",
        action="store_true",
        help=(
            "Measure the residual against the raw executable quote instead of "
            "its -110/-110 restatement. The raw line is what a bet actually "
            "settles into; the normalised one is comparable across books and "
            "horizons but is not a number anyone can take."
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
        # A truncated or empty file must never replace a valid dataset.
        if temporary.stat().st_size == 0:
            raise OSError(f"CSV write produced an empty file: {temporary}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    pregame = load_pregame_features(args.pregame_dir, seasons=args.seasons)
    games = read_table("games", partitions=args.seasons)
    odds_ticks = read_table("odds_ticks", partitions=args.seasons)

    training, scoring = build_intermediate_frame(
        odds_ticks,
        pregame,
        games,
        market=args.market,
        grid=tuple(args.grid),
        windows=tuple(args.windows),
        anchor_book=args.anchor_book,
        books=STABLE_BOOKS,
        use_normalized_anchor=not args.raw_anchor,
    )

    maximum_date = pd.to_datetime(training["GAME_DATE"], errors="raise").max()
    stem = (
        f"intermediate_{args.market}_{INTERMEDIATE_LINE_SCHEMA_VERSION}_"
        f"{maximum_date:%Y%m%d}"
    )
    training_path = args.output_dir / f"{stem}.csv"
    scoring_path = args.output_dir / f"{stem}_scoring.csv"
    _atomic_write_csv(training, training_path)
    _atomic_write_csv(scoring, scoring_path)

    horizons = sorted(training[HORIZON_COLUMN].unique())
    print(
        f"Intermediate {args.market}: {len(training):,} rows over "
        f"{training['GAME_ID'].nunique():,} games x {len(horizons)} horizons, "
        f"{training.shape[1]:,} columns "
        f"({len(feature_columns(training)):,} model features)."
    )
    print(f"Horizons: {horizons}")
    print(f"Scoring sidecar: {scoring.shape[1]:,} columns (never features).")
    for path in (training_path, scoring_path):
        print(path)
        print(f'sha256: "{_sha256(path)}"')


if __name__ == "__main__":
    main()
