#!/usr/bin/env python
"""Backfill Baseball Savant Statcast pitches for finished stored games."""

from __future__ import annotations

import argparse
import sys

from mlb_pred.config.settings import SETTINGS
from mlb_pred.local_store.statcast_updaters import statcast_coverage, update_statcast
from mlb_pred.utils.seasons import mlb_slate_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    parser.add_argument("--limit-games", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seasons = args.seasons or list(
        range(SETTINGS.first_season, mlb_slate_date().year + 1)
    )
    failures = 0
    for season in seasons:
        print(f"\n=== Statcast {season} ===")
        result = update_statcast(
            season,
            limit=args.limit_games,
            show_progress=not args.no_progress,
        )
        failures += result["failures"]
        print(result)
    coverage = statcast_coverage()
    if not coverage.empty:
        print("\n" + coverage.to_string(index=False))
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
