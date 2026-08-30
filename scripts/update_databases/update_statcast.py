#!/usr/bin/env python
"""Fill missing Statcast games for the current MLB season."""

from __future__ import annotations

import argparse
import sys

from mlb_pred.local_store.statcast_updaters import statcast_coverage, update_statcast
from mlb_pred.utils.seasons import mlb_slate_date


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-games", type=int, default=None)
    args = parser.parse_args()
    season = mlb_slate_date().year
    result = update_statcast(season, limit=args.limit_games)
    print(result)
    coverage = statcast_coverage()
    if not coverage.empty:
        print(coverage.tail(3).to_string(index=False))
    return 2 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
