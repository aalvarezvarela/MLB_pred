#!/usr/bin/env python
"""Archive today's point-in-time snapshots. Run this several times a day.

This is the one job whose missed runs cannot be repaired later. Lineups are
posted three to four hours before first pitch and revised after that, so a
single daily capture records one point on a curve that matters.

A sensible cron is roughly 11:00, 15:00, 17:00 and 19:00 local, plus one late
capture for west-coast night games.

Examples::

    python scripts/daily/capture_snapshots.py
    python scripts/daily/capture_snapshots.py --date 2026-08-29 --no-rosters
    python scripts/daily/capture_snapshots.py --coverage
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from mlb_pred.snapshots.archive import (
    capture_all,
    default_capture_dates,
    missing_snapshot_dates,
    snapshot_coverage,
)
from mlb_pred.utils.seasons import mlb_slate_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=1,
        help="Also capture this many future slates (tomorrow's probables are "
        "published today and are the earliest form of the signal).",
    )
    parser.add_argument("--no-rosters", action="store_true")
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Report what has been archived instead of capturing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.coverage:
        coverage = snapshot_coverage()
        if coverage.empty:
            print("No snapshots archived yet.")
        else:
            print(coverage.to_string(index=False))
            end = mlb_slate_date()
            start = end - timedelta(days=30)
            missing = missing_snapshot_dates("probables", start, end)
            if missing:
                print(
                    f"\nMissing probables for {len(missing)} slate(s) in the last 30 days:"
                )
                print("  " + ", ".join(d.isoformat() for d in missing))
                print("  These gaps are permanent -- they cannot be backfilled.")
        return 0

    targets = [args.date] if args.date else default_capture_dates(args.days_ahead)

    for slate_date in targets:
        results = capture_all(slate_date, include_rosters=not args.no_rosters)
        written = {k: v.name for k, v in results.items() if v is not None}
        skipped = sorted(k for k, v in results.items() if v is None)
        print(f"{slate_date}: {written}")
        if skipped:
            print(f"  (nothing to archive for: {', '.join(skipped)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
