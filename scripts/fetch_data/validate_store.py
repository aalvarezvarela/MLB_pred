#!/usr/bin/env python
"""Report data-quality findings for the local store.

Exit code 1 when anything is wrong, so this can gate a pipeline.

Examples::

    python scripts/fetch_data/validate_store.py
    python scripts/fetch_data/validate_store.py --seasons 2024 2025
"""

from __future__ import annotations

import argparse
import sys

from mlb_pred.local_store.validate import validate_store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    args = parser.parse_args()

    findings = validate_store(args.seasons)
    if not findings:
        print("No findings. The store is internally consistent.")
        return 0

    print(f"{len(findings)} finding(s):\n")
    for finding in findings:
        print(f"  {finding}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
