#!/usr/bin/env python
"""Archive older local Parquet seasons to S3 with a verifiable manifest.

Dry-run is the default. Nothing local is deleted, even after a verified upload.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from mlb_pred.archive.s3_archive import (
    discover_parquet,
    upload_and_verify,
    write_manifest,
)
from mlb_pred.config.settings import SETTINGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all",
        action="store_true",
        help="Archive every downloaded Parquet partition, including recent seasons.",
    )
    selection.add_argument(
        "--before-season",
        type=int,
        help="Archive season partitions older than this year (default: 2025).",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("MLB_ARCHIVE_S3_BUCKET")
        or SETTINGS.config.get("S3", "BUCKET", fallback=None),
        help="S3 bucket; defaults to MLB_ARCHIVE_S3_BUCKET or [S3] BUCKET.",
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("MLB_ARCHIVE_S3_PREFIX", "mlb-pred/parquet"),
        help="Key prefix (default: mlb-pred/parquet).",
    )
    parser.add_argument(
        "--profile",
        default=os.getenv("AWS_PROFILE")
        or SETTINGS.config.get("S3", "AWS_PROFILE", fallback=None),
        help="AWS CLI profile; defaults to AWS_PROFILE or [S3] AWS_PROFILE.",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION")
        or SETTINGS.config.get("S3", "AWS_REGION", fallback=None),
        help="AWS region; defaults to AWS_REGION or [S3] AWS_REGION.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest output path; defaults under data/archive_manifests/.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform uploads. Without this flag, only build the manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    before_season = None if args.all else (args.before_season or 2025)
    if before_season is not None and (
        before_season < 1876 or before_season > 2100
    ):
        print(f"Invalid --before-season: {args.before_season}")
        return 2
    if args.execute and not args.bucket:
        print("--bucket or MLB_ARCHIVE_S3_BUCKET is required with --execute.")
        return 2

    objects = discover_parquet(
        SETTINGS.raw_data_path,
        before_season=before_season,
        prefix=args.prefix,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    label = "all_parquet" if before_season is None else f"historical_before_{before_season}"
    manifest = args.manifest or (
        SETTINGS.raw_data_path.parent / "archive_manifests" / f"{label}_{stamp}.json"
    )
    write_manifest(
        objects,
        manifest,
        bucket=args.bucket,
        before_season=before_season,
    )

    rows = sum(obj.rows for obj in objects)
    size_mib = sum(obj.size_bytes for obj in objects) / (1024 * 1024)
    print(
        f"Manifested {len(objects)} object(s), {rows:,} row(s), "
        f"{size_mib:.1f} MiB: {manifest}"
    )
    if not args.execute:
        print("Dry run only; pass --execute with a bucket to upload.")
        return 0

    uploaded, skipped = upload_and_verify(
        objects,
        bucket=args.bucket,
        profile=args.profile,
        region=args.region,
    )
    print(f"S3 archive verified: {uploaded} uploaded, {skipped} already current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
