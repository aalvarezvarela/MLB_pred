"""Manifested, checksum-verified Parquet archival to Amazon S3."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

_SEASON_FILE = re.compile(r"_(\d{4})\.parquet$")


@dataclass(frozen=True)
class ArchiveObject:
    local_path: str
    relative_path: str
    s3_key: str
    season: int
    rows: int
    size_bytes: int
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_parquet(path: Path) -> tuple[int, int, str]:
    """Read metadata, size, and checksum from one immutable open file handle."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        rows = pq.ParquetFile(stream).metadata.num_rows
        stream.seek(0)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        size = stream.tell()
    return rows, size, digest.hexdigest()


def discover_parquet(
    raw_root: Path,
    *,
    prefix: str,
    before_season: int | None = None,
) -> list[ArchiveObject]:
    """Return validated Parquet objects, optionally restricted by season."""
    normalized_prefix = prefix.strip("/")
    objects: list[ArchiveObject] = []
    for path in sorted(raw_root.rglob("*.parquet")):
        match = _SEASON_FILE.search(path.name)
        if not match:
            continue
        season = int(match.group(1))
        if before_season is not None and season >= before_season:
            continue
        relative = path.relative_to(raw_root).as_posix()
        rows, size, checksum = _inspect_parquet(path)
        key = "/".join(part for part in (normalized_prefix, "raw", relative) if part)
        objects.append(
            ArchiveObject(
                local_path=str(path.resolve()),
                relative_path=relative,
                s3_key=key,
                season=season,
                rows=rows,
                size_bytes=size,
                sha256=checksum,
            )
        )
    return objects


def discover_historical_parquet(
    raw_root: Path,
    *,
    before_season: int,
    prefix: str,
) -> list[ArchiveObject]:
    """Return validated Parquet objects whose filename partition is older."""
    return discover_parquet(
        raw_root,
        before_season=before_season,
        prefix=prefix,
    )


def write_manifest(
    objects: list[ArchiveObject],
    path: Path,
    *,
    bucket: str | None,
    before_season: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "bucket": bucket,
        "selection": "all" if before_season is None else "historical",
        "before_season": before_season,
        "object_count": len(objects),
        "row_count": sum(obj.rows for obj in objects),
        "size_bytes": sum(obj.size_bytes for obj in objects),
        "objects": [asdict(obj) for obj in objects],
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _aws_command(
    profile: str | None,
    region: str | None,
    *arguments: str,
) -> list[str]:
    command = ["aws"]
    if profile:
        command.extend(("--profile", profile))
    if region:
        command.extend(("--region", region))
    command.extend(arguments)
    return command


def _run_aws(
    profile: str | None,
    region: str | None,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _aws_command(profile, region, *arguments),
        check=True,
        text=True,
        capture_output=True,
    )


def upload_and_verify(
    objects: list[ArchiveObject],
    *,
    bucket: str,
    profile: str | None = None,
    region: str | None = None,
) -> tuple[int, int]:
    """Upload missing/changed objects and verify length plus SHA-256 metadata."""
    _run_aws(profile, region, "s3api", "head-bucket", "--bucket", bucket)
    uploaded = 0
    skipped = 0

    for obj in objects:
        remote = None
        try:
            result = _run_aws(
                profile,
                region,
                "s3api",
                "head-object",
                "--bucket",
                bucket,
                "--key",
                obj.s3_key,
                "--query",
                "{size:ContentLength,checksum:Metadata.sha256}",
                "--output",
                "json",
            )
            remote = json.loads(result.stdout)
        except subprocess.CalledProcessError:
            remote = None

        if remote == {"size": obj.size_bytes, "checksum": obj.sha256}:
            skipped += 1
            continue

        # Copy one immutable generation before opening the network stream. A
        # concurrent backfill atomically replacing the source either leaves
        # this copy intact or is detected by the checksum comparison below.
        source = Path(obj.local_path)
        with tempfile.TemporaryDirectory(prefix="mlb-s3-archive-") as directory:
            staged = Path(directory) / source.name
            shutil.copyfile(source, staged)
            if staged.stat().st_size != obj.size_bytes or _sha256(staged) != obj.sha256:
                raise RuntimeError(
                    f"Local partition changed after manifesting: {obj.relative_path}. "
                    "Wait for the backfill to finish and regenerate the manifest."
                )
            _run_aws(
                profile,
                region,
                "s3",
                "cp",
                str(staged),
                f"s3://{bucket}/{obj.s3_key}",
                "--only-show-errors",
                "--sse",
                "AES256",
                "--metadata",
                f"sha256={obj.sha256}",
            )
        verification = _run_aws(
            profile,
            region,
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            obj.s3_key,
            "--query",
            "{size:ContentLength,checksum:Metadata.sha256}",
            "--output",
            "json",
        )
        remote = json.loads(verification.stdout)
        expected = {"size": obj.size_bytes, "checksum": obj.sha256}
        if remote != expected:
            raise RuntimeError(
                f"S3 verification failed for {obj.s3_key}: {remote!r} != {expected!r}"
            )
        uploaded += 1
    return uploaded, skipped
