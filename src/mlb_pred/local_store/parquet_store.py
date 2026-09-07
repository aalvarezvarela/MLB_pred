"""Local raw-data copy, as season-partitioned Parquet.

This is the working store for development: inspectable with pandas, cheap to
regenerate, and never committed (``data/`` is gitignored). The Postgres layer
in ``mlb_pred.postgre_db`` writes the same schemas, so switching
``[Database] DB_ENV`` moves the same pipeline onto Aiven without any change to
the fetchers.

Writes are **insert-only with key-level upsert**: an existing row for a
primary key is replaced by the incoming one, and everything else is left
alone. Combined with the gap-driven updaters, that makes any script in this
repo safe to re-run at any cadence and cheap when there is nothing to do.
"""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from mlb_pred.config.settings import SETTINGS
from mlb_pred.local_store.tables import TABLES, TableSpec, get_spec


def store_root() -> Path:
    root = SETTINGS.raw_data_path
    root.mkdir(parents=True, exist_ok=True)
    return root


def table_dir(table: str) -> Path:
    path = store_root() / table
    path.mkdir(parents=True, exist_ok=True)
    return path


def _partition_path(spec: TableSpec, partition: object | None) -> Path:
    if spec.partition_column is None or partition is None:
        return table_dir(spec.name) / f"{spec.name}.parquet"
    return table_dir(spec.name) / f"{spec.name}_{partition}.parquet"


def partition_paths(table: str) -> list[Path]:
    spec = get_spec(table)
    return sorted(table_dir(spec.name).glob(f"{spec.name}*.parquet"))


def read_table(table: str, partitions: list[int] | None = None) -> pd.DataFrame:
    """Read a table, optionally restricted to specific partitions."""
    spec = get_spec(table)

    if partitions is not None and spec.partition_column is not None:
        paths = [_partition_path(spec, p) for p in partitions]
        paths = [p for p in paths if p.exists()]
    else:
        paths = partition_paths(table)

    if not paths:
        return pd.DataFrame()

    frames = [pd.read_parquet(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    return _sorted(combined, spec)


def _sorted(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    columns = [c for c in spec.sort_columns if c in df.columns]
    if not columns:
        return df.reset_index(drop=True)
    return df.sort_values(columns, kind="mergesort").reset_index(drop=True)


def write_table(table: str, df: pd.DataFrame) -> int:
    """Upsert ``df`` into the store. Returns the number of rows written.

    Rows are routed to partitions by the spec's partition column, and within
    each partition an incoming row replaces any existing row with the same
    primary key.
    """
    if df is None or df.empty:
        return 0

    spec = get_spec(table)
    missing = [c for c in spec.primary_key if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot write {table!r}: primary-key column(s) {missing} absent. "
            f"Expected key {spec.primary_key}."
        )

    if spec.partition_column is None or spec.partition_column not in df.columns:
        return _write_partition(spec, None, df)

    written = 0
    for partition, chunk in df.groupby(spec.partition_column, dropna=False):
        written += _write_partition(spec, partition, chunk)
    return written


def _write_partition(
    spec: TableSpec, partition: object | None, incoming: pd.DataFrame
) -> int:
    path = _partition_path(spec, partition)

    # Drop duplicates within the incoming batch first, keeping the last
    # occurrence: a re-fetch of the same game should not fan out.
    incoming = incoming.drop_duplicates(subset=list(spec.primary_key), keep="last")

    with _partition_lock(path):
        if path.exists():
            existing = pd.read_parquet(path)
            keys = incoming[list(spec.primary_key)].apply(tuple, axis=1)
            existing_keys = existing[list(spec.primary_key)].apply(tuple, axis=1)
            retained = existing[~existing_keys.isin(set(keys))]
            combined = pd.concat([retained, incoming], ignore_index=True)
        else:
            combined = incoming

        combined = _sorted(combined, spec)
        _atomic_write_parquet(combined, path)
    return len(incoming)


@contextmanager
def _partition_lock(path: Path) -> Iterator[None]:
    """Serialize each partition's complete read/modify/replace transaction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a partition without ever exposing a half-written Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".parquet.tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        # Read the footer before promotion. A truncated Parquet file fails here.
        if pq.ParquetFile(temporary).metadata.num_rows != len(frame):
            raise OSError(f"Row-count validation failed for temporary file {temporary}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def replace_game_rows(
    table: str,
    df: pd.DataFrame,
    *,
    game_pks: set[str],
    season_year: int,
) -> int:
    """Replace complete per-game child collections for one season.

    Unlike a key upsert, this removes rows that disappeared from an authoritative
    refreshed collection (for example a player removed from a revised lineup).
    """
    spec = get_spec(table)
    if spec.game_column is None or spec.partition_column != "season_year":
        raise ValueError(f"{table!r} is not a season-partitioned game table")
    incoming = df.copy() if df is not None else pd.DataFrame()
    if not incoming.empty:
        missing = [c for c in spec.primary_key if c not in incoming.columns]
        if missing:
            raise ValueError(f"Cannot replace {table!r}: missing key columns {missing}")
        incoming = incoming[
            incoming[spec.game_column].astype(str).isin({str(v) for v in game_pks})
        ]
        incoming = incoming.drop_duplicates(list(spec.primary_key), keep="last")

    path = _partition_path(spec, season_year)
    with _partition_lock(path):
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        if not existing.empty:
            existing = existing[
                ~existing[spec.game_column].astype(str).isin({str(v) for v in game_pks})
            ]
        combined = pd.concat([existing, incoming], ignore_index=True)
        _atomic_write_parquet(_sorted(combined, spec), path)
    return len(incoming)


def rewrite_partition(table: str, partition: int, df: pd.DataFrame) -> None:
    """Atomically replace a whole partition for explicit repair migrations."""
    spec = get_spec(table)
    path = _partition_path(spec, partition)
    with _partition_lock(path):
        _atomic_write_parquet(_sorted(df, spec), path)


def existing_keys(table: str, column: str, partitions: list[int] | None = None) -> set:
    """Distinct values of ``column`` already present. The gap query's engine."""
    df = read_table(table, partitions=partitions)
    if df.empty or column not in df.columns:
        return set()
    return set(df[column].dropna().unique())


def existing_game_pks(table: str, season_year: int | None = None) -> set[str]:
    """Game ids already present in ``table``."""
    spec = get_spec(table)
    if spec.game_column is None:
        raise ValueError(f"Table {table!r} is not keyed on a game id.")
    partitions = None if season_year is None else [season_year]
    return {str(v) for v in existing_keys(table, spec.game_column, partitions)}


def table_summary() -> pd.DataFrame:
    """Row counts and season coverage per table -- a quick health check."""
    rows = []
    for name, spec in sorted(TABLES.items()):
        df = read_table(name)
        partition_column = spec.partition_column
        rows.append(
            {
                "table": name,
                "rows": len(df),
                "partitions": len(partition_paths(name)),
                "min_partition": (
                    df[partition_column].min()
                    if partition_column and partition_column in df.columns and len(df)
                    else None
                ),
                "max_partition": (
                    df[partition_column].max()
                    if partition_column and partition_column in df.columns and len(df)
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)
