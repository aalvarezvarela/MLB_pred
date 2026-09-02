from pathlib import Path

import pandas as pd

from mlb_pred.archive.s3_archive import (
    discover_historical_parquet,
    discover_parquet,
    write_manifest,
)


def test_aiven_standalone_password_overrides_embedded_url(monkeypatch):
    from mlb_pred.postgre_db.config import db_config

    monkeypatch.setenv(
        "AIVEN_DB_URL",
        "postgresql://user:stale@example.test:5432/defaultdb?sslmode=require",
    )
    monkeypatch.setenv("AIVEN_DB_PASSWORD", "new password/with?reserved#chars")

    params = db_config.conninfo_to_dict(db_config.get_db_dsn("aiven"))

    assert params["password"] == "new password/with?reserved#chars"


def _parquet(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"game_pk": [str(i) for i in range(rows)]}).to_parquet(
        path, index=False
    )


def test_archive_discovers_only_seasons_before_cutoff(tmp_path):
    _parquet(tmp_path / "games" / "games_2024.parquet", 2)
    _parquet(tmp_path / "games" / "games_2025.parquet", 3)
    _parquet(tmp_path / "games" / "games_2026.parquet", 4)

    objects = discover_historical_parquet(
        tmp_path,
        before_season=2025,
        prefix="mlb/parquet/",
    )

    assert len(objects) == 1
    assert objects[0].season == 2024
    assert objects[0].rows == 2
    assert objects[0].s3_key == "mlb/parquet/raw/games/games_2024.parquet"
    assert len(objects[0].sha256) == 64


def test_archive_all_discovers_recent_and_historical_seasons(tmp_path):
    _parquet(tmp_path / "games" / "games_2024.parquet", 2)
    _parquet(tmp_path / "games" / "games_2026.parquet", 4)

    objects = discover_parquet(tmp_path, prefix="mlb/parquet")

    assert [obj.season for obj in objects] == [2024, 2026]
    assert sum(obj.rows for obj in objects) == 6


def test_archive_manifest_contains_aggregate_counts(tmp_path):
    _parquet(tmp_path / "games" / "games_2023.parquet", 2)
    objects = discover_historical_parquet(
        tmp_path,
        before_season=2025,
        prefix="mlb",
    )
    manifest = tmp_path / "manifest.json"

    write_manifest(objects, manifest, bucket="archive-bucket", before_season=2025)

    payload = manifest.read_text(encoding="utf-8")
    assert '"object_count": 1' in payload
    assert '"row_count": 2' in payload
    assert '"bucket": "archive-bucket"' in payload
