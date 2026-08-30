"""The local store's insert-only-with-upsert semantics.

Every updater in this repo depends on re-running being safe and cheap.
"""

import pandas as pd
import pytest

from mlb_pred.local_store import parquet_store
from mlb_pred.local_store.tables import TABLES, get_spec


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(parquet_store.SETTINGS, "_cache", {}, raising=False)
    monkeypatch.setattr(
        type(parquet_store.SETTINGS),
        "raw_data_path",
        property(lambda self: tmp_path),
    )
    yield


def _games(pks, season=2025, date="2025-05-01"):
    return pd.DataFrame(
        {
            "game_pk": list(pks),
            "season_year": [season] * len(pks),
            "game_date": [date] * len(pks),
            "first_pitch_utc": [pd.Timestamp(f"{date}T23:05:00Z")] * len(pks),
        }
    )


def test_write_then_read_round_trip():
    parquet_store.write_table("games", _games(["1", "2"]))
    assert set(parquet_store.read_table("games")["game_pk"]) == {"1", "2"}


def test_rewriting_the_same_rows_does_not_duplicate_them():
    parquet_store.write_table("games", _games(["1", "2"]))
    parquet_store.write_table("games", _games(["1", "2"]))
    assert len(parquet_store.read_table("games")) == 2


def test_incoming_row_replaces_the_existing_one_for_the_same_key():
    parquet_store.write_table("games", _games(["1"], date="2025-05-01"))
    parquet_store.write_table("games", _games(["1"], date="2025-05-02"))

    stored = parquet_store.read_table("games")
    assert len(stored) == 1
    assert stored.iloc[0]["game_date"] == "2025-05-02"


def test_partitions_are_isolated_by_season():
    parquet_store.write_table("games", _games(["1"], season=2024, date="2024-05-01"))
    parquet_store.write_table("games", _games(["2"], season=2025))

    assert len(parquet_store.partition_paths("games")) == 2
    assert set(parquet_store.read_table("games", partitions=[2024])["game_pk"]) == {"1"}
    assert len(parquet_store.read_table("games")) == 2


def test_existing_game_pks_is_the_gap_query():
    parquet_store.write_table("games", _games(["1", "2"], season=2025))
    parquet_store.write_table("games", _games(["9"], season=2024, date="2024-05-01"))

    assert parquet_store.existing_game_pks("games", 2025) == {"1", "2"}
    assert parquet_store.existing_game_pks("games") == {"1", "2", "9"}


def test_missing_primary_key_column_raises():
    with pytest.raises(ValueError, match="primary-key"):
        parquet_store.write_table("games", pd.DataFrame({"season_year": [2025]}))


def test_writing_an_empty_frame_is_a_no_op():
    assert parquet_store.write_table("games", pd.DataFrame()) == 0
    assert parquet_store.read_table("games").empty


def test_duplicate_keys_within_one_batch_collapse():
    batch = pd.concat(
        [_games(["1"], date="2025-05-01"), _games(["1"], date="2025-05-02")]
    )
    parquet_store.write_table("games", batch)

    stored = parquet_store.read_table("games")
    assert len(stored) == 1
    assert stored.iloc[0]["game_date"] == "2025-05-02"


def test_every_registered_table_has_a_primary_key():
    for name in TABLES:
        assert get_spec(name).primary_key


def test_unknown_table_raises():
    with pytest.raises(KeyError):
        get_spec("no_such_table")


def test_replace_game_rows_removes_stale_collection_members():
    old = pd.DataFrame(
        {
            "game_pk": ["1", "1"],
            "team_id": ["147", "147"],
            "player_id": ["10", "11"],
            "season_year": [2025, 2025],
            "game_date": ["2025-05-01", "2025-05-01"],
        }
    )
    parquet_store.write_table("lineups", old)
    revised = old.iloc[[0]].copy()
    revised.loc[:, "player_id"] = "12"
    parquet_store.replace_game_rows(
        "lineups", revised, game_pks={"1"}, season_year=2025
    )
    stored = parquet_store.read_table("lineups")
    assert set(stored["player_id"]) == {"12"}
