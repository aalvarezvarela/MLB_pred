from datetime import date

import pandas as pd

from mlb_pred.fetch_data.statcast import client
from mlb_pred.fetch_data.statcast.client import normalize_statcast_frame
from mlb_pred.local_store import parquet_store, statcast_updaters


def _raw():
    return pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": "2025-05-01",
                "at_bat_number": 2,
                "pitch_number": 1,
                "batter": 100,
                "pitcher": 200,
                "release_speed": 97.2,
                "launch_speed": None,
            }
        ]
    )


def test_statcast_normalization_uses_mlb_ids_and_pitch_key():
    frame = normalize_statcast_frame(_raw(), game_pk="1", season_year=2025)
    row = frame.iloc[0]
    assert row["game_pk"] == "1"
    assert row["batter_id"] == "100"
    assert row["pitcher_id"] == "200"
    assert row["season_year"] == 2025


def test_empty_statcast_csv_is_an_explicit_empty_result(monkeypatch):
    monkeypatch.setattr(client, "_request_csv", lambda params: "")
    assert client.fetch_statcast_game("1", 2025).empty


def test_completion_marker_is_written_after_pitch_facts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        type(parquet_store.SETTINGS), "raw_data_path", property(lambda self: tmp_path)
    )
    games = pd.DataFrame(
        [
            {
                "game_pk": "1",
                "season_year": 2025,
                "game_date": date(2025, 5, 1),
                "first_pitch_utc": pd.Timestamp("2025-05-01T23:00:00Z"),
                "is_final": True,
            }
        ]
    )
    parquet_store.write_table("games", games)
    monkeypatch.setattr(
        statcast_updaters,
        "fetch_statcast_game",
        lambda game_pk, season: normalize_statcast_frame(
            _raw(), game_pk=game_pk, season_year=season
        ),
    )
    result = statcast_updaters.update_statcast(2025, show_progress=False)
    assert result["statcast_pitches"] == 1
    status = parquet_store.read_table("statcast_games").iloc[0]
    assert status["fetch_status"] == "complete"
    assert status["pitch_rows"] == 1
