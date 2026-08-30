from datetime import UTC, date, datetime

import pandas as pd

from mlb_pred.snapshots import archive


def test_empty_snapshot_is_persisted_as_negative_information(tmp_path, monkeypatch):
    monkeypatch.setattr(
        type(archive.SETTINGS), "snapshot_path", property(lambda self: tmp_path)
    )
    captured = datetime(2025, 8, 1, 14, tzinfo=UTC)
    path = archive._write_snapshot("lineups", date(2025, 8, 1), pd.DataFrame(), captured)
    assert path.exists()

    latest = archive.latest_snapshot_before(
        "lineups", date(2025, 8, 1), datetime(2025, 8, 1, 15, tzinfo=UTC)
    )
    assert latest.empty
    assert latest.attrs["snapshot_was_empty"] is True
    assert latest.attrs["captured_at_utc"] == pd.Timestamp(captured)


def test_league_today_uses_eastern_not_host_timezone():
    # Already the next day in Madrid/UTC, still the prior MLB slate in New York.
    now = datetime(2025, 8, 2, 2, 30, tzinfo=UTC)
    assert archive.league_today(now) == date(2025, 8, 1)
