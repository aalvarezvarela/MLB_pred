from __future__ import annotations

import pandas as pd
import pytest

from mlb_pred.features.context_features import build_context_features


def _games() -> pd.DataFrame:
    rows = [
        ("1", "2025-04-01", 19, "1", "147", "121", "night", 1, 1),
        ("2", "2025-04-02", 17, "2", "121", "147", "day", 1, 2),
        ("3", "2025-04-02", 21, "2", "121", "147", "day", 2, 2),
        ("4", "2025-04-03", 19, "1", "147", "121", "night", 1, 1),
    ]
    return pd.DataFrame(
        [
            {
                "game_pk": game_pk,
                "season_year": 2025,
                "game_date": pd.Timestamp(date).date(),
                "first_pitch_utc": pd.Timestamp(f"{date}T{hour:02d}:00:00Z"),
                "game_type": "R",
                "venue_id": venue_id,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "day_night": day_night,
                "series_game_number": series_game_number,
                "games_in_series": games_in_series,
            }
            for (
                game_pk,
                date,
                hour,
                venue_id,
                home_id,
                away_id,
                day_night,
                series_game_number,
                games_in_series,
            ) in rows
        ]
    )


def _team_games() -> pd.DataFrame:
    results = {
        "1": {"147": True, "121": False},
        "2": {"147": True, "121": False},
        "3": {"147": True, "121": False},
        # Deliberately spectacular current result: game 4 must not read it.
        "4": {"147": False, "121": True},
    }
    return pd.DataFrame(
        [
            {"game_pk": game_pk, "team_id": team_id, "win": win}
            for game_pk, teams in results.items()
            for team_id, win in teams.items()
        ]
    )


def _venues() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "venue_id": "1",
                "season": 2025,
                "city": "New York",
                "country": "USA",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "timezone_id": "America/New_York",
            },
            {
                "venue_id": "2",
                "season": 2025,
                "city": "Los Angeles",
                "country": "USA",
                "latitude": 34.0522,
                "longitude": -118.2437,
                "timezone_id": "America/Los_Angeles",
            },
        ]
    )


def _closing() -> pd.DataFrame:
    games = _games().set_index("game_pk")
    return pd.DataFrame(
        {
            "GAME_ID": ["2", "3", "4"],
            "GAME_HOME_TEAM_ID": [
                games.at[key, "home_team_id"] for key in ("2", "3", "4")
            ],
            "GAME_AWAY_TEAM_ID": [
                games.at[key, "away_team_id"] for key in ("2", "3", "4")
            ],
        }
    )


def test_team_identity_one_hots_cover_every_franchise_on_both_sides():
    """The 60-column block matches the NBA project's ``team_one_hot_features``."""
    features = build_context_features(
        _games(), _team_games(), _venues(), _closing()
    ).set_index("GAME_ID")

    identity = [c for c in features.columns if c.startswith("TEAM_IDENTITY_")]
    assert len(identity) == 60
    # Every emitted identity column still declares the pregame contract.
    assert all(column.endswith("_BEFORE") for column in identity)

    # Game 2: Mets (121) home, Yankees (147) away.
    assert features.at["2", "TEAM_IDENTITY_HOME_NEW_YORK_METS_BEFORE"] == 1
    assert features.at["2", "TEAM_IDENTITY_AWAY_NEW_YORK_YANKEES_BEFORE"] == 1
    assert features.at["2", "TEAM_IDENTITY_AWAY_NEW_YORK_METS_BEFORE"] == 0
    assert features.at["2", "TEAM_IDENTITY_HOME_NEW_YORK_YANKEES_BEFORE"] == 0

    # Game 4 flips the sides, so the same franchise moves column.
    assert features.at["4", "TEAM_IDENTITY_HOME_NEW_YORK_YANKEES_BEFORE"] == 1
    assert features.at["4", "TEAM_IDENTITY_AWAY_NEW_YORK_METS_BEFORE"] == 1

    # Exactly one home and one away franchise is hot on every row.
    home = [c for c in identity if c.startswith("TEAM_IDENTITY_HOME_")]
    away = [c for c in identity if c.startswith("TEAM_IDENTITY_AWAY_")]
    assert features[home].sum(axis=1).eq(1).all()
    assert features[away].sum(axis=1).eq(1).all()


def test_team_identity_block_can_be_switched_off():
    features = build_context_features(
        _games(),
        _team_games(),
        _venues(),
        _closing(),
        include_team_identity=False,
    )

    assert not [c for c in features.columns if c.startswith("TEAM_IDENTITY_")]


def test_doubleheader_record_and_schedule_features_are_date_gated():
    features = build_context_features(
        _games(), _team_games(), _venues(), _closing()
    ).set_index("GAME_ID")

    for game_pk in ("2", "3"):
        # Both games see game 1, but neither sees either same-date result.
        assert (
            features.at[game_pk, "TEAM_RECORD_CURRENT_WIN_STREAK_BEFORE_TEAM_AWAY"] == 1
        )
        # WINS_LAST_N was dropped: it correlates at r = 1.000000 with the
        # ratio because GAMES_LAST_N is constant once history exists.
        assert "TEAM_RECORD_WINS_LAST_5_GAMES_BEFORE_TEAM_AWAY" not in features.columns
        assert (
            features.at[game_pk, "TEAM_RECORD_WIN_RATIO_LAST_5_GAMES_BEFORE_TEAM_AWAY"]
            == 1
        )
        assert features.at[game_pk, "SCHEDULE_REST_DAYS_BEFORE_TEAM_AWAY"] == 0
        assert (
            features.at[game_pk, "SCHEDULE_GAMES_IN_LAST_3_DAYS_BEFORE_TEAM_AWAY"] == 1
        )

    # Once the date is complete, both doubleheader wins legally extend A's streak.
    assert features.at["4", "TEAM_RECORD_CURRENT_WIN_STREAK_BEFORE_TEAM_HOME"] == 3
    assert features.at["4", "TEAM_RECORD_WINS_SEASON_BEFORE_TEAM_HOME"] == 3
    assert features.at["4", "TEAM_RECORD_GAMES_SEASON_BEFORE_TEAM_HOME"] == 3


def test_series_travel_is_attached_to_every_game_without_double_counting():
    features = build_context_features(
        _games(), _team_games(), _venues(), _closing()
    ).set_index("GAME_ID")

    trip_game_1 = features.at["2", "TRAVEL_KM_FROM_PREVIOUS_SERIES_BEFORE_TEAM_AWAY"]
    trip_game_2 = features.at["3", "TRAVEL_KM_FROM_PREVIOUS_SERIES_BEFORE_TEAM_AWAY"]
    assert trip_game_1 == pytest.approx(trip_game_2)
    assert trip_game_1 > 3_900
    assert features.at[
        "2", "TRAVEL_LOG1P_KM_LAST_2_DAYS_BEFORE_TEAM_AWAY"
    ] == pytest.approx(features.at["3", "TRAVEL_LOG1P_KM_LAST_2_DAYS_BEFORE_TEAM_AWAY"])
    assert (
        features.at["2", "TRAVEL_JETLAG_HOURS_FROM_PREVIOUS_SERIES_BEFORE_TEAM_AWAY"]
        == 3
    )
    assert features.at["2", "SCHEDULE_DAY_GAME_AFTER_NIGHT_BEFORE_TEAM_AWAY"] == 1
    assert features.at["3", "SCHEDULE_IS_GETAWAY_DAY_BEFORE_TEAM_AWAY"] == 1


def test_current_result_perturbation_does_not_change_current_pregame_features():
    baseline = build_context_features(
        _games(), _team_games(), _venues(), _closing()
    ).set_index("GAME_ID")
    changed_results = _team_games()
    changed_results.loc[changed_results["game_pk"].eq("4"), "win"] = (
        ~changed_results.loc[changed_results["game_pk"].eq("4"), "win"]
    )
    changed = build_context_features(
        _games(), changed_results, _venues(), _closing()
    ).set_index("GAME_ID")

    record_columns = [
        column for column in baseline if column.startswith("TEAM_RECORD_")
    ]
    pd.testing.assert_series_equal(
        baseline.loc["4", record_columns], changed.loc["4", record_columns]
    )
