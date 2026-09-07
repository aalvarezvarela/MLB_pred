"""Tests for the feature families built from previously unread sources.

Each of these modules reads a table that was ingested from the start of the
project and never used by the feature layer: the odds tick path, venue
geometry, home-plate umpires, Statcast pitches and pitcher appearances.

The property that matters for all of them is the same one the rolling features
already enforce: a game must never see its own result, and games sharing a
calendar date must not see each other's.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlb_pred.features.bullpen_features import build_bullpen_features
from mlb_pred.features.environment_features import (
    build_park_features,
    build_umpire_features,
)
from mlb_pred.features.market_movement import (
    LATE_WINDOW_MINUTES,
    build_market_movement_features,
)
from mlb_pred.features.statcast_features import build_statcast_features


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("1", "2025-04-01", 2025, "V1", True, 5, 3, "A", "B"),
            ("2", "2025-04-02", 2025, "V1", True, 9, 8, "A", "B"),
            ("3", "2025-04-02", 2025, "V1", True, 1, 0, "A", "B"),
            ("4", "2025-04-03", 2025, "V1", True, 4, 4, "A", "B"),
        ],
        columns=[
            "game_pk",
            "game_date",
            "season_year",
            "venue_id",
            "is_final",
            "home_score",
            "away_score",
            "home_team_id",
            "away_team_id",
        ],
    )


def _venues() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "venue_id": "V1",
                "season": 2025,
                "elevation_ft": 5200,
                "roof_type": "Retractable",
                "turf_type": "Artificial",
                "capacity": 50000,
                "left_line": 347,
                "left": 360,
                "left_center": 390,
                "center": 415,
                "right_center": 375,
                "right": 350,
                "right_line": 350,
            }
        ]
    )


def _tick(game_pk, book, market, mins, line, pregame=True):
    return {
        "game_pk": game_pk,
        "market": market,
        "book_slug": book,
        "mins_to_tip": mins,
        "is_pregame": pregame,
        "left_line": line,
    }


# --------------------------------------------------------------------- market


def test_open_to_close_is_measured_per_book_then_pooled():
    """Two books that both moved +0.5 give a +0.5 consensus move, not +1.0.

    Pooling ticks across books first would make the statistic depend on how
    many books were quoting, which grows from four to seven across the store.
    """
    ticks = pd.DataFrame(
        [
            _tick("1", "bet365", "totals", -600, 17),
            _tick("1", "bet365", "totals", -60, 18),
            _tick("1", "fanduel", "totals", -600, 17),
            _tick("1", "fanduel", "totals", -60, 18),
        ]
    )

    features = build_market_movement_features(ticks).set_index("GAME_ID")

    assert features.at["1", "ODDS_MOVEMENT_TOTAL_OPEN_BEFORE"] == pytest.approx(8.5)
    assert features.at[
        "1", "ODDS_MOVEMENT_TOTAL_OPEN_TO_CLOSE_BEFORE"
    ] == pytest.approx(0.5)
    assert features.at["1", "ODDS_MOVEMENT_TOTAL_BOOKS_WITH_HISTORY_BEFORE"] == 2


def test_in_play_and_inside_the_safety_margin_ticks_never_reach_a_feature():
    """SBR stores in-play ticks in the same shape as pre-game ones."""
    ticks = pd.DataFrame(
        [
            _tick("1", "bet365", "totals", -600, 17),
            _tick("1", "bet365", "totals", -60, 18),
            _tick("1", "bet365", "totals", -1, 30),  # inside the 5-minute guard
            _tick("1", "bet365", "totals", 40, 40, pregame=False),  # in play
        ]
    )

    features = build_market_movement_features(ticks).set_index("GAME_ID")

    # The close is the -60 tick, so the move is 8.5 -> 9.0 and nothing wilder.
    assert features.at[
        "1", "ODDS_MOVEMENT_TOTAL_OPEN_TO_CLOSE_BEFORE"
    ] == pytest.approx(0.5)


def test_run_line_movement_is_expressed_as_a_home_handicap():
    """``left`` is AWAY for the run line, so the stored side must be negated."""
    ticks = pd.DataFrame(
        [
            _tick("1", "bet365", "run_line", -600, 3),  # away +1.5
            _tick("1", "bet365", "run_line", -60, -3),  # away -1.5
        ]
    )

    features = build_market_movement_features(ticks).set_index("GAME_ID")

    assert features.at["1", "ODDS_MOVEMENT_RUN_LINE_OPEN_BEFORE"] == pytest.approx(-1.5)
    assert features.at[
        "1", "ODDS_MOVEMENT_RUN_LINE_OPEN_TO_CLOSE_BEFORE"
    ] == pytest.approx(3.0)


def test_reversal_separates_a_round_trip_from_a_straight_drift():
    ticks = pd.DataFrame(
        [
            _tick("1", "bet365", "totals", -600, 17),
            _tick("1", "bet365", "totals", -300, 19),
            _tick("1", "bet365", "totals", -60, 18),
        ]
    )

    features = build_market_movement_features(ticks).set_index("GAME_ID")

    # Travelled 8.5 -> 9.5 -> 9.0: net +0.5 of a 1.0 range, so 0.5 given back.
    assert features.at["1", "ODDS_MOVEMENT_TOTAL_REVERSAL_BEFORE"] == pytest.approx(0.5)


def test_late_movement_measures_only_the_final_window():
    ticks = pd.DataFrame(
        [
            _tick("1", "bet365", "totals", -600, 17),
            _tick("1", "bet365", "totals", -(LATE_WINDOW_MINUTES - 10), 18),
            _tick("1", "bet365", "totals", -30, 19),
        ]
    )

    features = build_market_movement_features(ticks).set_index("GAME_ID")

    late = f"ODDS_MOVEMENT_TOTAL_LATE_{LATE_WINDOW_MINUTES}M_BEFORE"
    assert features.at["1", late] == pytest.approx(0.5)
    assert features.at[
        "1", "ODDS_MOVEMENT_TOTAL_OPEN_TO_CLOSE_BEFORE"
    ] == pytest.approx(1.0)


# ----------------------------------------------------------------- park / ump


def test_park_run_environment_excludes_the_current_game_and_its_date():
    """Games 2 and 3 share a date, so neither may see the other's 17 or 1."""
    features = build_park_features(_games(), _venues()).set_index("GAME_ID")

    assert pd.isna(features.at["1", "PARK_RUN_MEAN_EXPANDING_BEFORE"])
    # Both same-date games see only game 1's total of 8.
    assert features.at["2", "PARK_RUN_MEAN_EXPANDING_BEFORE"] == pytest.approx(8.0)
    assert features.at["3", "PARK_RUN_MEAN_EXPANDING_BEFORE"] == pytest.approx(8.0)
    assert features.at["2", "PARK_PRIOR_GAMES_BEFORE"] == 1
    assert features.at["3", "PARK_PRIOR_GAMES_BEFORE"] == 1
    # Game 4 sees all three earlier totals: (8 + 17 + 1) / 3.
    assert features.at["4", "PARK_RUN_MEAN_EXPANDING_BEFORE"] == pytest.approx(
        26.0 / 3.0
    )


def test_static_park_geometry_is_attached_from_venues():
    features = build_park_features(_games(), _venues()).set_index("GAME_ID")

    assert features.at["1", "PARK_ELEVATION_FT_BEFORE"] == 5200
    assert features.at["1", "PARK_FENCE_CENTER_BEFORE"] == 415
    assert features.at["1", "PARK_ROOF_CLOSEABLE_BEFORE"] == 1
    assert features.at["1", "PARK_TURF_ARTIFICIAL_BEFORE"] == 1


def test_umpire_history_is_per_official_and_strictly_prior():
    umpires = pd.DataFrame(
        [
            {"game_pk": "1", "official_id": "U1", "is_home_plate": True},
            {"game_pk": "2", "official_id": "U1", "is_home_plate": True},
            {"game_pk": "3", "official_id": "U2", "is_home_plate": True},
            {"game_pk": "4", "official_id": "U1", "is_home_plate": True},
            {"game_pk": "1", "official_id": "U9", "is_home_plate": False},
        ]
    )

    features = build_umpire_features(_games(), umpires).set_index("GAME_ID")

    assert features.at["1", "UMPIRE_HAS_HISTORY_BEFORE"] == 0
    assert features.at["2", "UMPIRE_RUN_MEAN_EXPANDING_BEFORE"] == pytest.approx(8.0)
    # U2's first game: no history of its own even though U1 has two by then.
    assert features.at["3", "UMPIRE_PRIOR_GAMES_BEFORE"] == 0
    # U1 again, now with games 1 and 2 behind it: (8 + 17) / 2.
    assert features.at["4", "UMPIRE_RUN_MEAN_EXPANDING_BEFORE"] == pytest.approx(12.5)


def test_a_second_home_plate_umpire_for_one_game_raises():
    umpires = pd.DataFrame(
        [
            {"game_pk": "1", "official_id": "U1", "is_home_plate": True},
            {"game_pk": "1", "official_id": "U2", "is_home_plate": True},
        ]
    )

    with pytest.raises(ValueError, match="more than one home-plate official"):
        build_umpire_features(_games(), umpires)


# -------------------------------------------------------------------- statcast


def _pitch(game_pk, topbot, xwoba, woba, speed=100.0, angle=28.0):
    return {
        "game_pk": game_pk,
        "inning_topbot": topbot,
        "estimated_woba_using_speedangle": xwoba,
        "woba_value": woba,
        "woba_denom": 1,
        "launch_speed": speed,
        "launch_angle": angle,
    }


def test_statcast_offense_and_allowed_are_opposite_sides_of_one_game():
    pitches = pd.DataFrame(
        [
            _pitch("1", "Bot", 0.400, 1.0),  # home batting
            _pitch("1", "Top", 0.200, 0.0),  # away batting
            _pitch("2", "Bot", 0.400, 1.0),
            _pitch("2", "Top", 0.200, 0.0),
            _pitch("3", "Bot", 0.400, 1.0),
            _pitch("3", "Top", 0.200, 0.0),
        ]
    )

    features = build_statcast_features(pitches, _games()).set_index("GAME_ID")

    # By game 3 the home team has two prior games of its own offence, and what
    # it allowed is exactly what the away team produced.
    assert features.at[
        "3", "STATCAST_OFFENSE_XWOBA_SEASON_BEFORE_TEAM_HOME"
    ] == pytest.approx(0.400)
    assert features.at[
        "3", "STATCAST_ALLOWED_XWOBA_SEASON_BEFORE_TEAM_HOME"
    ] == pytest.approx(0.200)
    assert features.at[
        "3", "STATCAST_OFFENSE_XWOBA_SEASON_BEFORE_TEAM_AWAY"
    ] == pytest.approx(0.200)


def test_statcast_windows_exclude_the_current_game():
    pitches = pd.DataFrame(
        [
            _pitch("1", "Bot", 0.300, 1.0),
            _pitch("1", "Top", 0.300, 1.0),
            _pitch("2", "Bot", 0.900, 1.0),  # a huge game that must not leak
            _pitch("2", "Top", 0.300, 1.0),
        ]
    )

    features = build_statcast_features(pitches, _games()).set_index("GAME_ID")

    assert pd.isna(features.at["1", "STATCAST_OFFENSE_XWOBA_SEASON_BEFORE_TEAM_HOME"])
    assert features.at[
        "2", "STATCAST_OFFENSE_XWOBA_SEASON_BEFORE_TEAM_HOME"
    ] == pytest.approx(0.300)


def test_woba_minus_xwoba_flags_results_running_ahead_of_contact():
    pitches = pd.DataFrame(
        [
            _pitch("1", "Bot", 0.200, 1.0),  # out-hit its contact quality
            _pitch("1", "Top", 0.200, 0.2),
            _pitch("2", "Bot", 0.200, 1.0),
            _pitch("2", "Top", 0.200, 0.2),
        ]
    )

    features = build_statcast_features(pitches, _games()).set_index("GAME_ID")

    assert (
        features.at["2", "STATCAST_OFFENSE_WOBA_MINUS_XWOBA_SEASON_BEFORE_TEAM_HOME"]
        > 0
    )


# --------------------------------------------------------------------- bullpen


def _appearance(game_pk, date, team, player, *, starter, outs, pitches, er=0, bf=3):
    return {
        "game_pk": game_pk,
        "team_id": team,
        "game_date": date,
        "player_id": player,
        "is_starter": starter,
        "outs_recorded": outs,
        "pitches_thrown": pitches,
        "batters_faced": bf,
        "earned_runs": er,
        "strikeouts": 1,
        "walks_allowed": 1,
        "home_runs_allowed": 0,
    }


def test_bullpen_workload_is_the_previous_games_not_this_one():
    appearances = pd.DataFrame(
        [
            _appearance(
                "1", "2025-04-01", "A", "SP1", starter=True, outs=18, pitches=90
            ),
            _appearance(
                "1", "2025-04-01", "A", "RP1", starter=False, outs=9, pitches=40
            ),
            _appearance(
                "2", "2025-04-02", "A", "SP2", starter=True, outs=21, pitches=95
            ),
            _appearance(
                "2", "2025-04-02", "A", "RP2", starter=False, outs=6, pitches=25
            ),
            _appearance(
                "4", "2025-04-03", "A", "SP3", starter=True, outs=15, pitches=80
            ),
            _appearance(
                "4", "2025-04-03", "A", "RP3", starter=False, outs=12, pitches=55
            ),
        ]
    )

    features = build_bullpen_features(appearances, _games()).set_index("GAME_ID")

    assert pd.isna(features.at["1", "BULLPEN_PITCHES_LAST_1_GAMES_BEFORE_TEAM_HOME"])
    assert features.at["2", "BULLPEN_PITCHES_LAST_1_GAMES_BEFORE_TEAM_HOME"] == 40
    assert features.at["4", "BULLPEN_PITCHES_LAST_1_GAMES_BEFORE_TEAM_HOME"] == 25
    assert features.at["4", "BULLPEN_PITCHES_LAST_3_GAMES_BEFORE_TEAM_HOME"] == 65
    # How deep the previous starter went, in outs -- never innings notation.
    assert features.at["4", "BULLPEN_STARTER_OUTS_LAST_GAME_BEFORE_TEAM_HOME"] == 21


def test_bullpen_quality_is_a_ratio_of_sums_not_a_mean_of_ratios():
    """One batter faced must not weigh the same as a three-inning outing."""
    appearances = pd.DataFrame(
        [
            _appearance(
                "1",
                "2025-04-01",
                "A",
                "RP1",
                starter=False,
                outs=3,
                pitches=15,
                er=0,
                bf=30,
            ),
            _appearance(
                "2",
                "2025-04-02",
                "A",
                "RP2",
                starter=False,
                outs=3,
                pitches=15,
                er=1,
                bf=1,
            ),
            _appearance(
                "3",
                "2025-04-02",
                "A",
                "RP3",
                starter=False,
                outs=3,
                pitches=15,
                er=1,
                bf=1,
            ),
            _appearance(
                "4",
                "2025-04-03",
                "A",
                "RP4",
                starter=False,
                outs=3,
                pitches=15,
                er=0,
                bf=1,
            ),
        ]
    )

    features = build_bullpen_features(appearances, _games()).set_index("GAME_ID")

    # Three prior games: 1 strikeout each over 30 + 1 + 1 batters faced, so the
    # rate is 3/32, not the mean of 1/30, 1/1 and 1/1.
    assert features.at["4", "BULLPEN_K_PCT_LAST_10_GAMES_BEFORE_TEAM_HOME"] == (
        pytest.approx(3.0 / 32.0)
    )
