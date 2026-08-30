"""Box-score parsing: the three accumulation grains."""

import pandas as pd
import pytest

from mlb_pred.fetch_data.statsapi.boxscore import (
    _batting_order_slot,
    innings_pitched_to_outs,
    parse_boxscore,
)

GAME_META = {
    "game_pk": "1",
    "season_year": 2025,
    "game_date": pd.Timestamp("2025-05-01").date(),
    "first_pitch_utc": pd.Timestamp("2025-05-01T23:05:00Z"),
    "game_type": "R",
    "venue_id": "3313",
}


def _side(team_id, runs, opponent_runs, *, players):
    return {
        "team": {"id": team_id, "name": "X"},
        "teamStats": {
            "batting": {
                "runs": runs,
                "plateAppearances": 38,
                "atBats": 34,
                "hits": 9,
                "doubles": 2,
                "triples": 0,
                "homeRuns": 1,
                "totalBases": 14,
                "baseOnBalls": 3,
                "strikeOuts": 8,
                "hitByPitch": 1,
                "sacFlies": 0,
                "leftOnBase": 7,
                "groundOuts": 8,
                "airOuts": 9,
            },
            "pitching": {
                "runs": opponent_runs,
                "outs": 27,
                "inningsPitched": "9.0",
                "battersFaced": 37,
                "pitchesThrown": 148,
                "strikes": 95,
                "hits": 8,
                "baseOnBalls": 2,
                "hitBatsmen": 0,
                "strikeOuts": 9,
                "homeRuns": 1,
                "earnedRuns": opponent_runs,
            },
            "fielding": {"errors": 0},
        },
        "players": players,
        "batters": [1, 2],
        "pitchers": [10, 11],
        "bullpen": [11, 12, 13],
        "bench": [3],
    }


def _boxscore():
    home_players = {
        "ID1": {
            "person": {"id": 1, "fullName": "Slot One"},
            "position": {"code": "8", "abbreviation": "CF"},
            "battingOrder": "100",
            "stats": {
                "batting": {
                    "plateAppearances": 5,
                    "atBats": 4,
                    "hits": 2,
                    "doubles": 1,
                    "triples": 0,
                    "homeRuns": 1,
                    "runs": 2,
                    "baseOnBalls": 1,
                    "strikeOuts": 1,
                },
                "pitching": {},
            },
        },
        "ID2": {
            "person": {"id": 2, "fullName": "Pinch Hitter"},
            "position": {"code": "10", "abbreviation": "DH"},
            "battingOrder": "101",
            "stats": {
                "batting": {"plateAppearances": 1, "atBats": 1, "hits": 0},
                "pitching": {},
            },
        },
        "ID10": {
            "person": {"id": 10, "fullName": "The Starter"},
            "position": {"code": "1", "abbreviation": "P"},
            "stats": {
                "batting": {},
                "pitching": {
                    "gamesStarted": 1,
                    "outs": 19,
                    "inningsPitched": "6.1",
                    "battersFaced": 25,
                    "pitchesThrown": 92,
                    "strikes": 60,
                    "strikeOuts": 7,
                    "baseOnBalls": 2,
                    "hits": 5,
                    "homeRuns": 1,
                    "runs": 3,
                    "earnedRuns": 3,
                },
            },
        },
        "ID11": {
            "person": {"id": 11, "fullName": "The Reliever"},
            "position": {"code": "1", "abbreviation": "P"},
            "stats": {
                "batting": {},
                "pitching": {
                    "gamesStarted": 0,
                    "outs": 8,
                    "inningsPitched": "2.2",
                    "battersFaced": 9,
                    "pitchesThrown": 34,
                    "strikes": 22,
                    "strikeOuts": 2,
                    "baseOnBalls": 0,
                    "hits": 3,
                    "runs": 1,
                    "earnedRuns": 1,
                },
            },
        },
    }
    return {
        "teams": {
            "home": _side(147, 5, 3, players=home_players),
            "away": _side(120, 3, 5, players={}),
        },
        "officials": [
            {
                "official": {"id": 900, "fullName": "Ump One"},
                "officialType": "Home Plate",
            },
            {
                "official": {"id": 901, "fullName": "Ump Two"},
                "officialType": "First Base",
            },
        ],
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("7.0", 21),
        ("7.1", 22),
        ("7.2", 23),
        ("0.1", 1),
        ("0.0", 0),
        (None, None),
        ("", None),
    ],
)
def test_innings_pitched_notation_is_thirds_not_tenths(value, expected):
    # 7.1 is seven innings and one out. Averaging the raw float treats a third
    # of an inning as a tenth, which is a real and easy mistake.
    assert innings_pitched_to_outs(value) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("100", (1, 0)),
        ("101", (1, 1)),
        ("900", (9, 0)),
        (None, (None, None)),
        ("", (None, None)),
    ],
)
def test_batting_order_splits_into_slot_and_substitution_index(raw, expected):
    assert _batting_order_slot(raw) == expected


def test_team_game_grain_is_two_rows_with_home_as_a_boolean():
    parsed = parse_boxscore(_boxscore(), GAME_META)
    team_games = pd.DataFrame(parsed["team_games"])

    assert len(team_games) == 2
    # Home/away is a column, never a column suffix.
    assert set(team_games["home"]) == {True, False}
    assert "home" in team_games.columns
    assert not any(
        c.endswith("_HOME") or c.endswith("_AWAY") for c in team_games.columns
    )


def test_team_game_outcome_fields_are_consistent_across_the_two_rows():
    team_games = pd.DataFrame(parse_boxscore(_boxscore(), GAME_META)["team_games"])
    home = team_games[team_games["home"]].iloc[0]
    away = team_games[~team_games["home"]].iloc[0]

    assert home["runs_scored"] == away["runs_allowed"] == 5
    assert away["runs_scored"] == home["runs_allowed"] == 3
    assert home["total_runs"] == away["total_runs"] == 8
    assert home["run_margin"] == -away["run_margin"] == 2
    assert bool(home["win"]) and not bool(away["win"])
    assert home["opponent_team_id"] == away["team_id"]


def test_team_game_stores_tempo_and_efficiency_separately():
    # A run total is expected_events x expected_value_per_event; both factors
    # have to survive storage for either to be recombined later.
    row = pd.DataFrame(parse_boxscore(_boxscore(), GAME_META)["team_games"]).iloc[0]
    assert row["plate_appearances"] == 38  # volume
    assert row["batters_faced"] == 37  # volume
    assert row["k_pct"] == pytest.approx(8 / 38, abs=5e-7)  # efficiency
    assert row["bb_pct"] == pytest.approx(3 / 38, abs=5e-7)
    assert row["obp"] == pytest.approx((9 + 3 + 1) / (34 + 3 + 1 + 0), abs=5e-7)
    assert row["slg"] == pytest.approx(14 / 34, abs=5e-7)


def test_pitcher_appearance_grain_separates_starters_from_relievers():
    pitchers = pd.DataFrame(
        parse_boxscore(_boxscore(), GAME_META)["pitcher_appearances"]
    )
    assert len(pitchers) == 2

    starter = pitchers[pitchers["is_starter"]].iloc[0]
    reliever = pitchers[~pitchers["is_starter"]].iloc[0]

    assert starter["player_name"] == "The Starter"
    assert starter["outs_recorded"] == 19
    assert starter["innings_pitched"] == pytest.approx(19 / 3, abs=5e-7)
    assert starter["appearance_number"] == 1
    assert reliever["appearance_number"] == 2
    assert reliever["outs_recorded"] == 8


def test_batter_grain_marks_starters_by_substitution_index():
    batters = pd.DataFrame(parse_boxscore(_boxscore(), GAME_META)["batter_games"])
    assert len(batters) == 2

    starter = batters[batters["is_starter"]].iloc[0]
    substitute = batters[~batters["is_starter"]].iloc[0]

    assert starter["lineup_slot"] == 1 and starter["substitution_index"] == 0
    assert substitute["lineup_slot"] == 1 and substitute["substitution_index"] == 1
    assert starter["singles"] == 2 - 1 - 0 - 1  # hits minus 2B/3B/HR


def test_pitchers_are_excluded_from_the_batter_grain():
    batters = pd.DataFrame(parse_boxscore(_boxscore(), GAME_META)["batter_games"])
    assert "The Starter" not in set(batters["player_name"])


def test_home_plate_umpire_is_flagged():
    umpires = pd.DataFrame(parse_boxscore(_boxscore(), GAME_META)["umpires"])
    assert umpires["is_home_plate"].sum() == 1
    assert umpires[umpires["is_home_plate"]].iloc[0]["official_id"] == "900"


def test_all_ids_are_stored_as_text():
    parsed = parse_boxscore(_boxscore(), GAME_META)
    for table in ("team_games", "batter_games", "pitcher_appearances"):
        for row in parsed[table]:
            for key in ("game_pk", "team_id", "opponent_team_id"):
                if row.get(key) is not None:
                    assert isinstance(row[key], str), f"{table}.{key}"
