"""Store-side data-quality checks.

The failure these exist for is a box score fetched while a game was still in
progress: it looks complete, and in an insert-only table it poisons that game
permanently. Per-row bounds catch some of it; the relational invariants
between the three grains catch the rest.
"""

import pandas as pd
import pytest

from mlb_pred.local_store.validate import (
    check_batter_games,
    check_games,
    check_pitcher_appearances,
    check_team_games,
    check_umpires,
)


@pytest.fixture()
def games():
    return pd.DataFrame(
        [
            {
                "game_pk": "1",
                "is_final": True,
                "home_score": 5,
                "away_score": 3,
                "total_runs": 8,
                "innings_played": 9,
                "home_team_id": "147",
                "away_team_id": "120",
            }
        ]
    )


@pytest.fixture()
def team_games():
    return pd.DataFrame(
        [
            {
                "game_pk": "1",
                "team_id": "147",
                "opponent_team_id": "120",
                "home": True,
                "runs_scored": 5,
                "runs_allowed": 3,
                "outs_recorded": 27,
            },
            {
                "game_pk": "1",
                "team_id": "120",
                "opponent_team_id": "147",
                "home": False,
                "runs_scored": 3,
                "runs_allowed": 5,
                "outs_recorded": 24,
            },
        ]
    )


def _names(findings):
    return {f.check for f in findings}


def test_a_clean_game_produces_no_findings(games, team_games):
    assert check_games(games) == []
    assert check_team_games(team_games, games) == []


def test_duplicate_game_pk_is_caught(games):
    doubled = pd.concat([games, games], ignore_index=True)
    assert "duplicate_game_pk" in _names(check_games(doubled))


def test_implausible_run_total_is_caught(games):
    games.loc[0, "home_score"] = 99
    assert "implausible_runs" in _names(check_games(games))


def test_final_game_without_a_score_is_caught(games):
    games.loc[0, "total_runs"] = None
    findings = _names(check_games(games))
    assert "final_without_score" in findings


def test_a_game_with_only_one_team_row_is_caught(games, team_games):
    partial = team_games.iloc[:1]
    assert "wrong_row_count" in _names(check_team_games(partial, games))


def test_mid_game_box_score_is_caught_by_the_outs_band(games, team_games):
    # Nine outs recorded means the fetch happened in the third inning.
    team_games.loc[0, "outs_recorded"] = 9
    assert "implausible_outs" in _names(check_team_games(team_games, games))


def test_a_rain_shortened_game_is_not_flagged(games, team_games):
    """An absolute out floor is wrong for baseball.

    A 24-out floor ("nine innings, home team winning") rejects every
    rain-shortened game and every 2020-21 seven-inning doubleheader -- 523
    legitimate rows, 1% of this repo's store.
    """
    games.loc[0, "innings_played"] = 7
    team_games.loc[0, "outs_recorded"] = 21
    team_games.loc[1, "outs_recorded"] = 18
    assert check_team_games(team_games, games) == []


def test_a_five_inning_official_game_is_not_flagged(games, team_games):
    # The shortest official game: home leads after five, away staff throws four.
    games.loc[0, "innings_played"] = 5
    team_games.loc[0, "outs_recorded"] = 15
    team_games.loc[1, "outs_recorded"] = 12
    assert check_team_games(team_games, games) == []


def test_outs_far_below_the_games_own_length_are_caught(games, team_games):
    """The check that actually finds a mid-game fetch.

    The game went nine innings but this team's box score records only three,
    which no completed game can produce.
    """
    games.loc[0, "innings_played"] = 9
    team_games.loc[0, "outs_recorded"] = 9
    findings = _names(check_team_games(team_games, games))
    assert "outs_inconsistent_with_innings" in findings


def test_outs_above_the_games_own_length_are_caught(games, team_games):
    games.loc[0, "innings_played"] = 9
    team_games.loc[0, "outs_recorded"] = 33
    assert "outs_inconsistent_with_innings" in _names(
        check_team_games(team_games, games)
    )


def test_extra_innings_are_not_flagged(games, team_games):
    games.loc[0, "innings_played"] = 13
    team_games.loc[0, "outs_recorded"] = 39
    team_games.loc[1, "outs_recorded"] = 38
    assert check_team_games(team_games, games) == []


def test_non_reciprocal_runs_are_caught(games, team_games):
    # One side updated, the other not: the signature of a partial ingest.
    team_games.loc[0, "runs_scored"] = 7
    assert "runs_not_reciprocal" in _names(check_team_games(team_games, games))


def test_team_game_runs_disagreeing_with_the_games_table_are_caught(games, team_games):
    games.loc[0, "home_score"] = 6
    assert "runs_disagree_with_games" in _names(check_team_games(team_games, games))


def test_orphan_team_game_is_caught(games, team_games):
    team_games.loc[:, "game_pk"] = "999"
    assert "orphan_game_pk" in _names(check_team_games(team_games, games))


def test_pitcher_outs_must_sum_to_the_team_game_total(team_games):
    pitchers = pd.DataFrame(
        [
            {"game_pk": "1", "team_id": "147", "is_starter": True, "outs_recorded": 21},
            {"game_pk": "1", "team_id": "147", "is_starter": False, "outs_recorded": 3},
            {"game_pk": "1", "team_id": "120", "is_starter": True, "outs_recorded": 24},
        ]
    )
    # Home team's pitchers recorded 24 outs but the team row claims 27.
    findings = _names(check_pitcher_appearances(pitchers, team_games))
    assert "outs_disagree_with_team_game" in findings


def test_exactly_one_starting_pitcher_per_team_per_game(team_games):
    pitchers = pd.DataFrame(
        [
            {"game_pk": "1", "team_id": "147", "is_starter": True, "outs_recorded": 27},
            {"game_pk": "1", "team_id": "147", "is_starter": True, "outs_recorded": 0},
            {"game_pk": "1", "team_id": "120", "is_starter": True, "outs_recorded": 24},
        ]
    )
    assert "wrong_starter_count" in _names(
        check_pitcher_appearances(pitchers, team_games)
    )


def test_batter_line_internal_consistency():
    batters = pd.DataFrame(
        [
            {
                "game_pk": "1",
                "team_id": "147",
                "is_starter": True,
                "at_bats": 3,
                "hits": 4,
                "doubles": 0,
                "triples": 0,
                "home_runs": 0,
            }
        ]
    )
    assert "hits_exceed_at_bats" in _names(check_batter_games(batters))


def test_extra_base_hits_cannot_exceed_hits():
    batters = pd.DataFrame(
        [
            {
                "game_pk": "1",
                "team_id": "147",
                "is_starter": True,
                "at_bats": 4,
                "hits": 1,
                "doubles": 1,
                "triples": 0,
                "home_runs": 1,
            }
        ]
    )
    assert "extra_base_hits_exceed_hits" in _names(check_batter_games(batters))


def test_missing_home_plate_umpire_is_caught(games):
    crew = pd.DataFrame(
        [{"game_pk": "1", "official_id": "901", "is_home_plate": False}]
    )
    assert "missing_home_plate_umpire" in _names(check_umpires(crew, games))


def test_present_home_plate_umpire_passes(games):
    crew = pd.DataFrame([{"game_pk": "1", "official_id": "900", "is_home_plate": True}])
    assert check_umpires(crew, games) == []
