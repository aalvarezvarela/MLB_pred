from __future__ import annotations

import pandas as pd
import pytest

from mlb_pred.features.availability_features import (
    HITTER_NEUTRALS,
    build_availability_features,
    build_hitter_history,
    classify_il_transaction,
)


def _game(
    game_pk: str,
    date: str,
    season: int,
    *,
    home_score: int = 3,
    away_score: int = 2,
) -> dict[str, object]:
    return {
        "game_pk": game_pk,
        "season_year": season,
        "game_date": pd.Timestamp(date).date(),
        "first_pitch_utc": pd.Timestamp(f"{date}T19:00:00Z"),
        "game_type": "R",
        "home_team_id": "A",
        "away_team_id": "B",
        "home_score": home_score,
        "away_score": away_score,
        "total_runs": home_score + away_score,
        "home_probable_pitcher_id": "SP-A",
        "away_probable_pitcher_id": "SP-B",
    }


def _batter(
    game_pk: str,
    date: str,
    season: int,
    team_id: str,
    player_id: str,
    *,
    hits: int = 1,
    home_runs: int = 0,
) -> dict[str, object]:
    return {
        "game_pk": game_pk,
        "team_id": team_id,
        "player_id": player_id,
        "season_year": season,
        "game_date": pd.Timestamp(date).date(),
        "plate_appearances": 4,
        "at_bats": 4,
        "hits": hits,
        "home_runs": home_runs,
        "walks": 0,
        "strikeouts": 1,
        "hit_by_pitch": 0,
        "sac_flies": 0,
        "total_bases": hits + 3 * home_runs,
    }


def _pitcher(
    game_pk: str,
    date: str,
    season: int,
    team_id: str,
    player_id: str,
    *,
    starter: bool,
) -> dict[str, object]:
    return {
        "game_pk": game_pk,
        "team_id": team_id,
        "player_id": player_id,
        "season_year": season,
        "game_date": pd.Timestamp(date).date(),
        "is_starter": starter,
        "outs_recorded": 18 if starter else 3,
        "batters_faced": 24 if starter else 4,
        "earned_runs": 2,
        "hits_allowed": 5,
        "walks_allowed": 1,
        "home_runs_allowed": 1,
        "strikeouts": 6,
        "pitches_thrown": 90 if starter else 15,
    }


def _lineup(
    game_pk: str,
    date: str,
    team_id: str,
    player_id: str,
    *,
    position_code: str = "8",
) -> dict[str, object]:
    return {
        "game_pk": game_pk,
        "season_year": pd.Timestamp(date).year,
        "game_date": pd.Timestamp(date).date(),
        "team_id": team_id,
        "home": team_id == "A",
        "player_id": player_id,
        "player_name": f"Player {player_id}",
        "lineup_slot": 1,
        "position_code": position_code,
    }


def test_hitter_ewma_excludes_target_and_both_doubleheader_games():
    games = pd.DataFrame(
        [
            _game("1", "2025-04-01", 2025),
            _game("2", "2025-04-02", 2025),
            _game("3", "2025-04-02", 2025),
        ]
    )
    batters = pd.DataFrame(
        [
            _batter("1", "2025-04-01", 2025, "A", "P", hits=1),
            _batter("2", "2025-04-02", 2025, "A", "P", hits=4),
            _batter("3", "2025-04-02", 2025, "A", "P", hits=0),
        ]
    )

    history = build_hitter_history(batters, games).set_index("game_pk")

    assert history.loc["2", "PLAYER_HISTORY_HITTER_OBP_BEFORE"] == 0.25
    assert history.loc["3", "PLAYER_HISTORY_HITTER_OBP_BEFORE"] == 0.25


def test_hitter_history_falls_back_to_previous_regular_season():
    games = pd.DataFrame(
        [
            _game("1", "2024-09-01", 2024),
            _game("2", "2025-04-01", 2025),
        ]
    )
    batters = pd.DataFrame(
        [
            _batter("1", "2024-09-01", 2024, "A", "P", hits=2),
            _batter("2", "2025-04-01", 2025, "A", "P", hits=0),
        ]
    )

    history = build_hitter_history(batters, games).set_index("game_pk")

    assert history.loc["2", "PLAYER_HISTORY_HITTER_OBP_BEFORE"] == 0.5
    assert history.loc["2", "PLAYER_HISTORY_HITTER_HISTORY_COVERAGE_BEFORE"] == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("placed OF P on the 10-day injured list", "open"),
        ("transferred OF P to the 60-day injured list", "maintain"),
        ("activated OF P from the 10-day injured list", "close"),
        ("placed OF P on the paternity list", None),
    ],
)
def test_il_classifier_interprets_transaction_direction(text, expected):
    assert (
        classify_il_transaction({"type_desc": "Status Change", "description": text})
        == expected
    )


def test_availability_distinguishes_rest_from_injury_and_ignores_relief_absence():
    games = pd.DataFrame(
        [
            _game("1", "2025-04-01", 2025),
            _game("2", "2025-04-02", 2025),
            _game("3", "2025-04-03", 2025),
        ]
    )
    lineups = pd.DataFrame(
        [
            _lineup("1", "2025-04-01", "A", "STAR"),
            _lineup("1", "2025-04-01", "B", "B1"),
            _lineup("2", "2025-04-02", "A", "REPLACEMENT"),
            _lineup("2", "2025-04-02", "B", "B1"),
            _lineup("3", "2025-04-03", "A", "REPLACEMENT"),
            _lineup("3", "2025-04-03", "B", "B1"),
        ]
    )
    batters = pd.DataFrame(
        [
            _batter("1", "2025-04-01", 2025, "A", "STAR", hits=2),
            _batter("1", "2025-04-01", 2025, "B", "B1"),
            _batter("2", "2025-04-02", 2025, "A", "REPLACEMENT"),
            _batter("2", "2025-04-02", 2025, "B", "B1"),
            _batter("3", "2025-04-03", 2025, "A", "REPLACEMENT"),
            _batter("3", "2025-04-03", 2025, "B", "B1"),
        ]
    )
    pitchers = pd.DataFrame(
        [
            _pitcher("1", "2025-04-01", 2025, "A", "SP-A", starter=True),
            _pitcher("1", "2025-04-01", 2025, "A", "RELIEF", starter=False),
            _pitcher("1", "2025-04-01", 2025, "B", "SP-B", starter=True),
            _pitcher("2", "2025-04-02", 2025, "A", "SP-A", starter=True),
            _pitcher("2", "2025-04-02", 2025, "B", "SP-B", starter=True),
            _pitcher("3", "2025-04-03", 2025, "A", "SP-A", starter=True),
            _pitcher("3", "2025-04-03", 2025, "B", "SP-B", starter=True),
        ]
    )
    transactions = pd.DataFrame(
        [
            {
                "transaction_id": "10",
                "player_id": "STAR",
                "known_date": pd.Timestamp("2025-04-03").date(),
                "from_team_id": "A",
                "to_team_id": "A",
                "type_desc": "Status Change",
                "description": "placed OF STAR on the 10-day injured list",
            }
        ]
    )
    closing = pd.DataFrame(
        {
            "GAME_ID": ["1", "2", "3"],
            "ODDS_TOTAL_CONSENSUS_LINE_NORMALIZED_MEDIAN": [8.0, 8.0, 8.0],
        }
    )

    features = build_availability_features(
        games,
        lineups,
        batters,
        pitchers,
        transactions,
        closing,
        target_game_ids=["2", "3"],
    ).set_index("GAME_ID")

    assert (
        features.loc["2", "TEAM_AVAILABILITY_HITTER_N_OBSERVED_ABSENT_BEFORE_TEAM_HOME"]
        == 1
    )
    assert features.loc["2", "TEAM_AVAILABILITY_HITTER_N_INJURED_BEFORE_TEAM_HOME"] == 0
    assert (
        features.loc["3", "TEAM_AVAILABILITY_HITTER_N_OBSERVED_ABSENT_BEFORE_TEAM_HOME"]
        == 0
    )
    assert features.loc["3", "TEAM_AVAILABILITY_HITTER_N_INJURED_BEFORE_TEAM_HOME"] == 1
    assert (
        features.loc["2", "TEAM_AVAILABILITY_PITCHER_N_INJURED_BEFORE_TEAM_HOME"] == 0
    )
    assert not any(
        "PLAYER_ID" in column or "PLAYER_NAME" in column for column in features
    )
    assert all(
        column == "GAME_ID" or column.startswith("TEAM_AVAILABILITY_")
        for column in features.columns
    )


def test_availability_schema_is_stable_across_seasons():
    games = pd.DataFrame(
        [_game("1", "2024-04-01", 2024), _game("2", "2025-04-01", 2025)]
    )
    lineups = pd.DataFrame(
        [
            _lineup("1", "2024-04-01", "A", "A1"),
            _lineup("1", "2024-04-01", "B", "B1"),
            _lineup("2", "2025-04-01", "A", "A1"),
            _lineup("2", "2025-04-01", "B", "B1"),
        ]
    )
    batters = pd.DataFrame(
        [
            _batter("1", "2024-04-01", 2024, "A", "A1"),
            _batter("1", "2024-04-01", 2024, "B", "B1"),
            _batter("2", "2025-04-01", 2025, "A", "A1"),
            _batter("2", "2025-04-01", 2025, "B", "B1"),
        ]
    )
    pitchers = pd.DataFrame(
        [
            _pitcher("1", "2024-04-01", 2024, "A", "SP-A", starter=True),
            _pitcher("1", "2024-04-01", 2024, "B", "SP-B", starter=True),
            _pitcher("2", "2025-04-01", 2025, "A", "SP-A", starter=True),
            _pitcher("2", "2025-04-01", 2025, "B", "SP-B", starter=True),
        ]
    )
    empty_transactions = pd.DataFrame()
    closing = pd.DataFrame({"GAME_ID": ["1", "2"]})

    first = build_availability_features(
        games,
        lineups,
        batters,
        pitchers,
        empty_transactions,
        closing,
        target_game_ids=["1"],
    )
    second = build_availability_features(
        games,
        lineups,
        batters,
        pitchers,
        empty_transactions,
        closing,
        target_game_ids=["2"],
    )

    assert list(first.columns) == list(second.columns)


def test_team_change_keeps_player_history_but_moves_membership():
    games = pd.DataFrame(
        [
            _game("1", "2025-04-01", 2025),
            {
                **_game("2", "2025-04-02", 2025),
                "home_team_id": "B",
                "away_team_id": "A",
                "home_probable_pitcher_id": "SP-B",
                "away_probable_pitcher_id": "SP-A",
            },
        ]
    )
    lineups = pd.DataFrame(
        [
            _lineup("1", "2025-04-01", "A", "TRADED"),
            _lineup("1", "2025-04-01", "B", "B1"),
            {**_lineup("2", "2025-04-02", "B", "TRADED"), "home": True},
            {**_lineup("2", "2025-04-02", "A", "A2"), "home": False},
        ]
    )
    batters = pd.DataFrame(
        [
            _batter("1", "2025-04-01", 2025, "A", "TRADED", hits=2),
            _batter("1", "2025-04-01", 2025, "B", "B1"),
            _batter("2", "2025-04-02", 2025, "B", "TRADED"),
            _batter("2", "2025-04-02", 2025, "A", "A2"),
        ]
    )
    pitchers = pd.DataFrame(
        [
            _pitcher("1", "2025-04-01", 2025, "A", "SP-A", starter=True),
            _pitcher("1", "2025-04-01", 2025, "B", "SP-B", starter=True),
            _pitcher("2", "2025-04-02", 2025, "A", "SP-A", starter=True),
            _pitcher("2", "2025-04-02", 2025, "B", "SP-B", starter=True),
        ]
    )
    transactions = pd.DataFrame(
        [
            {
                "transaction_id": "1",
                "player_id": "TRADED",
                "known_date": pd.Timestamp("2025-04-02").date(),
                "from_team_id": "A",
                "to_team_id": "B",
                "type_desc": "Trade",
                "description": "B acquired OF TRADED from A",
            }
        ]
    )

    features = build_availability_features(
        games,
        lineups,
        batters,
        pitchers,
        transactions,
        pd.DataFrame({"GAME_ID": ["1", "2"]}),
        target_game_ids=["2"],
    ).iloc[0]

    assert (
        features["TEAM_AVAILABILITY_HITTER_TOP1_AVAILABLE_OBP_BEFORE_TEAM_HOME"] == 0.5
    )
    assert features["TEAM_AVAILABILITY_HITTER_N_OBSERVED_ABSENT_BEFORE_TEAM_AWAY"] == 0


def test_two_way_player_stays_a_hitter_after_starting_on_the_mound():
    """A two-way starter must not vanish from his own lineup.

    An exclusive hitter/pitcher label made this player a "pitcher" on the date
    he started on the mound and only reverted on his next batting date, so the
    game in between lost him from both the available and the absent side.
    """
    games = pd.DataFrame(
        [
            _game("1", "2025-04-01", 2025),
            _game("2", "2025-04-02", 2025),
            _game("3", "2025-04-03", 2025),
        ]
    )
    lineups = pd.DataFrame(
        [
            # Game 1: two-way player bats as the designated hitter and starts.
            _lineup("1", "2025-04-01", "A", "TWOWAY", position_code="10"),
            _lineup("1", "2025-04-01", "A", "H1"),
            _lineup("1", "2025-04-01", "B", "B1"),
            # Game 2: the day after pitching. He is still a hitter.
            _lineup("2", "2025-04-02", "A", "TWOWAY", position_code="10"),
            _lineup("2", "2025-04-02", "A", "H1"),
            _lineup("2", "2025-04-02", "B", "B1"),
            # Game 3: listed in the pitcher's slot while batting, exactly as
            # the store records a two-way player's own start.  The away side
            # carries a pitcher who has never batted, who must stay excluded.
            _lineup("3", "2025-04-03", "A", "TWOWAY", position_code="1"),
            _lineup("3", "2025-04-03", "A", "H1"),
            _lineup("3", "2025-04-03", "B", "B1"),
            _lineup("3", "2025-04-03", "B", "NL-PITCHER", position_code="1"),
        ]
    )
    batters = pd.DataFrame(
        [
            _batter("1", "2025-04-01", 2025, "A", "TWOWAY", hits=3),
            _batter("1", "2025-04-01", 2025, "A", "H1"),
            _batter("1", "2025-04-01", 2025, "B", "B1"),
            _batter("2", "2025-04-02", 2025, "A", "TWOWAY", hits=3),
            _batter("2", "2025-04-02", 2025, "A", "H1"),
            _batter("2", "2025-04-02", 2025, "B", "B1"),
            _batter("3", "2025-04-03", 2025, "A", "TWOWAY", hits=3),
            _batter("3", "2025-04-03", 2025, "A", "H1"),
            _batter("3", "2025-04-03", 2025, "B", "B1"),
        ]
    )
    pitchers = pd.DataFrame(
        [
            _pitcher("1", "2025-04-01", 2025, "A", "TWOWAY", starter=True),
            _pitcher("1", "2025-04-01", 2025, "B", "SP-B", starter=True),
            _pitcher("2", "2025-04-02", 2025, "A", "SP-A", starter=True),
            _pitcher("2", "2025-04-02", 2025, "B", "SP-B", starter=True),
            _pitcher("3", "2025-04-03", 2025, "A", "TWOWAY", starter=True),
            _pitcher("3", "2025-04-03", 2025, "B", "NL-PITCHER", starter=True),
        ]
    )
    closing = pd.DataFrame({"GAME_ID": ["1", "2", "3"]})

    features = build_availability_features(
        games,
        lineups,
        batters,
        pitchers,
        pd.DataFrame(),
        closing,
        target_game_ids=["2", "3"],
    ).set_index("GAME_ID")

    home_available = "TEAM_AVAILABILITY_HITTER_N_AVAILABLE_BEFORE_TEAM_HOME"
    home_absent = "TEAM_AVAILABILITY_HITTER_N_OBSERVED_ABSENT_BEFORE_TEAM_HOME"

    # The day after a pitching start: both hitters present, nobody absent.
    assert features.loc["2", home_available] == 2
    assert features.loc["2", home_absent] == 0

    # On his own start he is in the pitcher's slot but still a hitter, while
    # the away side's genuine pitcher is not counted as one.
    assert features.loc["3", home_available] == 2
    assert features.loc["3", home_absent] == 0
    assert (
        features.loc["3", "TEAM_AVAILABILITY_HITTER_N_AVAILABLE_BEFORE_TEAM_AWAY"] == 1
    )


def test_effect_history_defers_same_date_outcomes_and_shrinks_by_sample():
    """Pin the empirical-effect block: shrinkage and same-date deferral.

    The second game of a doubleheader must not see the first game's result,
    and a player with evidence on only one side of the comparison must yield a
    zero effect with a zero sample size rather than a fabricated one.
    """
    games = pd.DataFrame(
        [
            _game("1", "2025-04-01", 2025, home_score=7, away_score=3),
            _game("2", "2025-04-02", 2025, home_score=1, away_score=5),
            _game("3", "2025-04-03", 2025, home_score=4, away_score=4),
            _game("4", "2025-04-03", 2025, home_score=9, away_score=0),
        ]
    )
    lineups = pd.DataFrame(
        [
            _lineup("1", "2025-04-01", "A", "STAR"),
            _lineup("1", "2025-04-01", "A", "H1"),
            _lineup("1", "2025-04-01", "B", "B1"),
            # STAR sits out entirely on the second date.
            _lineup("2", "2025-04-02", "A", "H1"),
            _lineup("2", "2025-04-02", "B", "B1"),
            _lineup("3", "2025-04-03", "A", "STAR"),
            _lineup("3", "2025-04-03", "A", "H1"),
            _lineup("3", "2025-04-03", "B", "B1"),
            _lineup("4", "2025-04-03", "A", "STAR"),
            _lineup("4", "2025-04-03", "A", "H1"),
            _lineup("4", "2025-04-03", "B", "B1"),
        ]
    )
    batters = pd.DataFrame(
        [
            _batter("1", "2025-04-01", 2025, "A", "STAR", hits=3),
            _batter("1", "2025-04-01", 2025, "A", "H1"),
            _batter("1", "2025-04-01", 2025, "B", "B1"),
            _batter("2", "2025-04-02", 2025, "A", "H1"),
            _batter("2", "2025-04-02", 2025, "B", "B1"),
            _batter("3", "2025-04-03", 2025, "A", "STAR", hits=3),
            _batter("3", "2025-04-03", 2025, "A", "H1"),
            _batter("3", "2025-04-03", 2025, "B", "B1"),
            _batter("4", "2025-04-03", 2025, "A", "STAR", hits=3),
            _batter("4", "2025-04-03", 2025, "A", "H1"),
            _batter("4", "2025-04-03", 2025, "B", "B1"),
        ]
    )
    pitchers = pd.DataFrame(
        [
            _pitcher(str(pk), date, 2025, team, f"SP-{team}", starter=True)
            for pk, date in (
                ("1", "2025-04-01"),
                ("2", "2025-04-02"),
                ("3", "2025-04-03"),
                ("4", "2025-04-03"),
            )
            for team in ("A", "B")
        ]
    )
    closing = pd.DataFrame({"GAME_ID": ["1", "2", "3", "4"]})

    features = build_availability_features(
        games,
        lineups,
        batters,
        pitchers,
        pd.DataFrame(),
        closing,
        target_game_ids=["3", "4"],
    ).set_index("GAME_ID")

    prefix = "TEAM_AVAILABILITY_HITTER_AVAILABLE"
    sample = f"{prefix}_TEAM_RUNS_EFFECT_SAMPLE_SIZE_BEFORE_TEAM_HOME"
    max_abs = f"{prefix}_TEAM_RUNS_EFFECT_MAX_ABS_BEFORE_TEAM_HOME"

    # Both halves of the doubleheader see identical evidence: the first game's
    # own outcome is not folded in before the second is featurised.
    assert features.loc["3", sample] == features.loc["4", sample]
    assert features.loc["3", max_abs] == pytest.approx(features.loc["4", max_abs])

    # STAR: home runs 7 available, 1 absent, so a raw effect of 6 over one
    # game on each side, shrunk by n_eff / (n_eff + 10) = 1 / 11.
    assert features.loc["3", sample] == 1
    assert features.loc["3", max_abs] == pytest.approx(6.0 * 1.0 / 11.0)

    # The TOTAL_RUNS effect outcome was dropped. Measured against the line,
    # TOTAL_LINE_ERROR says the same thing in the units the model works in --
    # deviation from the market expectation -- so keeping both was carrying one
    # idea twice. TEAM_RUNS and TOTAL_LINE_ERROR remain.
    assert (
        f"{prefix}_TOTAL_RUNS_EFFECT_MAX_ABS_BEFORE_TEAM_HOME" not in features.columns
    )
    assert (
        f"{prefix}_TOTAL_LINE_ERROR_EFFECT_MAX_ABS_BEFORE_TEAM_HOME" in features.columns
    )

    # The away side has one hitter who never missed a game, so there is no
    # absent sample to compare against and the effect must stay empty.
    away_sample = f"{prefix}_TEAM_RUNS_EFFECT_SAMPLE_SIZE_BEFORE_TEAM_AWAY"
    away_max_abs = f"{prefix}_TEAM_RUNS_EFFECT_MAX_ABS_BEFORE_TEAM_AWAY"
    assert features.loc["3", away_sample] == 0
    assert features.loc["3", away_max_abs] == 0.0


def test_player_without_history_falls_back_to_documented_neutrals():
    games = pd.DataFrame([_game("1", "2025-04-01", 2025)])
    lineups = pd.DataFrame(
        [
            _lineup("1", "2025-04-01", "A", "DEBUT"),
            _lineup("1", "2025-04-01", "B", "B1"),
        ]
    )
    batters = pd.DataFrame(
        [
            _batter("1", "2025-04-01", 2025, "A", "DEBUT"),
            _batter("1", "2025-04-01", 2025, "B", "B1"),
        ]
    )
    pitchers = pd.DataFrame(
        [
            _pitcher("1", "2025-04-01", 2025, "A", "SP-A", starter=True),
            _pitcher("1", "2025-04-01", 2025, "B", "SP-B", starter=True),
        ]
    )
    closing = pd.DataFrame({"GAME_ID": ["1"]})

    features = build_availability_features(
        games,
        lineups,
        batters,
        pitchers,
        pd.DataFrame(),
        closing,
        target_game_ids=["1"],
    ).set_index("GAME_ID")

    # Primary metrics are emitted as TOTAL plus TOP1. MEAN was dropped because
    # it is TOTAL divided by N_<state>, already a column of its own; MAX was
    # dropped because it is identical to TOP1 by construction -- verified equal
    # across all 24 column pairs over 17,638 games.
    primary = "TEAM_AVAILABILITY_HITTER_TOP1_AVAILABLE"
    assert features.loc["1", f"{primary}_OBP_BEFORE_TEAM_HOME"] == pytest.approx(
        HITTER_NEUTRALS["OBP"]
    )
    assert features.loc["1", f"{primary}_PA_BEFORE_TEAM_HOME"] == pytest.approx(
        HITTER_NEUTRALS["PA"]
    )
    assert (
        "TEAM_AVAILABILITY_HITTER_MEAN_AVAILABLE_OBP_BEFORE_TEAM_HOME" not in features
    )
    assert "TEAM_AVAILABILITY_HITTER_MAX_AVAILABLE_OBP_BEFORE_TEAM_HOME" not in features
    # Secondary metrics describe the shape of a plate appearance rather than
    # its value, so they stay MEAN-only and fall back identically.
    secondary = "TEAM_AVAILABILITY_HITTER_MEAN_AVAILABLE"
    assert features.loc["1", f"{secondary}_K_PCT_BEFORE_TEAM_HOME"] == pytest.approx(
        HITTER_NEUTRALS["K_PCT"]
    )
    assert (
        features.loc[
            "1", "TEAM_AVAILABILITY_HITTER_N_EXPECTED_WITH_HISTORY_BEFORE_TEAM_HOME"
        ]
        == 0
    )


def test_missing_lineup_reports_no_coverage_instead_of_inventing_absences():
    """A game with no lineup is missing data, not a squad-wide injury crisis."""
    games = pd.DataFrame(
        [_game("1", "2025-04-01", 2025), _game("2", "2025-04-02", 2025)]
    )
    lineups = pd.DataFrame(
        [
            _lineup("1", "2025-04-01", "A", "STAR"),
            _lineup("1", "2025-04-01", "A", "H1"),
            _lineup("1", "2025-04-01", "B", "B1"),
            # Game 2 has a lineup for the away side only.
            _lineup("2", "2025-04-02", "B", "B1"),
        ]
    )
    batters = pd.DataFrame(
        [
            _batter("1", "2025-04-01", 2025, "A", "STAR", hits=3),
            _batter("1", "2025-04-01", 2025, "A", "H1"),
            _batter("1", "2025-04-01", 2025, "B", "B1"),
            _batter("2", "2025-04-02", 2025, "B", "B1"),
        ]
    )
    pitchers = pd.DataFrame(
        [
            _pitcher("1", "2025-04-01", 2025, "A", "SP-A", starter=True),
            _pitcher("1", "2025-04-01", 2025, "B", "SP-B", starter=True),
            _pitcher("2", "2025-04-02", 2025, "B", "SP-B", starter=True),
        ]
    )
    closing = pd.DataFrame({"GAME_ID": ["1", "2"]})

    features = build_availability_features(
        games,
        lineups,
        batters,
        pitchers,
        pd.DataFrame(),
        closing,
        target_game_ids=["2"],
    ).set_index("GAME_ID")

    assert (
        features.loc["2", "TEAM_AVAILABILITY_HITTER_LINEUP_COVERAGE_BEFORE_TEAM_HOME"]
        == 0
    )
    assert (
        features.loc["2", "TEAM_AVAILABILITY_HITTER_N_LINEUP_STARTERS_BEFORE_TEAM_HOME"]
        == 0
    )
    assert (
        features.loc["2", "TEAM_AVAILABILITY_HITTER_N_OBSERVED_ABSENT_BEFORE_TEAM_HOME"]
        == 0
    )
    assert features.loc["2", "TEAM_AVAILABILITY_HITTER_N_INJURED_BEFORE_TEAM_HOME"] == 0
    # The away side does have a lineup, so its coverage flag stays set.
    assert (
        features.loc["2", "TEAM_AVAILABILITY_HITTER_LINEUP_COVERAGE_BEFORE_TEAM_AWAY"]
        == 1
    )
