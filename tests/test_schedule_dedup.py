"""The schedule feed lists a game once per slot it has occupied.

Left alone that silently doubles a team's game count for the affected dates --
the kind of quiet corruption that never surfaces in a rolling mean.
"""

import pandas as pd
import pytest

from mlb_pred.fetch_data.statsapi.schedule import deduplicate_games, finished_games


def _row(game_pk, first_pitch, is_final, total_runs, game_date="2025-08-19"):
    return {
        "game_pk": game_pk,
        "game_date": pd.Timestamp(game_date).date(),
        "first_pitch_utc": pd.Timestamp(first_pitch, tz="UTC"),
        "is_final": is_final,
        "total_runs": total_runs,
        "status_detailed": "Final" if is_final else "Postponed",
    }


def test_postponed_then_replayed_keeps_the_played_row():
    games = pd.DataFrame(
        [
            _row("1", "2025-08-19T00:05:00", False, None),
            _row("1", "2025-08-19T18:20:00", True, 10),
        ]
    )
    result = deduplicate_games(games)
    assert len(result) == 1
    assert result.iloc[0]["total_runs"] == 10
    assert bool(result.iloc[0]["is_final"])


def test_suspended_then_resumed_keeps_the_original_first_pitch():
    # Both rows are Final with the same score; the real first pitch is the
    # earlier one, not the continuation the next day.
    games = pd.DataFrame(
        [
            _row("2", "2025-08-03T17:05:00", True, 6),
            _row("2", "2025-08-02T23:15:00", True, 6),
        ]
    )
    result = deduplicate_games(games)
    assert len(result) == 1
    assert result.iloc[0]["first_pitch_utc"] == pd.Timestamp(
        "2025-08-02T23:15:00", tz="UTC"
    )


def test_distinct_games_are_untouched():
    games = pd.DataFrame(
        [
            _row("1", "2025-08-19T18:20:00", True, 10),
            _row("2", "2025-08-19T23:10:00", True, 7),
        ]
    )
    assert len(deduplicate_games(games)) == 2


def test_doubleheader_games_are_not_collapsed():
    # Distinct gamePks, so the compound-key problem the NBA repo flags for MLB
    # simply does not arise.
    games = pd.DataFrame(
        [
            _row("10", "2025-08-19T17:05:00", True, 5),
            _row("11", "2025-08-19T21:35:00", True, 8),
        ]
    )
    assert len(deduplicate_games(games)) == 2


def test_unplayed_games_are_dropped_rather_than_stored_with_null_scores():
    # A NULL-score row would reach a rolling mean as a real observation.
    games = pd.DataFrame(
        [
            _row("1", "2025-08-19T18:20:00", True, 10),
            _row("3", "2025-08-19T23:10:00", False, None),
        ]
    )
    result = finished_games(deduplicate_games(games))
    assert set(result["game_pk"]) == {"1"}
    assert result["total_runs"].notna().all()


def test_empty_input():
    assert deduplicate_games(pd.DataFrame()).empty


def test_games_table_scores_are_named_score_not_runs():
    """`home_runs` is dangerously overloaded in a baseball schema.

    In the games table it would mean "runs scored by the home team"; in
    team_games and batter_games it means "home runs hit". Two different
    quantities under one name is a merge waiting to break -- and it did break
    the store validator before the rename. The games table uses ``*_score``.
    """
    from mlb_pred.fetch_data.statsapi.schedule import GAME_COLUMNS

    assert "home_score" in GAME_COLUMNS
    assert "away_score" in GAME_COLUMNS
    assert "home_runs" not in GAME_COLUMNS
    assert "away_runs" not in GAME_COLUMNS


# ---------------------------------------------------------------------------
# Bracket placeholders vs. a genuinely unknown team
# ---------------------------------------------------------------------------
def _game_with_teams(home_id, home_name, away_id="147", game_type="F", state="Preview"):
    return {
        "gamePk": 1,
        "gameType": game_type,
        "status": {"abstractGameState": state},
        "teams": {
            "home": {"team": {"id": home_id, "name": home_name}},
            "away": {"team": {"id": away_id, "name": "New York Yankees"}},
        },
    }


def test_an_unplayed_postseason_placeholder_is_skipped():
    """The Stats API publishes an unplayed bracket with placeholder "teams".

    "AL Wild Card #1" is not a team with an unknown name -- it is a game with
    undetermined participants, and skipping it is correct.
    """
    from mlb_pred.fetch_data.statsapi.schedule import (
        UndeterminedParticipantsError,
        _check_participants,
    )

    with pytest.raises(UndeterminedParticipantsError):
        _check_participants(_game_with_teams("4618", "AL Wild Card #1"))


def test_a_real_matchup_passes_the_participant_check():
    from mlb_pred.fetch_data.statsapi.schedule import _check_participants

    _check_participants(
        _game_with_teams("111", "Boston Red Sox", game_type="R", state="Final")
    )


def test_an_unknown_id_on_a_regular_season_game_still_raises():
    """The escape hatch must stay narrow.

    The other reading of an unknown team id is an *expansion franchise*, which
    must never be silently skipped. A placeholder is only accepted as such on
    an unplayed postseason game.
    """
    from mlb_pred.config.constants import UnknownTeamNameError
    from mlb_pred.fetch_data.statsapi.schedule import _check_participants

    with pytest.raises(UnknownTeamNameError):
        _check_participants(
            _game_with_teams("9001", "Nashville Stars", game_type="R", state="Preview")
        )


def test_an_unknown_id_on_a_completed_game_still_raises():
    from mlb_pred.config.constants import UnknownTeamNameError
    from mlb_pred.fetch_data.statsapi.schedule import _check_participants

    with pytest.raises(UnknownTeamNameError):
        _check_participants(
            _game_with_teams("9001", "Nashville Stars", game_type="F", state="Final")
        )


def test_placeholders_are_detected_by_id_not_by_name():
    """Placeholder names vary freely between rounds ("AL Wild Card #1",
    "NL Higher Seed", "Higher Seed League Champion"); the id range is
    structural. Keying on the name would mean chasing bracket vocabulary."""
    from mlb_pred.config.constants import is_known_team_id

    for placeholder_id in ("4618", "5517", "2710", "4613"):
        assert not is_known_team_id(placeholder_id)
    for franchise_id in ("147", "111", "133", "119"):
        assert is_known_team_id(franchise_id)
