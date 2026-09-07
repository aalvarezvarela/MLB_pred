from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_pred.features.rolling_features import (
    DERIVED_RATE_SOURCE_COLUMNS,
    TEAM_SOURCE_COLUMNS,
    TIER1,
    _prepare_team_context,
    _rolling_source_features,
    _wide_team_features,
    build_pregame_features,
    build_team_rolling_features,
)


def _team_row(
    game_pk: str,
    game_date: str,
    team_id: str,
    opponent_team_id: str,
    *,
    home: bool,
    runs_scored: float,
    season_year: int = 2025,
    first_pitch_hour: int = 19,
) -> dict[str, object]:
    row: dict[str, object] = {
        column: 1.0 for column in set(TEAM_SOURCE_COLUMNS) | DERIVED_RATE_SOURCE_COLUMNS
    }
    row.update(
        {
            "game_pk": game_pk,
            "team_id": team_id,
            "season_year": season_year,
            "game_date": pd.Timestamp(game_date).date(),
            "first_pitch_utc": pd.Timestamp(
                f"{game_date}T{first_pitch_hour:02d}:00:00Z"
            ),
            "game_type": "R",
            "home": home,
            "opponent_team_id": opponent_team_id,
            "runs_scored": runs_scored,
            "runs_allowed": 2.0,
            "run_margin": runs_scored - 2.0,
            "total_runs": runs_scored + 2.0,
            "win": float(runs_scored > 2.0),
        }
    )
    return row


def _game_rows(team_games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for game_pk, group in team_games.groupby("game_pk", sort=False):
        home = group.loc[group["home"]].iloc[0]
        away = group.loc[~group["home"]].iloc[0]
        rows.append(
            {
                "game_pk": game_pk,
                "season_year": home["season_year"],
                "game_date": home["game_date"],
                "first_pitch_utc": home["first_pitch_utc"],
                "game_type": home["game_type"],
                "home_team_id": home["team_id"],
                "away_team_id": away["team_id"],
            }
        )
    return pd.DataFrame(rows)


def _single_source_team_rows(
    team_games: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    context = _prepare_team_context(team_games, games)
    features = _rolling_source_features(
        context, "runs_scored", "RUNS_SCORED", tier=TIER1
    )
    return pd.concat([context, pd.DataFrame(features, index=context.index)], axis=1)


def test_current_game_result_is_excluded_from_every_rolling_window():
    team_games = pd.DataFrame(
        [
            _team_row("1", "2025-04-01", "A", "B", home=True, runs_scored=2),
            _team_row("1", "2025-04-01", "B", "A", home=False, runs_scored=3),
            _team_row("2", "2025-04-02", "A", "B", home=True, runs_scored=4),
            _team_row("2", "2025-04-02", "B", "A", home=False, runs_scored=1),
            _team_row("3", "2025-04-03", "A", "B", home=True, runs_scored=999),
            _team_row("3", "2025-04-03", "B", "A", home=False, runs_scored=999),
        ]
    )
    games = _game_rows(team_games)
    rows = _single_source_team_rows(team_games, games)
    current = rows.loc[(rows["game_pk"] == "3") & rows["home"]].iloc[0]

    assert current["TEAM_ROLLING_RUNS_SCORED_LAST_ALL_1_GAMES_BEFORE"] == 4
    assert current["TEAM_ROLLING_RUNS_SCORED_LAST_ALL_5_GAMES_BEFORE"] == 3


def test_doubleheader_games_share_the_history_available_before_that_date():
    team_games = pd.DataFrame(
        [
            _team_row("1", "2025-04-01", "A", "B", home=True, runs_scored=2),
            _team_row("1", "2025-04-01", "B", "A", home=False, runs_scored=3),
            _team_row(
                "2",
                "2025-04-02",
                "A",
                "B",
                home=True,
                runs_scored=100,
                first_pitch_hour=17,
            ),
            _team_row(
                "2",
                "2025-04-02",
                "B",
                "A",
                home=False,
                runs_scored=1,
                first_pitch_hour=17,
            ),
            _team_row(
                "3",
                "2025-04-02",
                "A",
                "B",
                home=True,
                runs_scored=200,
                first_pitch_hour=21,
            ),
            _team_row(
                "3",
                "2025-04-02",
                "B",
                "A",
                home=False,
                runs_scored=1,
                first_pitch_hour=21,
            ),
        ]
    )
    rows = _single_source_team_rows(team_games, _game_rows(team_games))
    doubleheader = rows.loc[(rows["team_id"] == "A") & rows["game_pk"].isin(["2", "3"])]

    assert doubleheader[
        "TEAM_ROLLING_RUNS_SCORED_LAST_ALL_1_GAMES_BEFORE"
    ].tolist() == [2.0, 2.0]


def test_season_opening_home_average_falls_back_to_previous_regular_season():
    team_games = pd.DataFrame(
        [
            _team_row(
                "1",
                "2024-08-01",
                "A",
                "B",
                home=True,
                runs_scored=2,
                season_year=2024,
            ),
            _team_row(
                "1",
                "2024-08-01",
                "B",
                "A",
                home=False,
                runs_scored=3,
                season_year=2024,
            ),
            _team_row(
                "2",
                "2024-08-02",
                "A",
                "B",
                home=True,
                runs_scored=6,
                season_year=2024,
            ),
            _team_row(
                "2",
                "2024-08-02",
                "B",
                "A",
                home=False,
                runs_scored=1,
                season_year=2024,
            ),
            _team_row("3", "2025-04-01", "A", "B", home=True, runs_scored=999),
            _team_row("3", "2025-04-01", "B", "A", home=False, runs_scored=999),
        ]
    )
    rows = _single_source_team_rows(team_games, _game_rows(team_games))
    opener = rows.loc[(rows["game_pk"] == "3") & rows["home"]].iloc[0]

    assert opener["TEAM_ROLLING_RUNS_SCORED_SEASON_BEFORE_AVG"] == 4


def test_scheduled_rows_neither_create_zero_rates_nor_consume_history_slots():
    completed = pd.DataFrame(
        [
            _team_row("1", "2025-04-01", "A", "B", home=True, runs_scored=2),
            _team_row("1", "2025-04-01", "B", "A", home=False, runs_scored=3),
        ]
    )
    games = pd.concat(
        [
            _game_rows(completed),
            pd.DataFrame(
                [
                    {
                        "game_pk": "2",
                        "season_year": 2025,
                        "game_date": pd.Timestamp("2025-04-02").date(),
                        "first_pitch_utc": pd.Timestamp("2025-04-02T19:00:00Z"),
                        "game_type": "R",
                        "home_team_id": "A",
                        "away_team_id": "B",
                    },
                    {
                        "game_pk": "3",
                        "season_year": 2025,
                        "game_date": pd.Timestamp("2025-04-03").date(),
                        "first_pitch_utc": pd.Timestamp("2025-04-03T19:00:00Z"),
                        "game_type": "R",
                        "home_team_id": "A",
                        "away_team_id": "B",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    context = _prepare_team_context(completed, games)
    scheduled = context.loc[(context["game_pk"] == "2") & context["home"]].iloc[0]
    assert pd.isna(scheduled["__pitching_era_per_9"])

    features = _rolling_source_features(
        context,
        "__pitching_era_per_9",
        "PITCHING_ERA_PER_9",
        tier=TIER1,
    )
    rows = pd.concat([context, pd.DataFrame(features, index=context.index)], axis=1)
    later = rows.loc[(rows["game_pk"] == "3") & rows["home"]].iloc[0]
    assert later[
        "TEAM_ROLLING_PITCHING_ERA_PER_9_LAST_ALL_1_GAMES_BEFORE"
    ] == pytest.approx(27.0)


def test_public_builder_labels_families_and_keeps_outcomes_only_as_history():
    team_games = pd.DataFrame(
        [
            _team_row("1", "2025-04-01", "A", "B", home=True, runs_scored=2),
            _team_row("1", "2025-04-01", "B", "A", home=False, runs_scored=3),
            _team_row("2", "2025-04-02", "A", "B", home=True, runs_scored=8),
            _team_row("2", "2025-04-02", "B", "A", home=False, runs_scored=1),
        ]
    )
    games = _game_rows(team_games)
    closing = pd.DataFrame(
        {
            "GAME_ID": ["1", "2"],
            "GAME_SEASON_YEAR": [2025, 2025],
            "ODDS_TOTAL_CONSENSUS_LINE_NORMALIZED_MEDIAN": [8.0, 9.0],
        }
    )

    rolling = build_team_rolling_features(
        team_games, games, closing, target_game_ids=["2"]
    )
    features = build_pregame_features(closing, rolling)

    assert all(column.startswith(("GAME_", "ODDS_", "TEAM_")) for column in features)
    assert not any(column in features for column in ("runs_scored", "total_runs"))
    assert "TEAM_ROLLING_RUNS_SCORED_SEASON_BEFORE_AVG_TEAM_HOME" in features
    assert "TEAM_ROLLING_RUNS_SCORED_LAST_ALL_5_GAMES_BEFORE_TEAM_HOME" in features
    # HOME - AWAY is emitted only for the hand-picked DIFF_FEATURES, because
    # it is an exact linear combination of the two side columns.
    assert "TEAM_ROLLING_RUNS_SCORED_SEASON_BEFORE_AVG_DIFF_BEFORE" in features
    assert "TEAM_ROLLING_RUNS_SCORED_LAST_ALL_5_GAMES_DIFF_BEFORE" not in features
    assert not features.columns.duplicated().any()
    assert rolling["GAME_ID"].tolist() == ["2"]
    row = features.loc[features["GAME_ID"] == "2"].iloc[0]
    assert row["TEAM_ROLLING_RUNS_SCORED_LAST_ALL_1_GAMES_BEFORE_TEAM_HOME"] == 2
    values = pd.to_numeric(
        row.filter(like="TEAM_ROLLING_RUNS_SCORED"), errors="raise"
    ).to_numpy(dtype="float64")
    assert np.isfinite(values).all()


def test_widening_includes_season_columns_and_rejects_incomplete_game_pairs():
    team_rows = pd.DataFrame(
        {
            "game_pk": ["1"],
            "team_id": ["A"],
            "home": [True],
            "TEAM_ROLLING_RUNS_SCORED_SEASON_BEFORE_AVG": [3.0],
        }
    )

    with pytest.raises(ValueError, match="exactly one home and one away"):
        _wide_team_features(team_rows, {"1"})


def test_final_gate_rejects_family_labeled_columns_without_before_tag():
    closing = pd.DataFrame({"GAME_ID": ["1"], "ODDS_TOTAL_SAFE": [8.5]})
    unsafe_context = pd.DataFrame(
        {"GAME_ID": ["1"], "TEAM_RESULT": [1], "ODDS_LINE_ERROR": [2.0]}
    )

    with pytest.raises(ValueError, match="missing _BEFORE temporal tag"):
        build_pregame_features(
            closing,
            pd.DataFrame({"GAME_ID": ["1"]}),
            unsafe_context,
        )
