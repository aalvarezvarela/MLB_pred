"""Bullpen workload and quality, derived from pitcher appearances.

``pitcher_appearances`` exists precisely because a pitcher's history does not
accumulate on the team-game grain, but no feature module had used it to
describe the relief staff. For a totals model this is a real gap: the market
prices a team's bullpen *strength* over a season, while what varies game to
game -- and is knowable before first pitch -- is how much of that bullpen was
used in the last few days and whether the best arms are likely unavailable.

Two kinds of column:

* **Workload** -- relief pitches, outs and distinct arms used over the previous
  1, 3 and 5 team games, plus how deep the last starter went. All shifted, so
  the current game is never part of its own workload.
* **Quality** -- relief ERA per nine, strikeout and walk rates over the last 10
  and 30 team games, computed as ratios of shifted rolling sums rather than
  means of per-game ratios, so a one-batter appearance does not weigh the same
  as a three-inning one.

``is_starter`` separates the two staffs. Innings are read as ``outs_recorded``
per the store's convention, never as the ``7.1`` innings-pitched notation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_PREFIX = "BULLPEN_"

WORKLOAD_WINDOWS = (1, 3, 5)
QUALITY_WINDOWS = (10, 30)
MIN_QUALITY_GAMES = 3
OUTS_PER_NINE = 27.0

_APPEARANCE_COLUMNS = {
    "game_pk",
    "team_id",
    "game_date",
    "is_starter",
    "outs_recorded",
    "pitches_thrown",
    "batters_faced",
    "earned_runs",
    "strikeouts",
    "walks_allowed",
    "home_runs_allowed",
    "player_id",
}
_GAME_COLUMNS = {"game_pk", "home_team_id", "away_team_id"}

_QUALITY_RATES: tuple[tuple[str, str, str, float], ...] = (
    ("ERA9", "__earned_runs", "__outs", OUTS_PER_NINE),
    ("K_PCT", "__strikeouts", "__batters_faced", 1.0),
    ("BB_PCT", "__walks", "__batters_faced", 1.0),
    ("HR_RATE", "__home_runs", "__batters_faced", 1.0),
)


def bullpen_feature_columns() -> list[str]:
    """Return the stable schema this module emits, in order."""
    stems: list[str] = []
    for window in WORKLOAD_WINDOWS:
        stems.extend(
            [
                f"{FEATURE_PREFIX}PITCHES_LAST_{window}_GAMES_BEFORE",
                f"{FEATURE_PREFIX}ARMS_USED_LAST_{window}_GAMES_BEFORE",
            ]
        )
    stems.extend(
        [
            f"{FEATURE_PREFIX}OUTS_LAST_3_GAMES_BEFORE",
            f"{FEATURE_PREFIX}STARTER_OUTS_LAST_GAME_BEFORE",
        ]
    )
    for name, _, _, _ in _QUALITY_RATES:
        for window in QUALITY_WINDOWS:
            stems.append(f"{FEATURE_PREFIX}{name}_LAST_{window}_GAMES_BEFORE")
    return [f"{stem}_{role}" for stem in stems for role in ("TEAM_HOME", "TEAM_AWAY")]


def _require_columns(
    frame: pd.DataFrame, required: set[str], *, frame_name: str
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _team_game_totals(pitcher_appearances: pd.DataFrame) -> pd.DataFrame:
    """Collapse appearances to one relief line plus one starter line per team-game."""
    _require_columns(
        pitcher_appearances, _APPEARANCE_COLUMNS, frame_name="pitcher appearances"
    )
    frame = pitcher_appearances.copy()
    frame["game_pk"] = frame["game_pk"].astype(str)
    frame["team_id"] = frame["team_id"].astype(str)
    frame["game_date"] = pd.to_datetime(
        frame["game_date"], errors="raise"
    ).dt.normalize()
    is_starter = frame["is_starter"].fillna(False).astype(bool)

    relief = (
        frame.loc[~is_starter]
        .groupby(["team_id", "game_date", "game_pk"], sort=False)
        .agg(
            __pitches=("pitches_thrown", "sum"),
            __outs=("outs_recorded", "sum"),
            __arms=("player_id", "nunique"),
            __earned_runs=("earned_runs", "sum"),
            __batters_faced=("batters_faced", "sum"),
            __strikeouts=("strikeouts", "sum"),
            __walks=("walks_allowed", "sum"),
            __home_runs=("home_runs_allowed", "sum"),
        )
        .reset_index()
    )
    starters = (
        frame.loc[is_starter]
        .groupby(["team_id", "game_date", "game_pk"], sort=False)
        .agg(__starter_outs=("outs_recorded", "sum"))
        .reset_index()
    )
    totals = relief.merge(starters, on=["team_id", "game_date", "game_pk"], how="outer")
    return totals.sort_values(
        ["team_id", "game_date", "game_pk"], kind="mergesort"
    ).reset_index(drop=True)


def _shifted_bullpen_state(totals: pd.DataFrame) -> pd.DataFrame:
    """Add the shifted workload and quality columns for each team-game."""
    grouped = totals.groupby("team_id", sort=False)
    built: dict[str, pd.Series] = {}
    for window in WORKLOAD_WINDOWS:
        built[f"{FEATURE_PREFIX}PITCHES_LAST_{window}_GAMES_BEFORE"] = grouped[
            "__pitches"
        ].transform(lambda s, w=window: s.shift().rolling(w, min_periods=1).sum())
        built[f"{FEATURE_PREFIX}ARMS_USED_LAST_{window}_GAMES_BEFORE"] = grouped[
            "__arms"
        ].transform(lambda s, w=window: s.shift().rolling(w, min_periods=1).sum())
    built[f"{FEATURE_PREFIX}OUTS_LAST_3_GAMES_BEFORE"] = grouped["__outs"].transform(
        lambda s: s.shift().rolling(3, min_periods=1).sum()
    )
    built[f"{FEATURE_PREFIX}STARTER_OUTS_LAST_GAME_BEFORE"] = grouped[
        "__starter_outs"
    ].transform(lambda s: s.shift())

    # Ratio of shifted sums, not a mean of per-game ratios: a one-batter
    # appearance must not carry the same weight as a three-inning one.
    for name, numerator, denominator, scale in _QUALITY_RATES:
        for window in QUALITY_WINDOWS:
            top = grouped[numerator].transform(
                lambda s, w=window: s.shift()
                .rolling(w, min_periods=MIN_QUALITY_GAMES)
                .sum()
            )
            bottom = grouped[denominator].transform(
                lambda s, w=window: s.shift()
                .rolling(w, min_periods=MIN_QUALITY_GAMES)
                .sum()
            )
            built[f"{FEATURE_PREFIX}{name}_LAST_{window}_GAMES_BEFORE"] = (
                top / bottom.replace(0.0, np.nan)
            ) * scale
    return pd.concat([totals, pd.DataFrame(built, index=totals.index)], axis=1)


def build_bullpen_features(
    pitcher_appearances: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Build shifted bullpen workload and quality, pivoted to one row per game."""
    _require_columns(games, _GAME_COLUMNS, frame_name="games")
    totals = _team_game_totals(pitcher_appearances)
    if totals.empty:
        return pd.DataFrame(columns=["GAME_ID", *bullpen_feature_columns()])
    state = _shifted_bullpen_state(totals)
    stems = [
        column
        for column in state.columns
        if column.startswith(FEATURE_PREFIX) and column.endswith("_BEFORE")
    ]
    keyed = state[["game_pk", "team_id", *stems]]

    schedule = games[["game_pk", "home_team_id", "away_team_id"]].copy()
    schedule["game_pk"] = schedule["game_pk"].astype(str)
    sides = []
    for role, id_column in (
        ("TEAM_HOME", "home_team_id"),
        ("TEAM_AWAY", "away_team_id"),
    ):
        side = schedule[["game_pk", id_column]].copy()
        side[id_column] = side[id_column].astype(str)
        side = side.merge(
            keyed,
            left_on=["game_pk", id_column],
            right_on=["game_pk", "team_id"],
            how="left",
            validate="one_to_one",
        )
        side = side.drop(columns=[id_column, "team_id"]).rename(
            columns={stem: f"{stem}_{role}" for stem in stems}
        )
        sides.append(side.set_index("game_pk"))

    output = pd.concat(sides, axis=1)
    output.index.name = "GAME_ID"
    output = output.reset_index()
    for column in bullpen_feature_columns():
        if column not in output:
            output[column] = np.nan
    return output[["GAME_ID", *bullpen_feature_columns()]]


__all__ = [
    "FEATURE_PREFIX",
    "bullpen_feature_columns",
    "build_bullpen_features",
]
