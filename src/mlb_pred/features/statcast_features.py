"""Team form measured from Statcast batted-ball quality rather than results.

``data/raw/statcast_pitches`` has been ingested since the beginning of the
project and no feature module read it. It carries the one thing a box score
cannot give: what a team *deserved* from its contact, independent of whether
the ball found a glove.

That matters here specifically because the model is given the sportsbook line.
Realised runs over five games are extremely noisy, and the market has already
priced the underlying quality. The useful residual signal, if there is one, is
where recent results and underlying contact quality disagree -- so this module
emits both an expected measure (xwOBA) and the gap between realised and
expected outcomes.

Grain and orientation
---------------------
A pitch's ``inning_topbot`` says who is batting: ``Top`` is the away team,
``Bot`` the home team. Aggregating to (game, batting side) therefore yields
each team's offensive line, and the same rows read from the opposing side give
what its pitching staff allowed. Windows follow the NBA scheme -- last 5, last
10, season to date -- and every one is shifted so the current game is excluded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_PREFIX = "STATCAST_"

SHORT_WINDOW = 5
LONG_WINDOW = 10
MIN_SHORT_GAMES = 2
MIN_LONG_GAMES = 3

# Statcast's own thresholds: 95 mph is the hard-hit cutoff, and a barrel is the
# launch-speed/angle combination that historically produces extra-base damage.
HARD_HIT_MPH = 95.0
BARREL_MIN_MPH = 98.0
BARREL_ANGLE_RANGE = (26.0, 30.0)

# Metrics kept per side. Each is a rate or a per-game expectation, so a team
# with a suspended or shortened game is not penalised by a volume measure.
_METRICS = ("XWOBA", "WOBA", "WOBA_MINUS_XWOBA", "HARD_HIT_PCT", "BARREL_PCT")
_SIDES = ("OFFENSE", "ALLOWED")

_PITCH_COLUMNS = {
    "game_pk",
    "inning_topbot",
    "estimated_woba_using_speedangle",
    "woba_value",
    "woba_denom",
    "launch_speed",
    "launch_angle",
}
_GAME_COLUMNS = {
    "game_pk",
    "game_date",
    "season_year",
    "home_team_id",
    "away_team_id",
}


def statcast_feature_columns() -> list[str]:
    """Return the stable schema this module emits, in order."""
    columns: list[str] = []
    for side in _SIDES:
        for metric in _METRICS:
            for window in (f"LAST_{SHORT_WINDOW}", f"LAST_{LONG_WINDOW}", "SEASON"):
                stem = f"{FEATURE_PREFIX}{side}_{metric}_{window}_BEFORE"
                columns.extend([f"{stem}_TEAM_HOME", f"{stem}_TEAM_AWAY"])
    return columns


def _require_columns(
    frame: pd.DataFrame, required: set[str], *, frame_name: str
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _batting_side_lines(statcast_pitches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitches to one offensive line per (game, batting side)."""
    _require_columns(statcast_pitches, _PITCH_COLUMNS, frame_name="statcast pitches")
    pitches = statcast_pitches
    launch_speed = pd.to_numeric(pitches["launch_speed"], errors="coerce")
    launch_angle = pd.to_numeric(pitches["launch_angle"], errors="coerce")
    in_play = launch_speed.notna()
    frame = pd.DataFrame(
        {
            "game_pk": pitches["game_pk"].astype(str),
            "__is_home_batting": pitches["inning_topbot"]
            .astype("string")
            .str.lower()
            .str.startswith("b"),
            "__xwoba": pd.to_numeric(
                pitches["estimated_woba_using_speedangle"], errors="coerce"
            ),
            "__woba_value": pd.to_numeric(pitches["woba_value"], errors="coerce"),
            "__woba_denom": pd.to_numeric(pitches["woba_denom"], errors="coerce"),
            "__hard_hit": launch_speed.ge(HARD_HIT_MPH).where(in_play),
            "__barrel": (
                launch_speed.ge(BARREL_MIN_MPH)
                & launch_angle.between(*BARREL_ANGLE_RANGE)
            ).where(in_play),
        }
    )
    grouped = frame.groupby(["game_pk", "__is_home_batting"], sort=False)
    lines = grouped.agg(
        XWOBA=("__xwoba", "mean"),
        __woba_numerator=("__woba_value", "sum"),
        __woba_denominator=("__woba_denom", "sum"),
        HARD_HIT_PCT=("__hard_hit", "mean"),
        BARREL_PCT=("__barrel", "mean"),
    ).reset_index()
    lines["WOBA"] = lines["__woba_numerator"] / lines["__woba_denominator"].replace(
        0.0, np.nan
    )
    # Positive means the team out-hit its contact quality -- results running
    # ahead of what the batted balls deserved, and therefore likely to regress.
    lines["WOBA_MINUS_XWOBA"] = lines["WOBA"] - lines["XWOBA"]
    return lines.drop(columns=["__woba_numerator", "__woba_denominator"])


def _team_game_lines(lines: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach team identity and pair each offensive line with what was allowed."""
    _require_columns(games, _GAME_COLUMNS, frame_name="games")
    schedule = games[list(_GAME_COLUMNS)].copy()
    schedule["game_pk"] = schedule["game_pk"].astype(str)
    schedule["home_team_id"] = schedule["home_team_id"].astype(str)
    schedule["away_team_id"] = schedule["away_team_id"].astype(str)
    schedule["game_date"] = pd.to_datetime(
        schedule["game_date"], errors="raise"
    ).dt.normalize()

    offense = lines.merge(schedule, on="game_pk", how="inner", validate="many_to_one")
    offense["team_id"] = np.where(
        offense["__is_home_batting"],
        offense["home_team_id"],
        offense["away_team_id"],
    )
    offense["opponent_team_id"] = np.where(
        offense["__is_home_batting"],
        offense["away_team_id"],
        offense["home_team_id"],
    )
    offense = offense.rename(columns={m: f"OFFENSE_{m}" for m in _METRICS})

    # What a team allowed is exactly what its opponent produced in that game.
    allowed = offense[["game_pk", "opponent_team_id"]].copy()
    for metric in _METRICS:
        allowed[f"ALLOWED_{metric}"] = offense[f"OFFENSE_{metric}"]
    allowed = allowed.rename(columns={"opponent_team_id": "team_id"})

    merged = offense.merge(
        allowed, on=["game_pk", "team_id"], how="inner", validate="one_to_one"
    )
    return merged.sort_values(
        ["team_id", "game_date", "game_pk"], kind="mergesort"
    ).reset_index(drop=True)


def _shifted_windows(team_lines: pd.DataFrame) -> pd.DataFrame:
    """Add shifted last-5, last-10 and season-to-date means per team."""
    out = team_lines
    grouped = out.groupby("team_id", sort=False)
    season_grouped = out.groupby(["team_id", "season_year"], sort=False)
    built: dict[str, pd.Series] = {}
    for side in _SIDES:
        for metric in _METRICS:
            source = f"{side}_{metric}"
            stem = f"{FEATURE_PREFIX}{side}_{metric}"
            built[f"{stem}_LAST_{SHORT_WINDOW}_BEFORE"] = grouped[source].transform(
                lambda s: s.shift()
                .rolling(SHORT_WINDOW, min_periods=MIN_SHORT_GAMES)
                .mean()
            )
            built[f"{stem}_LAST_{LONG_WINDOW}_BEFORE"] = grouped[source].transform(
                lambda s: s.shift()
                .rolling(LONG_WINDOW, min_periods=MIN_LONG_GAMES)
                .mean()
            )
            built[f"{stem}_SEASON_BEFORE"] = season_grouped[source].transform(
                lambda s: s.shift().expanding().mean()
            )
    return pd.concat([out, pd.DataFrame(built, index=out.index)], axis=1)


def build_statcast_features(
    statcast_pitches: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Build shifted Statcast team form, pivoted to one row per game."""
    lines = _batting_side_lines(statcast_pitches)
    if lines.empty:
        return pd.DataFrame(columns=["GAME_ID", *statcast_feature_columns()])
    team_lines = _shifted_windows(_team_game_lines(lines, games))

    feature_stems = [
        column
        for column in team_lines.columns
        if column.startswith(FEATURE_PREFIX) and column.endswith("_BEFORE")
    ]
    schedule = games[["game_pk", "home_team_id", "away_team_id"]].copy()
    schedule["game_pk"] = schedule["game_pk"].astype(str)
    keyed = team_lines[["game_pk", "team_id", *feature_stems]]

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
            columns={stem: f"{stem}_{role}" for stem in feature_stems}
        )
        sides.append(side.set_index("game_pk"))

    output = pd.concat(sides, axis=1)
    output.index.name = "GAME_ID"
    output = output.reset_index()
    for column in statcast_feature_columns():
        if column not in output:
            output[column] = np.nan
    return output[["GAME_ID", *statcast_feature_columns()]]


__all__ = [
    "FEATURE_PREFIX",
    "build_statcast_features",
    "statcast_feature_columns",
]
