"""Create the one-row-per-game MLB training dataframe.

This is the MLB counterpart of the NBA project's historical training-data
builder.  Pregame feature partitions stay free of realised results.  This
module is the explicit boundary that joins those features to a scoring sidecar
and derives targets for total runs, home margin, total-line error, and
run-line/spread error.

Outcome columns deliberately remain in the returned dataframe so every target
can be reproduced and games can later be settled.  They are never model
features: callers must form ``X`` from :data:`feature_columns` or drop every
name in :data:`OUTCOME_ONLY_COLUMNS`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from mlb_pred.config.settings import PROJECT_ROOT

DEFAULT_PREGAME_DIR = PROJECT_ROOT / "data" / "features" / "pregame"

TOTAL_LINE_COLUMN = "ODDS_TOTAL_CONSENSUS_LINE_NORMALIZED_MEDIAN"
RUN_LINE_HANDICAP_COLUMN = "ODDS_RUN_LINE_CONSENSUS_HOME_HANDICAP_NORMALIZED_MEDIAN"

# The raw executable close, for measuring residuals against the price a bet
# could actually have been struck at rather than the -110/-110 restatement.
RAW_TOTAL_LINE_COLUMN = "ODDS_TOTAL_CONSENSUS_LINE_RAW_MEDIAN"
RAW_RUN_LINE_HANDICAP_COLUMN = "ODDS_RUN_LINE_CONSENSUS_HOME_HANDICAP_RAW_MEDIAN"

# Canonical targets/outcomes.  HOME_MARGIN is retained as the NBA-compatible
# spelling; RUN_LINE_MARGIN is the native MLB spelling and must match it.
TOTAL_RUNS_COLUMN = "TOTAL_RUNS"
RUN_LINE_MARGIN_COLUMN = "RUN_LINE_MARGIN"
HOME_MARGIN_COLUMN = "HOME_MARGIN"
LINE_ERROR_COLUMN = "LINE_ERROR"
SPREAD_LINE_HOME_COLUMN = "SPREAD_LINE_HOME"
SPREAD_ERROR_COLUMN = "SPREAD_ERROR"
RUNS_HOME_COLUMN = "RUNS_TEAM_HOME"
RUNS_AWAY_COLUMN = "RUNS_TEAM_AWAY"
EXTRA_INNINGS_COLUMN = "IS_EXTRA_INNINGS"

TARGET_COLUMNS: tuple[str, ...] = (
    TOTAL_RUNS_COLUMN,
    RUN_LINE_MARGIN_COLUMN,
    HOME_MARGIN_COLUMN,
    LINE_ERROR_COLUMN,
    SPREAD_ERROR_COLUMN,
)

# Exact matches only.  Historical ``*_BEFORE`` features are allowed to contain
# words such as TOTAL_RUNS or LINE_ERROR in their descriptive names.
OUTCOME_ONLY_COLUMNS: tuple[str, ...] = (
    *TARGET_COLUMNS,
    RUNS_HOME_COLUMN,
    RUNS_AWAY_COLUMN,
    EXTRA_INNINGS_COLUMN,
    # Derived here as ``-home_handicap`` purely so ``SPREAD_ERROR`` can be
    # formed and rechecked.  It is the quantity the spread target is defined
    # against, so it is stored and excluded rather than offered as a feature;
    # the closing handicap itself remains available under its ``ODDS_*`` name.
    SPREAD_LINE_HOME_COLUMN,
)

METADATA_COLUMNS: tuple[str, ...] = (
    "GAME_ID",
    "GAME_DATE",
    "GAME_SEASON_YEAR",
    "GAME_FIRST_PITCH_UTC",
    "GAME_HOME_TEAM_ID",
    "GAME_AWAY_TEAM_ID",
    "GAME_HOME_TEAM_NAME",
    "GAME_AWAY_TEAM_NAME",
)

_PREGAME_REQUIRED = {
    "GAME_ID",
    "GAME_DATE",
    "GAME_SEASON_YEAR",
    "GAME_FIRST_PITCH_UTC",
    "GAME_HOME_TEAM_ID",
    "GAME_AWAY_TEAM_ID",
}
_GAMES_REQUIRED = {
    "game_pk",
    "game_date",
    "season_year",
    "home_team_id",
    "away_team_id",
    "is_final",
    "home_score",
    "away_score",
}


def _require(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _schema_difference(
    expected: list[str], actual: list[str]
) -> tuple[list[str], list[str]]:
    expected_set = set(expected)
    actual_set = set(actual)
    return sorted(expected_set - actual_set), sorted(actual_set - expected_set)


def load_pregame_features(
    input_dir: Path = DEFAULT_PREGAME_DIR,
    *,
    seasons: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Read selected pregame partitions and enforce one stable schema."""
    selected = None if seasons is None else {int(season) for season in seasons}
    paths = sorted(input_dir.glob("pregame_features_*.parquet"))
    if selected is not None:
        paths = [
            path
            for path in paths
            if path.stem.removeprefix("pregame_features_").isdigit()
            and int(path.stem.removeprefix("pregame_features_")) in selected
        ]
    if not paths:
        suffix = "" if selected is None else f" for seasons {sorted(selected)}"
        raise FileNotFoundError(
            f"No pregame feature partitions in {input_dir}{suffix}."
        )

    frames: list[pd.DataFrame] = []
    expected_columns: list[str] | None = None
    for path in paths:
        frame = pd.read_parquet(path)
        _require(frame, _PREGAME_REQUIRED, path.name)
        if expected_columns is None:
            expected_columns = list(frame.columns)
        elif list(frame.columns) != expected_columns:
            missing, extra = _schema_difference(expected_columns, list(frame.columns))
            order_only = not missing and not extra
            raise ValueError(
                f"Pregame feature schema changed in {path.name}: missing={missing[:10]}, "
                f"extra={extra[:10]}, order_only={order_only}. Rebuild all selected "
                "partitions with the same feature code before creating training data."
            )
        frames.append(frame)

    features = pd.concat(frames, ignore_index=True)
    features["GAME_ID"] = features["GAME_ID"].astype(str)
    if features["GAME_ID"].duplicated().any():
        duplicate_ids = features.loc[
            features["GAME_ID"].duplicated(keep=False), "GAME_ID"
        ].head(10)
        raise ValueError(
            "Pregame features are not unique by GAME_ID; examples: "
            f"{duplicate_ids.tolist()}"
        )
    return features


def feature_columns(training_frame: pd.DataFrame) -> list[str]:
    """Return the allow-list of numeric model features, excluding row metadata.

    Non-numeric columns are excluded by dtype rather than by name.  The
    per-book ``*_CLOSE_TIMESTAMP_UTC`` quote times are legitimate pre-game
    facts and stay in the frame, but a tz-aware datetime is not a model
    feature; ``*_CLOSE_MINUTES_BEFORE_START`` already carries that information
    numerically.
    """
    excluded = set(OUTCOME_ONLY_COLUMNS).union(METADATA_COLUMNS)
    return [
        column
        for column in training_frame.columns
        if column not in excluded and is_numeric_dtype(training_frame[column])
    ]


def assert_no_outcome_features(columns: Iterable[str]) -> None:
    """Reject an attempted model feature list containing realised outcomes."""
    leaked = sorted(set(columns).intersection(OUTCOME_ONLY_COLUMNS))
    if leaked:
        raise ValueError(
            f"Outcome-derived columns reached the feature list: {leaked}. Use "
            "feature_columns(training_frame) to build X."
        )


def _assert_source_has_no_targets(features: pd.DataFrame) -> None:
    leaked = sorted(set(features.columns).intersection(OUTCOME_ONLY_COLUMNS))
    if leaked:
        raise ValueError(
            "Pregame feature input already contains scoring outcomes: "
            f"{leaked}. Targets must be joined only at the training-data boundary."
        )


def _assert_equal(
    expected: pd.Series,
    actual: pd.Series,
    *,
    label: str,
) -> None:
    comparable = expected.notna() & actual.notna()
    mismatch = comparable & ~np.isclose(
        expected, actual, rtol=0.0, atol=1e-10, equal_nan=True
    )
    if mismatch.any():
        sample = pd.DataFrame(
            {"expected": expected.loc[mismatch], "stored": actual.loc[mismatch]}
        ).head()
        raise ValueError(
            f"Stored {label} disagrees with recomputed scores for "
            f"{int(mismatch.sum())} games. First rows:\n{sample.to_string()}"
        )


def build_training_frame(
    pregame_features: pd.DataFrame,
    games: pd.DataFrame,
    *,
    seasons: Iterable[int] | None = None,
    limit_date: str | pd.Timestamp | None = None,
    total_line_column: str = TOTAL_LINE_COLUMN,
    run_line_handicap_column: str = RUN_LINE_HANDICAP_COLUMN,
) -> pd.DataFrame:
    """Join final scores to pregame features and derive all supported targets.

    ``LINE_ERROR`` is ``TOTAL_RUNS - total_line``.  The stored run-line column
    is a home handicap (negative when the home team is favoured), so
    ``SPREAD_LINE_HOME`` is its negation and ``SPREAD_ERROR`` is
    ``RUN_LINE_MARGIN - SPREAD_LINE_HOME``.  Positive spread error therefore
    means the home team beat the run line; zero is a push and is retained.
    """
    _require(pregame_features, _PREGAME_REQUIRED, "pregame features")
    _require(games, _GAMES_REQUIRED, "games")
    _assert_source_has_no_targets(pregame_features)
    missing_lines = [
        column
        for column in (total_line_column, run_line_handicap_column)
        if column not in pregame_features
    ]
    if missing_lines:
        raise ValueError(f"Pregame features are missing target lines: {missing_lines}")

    features = pregame_features.copy()
    features["GAME_ID"] = features["GAME_ID"].astype(str)
    if features["GAME_ID"].duplicated().any():
        raise ValueError("Pregame features are not unique by GAME_ID.")

    scoring = games.copy()
    scoring["game_pk"] = scoring["game_pk"].astype(str)
    if scoring["game_pk"].duplicated().any():
        raise ValueError("Games are not unique by game_pk.")
    scoring["game_date"] = pd.to_datetime(
        scoring["game_date"], errors="raise"
    ).dt.normalize()
    home_runs = pd.to_numeric(scoring["home_score"], errors="coerce")
    away_runs = pd.to_numeric(scoring["away_score"], errors="coerce")
    scoring[RUNS_HOME_COLUMN] = home_runs
    scoring[RUNS_AWAY_COLUMN] = away_runs
    scoring[TOTAL_RUNS_COLUMN] = home_runs + away_runs
    scoring[RUN_LINE_MARGIN_COLUMN] = home_runs - away_runs
    scoring[HOME_MARGIN_COLUMN] = scoring[RUN_LINE_MARGIN_COLUMN]
    scoring[EXTRA_INNINGS_COLUMN] = scoring.get(
        "extra_innings", pd.Series(False, index=scoring.index)
    ).astype("boolean")

    if "total_runs" in scoring:
        _assert_equal(
            scoring[TOTAL_RUNS_COLUMN],
            pd.to_numeric(scoring["total_runs"], errors="coerce"),
            label="total_runs",
        )
    if "run_line_margin" in scoring:
        _assert_equal(
            scoring[RUN_LINE_MARGIN_COLUMN],
            pd.to_numeric(scoring["run_line_margin"], errors="coerce"),
            label="run_line_margin",
        )

    score_columns = [
        "game_pk",
        "game_date",
        "season_year",
        "home_team_id",
        "away_team_id",
        "is_final",
        RUNS_HOME_COLUMN,
        RUNS_AWAY_COLUMN,
        TOTAL_RUNS_COLUMN,
        RUN_LINE_MARGIN_COLUMN,
        HOME_MARGIN_COLUMN,
        EXTRA_INNINGS_COLUMN,
    ]
    scoring = scoring.loc[
        scoring["is_final"].fillna(False).astype(bool)
        & scoring[RUNS_HOME_COLUMN].notna()
        & scoring[RUNS_AWAY_COLUMN].notna(),
        score_columns,
    ].rename(columns={"game_pk": "GAME_ID"})

    frame = features.merge(scoring, on="GAME_ID", how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError("No completed games overlap the pregame feature partitions.")

    feature_date = pd.to_datetime(frame["GAME_DATE"], errors="raise").dt.normalize()
    if not feature_date.eq(frame["game_date"]).all():
        raise ValueError("GAME_DATE disagrees with the games scoring sidecar.")
    if (
        not pd.to_numeric(frame["GAME_SEASON_YEAR"], errors="coerce")
        .eq(pd.to_numeric(frame["season_year"], errors="coerce"))
        .all()
    ):
        raise ValueError("GAME_SEASON_YEAR disagrees with the games scoring sidecar.")
    for feature_column, scoring_column in (
        ("GAME_HOME_TEAM_ID", "home_team_id"),
        ("GAME_AWAY_TEAM_ID", "away_team_id"),
    ):
        if (
            not frame[feature_column]
            .astype(str)
            .eq(frame[scoring_column].astype(str))
            .all()
        ):
            raise ValueError(
                f"{feature_column} disagrees with {scoring_column} in games."
            )

    selected_seasons = None if seasons is None else {int(season) for season in seasons}
    if selected_seasons is not None:
        frame = frame.loc[
            pd.to_numeric(frame["GAME_SEASON_YEAR"], errors="coerce")
            .astype("Int64")
            .isin(selected_seasons)
        ].copy()
    if limit_date is not None:
        cutoff = pd.to_datetime(limit_date, errors="raise").normalize()
        frame = frame.loc[feature_date.loc[frame.index].le(cutoff)].copy()
    if frame.empty:
        raise ValueError("No completed training rows remain after date/season filters.")

    total_line = pd.to_numeric(frame[total_line_column], errors="coerce")
    home_handicap = pd.to_numeric(frame[run_line_handicap_column], errors="coerce")
    frame[LINE_ERROR_COLUMN] = frame[TOTAL_RUNS_COLUMN] - total_line
    frame[SPREAD_LINE_HOME_COLUMN] = -home_handicap
    frame[SPREAD_ERROR_COLUMN] = (
        frame[RUN_LINE_MARGIN_COLUMN] - frame[SPREAD_LINE_HOME_COLUMN]
    )

    # Recheck the formulas at the output boundary.  This intentionally retains
    # pushes and permits missing residuals where that market was not quoted.
    _assert_equal(
        frame[TOTAL_RUNS_COLUMN] - total_line,
        frame[LINE_ERROR_COLUMN],
        label=LINE_ERROR_COLUMN,
    )
    _assert_equal(
        frame[RUN_LINE_MARGIN_COLUMN] - frame[SPREAD_LINE_HOME_COLUMN],
        frame[SPREAD_ERROR_COLUMN],
        label=SPREAD_ERROR_COLUMN,
    )

    frame = frame.drop(
        columns=[
            "game_date",
            "season_year",
            "home_team_id",
            "away_team_id",
            "is_final",
        ]
    )
    outcome_order = [
        RUNS_HOME_COLUMN,
        RUNS_AWAY_COLUMN,
        TOTAL_RUNS_COLUMN,
        RUN_LINE_MARGIN_COLUMN,
        HOME_MARGIN_COLUMN,
        LINE_ERROR_COLUMN,
        SPREAD_LINE_HOME_COLUMN,
        SPREAD_ERROR_COLUMN,
        EXTRA_INNINGS_COLUMN,
    ]
    feature_order = [column for column in frame if column not in outcome_order]
    frame = frame[[*feature_order, *outcome_order]]
    frame = frame.sort_values(
        ["GAME_DATE", "GAME_FIRST_PITCH_UTC", "GAME_ID"], kind="mergesort"
    ).reset_index(drop=True)
    assert_no_outcome_features(feature_columns(frame))
    return frame


__all__ = [
    "DEFAULT_PREGAME_DIR",
    "EXTRA_INNINGS_COLUMN",
    "HOME_MARGIN_COLUMN",
    "LINE_ERROR_COLUMN",
    "METADATA_COLUMNS",
    "OUTCOME_ONLY_COLUMNS",
    "RAW_RUN_LINE_HANDICAP_COLUMN",
    "RAW_TOTAL_LINE_COLUMN",
    "RUN_LINE_MARGIN_COLUMN",
    "SPREAD_ERROR_COLUMN",
    "SPREAD_LINE_HOME_COLUMN",
    "TARGET_COLUMNS",
    "TOTAL_RUNS_COLUMN",
    "assert_no_outcome_features",
    "build_training_frame",
    "feature_columns",
    "load_pregame_features",
]
