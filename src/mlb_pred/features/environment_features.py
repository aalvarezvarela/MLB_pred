"""Run-environment features: the ballpark and the home-plate umpire.

Both families describe the conditions a game is played under rather than the
teams playing it, and both are already in the store and previously unused by
any feature module. ``venues`` was read only for travel distance; ``umpires``
was ingested and never read at all.

Two kinds of column are produced:

* **Static park geometry** -- elevation, roof, turf, and the outfield fence
  distances that ``venues`` carries. These are properties of the stadium, not
  of any game, so they need no temporal treatment.
* **Expanding run environment** -- how many runs have historically been scored
  at this venue, and in games worked by this plate umpire, *strictly before*
  the current game. Each is expressed as a ratio to the league's expanding mean
  over the same prior games, so a scoring-era shift moves numerator and
  denominator together.

The expanding statistics are shifted within an ordering by ``game_date`` and
then gated so that games sharing a date cannot enter one another's history.
That is the same doubleheader rule the rolling team features use, and it also
prevents the second game of a day from seeing the first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARK_PREFIX = "PARK_"
UMPIRE_PREFIX = "UMPIRE_"

# Neutral fallbacks. A run *factor* is a ratio to the league mean, so 1.0 is
# "no information"; a count of prior games is genuinely zero.
NEUTRAL_RUN_FACTOR = 1.0

_GAME_COLUMNS = {
    "game_pk",
    "game_date",
    "venue_id",
    "is_final",
    "home_score",
    "away_score",
}
_VENUE_COLUMNS = {
    "venue_id",
    "season",
    "elevation_ft",
    "roof_type",
    "turf_type",
    "capacity",
    "left_line",
    "left",
    "left_center",
    "center",
    "right_center",
    "right",
    "right_line",
}
_UMPIRE_COLUMNS = {"game_pk", "official_id", "is_home_plate"}

_FENCE_COLUMNS = (
    "left_line",
    "left",
    "left_center",
    "center",
    "right_center",
    "right",
    "right_line",
)

PARK_FEATURE_COLUMNS: tuple[str, ...] = (
    f"{PARK_PREFIX}RUN_FACTOR_EXPANDING_BEFORE",
    f"{PARK_PREFIX}RUN_MEAN_EXPANDING_BEFORE",
    f"{PARK_PREFIX}PRIOR_GAMES_BEFORE",
    f"{PARK_PREFIX}ELEVATION_FT_BEFORE",
    f"{PARK_PREFIX}CAPACITY_BEFORE",
    f"{PARK_PREFIX}FENCE_LEFT_LINE_BEFORE",
    f"{PARK_PREFIX}FENCE_CENTER_BEFORE",
    f"{PARK_PREFIX}FENCE_RIGHT_LINE_BEFORE",
    f"{PARK_PREFIX}FENCE_MEAN_BEFORE",
    f"{PARK_PREFIX}ROOF_CLOSEABLE_BEFORE",
    f"{PARK_PREFIX}TURF_ARTIFICIAL_BEFORE",
)

UMPIRE_FEATURE_COLUMNS: tuple[str, ...] = (
    f"{UMPIRE_PREFIX}RUN_FACTOR_EXPANDING_BEFORE",
    f"{UMPIRE_PREFIX}RUN_MEAN_EXPANDING_BEFORE",
    f"{UMPIRE_PREFIX}RUN_STD_EXPANDING_BEFORE",
    f"{UMPIRE_PREFIX}PRIOR_GAMES_BEFORE",
    f"{UMPIRE_PREFIX}HAS_HISTORY_BEFORE",
)


def _require_columns(
    frame: pd.DataFrame, required: set[str], *, frame_name: str
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _completed_game_history(games: pd.DataFrame) -> pd.DataFrame:
    """Order completed games by date and attach realised total runs.

    Only completed games enter a history. The result of the current game is
    never read for its own row: every statistic below is shifted first and then
    date-gated.
    """
    _require_columns(games, _GAME_COLUMNS, frame_name="games")
    history = games.loc[games["is_final"].fillna(False).astype(bool)].copy()
    history["game_pk"] = history["game_pk"].astype(str)
    history["venue_id"] = history["venue_id"].astype(str)
    history["game_date"] = pd.to_datetime(
        history["game_date"], errors="raise"
    ).dt.normalize()
    history["__total_runs"] = pd.to_numeric(
        history["home_score"], errors="coerce"
    ) + pd.to_numeric(history["away_score"], errors="coerce")
    history = history.loc[history["__total_runs"].notna()]
    return history.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(
        drop=True
    )


def _date_gate(frame: pd.DataFrame, values: pd.Series, *, key: str) -> pd.Series:
    """Give every same-(key, date) group the value held before that date began.

    Without this, the second game of a doubleheader -- or two games at the same
    venue on one day -- would see each other's results.
    """
    if frame.empty:
        return pd.Series(index=frame.index, dtype="float64")
    codes = frame.groupby([key, "game_date"], sort=False, dropna=False).ngroup()
    codes = codes.to_numpy()
    first_positions = np.flatnonzero(~pd.Series(codes).duplicated().to_numpy())
    first_by_code = np.empty(int(codes.max()) + 1, dtype=np.int64)
    first_by_code[codes[first_positions]] = first_positions
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    return pd.Series(numeric[first_by_code[codes]], index=frame.index)


def _expanding_environment(
    history: pd.DataFrame, key: str, *, prefix: str, with_std: bool
) -> pd.DataFrame:
    """Expanding pre-game run environment for one grouping key."""
    grouped = history.groupby(key, sort=False)["__total_runs"]
    league = history["__total_runs"].expanding().mean().shift(1)
    group_mean = grouped.transform(lambda s: s.shift(1).expanding().mean())
    prior_games = grouped.cumcount().astype("float64")

    values = {
        f"{prefix}RUN_MEAN_EXPANDING_BEFORE": _date_gate(history, group_mean, key=key),
        f"{prefix}PRIOR_GAMES_BEFORE": _date_gate(history, prior_games, key=key),
    }
    gated_league = _date_gate(history, league, key=key)
    values[f"{prefix}RUN_FACTOR_EXPANDING_BEFORE"] = (
        values[f"{prefix}RUN_MEAN_EXPANDING_BEFORE"] / gated_league.replace(0.0, np.nan)
    ).fillna(NEUTRAL_RUN_FACTOR)
    if with_std:
        values[f"{prefix}RUN_STD_EXPANDING_BEFORE"] = _date_gate(
            history,
            grouped.transform(lambda s: s.shift(1).expanding().std(ddof=0)),
            key=key,
        )
    output = pd.DataFrame(values, index=history.index)
    output.insert(0, "GAME_ID", history["game_pk"].to_numpy())
    return output


def _static_park_columns(venues: pd.DataFrame) -> pd.DataFrame:
    """One row of stadium geometry per venue, from its most recent season."""
    _require_columns(venues, _VENUE_COLUMNS, frame_name="venues")
    ordered = venues.copy()
    ordered["venue_id"] = ordered["venue_id"].astype(str)
    latest = ordered.sort_values("season").groupby("venue_id", sort=False).last()
    fences = latest[list(_FENCE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    roof = latest["roof_type"].astype("string").str.lower().fillna("")
    turf = latest["turf_type"].astype("string").str.lower().fillna("")
    static = pd.DataFrame(
        {
            f"{PARK_PREFIX}ELEVATION_FT_BEFORE": pd.to_numeric(
                latest["elevation_ft"], errors="coerce"
            ),
            f"{PARK_PREFIX}CAPACITY_BEFORE": pd.to_numeric(
                latest["capacity"], errors="coerce"
            ),
            f"{PARK_PREFIX}FENCE_LEFT_LINE_BEFORE": fences["left_line"],
            f"{PARK_PREFIX}FENCE_CENTER_BEFORE": fences["center"],
            f"{PARK_PREFIX}FENCE_RIGHT_LINE_BEFORE": fences["right_line"],
            f"{PARK_PREFIX}FENCE_MEAN_BEFORE": fences.mean(axis=1),
            # A closeable roof is the part of "roof" that changes conditions;
            # an open-air park and a park with the roof open behave alike.
            f"{PARK_PREFIX}ROOF_CLOSEABLE_BEFORE": roof.str.contains(
                "dome|retract|closed", regex=True, na=False
            ).astype("float64"),
            f"{PARK_PREFIX}TURF_ARTIFICIAL_BEFORE": turf.str.contains(
                "artificial|turf", regex=True, na=False
            ).astype("float64"),
        }
    )
    static.index.name = "venue_id"
    return static.reset_index()


def build_park_features(games: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    """Build static park geometry plus the expanding venue run environment."""
    history = _completed_game_history(games)
    expanding = _expanding_environment(
        history, "venue_id", prefix=PARK_PREFIX, with_std=False
    )
    expanding["venue_id"] = history["venue_id"].to_numpy()
    output = expanding.merge(
        _static_park_columns(venues), on="venue_id", how="left", validate="many_to_one"
    ).drop(columns="venue_id")
    return output[["GAME_ID", *PARK_FEATURE_COLUMNS]]


def build_umpire_features(games: pd.DataFrame, umpires: pd.DataFrame) -> pd.DataFrame:
    """Build the expanding run environment of each game's home-plate umpire."""
    _require_columns(umpires, _UMPIRE_COLUMNS, frame_name="umpires")
    plate = umpires.loc[umpires["is_home_plate"].fillna(False).astype(bool)].copy()
    plate["game_pk"] = plate["game_pk"].astype(str)
    plate["official_id"] = plate["official_id"].astype(str)
    if plate.duplicated("game_pk").any():
        raise ValueError("umpires has more than one home-plate official per game.")

    history = _completed_game_history(games).merge(
        plate[["game_pk", "official_id"]],
        left_on="game_pk",
        right_on="game_pk",
        how="inner",
        validate="one_to_one",
    )
    if history.empty:
        return pd.DataFrame(columns=["GAME_ID", *UMPIRE_FEATURE_COLUMNS])
    output = _expanding_environment(
        history, "official_id", prefix=UMPIRE_PREFIX, with_std=True
    )
    output[f"{UMPIRE_PREFIX}HAS_HISTORY_BEFORE"] = (
        output[f"{UMPIRE_PREFIX}PRIOR_GAMES_BEFORE"].gt(0).astype("float64")
    )
    return output[["GAME_ID", *UMPIRE_FEATURE_COLUMNS]]


__all__ = [
    "PARK_FEATURE_COLUMNS",
    "PARK_PREFIX",
    "UMPIRE_FEATURE_COLUMNS",
    "UMPIRE_PREFIX",
    "build_park_features",
    "build_umpire_features",
]
