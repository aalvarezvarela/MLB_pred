"""Player availability features built from retrospective MLB data.

The historical lineup is the final lineup, not a timestamped pregame snapshot.
It is therefore a useful backtest proxy, but a potentially optimistic one.  The
transaction path is point-in-time constrained by ``known_date`` and the player
statistics and empirical effects exclude the target game and every other game
on the same calendar date.

Pitchers deliberately follow a separate path.  An announced starter receives
historical quality features and pitchers on the injured list contribute depth
counts.  A reliever is never called absent merely because he did not appear.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

REGULAR_SEASON_GAME_TYPE = "R"
PLAYER_EWMA_HALF_LIFE = 10.0
RECENT_LINEUP_GAMES = 20
EXPECTED_HITTERS = 9
EFFECT_SHRINKAGE = 10.0

# Neutral values are used only after current-season history and the immediately
# preceding regular season are both unavailable.  They are deliberately
# explicit so a cold-start value cannot silently change with the input sample.
HITTER_NEUTRALS: dict[str, float] = {
    "PA": 0.0,
    "OBP": 0.320,
    "SLG": 0.400,
    "HR_PER_PA": 0.030,
    "BB_PCT": 0.080,
    "K_PCT": 0.220,
    "TOTAL_BASES_PER_PA": 0.360,
    "BASERUNNERS_PER_PA": 0.320,
}

PITCHER_NEUTRALS: dict[str, float] = {
    "OUTS": 0.0,
    "BF": 0.0,
    "ERA_PER_9": 4.50,
    "WHIP": 1.30,
    "K_PCT": 0.220,
    "BB_PCT": 0.080,
    "HR_PER_BF": 0.030,
    "PITCHES_PER_BF": 3.90,
}

# Primary metrics carry the sum and the top rank, because a single star
# dominates them.  Secondary metrics describe the shape of a plate appearance
# rather than its value, so they are emitted only as MEAN: a sum over rate
# statistics has no interpretation and a per-rank ordering of them is not a
# quality ordering.
HITTER_OUTPUT_METRICS = ("PA", "OBP", "SLG", "HR_PER_PA")
HITTER_SECONDARY_METRICS = ("K_PCT", "BASERUNNERS_PER_PA")
AVAILABILITY_STATES = ("AVAILABLE", "OBSERVED_ABSENT", "INJURED")

# Which availability quantities are worth a home-minus-away contrast. The rest
# keep only their two side columns.
DIFF_FEATURES: tuple[str, ...] = (
    "TEAM_AVAILABILITY_HITTER_N_INJURED_BEFORE",
    "TEAM_AVAILABILITY_HITTER_N_OBSERVED_ABSENT_BEFORE",
    "TEAM_AVAILABILITY_HITTER_TOTAL_INJURED_PA_BEFORE",
    "TEAM_AVAILABILITY_STARTING_PITCHER_ERA_PER_9_BEFORE",
    "TEAM_AVAILABILITY_STARTING_PITCHER_K_PCT_BEFORE",
    "TEAM_AVAILABILITY_STARTING_PITCHER_WHIP_BEFORE",
)
EFFECT_OUTCOMES = ("TEAM_RUNS", "TOTAL_LINE_ERROR")

# Rank depth for the per-player ordering.  A lineup is nine hitters, so the
# single best available bat and the single biggest absence carry nearly all of
# the signal a rank ordering can express; TOP2..TOP5 were a long tail.
TOP_RANKS = 1
# Only the sum. MEAN is TOTAL divided by ``N_<state>``, which is already a
# column; MAX is *identical* to TOP1 by construction -- verified equal across
# all 24 column pairs over 17,638 games -- so it was a deterministic duplicate.
# STREAK keeps its own MAX because it has no per-rank ordering.
HITTER_AGGREGATES = ("TOTAL",)

_GAME_COLUMNS = {
    "game_pk",
    "season_year",
    "game_date",
    "game_type",
    "home_team_id",
    "away_team_id",
}
_LINEUP_COLUMNS = {"game_pk", "team_id", "player_id"}
_BATTER_COLUMNS = {
    "game_pk",
    "team_id",
    "player_id",
    "season_year",
    "game_date",
    "plate_appearances",
    "at_bats",
    "hits",
    "home_runs",
    "walks",
    "strikeouts",
    "hit_by_pitch",
    "sac_flies",
    "total_bases",
}
_PITCHER_COLUMNS = {
    "game_pk",
    "team_id",
    "player_id",
    "season_year",
    "game_date",
    "is_starter",
    "outs_recorded",
    "batters_faced",
    "earned_runs",
    "hits_allowed",
    "walks_allowed",
    "home_runs_allowed",
    "strikeouts",
    "pitches_thrown",
}


def _ids(values: Iterable[Any]) -> set[str]:
    return {str(value) for value in values if pd.notna(value) and str(value)}


def _number(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0.0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan)


def _require(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _prepare_hitter_games(
    batter_games: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    _require(batter_games, _BATTER_COLUMNS, "batter games")
    out = batter_games.copy()
    out["game_pk"] = out["game_pk"].astype(str)
    out["player_id"] = out["player_id"].astype(str)
    out["team_id"] = out["team_id"].astype(str)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="raise").dt.normalize()
    game_types = games.assign(game_pk=games["game_pk"].astype(str)).set_index(
        "game_pk"
    )["game_type"]
    out["game_type"] = out["game_pk"].map(game_types)

    pa = _number(out, "plate_appearances")
    ab = _number(out, "at_bats")
    hits = _number(out, "hits")
    walks = _number(out, "walks")
    hbp = _number(out, "hit_by_pitch")
    sac_flies = _number(out, "sac_flies")
    out["PA"] = pa
    out["OBP"] = _safe_ratio(hits + walks + hbp, ab + walks + hbp + sac_flies)
    out["SLG"] = _safe_ratio(_number(out, "total_bases"), ab)
    out["HR_PER_PA"] = _safe_ratio(_number(out, "home_runs"), pa)
    out["BB_PCT"] = _safe_ratio(walks, pa)
    out["K_PCT"] = _safe_ratio(_number(out, "strikeouts"), pa)
    out["TOTAL_BASES_PER_PA"] = _safe_ratio(_number(out, "total_bases"), pa)
    out["BASERUNNERS_PER_PA"] = _safe_ratio(hits + walks + hbp, pa)

    # Provider row inclusion changed over time; a row without a PA is not a
    # batting appearance and must not advance a ten-appearance EWMA.
    out = out.loc[pa.gt(0)].copy()
    return out.sort_values(
        ["player_id", "season_year", "game_date", "game_pk"], kind="mergesort"
    ).reset_index(drop=True)


def _prepare_pitcher_games(
    pitcher_appearances: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    _require(pitcher_appearances, _PITCHER_COLUMNS, "pitcher appearances")
    out = pitcher_appearances.copy()
    out["game_pk"] = out["game_pk"].astype(str)
    out["player_id"] = out["player_id"].astype(str)
    out["team_id"] = out["team_id"].astype(str)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="raise").dt.normalize()
    game_types = games.assign(game_pk=games["game_pk"].astype(str)).set_index(
        "game_pk"
    )["game_type"]
    out["game_type"] = out["game_pk"].map(game_types)

    outs = _number(out, "outs_recorded")
    bf = _number(out, "batters_faced")
    walks = _number(out, "walks_allowed")
    out["OUTS"] = outs
    out["BF"] = bf
    out["ERA_PER_9"] = _safe_ratio(_number(out, "earned_runs") * 27.0, outs)
    out["WHIP"] = _safe_ratio((_number(out, "hits_allowed") + walks) * 3.0, outs)
    out["K_PCT"] = _safe_ratio(_number(out, "strikeouts"), bf)
    out["BB_PCT"] = _safe_ratio(walks, bf)
    out["HR_PER_BF"] = _safe_ratio(_number(out, "home_runs_allowed"), bf)
    out["PITCHES_PER_BF"] = _safe_ratio(_number(out, "pitches_thrown"), bf)
    return out.sort_values(
        ["player_id", "season_year", "game_date", "game_pk"], kind="mergesort"
    ).reset_index(drop=True)


@dataclass
class _MetricHistory:
    metrics: tuple[str, ...]
    neutrals: Mapping[str, float]
    # Values are the EWM state *after* all appearances on the stored date.
    by_player_season: dict[
        tuple[str, int], tuple[list[pd.Timestamp], list[dict[str, float]]]
    ] = field(default_factory=dict)
    previous_regular: dict[tuple[str, int], dict[str, float]] = field(
        default_factory=dict
    )

    def get(
        self, player_id: str, season_year: int, game_date: pd.Timestamp
    ) -> tuple[dict[str, float], bool]:
        key = (str(player_id), int(season_year))
        dates, values = self.by_player_season.get(key, ([], []))
        position = bisect_left(dates, pd.Timestamp(game_date)) - 1
        if position >= 0:
            return values[position].copy(), True
        previous = self.previous_regular.get(key)
        if previous is not None:
            return previous.copy(), True
        return dict(self.neutrals), False


def _build_metric_history(
    frame: pd.DataFrame,
    *,
    metrics: Iterable[str],
    neutrals: Mapping[str, float],
) -> _MetricHistory:
    metric_names = tuple(metrics)
    result = _MetricHistory(metric_names, neutrals)
    alpha = 1.0 - np.exp(np.log(0.5) / PLAYER_EWMA_HALF_LIFE)

    for (player_id, season), group in frame.groupby(
        ["player_id", "season_year"], sort=False
    ):
        group = group.sort_values(["game_date", "game_pk"], kind="mergesort")
        ewm = group[list(metric_names)].ewm(alpha=alpha, adjust=False).mean()
        dated = (
            pd.concat(
                [
                    group[["game_date"]].reset_index(drop=True),
                    ewm.reset_index(drop=True),
                ],
                axis=1,
            )
            .groupby("game_date", sort=True, as_index=False)
            .last()
        )
        dates = dated["game_date"].tolist()
        values = [
            {
                metric: (
                    float(row[metric])
                    if pd.notna(row[metric])
                    else float(neutrals[metric])
                )
                for metric in metric_names
            }
            for row in dated.to_dict("records")
        ]
        result.by_player_season[(str(player_id), int(season))] = (dates, values)

        regular = group.loc[group["game_type"].eq(REGULAR_SEASON_GAME_TYPE)]
        if not regular.empty:
            final = (
                regular[list(metric_names)]
                .ewm(alpha=alpha, adjust=False)
                .mean()
                .iloc[-1]
            )
            result.previous_regular[(str(player_id), int(season) + 1)] = {
                metric: (
                    float(final[metric])
                    if pd.notna(final[metric])
                    else float(neutrals[metric])
                )
                for metric in metric_names
            }
    return result


def _history_before_source_games(
    prepared: pd.DataFrame,
    history: _MetricHistory,
    *,
    prefix: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in prepared.itertuples(index=False):
        values, covered = history.get(
            str(row.player_id), int(row.season_year), pd.Timestamp(row.game_date)
        )
        output: dict[str, Any] = {
            "game_pk": str(row.game_pk),
            "team_id": str(row.team_id),
            "player_id": str(row.player_id),
            f"{prefix}_HISTORY_COVERAGE_BEFORE": int(covered),
        }
        output.update(
            {f"{prefix}_{metric}_BEFORE": value for metric, value in values.items()}
        )
        rows.append(output)
    columns = [
        "game_pk",
        "team_id",
        "player_id",
        f"{prefix}_HISTORY_COVERAGE_BEFORE",
        *(f"{prefix}_{metric}_BEFORE" for metric in history.metrics),
    ]
    return pd.DataFrame(rows, columns=columns)


def build_hitter_history(
    batter_games: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    """Return leakage-safe hitter history at each recorded appearance."""
    prepared = _prepare_hitter_games(batter_games, games)
    history = _build_metric_history(
        prepared, metrics=HITTER_NEUTRALS, neutrals=HITTER_NEUTRALS
    )
    return _history_before_source_games(
        prepared, history, prefix="PLAYER_HISTORY_HITTER"
    )


def build_pitcher_history(
    pitcher_appearances: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    """Return leakage-safe pitcher history at each recorded appearance."""
    prepared = _prepare_pitcher_games(pitcher_appearances, games)
    history = _build_metric_history(
        prepared, metrics=PITCHER_NEUTRALS, neutrals=PITCHER_NEUTRALS
    )
    return _history_before_source_games(
        prepared, history, prefix="PLAYER_HISTORY_PITCHER"
    )


def classify_il_transaction(transaction: Mapping[str, Any] | pd.Series) -> str | None:
    """Classify an IL record as ``open``, ``maintain`` or ``close``.

    ``is_injury_related`` cannot determine direction because an activation also
    says "injured list".  Disabled-list wording is retained for pre-2019 data.
    """
    parts = []
    for column in ("type_desc", "description"):
        value = transaction.get(column)
        parts.append("" if value is None or pd.isna(value) else str(value))
    text = " ".join(parts).lower()
    if not any(marker in text for marker in ("injured list", "disabled list")):
        return None
    if any(
        marker in text
        for marker in (
            "activated",
            "reinstated",
            "returned from",
            "removed from",
        )
    ):
        return "close"
    if any(marker in text for marker in ("transferred", "transferred from")):
        return "maintain"
    if any(
        marker in text
        for marker in ("placed", "added", "selected to", "transferred to")
    ):
        return "open"
    return None


@dataclass
class _EffectCounts:
    available_count: int = 0
    absent_count: int = 0
    available_sum: float = 0.0
    absent_sum: float = 0.0

    def add(self, available: bool, value: float) -> None:
        if available:
            self.available_count += 1
            self.available_sum += value
        else:
            self.absent_count += 1
            self.absent_sum += value

    def effect(self) -> tuple[float, int]:
        n_eff = min(self.available_count, self.absent_count)
        if n_eff == 0:
            return 0.0, 0
        raw = self.available_sum / self.available_count
        raw -= self.absent_sum / self.absent_count
        return raw * n_eff / (n_eff + EFFECT_SHRINKAGE), n_eff


def _empty_feature_row() -> dict[str, float]:
    row: dict[str, float] = {}
    prefix = "TEAM_AVAILABILITY_HITTER"
    for state in AVAILABILITY_STATES:
        row[f"{prefix}_N_{state}_BEFORE"] = 0.0
        for metric in HITTER_OUTPUT_METRICS:
            for aggregate in HITTER_AGGREGATES:
                row[f"{prefix}_{aggregate}_{state}_{metric}_BEFORE"] = 0.0
            for rank in range(1, TOP_RANKS + 1):
                row[f"{prefix}_TOP{rank}_{state}_{metric}_BEFORE"] = 0.0
        for metric in HITTER_SECONDARY_METRICS:
            row[f"{prefix}_MEAN_{state}_{metric}_BEFORE"] = 0.0
        if state != "AVAILABLE":
            row[f"{prefix}_TOTAL_{state}_STREAK_BEFORE"] = 0.0
            row[f"{prefix}_MAX_{state}_STREAK_BEFORE"] = 0.0

    for group in ("AVAILABLE", "ABSENT"):
        for outcome in EFFECT_OUTCOMES:
            row[f"{prefix}_{group}_{outcome}_EFFECT_MEAN_BEFORE"] = 0.0
            row[f"{prefix}_{group}_{outcome}_EFFECT_MAX_ABS_BEFORE"] = 0.0
            row[f"{prefix}_{group}_{outcome}_EFFECT_SAMPLE_SIZE_BEFORE"] = 0.0

    row.update(
        {
            f"{prefix}_N_LINEUP_STARTERS_BEFORE": 0.0,
            f"{prefix}_N_EXPECTED_CANDIDATES_BEFORE": 0.0,
            f"{prefix}_N_EXPECTED_WITH_HISTORY_BEFORE": 0.0,
            f"{prefix}_LINEUP_COVERAGE_BEFORE": 0.0,
            f"{prefix}_EXPECTED_COVERAGE_BEFORE": 0.0,
            "TEAM_AVAILABILITY_TRANSACTION_COVERAGE_BEFORE": 0.0,
            "TEAM_AVAILABILITY_PITCHER_N_INJURED_BEFORE": 0.0,
            "TEAM_AVAILABILITY_STARTING_PITCHER_HAS_ANNOUNCED_BEFORE": 0.0,
            "TEAM_AVAILABILITY_STARTING_PITCHER_HISTORY_COVERAGE_BEFORE": 0.0,
            # A starter-scratch flag is deliberately absent.  The only
            # leakage-safe evidence would be a change between an archived
            # pre-game probable and the announced starter, and the snapshot
            # archive does not yet reach back over the backfill.  Comparing the
            # schedule's probable against who actually pitched is post-game
            # information, so no scratch column is emitted at all rather than
            # one that is constantly zero.
        }
    )
    for metric in PITCHER_NEUTRALS:
        row[f"TEAM_AVAILABILITY_STARTING_PITCHER_{metric}_BEFORE"] = float(
            PITCHER_NEUTRALS[metric]
        )
    return row


def _aggregate_players(
    row: dict[str, float],
    state: str,
    players: list[dict[str, Any]],
) -> None:
    prefix = "TEAM_AVAILABILITY_HITTER"
    row[f"{prefix}_N_{state}_BEFORE"] = float(len(players))
    for metric in HITTER_OUTPUT_METRICS:
        values = [float(player["metrics"][metric]) for player in players]
        row[f"{prefix}_TOTAL_{state}_{metric}_BEFORE"] = float(sum(values))
        ordered = sorted(values, reverse=True)
        for rank in range(1, TOP_RANKS + 1):
            row[f"{prefix}_TOP{rank}_{state}_{metric}_BEFORE"] = (
                ordered[rank - 1] if rank <= len(ordered) else 0.0
            )
    for metric in HITTER_SECONDARY_METRICS:
        values = [float(player["metrics"][metric]) for player in players]
        row[f"{prefix}_MEAN_{state}_{metric}_BEFORE"] = (
            float(np.mean(values)) if values else 0.0
        )
    if state != "AVAILABLE":
        streaks = [float(player["streak"]) for player in players]
        row[f"{prefix}_TOTAL_{state}_STREAK_BEFORE"] = float(sum(streaks))
        row[f"{prefix}_MAX_{state}_STREAK_BEFORE"] = max(streaks, default=0.0)


def _aggregate_effects(
    row: dict[str, float],
    group: str,
    players: list[dict[str, Any]],
    effect_history: Mapping[tuple[str, int, str], _EffectCounts],
    season: int,
) -> None:
    prefix = "TEAM_AVAILABILITY_HITTER"
    selected = sorted(players, key=lambda item: item["role"], reverse=True)
    selected = selected[: 5 if group == "AVAILABLE" else 4]
    for outcome in EFFECT_OUTCOMES:
        effects: list[float] = []
        samples: list[int] = []
        for player in selected:
            counts = _EffectCounts()
            for effect_season in (season - 1, season):
                historical = effect_history.get(
                    (str(player["player_id"]), effect_season, outcome)
                )
                if historical is not None:
                    counts.available_count += historical.available_count
                    counts.absent_count += historical.absent_count
                    counts.available_sum += historical.available_sum
                    counts.absent_sum += historical.absent_sum
            effect, sample = counts.effect()
            effects.append(effect)
            samples.append(sample)
        row[f"{prefix}_{group}_{outcome}_EFFECT_MEAN_BEFORE"] = (
            float(np.mean(effects)) if effects else 0.0
        )
        row[f"{prefix}_{group}_{outcome}_EFFECT_MAX_ABS_BEFORE"] = max(
            (abs(value) for value in effects), default=0.0
        )
        row[f"{prefix}_{group}_{outcome}_EFFECT_SAMPLE_SIZE_BEFORE"] = float(
            sum(samples)
        )


def _wide_team_features(team_rows: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        column
        for column in team_rows.columns
        if column.startswith("TEAM_AVAILABILITY_") and "_BEFORE" in column
    ]
    counts = team_rows.groupby("game_pk").agg(
        rows=("team_id", "size"), homes=("home", "sum")
    )
    invalid = counts.loc[(counts["rows"] != 2) | (counts["homes"] != 1)]
    if not invalid.empty:
        raise ValueError(
            "Availability targets require one home and one away row per game: "
            f"{invalid.head()}"
        )
    home = team_rows.loc[team_rows["home"], ["game_pk", *feature_columns]].rename(
        columns={column: f"{column}_TEAM_HOME" for column in feature_columns}
    )
    away = team_rows.loc[~team_rows["home"], ["game_pk", *feature_columns]].rename(
        columns={column: f"{column}_TEAM_AWAY" for column in feature_columns}
    )
    wide = home.merge(away, on="game_pk", how="inner", validate="one_to_one")
    wide = wide.rename(columns={"game_pk": "GAME_ID"})
    # HOME - AWAY is exactly recoverable from the two side columns, so only the
    # contrasts that are genuinely read as a contrast are materialised. Sixty-six
    # availability diffs were sixty-six columns of no added rank.
    missing = sorted(set(DIFF_FEATURES).difference(feature_columns))
    if missing:
        raise ValueError(f"DIFF_FEATURES names unknown availability columns: {missing}")
    differences = {
        column.removesuffix("_BEFORE")
        + "_DIFF_BEFORE": wide[f"{column}_TEAM_HOME"]
        - wide[f"{column}_TEAM_AWAY"]
        for column in DIFF_FEATURES
    }
    return pd.concat([wide, pd.DataFrame(differences, index=wide.index)], axis=1)


def build_availability_features(
    games: pd.DataFrame,
    lineups: pd.DataFrame,
    batter_games: pd.DataFrame,
    pitcher_appearances: pd.DataFrame,
    transactions: pd.DataFrame,
    closing_lines: pd.DataFrame,
    *,
    target_game_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build one wide, model-ready availability row per target ``GAME_ID``.

    Player identifiers and names are used only as temporary state keys.  No raw
    player identity or realised outcome is returned.
    """
    _require(games, _GAME_COLUMNS, "games")
    _require(lineups, _LINEUP_COLUMNS, "lineups")
    hitter_games = _prepare_hitter_games(batter_games, games)
    pitcher_games = _prepare_pitcher_games(pitcher_appearances, games)
    hitter_history = _build_metric_history(
        hitter_games, metrics=HITTER_NEUTRALS, neutrals=HITTER_NEUTRALS
    )
    pitcher_history = _build_metric_history(
        pitcher_games, metrics=PITCHER_NEUTRALS, neutrals=PITCHER_NEUTRALS
    )

    schedule = games.copy()
    schedule["game_pk"] = schedule["game_pk"].astype(str)
    schedule["game_date"] = pd.to_datetime(
        schedule["game_date"], errors="raise"
    ).dt.normalize()
    if "first_pitch_utc" in schedule:
        schedule["first_pitch_utc"] = pd.to_datetime(
            schedule["first_pitch_utc"], utc=True, errors="coerce"
        )
    else:
        schedule["first_pitch_utc"] = pd.NaT
    schedule = schedule.sort_values(
        ["game_date", "first_pitch_utc", "game_pk"], kind="mergesort"
    )
    targets = (
        _ids(target_game_ids)
        if target_game_ids is not None
        else _ids(schedule["game_pk"])
    )
    unknown_targets = sorted(targets.difference(_ids(schedule["game_pk"])))
    if unknown_targets:
        raise ValueError(
            f"Availability target games are absent from games: {unknown_targets}"
        )

    lineup_frame = lineups.copy()
    lineup_frame["game_pk"] = lineup_frame["game_pk"].astype(str)
    lineup_frame["team_id"] = lineup_frame["team_id"].astype(str)
    lineup_frame["player_id"] = lineup_frame["player_id"].astype(str)
    lineup_map = {
        (game_pk, team_id): _ids(group["player_id"])
        for (game_pk, team_id), group in lineup_frame.groupby(
            ["game_pk", "team_id"], sort=False
        )
    }
    lineup_pitchers = {
        (game_pk, team_id): _ids(
            group.loc[
                group.get("position_code", pd.Series(index=group.index, dtype=object))
                .astype(str)
                .eq("1"),
                "player_id",
            ]
        )
        for (game_pk, team_id), group in lineup_frame.groupby(
            ["game_pk", "team_id"], sort=False
        )
    }
    # A per-game appearance map is deliberately NOT built. Who came off the
    # bench is decided by the game being played, so it can never be read while
    # featurising that game; ``has_batted`` below carries the strictly prior
    # "is this player a hitter at all" evidence instead.

    batter_appearances_by_date = {
        date: group
        for date, group in batter_games.assign(
            game_date=pd.to_datetime(batter_games["game_date"]).dt.normalize(),
            game_pk=batter_games["game_pk"].astype(str),
            team_id=batter_games["team_id"].astype(str),
            player_id=batter_games["player_id"].astype(str),
        ).groupby("game_date", sort=False)
    }
    pitcher_appearances_by_date = {
        date: group
        for date, group in pitcher_appearances.assign(
            game_date=pd.to_datetime(pitcher_appearances["game_date"]).dt.normalize(),
            game_pk=pitcher_appearances["game_pk"].astype(str),
            team_id=pitcher_appearances["team_id"].astype(str),
            player_id=pitcher_appearances["player_id"].astype(str),
        ).groupby("game_date", sort=False)
    }

    transaction_coverage = not transactions.empty
    transaction_records: list[dict[str, Any]] = []
    transaction_min_date: pd.Timestamp | None = None
    transaction_max_date: pd.Timestamp | None = None
    if transaction_coverage:
        _require(
            transactions,
            {"transaction_id", "player_id", "known_date"},
            "transactions",
        )
        tx = transactions.copy()
        tx["known_date"] = pd.to_datetime(
            tx["known_date"], errors="raise"
        ).dt.normalize()
        tx["transaction_id"] = tx["transaction_id"].astype(str)
        tx["__transaction_id_numeric"] = pd.to_numeric(
            tx["transaction_id"], errors="coerce"
        )
        tx = tx.sort_values(
            [
                "known_date",
                "player_id",
                "__transaction_id_numeric",
                "transaction_id",
            ],
            kind="mergesort",
            na_position="last",
        )
        transaction_min_date = tx["known_date"].min()
        transaction_max_date = tx["known_date"].max()
        transaction_records = tx.to_dict("records")

    total_line_by_game: dict[str, float] = {}
    if not closing_lines.empty and "GAME_ID" in closing_lines:
        line_column = "ODDS_TOTAL_CONSENSUS_LINE_NORMALIZED_MEDIAN"
        if line_column in closing_lines:
            clean = closing_lines.assign(GAME_ID=closing_lines["GAME_ID"].astype(str))
            if clean["GAME_ID"].duplicated().any():
                raise ValueError("closing lines is not unique by GAME_ID")
            total_line_by_game = pd.to_numeric(
                clean.set_index("GAME_ID")[line_column], errors="coerce"
            ).to_dict()

    player_team: dict[str, str] = {}
    last_seen: dict[str, pd.Timestamp] = {}
    # Roles are monotone sets rather than one exclusive label.  A two-way
    # player is genuinely both, and an exclusive label forces him to flip: he
    # became a "pitcher" on the date he started on the mound and only reverted
    # on his next batting date, which erased the best hitter on the roster from
    # the following game.  Membership here only ever accumulates, so a role can
    # never oscillate with the calendar.
    has_batted: set[str] = set()
    has_pitched: set[str] = set()
    has_started_on_mound: set[str] = set()
    roster: defaultdict[str, set[str]] = defaultdict(set)
    il_state: dict[tuple[str, str], bool] = {}
    recent_starts: defaultdict[tuple[str, str], deque[int]] = defaultdict(
        lambda: deque(maxlen=RECENT_LINEUP_GAMES)
    )
    absence_streak: defaultdict[tuple[str, str], int] = defaultdict(int)
    effect_history: defaultdict[tuple[str, int, str], _EffectCounts] = defaultdict(
        _EffectCounts
    )
    team_rows: list[dict[str, Any]] = []

    def assign_team(player_id: str, team_id: str, seen_date: pd.Timestamp) -> None:
        old_team = player_team.get(player_id)
        if old_team and old_team != team_id:
            roster[old_team].discard(player_id)
        player_team[player_id] = team_id
        roster[team_id].add(player_id)
        last_seen[player_id] = seen_date

    transaction_index = 0
    for game_date, date_games in schedule.groupby("game_date", sort=True):
        # The plan accepts known_date <= game_date because intraday transaction
        # time is unavailable.  Every game on the date receives the same state.
        while (
            transaction_index < len(transaction_records)
            and transaction_records[transaction_index]["known_date"] <= game_date
        ):
            transaction = transaction_records[transaction_index]
            transaction_index += 1
            if pd.isna(transaction.get("player_id")):
                continue
            player_id = str(transaction["player_id"])
            from_team = transaction.get("from_team_id")
            to_team = transaction.get("to_team_id")
            from_team = str(from_team) if pd.notna(from_team) else None
            to_team = str(to_team) if pd.notna(to_team) else None
            team_id = to_team or from_team or player_team.get(player_id)
            if to_team and (from_team is None or to_team != from_team):
                assign_team(player_id, to_team, game_date)
            elif team_id and player_id not in player_team:
                assign_team(player_id, team_id, game_date)
            direction = classify_il_transaction(transaction)
            if direction and team_id:
                il_state[(player_id, team_id)] = direction != "close"
            description_value = transaction.get("description")
            description = (
                ""
                if description_value is None or pd.isna(description_value)
                else str(description_value)
            )
            # A transaction is the only evidence for a player who has not yet
            # appeared, so it can add a pitching role but never remove a
            # batting one.
            if re.search(r"\b[RL]HP\b", description):
                has_pitched.add(player_id)

        pending_states: list[
            tuple[str, str, int, set[str], set[str], dict[str, float]]
        ] = []
        for game in date_games.itertuples(index=False):
            game_pk = str(game.game_pk)
            season = int(game.season_year)
            for home, team_value in (
                (True, game.home_team_id),
                (False, game.away_team_id),
            ):
                team_id = str(team_value)
                starters = lineup_map.get((game_pk, team_id), set())
                current_pitchers = lineup_pitchers.get((game_pk, team_id), set())

                # The lineup slot marked P is the pitcher taking his own turn
                # at bat, which is the 2015-2021 National League default and
                # not a hitter.  A two-way player occupies that same slot while
                # batting as a real hitter, so an established batting role
                # -- evidence from strictly earlier dates -- overrides it.
                # Coverage asks whether a lineup exists at all, not whether it
                # is complete: absence is inferred by differencing against the
                # starters, so any lineup is evidence and none is not.  How
                # complete it is stays recoverable from N_LINEUP_STARTERS.
                lineup_covered = len(starters) > 0
                available_ids = {
                    player_id
                    for player_id in starters
                    if player_id not in current_pitchers or player_id in has_batted
                }
                candidate_ids = {
                    player_id
                    for player_id in roster[team_id]
                    if player_id in has_batted
                    and (
                        player_id not in last_seen
                        or (game_date - last_seen[player_id]).days <= 370
                    )
                }.union(available_ids)

                candidates: list[dict[str, Any]] = []
                for player_id in candidate_ids:
                    metrics, covered = hitter_history.get(player_id, season, game_date)
                    starts = recent_starts[(player_id, team_id)]
                    lineup_share = sum(starts) / len(starts) if starts else 0.0
                    candidates.append(
                        {
                            "player_id": player_id,
                            "metrics": metrics,
                            "covered": covered,
                            "role": (float(metrics["PA"]), lineup_share),
                            "streak": absence_streak[(player_id, team_id)],
                        }
                    )
                expected = sorted(
                    candidates, key=lambda item: item["role"], reverse=True
                )[:EXPECTED_HITTERS]
                expected_ids = {str(item["player_id"]) for item in expected}
                by_id = {str(item["player_id"]): item for item in candidates}

                available = [by_id[player_id] for player_id in available_ids]
                # Absence is decided from the starting lineup alone.
                #
                # It previously also subtracted this game's appearances, so a
                # rested regular who later pinch-hit was not counted absent.
                # Whether a substitute enters is decided *by the game*: a team
                # empties its bench in a long or high-scoring game, so
                # subtracting appearances let the realised outcome back into a
                # ``_BEFORE`` column. Measured over 55,434 team-games,
                # non-starters appeared in 8.41-run games 0 times on average
                # against 9.42 runs when four or more appeared.
                #
                # Without a lineup there is no evidence of absence.  Differencing
                # against an empty starter set would mark every expected hitter
                # absent, which reads as a squad-wide injury crisis rather than
                # as missing data.  The coverage flag is what carries that.
                missing = expected_ids.difference(starters) if lineup_covered else set()
                # The emitted streak includes the target game's observed
                # absence; the stored counter remains strictly prior state.
                for player_id in missing:
                    by_id[player_id]["streak"] += 1
                injured = [
                    by_id[player_id]
                    for player_id in missing
                    if il_state.get((player_id, team_id), False)
                ]
                observed = [
                    by_id[player_id]
                    for player_id in missing
                    if not il_state.get((player_id, team_id), False)
                ]

                feature_row = _empty_feature_row()
                feature_row.update(
                    {
                        "game_pk": game_pk,
                        "team_id": team_id,
                        "home": home,
                        "TEAM_AVAILABILITY_HITTER_N_LINEUP_STARTERS_BEFORE": float(
                            len(starters)
                        ),
                        "TEAM_AVAILABILITY_HITTER_N_EXPECTED_CANDIDATES_BEFORE": float(
                            len(expected)
                        ),
                        "TEAM_AVAILABILITY_HITTER_N_EXPECTED_WITH_HISTORY_BEFORE": float(
                            sum(bool(item["covered"]) for item in expected)
                        ),
                        "TEAM_AVAILABILITY_HITTER_LINEUP_COVERAGE_BEFORE": float(
                            lineup_covered
                        ),
                        "TEAM_AVAILABILITY_HITTER_EXPECTED_COVERAGE_BEFORE": float(
                            len(expected) >= EXPECTED_HITTERS
                        ),
                        "TEAM_AVAILABILITY_TRANSACTION_COVERAGE_BEFORE": float(
                            transaction_min_date is not None
                            and transaction_max_date is not None
                            and transaction_min_date <= game_date
                            and game_date <= transaction_max_date
                        ),
                    }
                )
                _aggregate_players(feature_row, "AVAILABLE", available)
                _aggregate_players(feature_row, "OBSERVED_ABSENT", observed)
                _aggregate_players(feature_row, "INJURED", injured)
                _aggregate_effects(
                    feature_row,
                    "AVAILABLE",
                    available,
                    effect_history,
                    season,
                )
                _aggregate_effects(
                    feature_row,
                    "ABSENT",
                    [*observed, *injured],
                    effect_history,
                    season,
                )

                # Counts a genuine pitcher: someone who has started on the
                # mound, or who has pitched without ever holding a batting
                # role.  That admits a two-way starter while still excluding
                # the position player who mopped up one blowout inning.
                injured_pitchers = {
                    player_id
                    for player_id in roster[team_id]
                    if player_id in has_pitched
                    and (
                        player_id in has_started_on_mound or player_id not in has_batted
                    )
                    and il_state.get((player_id, team_id), False)
                    and (
                        player_id not in last_seen
                        or (game_date - last_seen[player_id]).days <= 370
                    )
                }
                feature_row["TEAM_AVAILABILITY_PITCHER_N_INJURED_BEFORE"] = float(
                    len(injured_pitchers)
                )
                probable_column = (
                    "home_probable_pitcher_id" if home else "away_probable_pitcher_id"
                )
                probable_id = getattr(game, probable_column, None)
                if pd.notna(probable_id) and str(probable_id):
                    probable_id = str(probable_id)
                    metrics, covered = pitcher_history.get(
                        probable_id, season, game_date
                    )
                    feature_row[
                        "TEAM_AVAILABILITY_STARTING_PITCHER_HAS_ANNOUNCED_BEFORE"
                    ] = 1.0
                    feature_row[
                        "TEAM_AVAILABILITY_STARTING_PITCHER_HISTORY_COVERAGE_BEFORE"
                    ] = float(covered)
                    for metric, value in metrics.items():
                        feature_row[
                            f"TEAM_AVAILABILITY_STARTING_PITCHER_{metric}_BEFORE"
                        ] = float(value)

                if game_pk in targets:
                    team_rows.append(feature_row)

                team_runs = getattr(
                    game, "home_score" if home else "away_score", np.nan
                )
                total_runs = getattr(game, "total_runs", np.nan)
                if pd.isna(total_runs):
                    home_score = getattr(game, "home_score", np.nan)
                    away_score = getattr(game, "away_score", np.nan)
                    if pd.notna(home_score) and pd.notna(away_score):
                        total_runs = float(home_score) + float(away_score)
                outcomes = {
                    "TEAM_RUNS": float(team_runs) if pd.notna(team_runs) else np.nan,
                    "TOTAL_RUNS": (
                        float(total_runs) if pd.notna(total_runs) else np.nan
                    ),
                    "TOTAL_LINE_ERROR": np.nan,
                }
                total_line = total_line_by_game.get(game_pk, np.nan)
                if pd.notna(total_runs) and pd.notna(total_line):
                    outcomes["TOTAL_LINE_ERROR"] = float(total_runs) - float(total_line)
                pending_states.append(
                    (game_pk, team_id, season, available_ids, missing, outcomes)
                )

        # Update team membership only after all games on the date have been
        # featurized, so game two of a doubleheader cannot alter game one/two.
        for row in batter_appearances_by_date.get(game_date, pd.DataFrame()).itertuples(
            index=False
        ):
            assign_team(str(row.player_id), str(row.team_id), game_date)
            # A batting role requires a plate appearance taken somewhere other
            # than the pitcher's slot.  Aligning this with the plate-appearance
            # filter used to build the EWMA keeps the candidate pool and the
            # history in step, so a defensive substitute who never bats cannot
            # become an expected hitter with no statistics behind him.
            position_code = getattr(row, "position_code", None)
            plate_appearances = pd.to_numeric(
                getattr(row, "plate_appearances", 0), errors="coerce"
            )
            if (
                str(position_code) != "1"
                and pd.notna(plate_appearances)
                and float(plate_appearances) > 0
            ):
                has_batted.add(str(row.player_id))
        for row in pitcher_appearances_by_date.get(
            game_date, pd.DataFrame()
        ).itertuples(index=False):
            assign_team(str(row.player_id), str(row.team_id), game_date)
            player_id = str(row.player_id)
            has_pitched.add(player_id)
            if bool(row.is_starter):
                has_started_on_mound.add(player_id)

        for (
            game_pk,
            team_id,
            season,
            available_ids,
            missing,
            outcomes,
        ) in pending_states:
            relevant = available_ids.union(missing)
            lineup_ids = lineup_map.get((game_pk, team_id), set())
            fresh_roster = {
                player_id
                for player_id in roster[team_id]
                if player_id not in last_seen
                or (game_date - last_seen[player_id]).days <= 370
            }
            for player_id in fresh_roster.union(relevant):
                if player_id in has_batted:
                    recent_starts[(player_id, team_id)].append(
                        int(player_id in lineup_ids)
                    )
            for player_id in relevant:
                is_available = player_id in available_ids
                # The streak counts consecutive non-starts, matching how
                # ``missing`` is now defined. Resetting it on a substitute
                # appearance would make the streak mean something the emitted
                # absence count does not.
                if is_available:
                    absence_streak[(player_id, team_id)] = 0
                elif player_id in missing:
                    absence_streak[(player_id, team_id)] += 1
                for outcome_name, value in outcomes.items():
                    if pd.notna(value):
                        effect_history[(player_id, season, outcome_name)].add(
                            is_available, float(value)
                        )

    if not team_rows:
        columns = ["GAME_ID"]
        for column in _empty_feature_row():
            columns.extend([f"{column}_TEAM_HOME", f"{column}_TEAM_AWAY"])
        columns.extend(
            column.removesuffix("_BEFORE") + "_DIFF_BEFORE" for column in DIFF_FEATURES
        )
        return pd.DataFrame(columns=columns)
    result = _wide_team_features(pd.DataFrame(team_rows))
    identity_fragments = ("PLAYER_ID", "PLAYER_NAME")
    leaking_fragments = ("HOME_SCORE", "AWAY_SCORE", "RAW_OUTCOME")
    forbidden = [
        column
        for column in result.columns
        if any(
            fragment in column for fragment in (*identity_fragments, *leaking_fragments)
        )
    ]
    if forbidden:
        raise ValueError(
            f"Private player/outcome columns reached availability: {forbidden}"
        )
    return result


__all__ = [
    "HITTER_NEUTRALS",
    "HITTER_OUTPUT_METRICS",
    "HITTER_SECONDARY_METRICS",
    "PITCHER_NEUTRALS",
    "build_availability_features",
    "build_hitter_history",
    "build_pitcher_history",
    "classify_il_transaction",
]
