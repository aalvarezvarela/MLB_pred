"""Data-quality checks for the local store.

Postgres enforces plausibility with CHECK constraints at the schema boundary.
The Parquet store has no such boundary, so the same guards live here and are
run explicitly. They are cheap and they catch the one failure mode that
matters: a box score fetched while a game was still in progress looks
complete, and written into an insert-only table it poisons that game forever.

Beyond plausibility these checks assert the *relational* invariants the three
grains have to satisfy -- two team rows per game, runs that reconcile between
the games table and the team-game rows, no orphans. Those are what catch a
partial ingest, which no per-row bound can see.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlb_pred.config.constants import (
    MAX_PLAUSIBLE_GAME_OUTS,
    MAX_PLAUSIBLE_TEAM_RUNS,
    MIN_PLAUSIBLE_GAME_OUTS,
)
from mlb_pred.local_store.parquet_store import read_table


@dataclass
class Finding:
    check: str
    table: str
    rows: int
    detail: str

    def __str__(self) -> str:
        return f"[{self.table}] {self.check}: {self.rows} row(s) -- {self.detail}"


def _finding(check, table, offenders, detail) -> list[Finding]:
    count = len(offenders)
    return [Finding(check, table, count, detail)] if count else []


def check_games(games: pd.DataFrame) -> list[Finding]:
    if games.empty:
        return []

    findings: list[Finding] = []
    findings += _finding(
        "duplicate_game_pk",
        "games",
        games[games["game_pk"].duplicated()],
        "gamePk must be unique; duplicates mean the schedule feed's "
        "postponed/suspended double-listing was not resolved",
    )
    findings += _finding(
        "implausible_runs",
        "games",
        games[
            (games["home_score"] > MAX_PLAUSIBLE_TEAM_RUNS)
            | (games["away_score"] > MAX_PLAUSIBLE_TEAM_RUNS)
            | (games["home_score"] < 0)
            | (games["away_score"] < 0)
        ],
        f"team runs outside 0..{MAX_PLAUSIBLE_TEAM_RUNS}",
    )
    findings += _finding(
        "final_without_score",
        "games",
        games[games["is_final"] & games["total_runs"].isna()],
        "marked final but has no score",
    )
    findings += _finding(
        "total_runs_mismatch",
        "games",
        games[games["total_runs"] != games["home_score"] + games["away_score"]].dropna(
            subset=["total_runs"]
        ),
        "total_runs does not equal home + away",
    )
    findings += _finding(
        "short_game",
        "games",
        games[games["innings_played"].notna() & (games["innings_played"] < 5)],
        "fewer than five innings; not an official game",
    )
    findings += _finding(
        "same_team_both_sides",
        "games",
        games[games["home_team_id"] == games["away_team_id"]],
        "home and away are the same team",
    )
    return findings


def check_team_games(team_games: pd.DataFrame, games: pd.DataFrame) -> list[Finding]:
    if team_games.empty:
        return []

    findings: list[Finding] = []

    per_game = team_games.groupby("game_pk").size()
    findings += _finding(
        "wrong_row_count",
        "team_games",
        per_game[per_game != 2],
        "every game must have exactly two team rows",
    )
    findings += _finding(
        "implausible_runs",
        "team_games",
        team_games[
            (team_games["runs_scored"] > MAX_PLAUSIBLE_TEAM_RUNS)
            | (team_games["runs_allowed"] > MAX_PLAUSIBLE_TEAM_RUNS)
        ],
        f"runs above {MAX_PLAUSIBLE_TEAM_RUNS}",
    )
    findings += _finding(
        "implausible_outs",
        "team_games",
        team_games[
            team_games["outs_recorded"].notna()
            & (
                (team_games["outs_recorded"] < MIN_PLAUSIBLE_GAME_OUTS)
                | (team_games["outs_recorded"] > MAX_PLAUSIBLE_GAME_OUTS)
            )
        ],
        f"outs outside the absolute band {MIN_PLAUSIBLE_GAME_OUTS}.."
        f"{MAX_PLAUSIBLE_GAME_OUTS}",
    )
    findings += _finding(
        "self_opponent",
        "team_games",
        team_games[team_games["team_id"] == team_games["opponent_team_id"]],
        "team is its own opponent",
    )

    if not games.empty and "innings_played" in games.columns:
        # The check that actually catches a mid-game fetch. An absolute floor
        # cannot: baseball games legitimately run 5, 6 or 7 innings (rain, and
        # the 2020-21 seven-inning doubleheaders), so any floor high enough to
        # flag a partial nine-inning box score also flags a real short game.
        # Relative to the game's own length, both problems disappear.
        lengths = games[["game_pk", "innings_played"]]
        joined = team_games.merge(lengths, on="game_pk", how="inner").dropna(
            subset=["outs_recorded", "innings_played"]
        )
        findings += _finding(
            "outs_inconsistent_with_innings",
            "team_games",
            joined[
                (joined["outs_recorded"] < 3 * (joined["innings_played"] - 1))
                | (joined["outs_recorded"] > 3 * joined["innings_played"])
            ],
            "outs recorded are not within one inning of the game's length; "
            "the signature of a box score fetched mid-game",
        )

    # The reciprocity check: one side's runs scored must equal the other's
    # runs allowed. A partial fetch breaks this even when both rows exist.
    paired = team_games.merge(
        team_games,
        left_on=["game_pk", "team_id"],
        right_on=["game_pk", "opponent_team_id"],
        suffixes=("", "_opp"),
    )
    findings += _finding(
        "runs_not_reciprocal",
        "team_games",
        paired[paired["runs_scored"] != paired["runs_allowed_opp"]],
        "one side's runs scored does not match the other's runs allowed",
    )

    if not games.empty:
        joined = team_games.merge(
            games[["game_pk", "home_score", "away_score"]], on="game_pk", how="left"
        )
        expected = joined["home_score"].where(joined["home"], joined["away_score"])
        findings += _finding(
            "runs_disagree_with_games",
            "team_games",
            joined[joined["runs_scored"] != expected].dropna(subset=["home_score"]),
            "team-game runs disagree with the games table",
        )
        findings += _finding(
            "orphan_game_pk",
            "team_games",
            team_games[~team_games["game_pk"].isin(set(games["game_pk"]))],
            "references a game that is not in the games table",
        )
    return findings


def check_pitcher_appearances(
    pitchers: pd.DataFrame, team_games: pd.DataFrame
) -> list[Finding]:
    if pitchers.empty:
        return []

    findings: list[Finding] = []

    starters = pitchers[pitchers["is_starter"]].groupby(["game_pk", "team_id"]).size()
    findings += _finding(
        "wrong_starter_count",
        "pitcher_appearances",
        starters[starters != 1],
        "each team must have exactly one starting pitcher per game",
    )
    findings += _finding(
        "implausible_outs",
        "pitcher_appearances",
        pitchers[
            pitchers["outs_recorded"].notna()
            & (pitchers["outs_recorded"] > MAX_PLAUSIBLE_GAME_OUTS)
        ],
        f"outs above {MAX_PLAUSIBLE_GAME_OUTS}",
    )

    if not team_games.empty:
        # A team's pitchers must collectively record exactly the outs the
        # team-game row claims. This is the sharpest single test that a game
        # was ingested completely.
        summed = (
            pitchers.groupby(["game_pk", "team_id"])["outs_recorded"]
            .sum()
            .rename("outs_from_pitchers")
            .reset_index()
        )
        merged = team_games.merge(summed, on=["game_pk", "team_id"], how="inner")
        findings += _finding(
            "outs_disagree_with_team_game",
            "pitcher_appearances",
            merged[merged["outs_recorded"] != merged["outs_from_pitchers"]],
            "pitcher outs do not sum to the team-game total",
        )
    return findings


def check_batter_games(batters: pd.DataFrame) -> list[Finding]:
    if batters.empty:
        return []

    findings: list[Finding] = []
    slots = batters[batters["is_starter"]].groupby(["game_pk", "team_id"]).size()
    findings += _finding(
        "wrong_starting_lineup_size",
        "batter_games",
        slots[~slots.isin({9, 10})],
        "a starting lineup is nine batters (ten where a pitcher also bats)",
    )
    findings += _finding(
        "hits_exceed_at_bats",
        "batter_games",
        batters[batters["hits"] > batters["at_bats"]],
        "more hits than at-bats",
    )
    findings += _finding(
        "extra_base_hits_exceed_hits",
        "batter_games",
        batters[
            (batters["doubles"] + batters["triples"] + batters["home_runs"])
            > batters["hits"]
        ],
        "extra-base hits exceed total hits",
    )
    return findings


def check_umpires(umpires: pd.DataFrame, games: pd.DataFrame) -> list[Finding]:
    if umpires.empty or games.empty:
        return []

    with_home_plate = set(umpires[umpires["is_home_plate"]]["game_pk"])
    missing = games[~games["game_pk"].isin(with_home_plate)]
    return _finding(
        "missing_home_plate_umpire",
        "umpires",
        missing,
        "no home-plate umpire recorded; the only crew position that moves "
        "run scoring",
    )


def check_odds_ticks(ticks: pd.DataFrame) -> list[Finding]:
    if ticks.empty:
        return []
    opener_counts = ticks[ticks["is_opener"]].groupby(
        ["game_pk", "market", "book_slug"]
    ).size()
    all_groups = ticks.groupby(["game_pk", "market", "book_slug"]).size()
    missing = all_groups[~all_groups.index.isin(opener_counts.index)]
    findings = _finding(
        "multiple_openers",
        "odds_ticks",
        opener_counts[opener_counts != 1],
        "each game/market/book history must derive exactly one opener",
    )
    findings += _finding(
        "missing_opener",
        "odds_ticks",
        missing,
        "each game/market/book history must have an opener",
    )
    return findings


def check_statcast(
    statcast_games: pd.DataFrame, pitches: pd.DataFrame, games: pd.DataFrame
) -> list[Finding]:
    findings: list[Finding] = []
    if not pitches.empty:
        duplicate = pitches.duplicated(
            ["game_pk", "at_bat_number", "pitch_number"], keep=False
        )
        findings += _finding(
            "duplicate_pitch_key",
            "statcast_pitches",
            pitches[duplicate],
            "pitch identity must be unique within a plate appearance",
        )
        if not games.empty:
            findings += _finding(
                "orphan_game_pk",
                "statcast_pitches",
                pitches[~pitches["game_pk"].isin(set(games["game_pk"]))],
                "Statcast pitch references a game absent from games",
            )
    if not statcast_games.empty:
        actual = pitches.groupby("game_pk").size() if not pitches.empty else pd.Series(dtype=int)
        complete = statcast_games[statcast_games["fetch_status"] == "complete"].copy()
        complete["actual_rows"] = complete["game_pk"].map(actual).fillna(0).astype(int)
        findings += _finding(
            "pitch_row_count_mismatch",
            "statcast_games",
            complete[complete["pitch_rows"] != complete["actual_rows"]],
            "completion metadata disagrees with stored pitch rows",
        )
    return findings


def validate_store(seasons: list[int] | None = None) -> list[Finding]:
    """Run every check. An empty result means the store is internally sound."""
    games = read_table("games", partitions=seasons)
    team_games = read_table("team_games", partitions=seasons)
    pitchers = read_table("pitcher_appearances", partitions=seasons)
    batters = read_table("batter_games", partitions=seasons)
    umpires = read_table("umpires", partitions=seasons)
    odds_ticks = read_table("odds_ticks", partitions=seasons)
    statcast_games = read_table("statcast_games", partitions=seasons)
    statcast_pitches = read_table("statcast_pitches", partitions=seasons)

    findings: list[Finding] = []
    findings += check_games(games)
    findings += check_team_games(team_games, games)
    findings += check_pitcher_appearances(pitchers, team_games)
    findings += check_batter_games(batters)
    findings += check_umpires(umpires, games)
    findings += check_odds_ticks(odds_ticks)
    findings += check_statcast(statcast_games, statcast_pitches, games)
    return findings
