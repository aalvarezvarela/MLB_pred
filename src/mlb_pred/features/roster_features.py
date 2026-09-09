"""Playing-time-weighted roster continuity features.

Baseball's answer to the NBA project's ``roster_continuity`` module. The idea is
unchanged: measure how much of the playing time a team is fielding today was
already on the roster at an earlier reference point, how much of it was brought
in from somewhere else, and net the two.

Three things had to be reframed for baseball.

**There is no single playing-time currency.** A basketball roster shares one
pool of 240 minutes. A baseball roster is two disjoint units -- the nine who bat
and the staff who pitch -- and a player almost never draws from both. Continuity
is therefore computed twice, weighting hitters by plate appearances and pitchers
by batters faced, and the two are never pooled into one number.

**Roster movement is published, not inferred.** ``transactions`` carries trades,
waiver claims, signings, selections, recalls, options and releases, each with the
``known_date`` on which it was announced. That feed is the MLB counterpart of the
NBA module's injury-report assignments, and it is what lets a player acquired at
the deadline count as incoming before he has played a game. Movement that never
touches a major-league roster -- an option to Triple-A, a minor-league signing --
is read as a *departure*, because ``to_team_id`` is then an affiliate rather than
one of the thirty clubs.

**The season is one calendar year.** The NBA module anchors its season-over-season
window to March 15 of the preceding season. The MLB equivalent is August 1 of the
previous calendar year: past the trade deadline, so the baseline is the roster the
club actually finished the prior season with.

Every read is gated on strictly earlier calendar dates, so the target game and
every other game sharing its date are excluded, and transactions are read on
``known_date`` -- never ``effective_date``, which is routinely backdated.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd

from mlb_pred.config.constants import is_known_team_id

FEATURE_PREFIX = "TEAM_ROSTER_"

# Measured on ``data/raw/team_games``, regular season 2015-2026: a team sends
# 37.811 batters to the plate per game and faces the same number, because every
# plate appearance taken by one side is a batter faced by the other. One
# constant therefore normalises both units.
TEAM_PLAYING_TIME_PER_GAME = 37.811

BATTING = "BATTING"
PITCHING = "PITCHING"
UNITS: tuple[str, ...] = (BATTING, PITCHING)

# Months in which no major-league games are played. A two-month lookback that
# lands here is snapped back to the previous season's post-deadline roster
# rather than being allowed to cover an empty stretch of winter.
OFFSEASON_MONTHS = frozenset({11, 12, 1, 2, 3})

# Within one calendar date an announced transaction supersedes a box score: a
# player can appear for his old club in the afternoon and be traded that night.
_APPEARANCE_PRIORITY = 0
_TRANSACTION_PRIORITY = 1

# ``HOME - AWAY`` is an exact linear combination of the two side columns, so only
# the contrasts actually read as a contrast are materialised.
DIFF_FEATURES: tuple[str, ...] = (
    "TEAM_ROSTER_BATTING_CONTINUITY_PCT_BEFORE",
    "TEAM_ROSTER_PITCHING_CONTINUITY_PCT_BEFORE",
)

_TEAM_GAME_COLUMNS = {"game_pk", "team_id", "season_year", "game_date", "home"}
_BATTER_COLUMNS = {
    "game_pk",
    "team_id",
    "player_id",
    "season_year",
    "game_date",
    "plate_appearances",
}
_PITCHER_COLUMNS = {
    "game_pk",
    "team_id",
    "player_id",
    "season_year",
    "game_date",
    "batters_faced",
}
_TRANSACTION_COLUMNS = {
    "player_id",
    "known_date",
    "from_team_id",
    "to_team_id",
}


def roster_feature_columns() -> list[str]:
    """Every team-game column this module emits, in a stable order."""
    columns: list[str] = []
    for unit in UNITS:
        for metric in ("CONTINUITY", "INCOMING", "NET"):
            for horizon in ("", "_2M"):
                columns.append(f"{FEATURE_PREFIX}{unit}_{metric}{horizon}_PCT_BEFORE")
    return columns


def _require_columns(
    frame: pd.DataFrame, required: set[str], *, frame_name: str
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _dates(values: pd.Series) -> pd.Series:
    """Normalise to midnight so a date comparison never sees a clock time."""
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _date_ns(values: pd.Series) -> pd.Series:
    """Epoch nanoseconds for already-validated dates.

    ``astype("int64")`` turns ``NaT`` into a large negative sentinel rather than
    a null, so anything that can carry a missing date must be dropped with
    :func:`_dates` *before* it reaches here.
    """
    dates = _dates(values)
    if dates.isna().any():
        raise ValueError("Cannot convert a missing date to an ordering key.")
    return dates.astype("int64")


def full_window_start(season_year: int) -> int:
    """The prior season's post-deadline roster, as an epoch-nanosecond bound."""
    return pd.Timestamp(year=season_year - 1, month=8, day=1).value


def immediate_window_start(game_date_ns: int, season_year: int) -> int:
    """Two calendar months back, snapped out of the winter when it lands there."""
    start = (pd.Timestamp(game_date_ns) - pd.DateOffset(months=2)).normalize()
    if start.month in OFFSEASON_MONTHS:
        return full_window_start(season_year)
    return start.value


class _RosterHistory:
    """Assignment timelines and playing-time weights, indexed for bisect reads."""

    def __init__(self) -> None:
        # player -> parallel arrays of assignment dates and the team assigned to,
        # sorted by (date, priority). ``None`` means "no major-league roster".
        self.player_dates: dict[str, list[int]] = {}
        self.player_teams: dict[str, list[str | None]] = {}
        # team -> arrivals, sorted by date, used to enumerate window candidates.
        self.arrival_dates: dict[str, list[int]] = {}
        self.arrival_players: dict[str, list[str]] = {}
        # (player, team, unit, season) -> sorted dates and cumulative playing time.
        self.weight_dates: dict[tuple[str, str, str, int], list[int]] = {}
        self.weight_sums: dict[tuple[str, str, str, int], list[float]] = {}
        # (team, season) -> the team's own game dates, sorted.
        self.team_game_dates: dict[tuple[str, int], list[int]] = {}

    def latest_assignment(self, player_id: str, before_ns: int) -> tuple[int, int]:
        """Return (index of first event at or after ``before_ns``, count)."""
        dates = self.player_dates.get(player_id)
        if not dates:
            return 0, 0
        return bisect_left(dates, before_ns), len(dates)

    def playing_time_per_game(
        self, player_id: str, team_id: str, unit: str, season_year: int, before_ns: int
    ) -> float:
        """The player's playing time per team game, prior season as fallback."""
        for season in (season_year, season_year - 1):
            key = (player_id, team_id, unit, season)
            dates = self.weight_dates.get(key)
            if not dates:
                continue
            count = bisect_left(dates, before_ns)
            if not count:
                continue
            team_games = bisect_left(
                self.team_game_dates.get((team_id, season), []), before_ns
            )
            if team_games:
                return self.weight_sums[key][count - 1] / team_games
        return 0.0

    def playing_time_during_stint(
        self, player_id: str, team_id: str, unit: str, season_year: int, before_ns: int
    ) -> float:
        """The player's playing time per team game *while he was on that club*.

        A departed player is weighted by the role he actually held, so the
        denominator is the club's games between his first and last appearance
        rather than every game it has played. Dividing by his own appearances
        instead -- what the NBA module does, where everyone is available every
        night -- breaks on a pitching staff: a starter faces twenty-five batters
        every fifth day, and counting that as a per-game rate makes one incoming
        starter look like two thirds of a team's pitching.
        """
        for season in (season_year, season_year - 1):
            key = (player_id, team_id, unit, season)
            dates = self.weight_dates.get(key)
            if not dates:
                continue
            count = bisect_left(dates, before_ns)
            if not count:
                continue
            team_dates = self.team_game_dates.get((team_id, season), [])
            stint_games = bisect_right(team_dates, dates[count - 1]) - bisect_left(
                team_dates, dates[0]
            )
            if stint_games > 0:
                return self.weight_sums[key][count - 1] / stint_games
        return 0.0


def _add_appearance_events(
    history: _RosterHistory,
    appearances: pd.DataFrame,
    *,
    unit: str,
    playing_time_column: str,
    events: list[tuple[int, int, str, str | None]],
) -> None:
    """Fold one appearance table into assignment events and playing-time weights."""
    frame = appearances.copy()
    frame["team_id"] = frame["team_id"].astype(str)
    frame["player_id"] = frame["player_id"].astype(str)
    frame["season_year"] = pd.to_numeric(frame["season_year"], errors="raise").astype(
        int
    )
    frame["__date_ns"] = _date_ns(frame["game_date"])
    frame["__playing_time"] = pd.to_numeric(
        frame[playing_time_column], errors="coerce"
    ).fillna(0.0)

    for date_ns, player_id, team_id in zip(
        frame["__date_ns"].to_numpy(),
        frame["player_id"].to_numpy(),
        frame["team_id"].to_numpy(),
        strict=True,
    ):
        events.append(
            (int(date_ns), _APPEARANCE_PRIORITY, str(player_id), str(team_id))
        )

    # Playing-time weights only count games the player actually worked in.
    worked = frame.loc[frame["__playing_time"] > 0]
    worked = worked.sort_values("__date_ns", kind="mergesort")
    grouped = worked.groupby(
        ["player_id", "team_id", "season_year"], sort=False, dropna=False
    )
    for (player_id, team_id, season_year), block in grouped:
        key = (str(player_id), str(team_id), unit, int(season_year))
        dates = block["__date_ns"].astype("int64").tolist()
        history.weight_dates[key] = dates
        history.weight_sums[key] = (
            block["__playing_time"].cumsum().astype(float).tolist()
        )


def _add_transaction_events(
    transactions: pd.DataFrame,
    events: list[tuple[int, int, str, str | None]],
) -> None:
    """Read announced roster movement, on ``known_date`` and never on effect date."""
    frame = transactions[sorted(_TRANSACTION_COLUMNS)].copy()
    frame["player_id"] = frame["player_id"].astype("string")
    # A transaction with no announcement date cannot be placed on a timeline,
    # and an unattributed one has nobody to move. Both are dropped before the
    # ordering key is built.
    frame["__known"] = _dates(frame["known_date"])
    frame = frame.loc[frame["player_id"].notna() & frame["__known"].notna()]
    frame["__date_ns"] = frame["__known"].astype("int64")

    def _team(value: object) -> str | None:
        if pd.isna(value):
            return None
        team_id = str(value)
        return team_id if is_known_team_id(team_id) else None

    for date_ns, player_id, from_team, to_team in zip(
        frame["__date_ns"].to_numpy(),
        frame["player_id"].to_numpy(),
        frame["from_team_id"].to_numpy(),
        frame["to_team_id"].to_numpy(),
        strict=True,
    ):
        arrival = _team(to_team)
        if arrival is not None:
            events.append(
                (int(date_ns), _TRANSACTION_PRIORITY, str(player_id), arrival)
            )
        elif _team(from_team) is not None:
            # Left a major-league roster for an affiliate, waivers or free
            # agency. The player is off the club until something says otherwise.
            events.append((int(date_ns), _TRANSACTION_PRIORITY, str(player_id), None))


def _add_team_game_dates(history: _RosterHistory, team_games: pd.DataFrame) -> None:
    frame = team_games[["team_id", "season_year", "game_date"]].copy()
    frame["team_id"] = frame["team_id"].astype(str)
    frame["season_year"] = pd.to_numeric(frame["season_year"], errors="raise").astype(
        int
    )
    frame["__date_ns"] = _date_ns(frame["game_date"])
    for (team_id, season_year), block in frame.groupby(
        ["team_id", "season_year"], sort=False
    ):
        history.team_game_dates[(str(team_id), int(season_year))] = sorted(
            block["__date_ns"].astype("int64").tolist()
        )


def build_roster_history(
    team_games: pd.DataFrame,
    batter_games: pd.DataFrame,
    pitcher_appearances: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
) -> _RosterHistory:
    """Index every assignment and playing-time fact this module needs.

    ``team_games`` is not optional: every playing-time weight divides by the
    number of games the club played, so a history built without it would return
    a silent zero for each one.
    """
    _require_columns(team_games, _TEAM_GAME_COLUMNS, frame_name="team games")
    _require_columns(batter_games, _BATTER_COLUMNS, frame_name="batter games")
    _require_columns(
        pitcher_appearances, _PITCHER_COLUMNS, frame_name="pitcher appearances"
    )

    history = _RosterHistory()
    events: list[tuple[int, int, str, str | None]] = []
    _add_appearance_events(
        history,
        batter_games[sorted(_BATTER_COLUMNS)],
        unit=BATTING,
        playing_time_column="plate_appearances",
        events=events,
    )
    _add_appearance_events(
        history,
        pitcher_appearances[sorted(_PITCHER_COLUMNS)],
        unit=PITCHING,
        playing_time_column="batters_faced",
        events=events,
    )
    if transactions is not None and not transactions.empty:
        _require_columns(transactions, _TRANSACTION_COLUMNS, frame_name="transactions")
        _add_transaction_events(transactions, events)

    # Sort on (date, priority) only: a departure carries ``None`` for the team
    # and must not be compared against a club id. The sort is stable, so
    # equal keys keep the order the tables were folded in.
    events.sort(key=lambda event: (event[0], event[1]))
    player_dates: dict[str, list[int]] = defaultdict(list)
    player_teams: dict[str, list[str | None]] = defaultdict(list)
    arrivals: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for date_ns, _priority, player_id, team_id in events:
        timeline = player_dates[player_id]
        teams = player_teams[player_id]
        # Consecutive repeats of the same assignment carry no information and
        # would only slow the backward scan for the previous distinct club.
        if timeline and timeline[-1] == date_ns and teams[-1] == team_id:
            continue
        timeline.append(date_ns)
        teams.append(team_id)
        if team_id is not None:
            arrivals[team_id].add((date_ns, player_id))

    history.player_dates = dict(player_dates)
    history.player_teams = dict(player_teams)
    for team_id, entries in arrivals.items():
        ordered = sorted(entries)
        history.arrival_dates[team_id] = [entry[0] for entry in ordered]
        history.arrival_players[team_id] = [entry[1] for entry in ordered]
    _add_team_game_dates(history, team_games)
    return history


def _window_metrics(
    history: _RosterHistory,
    *,
    team_id: str,
    season_year: int,
    date_ns: int,
    window_start_ns: int,
    unit: str,
    require_prior_season: bool,
) -> tuple[float, float]:
    """Continuity and incoming share for one team, unit and window."""
    arrival_dates = history.arrival_dates.get(team_id)
    if not arrival_dates:
        return np.nan, np.nan
    arrival_players = history.arrival_players[team_id]
    low = bisect_left(arrival_dates, window_start_ns)
    high = bisect_left(arrival_dates, date_ns)
    if low >= high:
        return np.nan, np.nan

    season_start_ns = pd.Timestamp(year=season_year, month=1, day=1).value
    if require_prior_season and arrival_dates[low] >= season_start_ns:
        # No pre-winter observation of this club, so there is no season-over-
        # season baseline to compare today's roster against.
        return np.nan, np.nan

    candidates = set(arrival_players[low:high])
    candidate_weight = 0.0
    lost_weight = 0.0
    incoming_weight = 0.0
    for player_id in candidates:
        index, _count = history.latest_assignment(player_id, date_ns)
        if index == 0:
            continue
        weight = history.playing_time_per_game(
            player_id, team_id, unit, season_year, date_ns
        )
        candidate_weight += weight
        teams = history.player_teams[player_id]
        if teams[index - 1] != team_id:
            lost_weight += weight
            continue

        # Still here. Was he somewhere else earlier inside this window?
        dates = history.player_dates[player_id]
        previous_team: str | None = None
        for position in range(index - 2, -1, -1):
            if dates[position] < window_start_ns:
                break
            if teams[position] != team_id:
                previous_team = teams[position]
                break
        if previous_team is not None:
            incoming_weight += history.playing_time_during_stint(
                player_id, previous_team, unit, season_year, date_ns
            )

    if candidate_weight <= 0.0:
        return np.nan, np.nan
    continuity = float(np.clip(1.0 - lost_weight / candidate_weight, 0.0, 1.0))
    incoming = float(np.clip(incoming_weight / TEAM_PLAYING_TIME_PER_GAME, 0.0, 1.0))
    return continuity, incoming


def build_team_roster_features(
    team_games: pd.DataFrame,
    batter_games: pd.DataFrame,
    pitcher_appearances: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Continuity, incoming and net playing-time shares per team game."""
    _require_columns(team_games, _TEAM_GAME_COLUMNS, frame_name="team games")
    history = build_roster_history(
        team_games, batter_games, pitcher_appearances, transactions
    )

    rows = team_games[["game_pk", "team_id", "season_year", "game_date", "home"]].copy()
    rows["game_pk"] = rows["game_pk"].astype(str)
    rows["team_id"] = rows["team_id"].astype(str)
    rows["season_year"] = pd.to_numeric(rows["season_year"], errors="raise").astype(int)
    rows["__date_ns"] = _date_ns(rows["game_date"])

    columns: dict[str, list[float]] = {
        column: [] for column in roster_feature_columns()
    }
    for team_id, season_year, date_ns in zip(
        rows["team_id"].to_numpy(),
        rows["season_year"].to_numpy(),
        rows["__date_ns"].to_numpy(),
        strict=True,
    ):
        windows = (
            ("", full_window_start(int(season_year)), True),
            ("_2M", immediate_window_start(int(date_ns), int(season_year)), False),
        )
        for unit in UNITS:
            for suffix, window_start_ns, require_prior in windows:
                continuity, incoming = _window_metrics(
                    history,
                    team_id=str(team_id),
                    season_year=int(season_year),
                    date_ns=int(date_ns),
                    window_start_ns=window_start_ns,
                    unit=unit,
                    require_prior_season=require_prior,
                )
                base = f"{FEATURE_PREFIX}{unit}"
                columns[f"{base}_CONTINUITY{suffix}_PCT_BEFORE"].append(continuity)
                columns[f"{base}_INCOMING{suffix}_PCT_BEFORE"].append(incoming)
                # Net is incoming minus lost, and lost is 1 - continuity.
                columns[f"{base}_NET{suffix}_PCT_BEFORE"].append(
                    incoming + continuity - 1.0
                )

    result = rows[["game_pk", "team_id", "home"]].copy()
    return pd.concat([result, pd.DataFrame(columns, index=result.index)], axis=1)


def build_roster_features(
    team_games: pd.DataFrame,
    batter_games: pd.DataFrame,
    pitcher_appearances: pd.DataFrame,
    closing_features: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
    *,
    target_game_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Pivot roster continuity onto one row per target game."""
    targets = (
        {str(value) for value in target_game_ids}
        if target_game_ids is not None
        else {str(value) for value in closing_features["GAME_ID"]}
    )
    team_rows = build_team_roster_features(
        team_games, batter_games, pitcher_appearances, transactions
    )
    feature_columns = roster_feature_columns()
    target = team_rows.loc[team_rows["game_pk"].isin(targets)]
    counts = target.groupby("game_pk").agg(
        rows=("team_id", "size"), homes=("home", "sum")
    )
    invalid = counts.loc[(counts["rows"] != 2) | (counts["homes"] != 1)]
    if not invalid.empty:
        raise ValueError(
            f"Target games must have exactly one home and one away row: {invalid.head()}"
        )
    home = target.loc[
        target["home"].astype(bool), ["game_pk", *feature_columns]
    ].rename(columns={column: f"{column}_TEAM_HOME" for column in feature_columns})
    away = target.loc[
        ~target["home"].astype(bool), ["game_pk", *feature_columns]
    ].rename(columns={column: f"{column}_TEAM_AWAY" for column in feature_columns})
    wide = home.merge(away, on="game_pk", how="inner", validate="one_to_one")
    wide = wide.rename(columns={"game_pk": "GAME_ID"})

    missing = sorted(set(DIFF_FEATURES).difference(feature_columns))
    if missing:
        raise ValueError(f"DIFF_FEATURES names unknown roster columns: {missing}")
    differences = {
        column.removesuffix("_BEFORE")
        + "_DIFF_BEFORE": wide[f"{column}_TEAM_HOME"]
        - wide[f"{column}_TEAM_AWAY"]
        for column in DIFF_FEATURES
    }
    return pd.concat([wide, pd.DataFrame(differences, index=wide.index)], axis=1)
