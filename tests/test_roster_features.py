"""Roster-continuity feature tests.

The module's whole value is that it reads roster movement without reading the
future, so most of these pin a temporal rule rather than a number.
"""

import pandas as pd
import pytest

from mlb_pred.features.roster_features import (
    OFFSEASON_MONTHS,
    TEAM_PLAYING_TIME_PER_GAME,
    build_roster_features,
    build_team_roster_features,
    full_window_start,
    immediate_window_start,
    roster_feature_columns,
)

METS = "121"
YANKEES = "147"
DODGERS = "119"


def _team_games() -> pd.DataFrame:
    """Two clubs, alternating home, across 2024 and 2025."""
    rows = []
    game_pk = 1
    for season, start in ((2024, "2024-08-01"), (2025, "2025-04-01")):
        for offset, day in enumerate(pd.date_range(start, periods=40, freq="D")):
            home, away = (METS, YANKEES) if offset % 2 == 0 else (YANKEES, METS)
            rows.append((str(game_pk), home, season, day.date(), True))
            rows.append((str(game_pk), away, season, day.date(), False))
            game_pk += 1
    return pd.DataFrame(
        rows, columns=["game_pk", "team_id", "season_year", "game_date", "home"]
    )


def _appearances(
    team_games: pd.DataFrame, assignments: dict[str, list[tuple[str, str]]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give each named player a fixed workload in every game of his club."""
    batters, pitchers = [], []
    for _, row in team_games.iterrows():
        for player_id, team_id in assignments.get(row["team_id"], []):
            base = {
                "game_pk": row["game_pk"],
                "team_id": row["team_id"],
                "player_id": player_id,
                "season_year": row["season_year"],
                "game_date": row["game_date"],
            }
            if team_id == "bat":
                batters.append({**base, "plate_appearances": 4})
            else:
                pitchers.append({**base, "batters_faced": 25})
    return pd.DataFrame(batters), pd.DataFrame(pitchers)


def _simple_history() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    team_games = _team_games()
    assignments = {
        METS: [(f"m{i}", "bat") for i in range(9)],
        YANKEES: [(f"y{i}", "bat") for i in range(9)],
    }
    batters, pitchers = _appearances(team_games, assignments)
    pitchers = pd.DataFrame(
        columns=[
            "game_pk",
            "team_id",
            "player_id",
            "season_year",
            "game_date",
            "batters_faced",
        ]
    )
    return team_games, batters, pitchers


def test_window_bounds_follow_the_baseball_calendar():
    # The season-over-season baseline is the previous club's post-deadline roster.
    assert full_window_start(2025) == pd.Timestamp("2024-08-01").value

    # A two-month lookback from June lands in April and is used as-is.
    june = pd.Timestamp("2025-06-15").value
    assert immediate_window_start(june, 2025) == pd.Timestamp("2025-04-15").value

    # From April it lands in February, where no games are played, so it snaps
    # back to the previous season rather than covering an empty winter.
    april = pd.Timestamp("2025-04-10").value
    assert immediate_window_start(april, 2025) == full_window_start(2025)
    assert 2 in OFFSEASON_MONTHS


def test_an_unchanged_roster_is_fully_continuous_and_takes_nobody_in():
    team_games, batters, pitchers = _simple_history()
    features = build_team_roster_features(team_games, batters, pitchers)

    # Restrict to 2025, where a previous-season baseline exists.
    rows = features.merge(
        team_games[["game_pk", "team_id", "season_year"]], on=["game_pk", "team_id"]
    )
    late = rows.loc[rows["season_year"] == 2025]

    assert (late["TEAM_ROSTER_BATTING_CONTINUITY_PCT_BEFORE"] == 1.0).all()
    assert (late["TEAM_ROSTER_BATTING_INCOMING_PCT_BEFORE"] == 0.0).all()
    assert (late["TEAM_ROSTER_BATTING_NET_PCT_BEFORE"] == 0.0).all()


def test_a_traded_hitter_is_lost_by_one_club_and_incoming_at_the_other():
    team_games, batters, pitchers = _simple_history()
    # m0 plays for the Mets all of 2024 and the first half of 2025, then moves.
    trade_date = pd.Timestamp("2025-04-20")
    moved = (batters["player_id"] == "m0") & (
        pd.to_datetime(batters["game_date"]) >= trade_date
    )
    batters = batters.loc[~moved].copy()

    transactions = pd.DataFrame(
        [
            {
                "player_id": "m0",
                "known_date": trade_date.date(),
                "from_team_id": METS,
                "to_team_id": YANKEES,
            }
        ]
    )
    features = build_team_roster_features(
        team_games, batters, pitchers, transactions
    ).merge(
        team_games[["game_pk", "team_id", "season_year", "game_date"]],
        on=["game_pk", "team_id"],
    )
    after = features.loc[pd.to_datetime(features["game_date"]) > trade_date]
    mets = after.loc[after["team_id"] == METS].iloc[0]
    yankees = after.loc[after["team_id"] == YANKEES].iloc[0]

    # The Mets lost one of nine hitters, so continuity drops below 1 but stays
    # high. It is not exactly 8/9: a departed player is weighted by his plate
    # appearances per *team* game, which keeps falling as the club plays on
    # without him. That decay is inherited from the NBA module and is pinned
    # separately in ``test_a_departed_players_weight_decays_after_he_leaves``.
    assert 8 / 9 <= mets["TEAM_ROSTER_BATTING_CONTINUITY_PCT_BEFORE"] < 1.0
    # The Yankees kept everyone and took him in at his Mets workload.
    assert yankees["TEAM_ROSTER_BATTING_CONTINUITY_PCT_BEFORE"] == 1.0
    assert yankees["TEAM_ROSTER_BATTING_INCOMING_PCT_BEFORE"] == pytest.approx(
        4.0 / TEAM_PLAYING_TIME_PER_GAME, abs=1e-6
    )
    # Net is incoming minus lost, and the Yankees lost nobody.
    assert yankees["TEAM_ROSTER_BATTING_NET_PCT_BEFORE"] == pytest.approx(
        yankees["TEAM_ROSTER_BATTING_INCOMING_PCT_BEFORE"], abs=1e-6
    )
    assert mets["TEAM_ROSTER_BATTING_NET_PCT_BEFORE"] < 0


def test_a_departed_players_weight_decays_after_he_leaves():
    """Continuity recovers as the club establishes playing time without him.

    A hitter lost yesterday still carries close to his full workload; one lost
    three months ago has been replaced, and the roster on the field today really
    is more settled. The two-month column is what keeps the recent shock visible.
    """
    team_games, batters, pitchers = _simple_history()
    trade_date = pd.Timestamp("2025-04-10")
    moved = (batters["player_id"] == "m0") & (
        pd.to_datetime(batters["game_date"]) >= trade_date
    )
    batters = batters.loc[~moved].copy()
    transactions = pd.DataFrame(
        [
            {
                "player_id": "m0",
                "known_date": trade_date.date(),
                "from_team_id": METS,
                "to_team_id": YANKEES,
            }
        ]
    )
    features = build_team_roster_features(
        team_games, batters, pitchers, transactions
    ).merge(
        team_games[["game_pk", "team_id", "season_year", "game_date"]],
        on=["game_pk", "team_id"],
    )
    mets = features.loc[
        (features["team_id"] == METS) & (features["season_year"] == 2025)
    ].copy()
    mets["game_date"] = pd.to_datetime(mets["game_date"])
    after = mets.loc[mets["game_date"] > trade_date].sort_values("game_date")
    continuity = after["TEAM_ROSTER_BATTING_CONTINUITY_PCT_BEFORE"]

    assert (continuity < 1.0).all()
    assert continuity.is_monotonic_increasing


def test_a_transaction_is_read_on_its_known_date_never_its_effective_date():
    """IL and assignment rows are routinely backdated; only ``known_date`` counts.

    The fixture is shaped like a real backdated move: the hitter stops appearing
    on 6 April, but the transaction is not announced until 20 April. Nothing
    knowable on 10 April says he has gone -- a missing name in a box score is not
    a departure -- so continuity must hold at 1.0 right up to the announcement.
    """
    team_games, batters, pitchers = _simple_history()
    effective = pd.Timestamp("2025-04-06")
    announced = pd.Timestamp("2025-04-20")
    stopped = (batters["player_id"] == "m0") & (
        pd.to_datetime(batters["game_date"]) >= effective
    )
    batters = batters.loc[~stopped].copy()

    transactions = pd.DataFrame(
        [
            {
                "player_id": "m0",
                "known_date": announced.date(),
                "effective_date": effective.date(),
                "from_team_id": METS,
                "to_team_id": DODGERS,
            }
        ]
    )
    features = build_team_roster_features(
        team_games, batters, pitchers, transactions
    ).merge(team_games[["game_pk", "team_id", "game_date"]], on=["game_pk", "team_id"])
    mets = features.loc[features["team_id"] == METS].copy()
    mets["game_date"] = pd.to_datetime(mets["game_date"])
    # 2024 is the first season here and has no season-over-season baseline.
    mets = mets.loc[mets["game_date"] >= pd.Timestamp("2025-01-01")]

    backdated = mets.loc[
        (mets["game_date"] > effective) & (mets["game_date"] < announced)
    ]
    after = mets.loc[mets["game_date"] > announced]

    assert not backdated.empty
    assert (backdated["TEAM_ROSTER_BATTING_CONTINUITY_PCT_BEFORE"] == 1.0).all()
    assert (after["TEAM_ROSTER_BATTING_CONTINUITY_PCT_BEFORE"] < 1.0).all()


def test_same_date_games_share_one_pregame_state():
    """A doubleheader's second game must not see the first game's roster."""
    team_games, batters, pitchers = _simple_history()
    # Duplicate one date as a doubleheader for both clubs.
    doubleheader = team_games.loc[team_games["game_pk"] == "50"].copy()
    doubleheader["game_pk"] = "5000"
    team_games = pd.concat([team_games, doubleheader], ignore_index=True)

    features = build_team_roster_features(team_games, batters, pitchers)
    indexed = features.set_index(["game_pk", "team_id"])
    for column in roster_feature_columns():
        assert indexed.at[("50", METS), column] == pytest.approx(
            indexed.at[("5000", METS), column], nan_ok=True
        )


def test_a_first_season_has_no_season_over_season_baseline():
    team_games, batters, pitchers = _simple_history()
    features = build_team_roster_features(team_games, batters, pitchers).merge(
        team_games[["game_pk", "team_id", "season_year"]], on=["game_pk", "team_id"]
    )
    first = features.loc[features["season_year"] == 2024]

    # 2024 is the earliest season here, so the full window has nothing to
    # compare against and must be NaN rather than a fabricated 1.0.
    assert first["TEAM_ROSTER_BATTING_CONTINUITY_PCT_BEFORE"].isna().all()


def test_batting_and_pitching_are_never_pooled():
    """A pitching-only change must not move the batting columns, or vice versa."""
    team_games = _team_games()
    assignments = {
        METS: [(f"m{i}", "bat") for i in range(9)] + [("mp0", "pitch")],
        YANKEES: [(f"y{i}", "bat") for i in range(9)] + [("yp0", "pitch")],
    }
    batters, pitchers = _appearances(team_games, assignments)

    trade_date = pd.Timestamp("2025-04-20")
    moved = (pitchers["player_id"] == "mp0") & (
        pd.to_datetime(pitchers["game_date"]) >= trade_date
    )
    pitchers = pitchers.loc[~moved].copy()
    transactions = pd.DataFrame(
        [
            {
                "player_id": "mp0",
                "known_date": trade_date.date(),
                "from_team_id": METS,
                "to_team_id": YANKEES,
            }
        ]
    )
    features = build_team_roster_features(
        team_games, batters, pitchers, transactions
    ).merge(team_games[["game_pk", "team_id", "game_date"]], on=["game_pk", "team_id"])
    after = features.loc[pd.to_datetime(features["game_date"]) > trade_date]
    mets = after.loc[after["team_id"] == METS].iloc[0]

    assert mets["TEAM_ROSTER_PITCHING_CONTINUITY_PCT_BEFORE"] < 1.0
    assert mets["TEAM_ROSTER_BATTING_CONTINUITY_PCT_BEFORE"] == 1.0


def test_the_wide_pivot_emits_both_sides_and_the_named_diffs():
    team_games, batters, pitchers = _simple_history()
    targets = ["60", "61", "62"]
    closing = pd.DataFrame({"GAME_ID": targets})
    wide = build_roster_features(
        team_games, batters, pitchers, closing, target_game_ids=targets
    )

    assert len(wide) == len(targets)
    for column in roster_feature_columns():
        assert f"{column}_TEAM_HOME" in wide.columns
        assert f"{column}_TEAM_AWAY" in wide.columns
    assert "TEAM_ROSTER_BATTING_CONTINUITY_PCT_DIFF_BEFORE" in wide.columns
    assert "TEAM_ROSTER_PITCHING_CONTINUITY_PCT_DIFF_BEFORE" in wide.columns
    # Every emitted column declares the pregame contract.
    assert all(column == "GAME_ID" or "_BEFORE" in column for column in wide.columns)
