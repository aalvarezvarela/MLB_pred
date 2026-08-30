"""Live contract tests against the MLB Stats API.

Skipped by default -- they need network and take a few seconds. Run with::

    pytest -m live

They exist because the failure they catch is not a code bug: the provider
changing a field name or dropping a hydration silently produces empty columns
that no unit test would notice.
"""

from datetime import date

import pandas as pd
import pytest

pytestmark = pytest.mark.live

SAMPLE_DATE = date(2025, 8, 27)


@pytest.fixture(scope="module")
def slate():
    from mlb_pred.fetch_data.statsapi.schedule import fetch_schedule

    return fetch_schedule(SAMPLE_DATE, SAMPLE_DATE)


def test_schedule_returns_games_with_both_date_forms(slate):
    games, _, _ = slate
    assert len(games) > 10

    # The slate date joins; the UTC instant orders. Both must be populated.
    assert games["game_date"].notna().all()
    assert games["first_pitch_utc"].notna().all()
    assert pd.api.types.is_datetime64_any_dtype(games["first_pitch_utc"])


def test_game_ids_are_unique_after_deduplication(slate):
    games, _, _ = slate
    assert not games["game_pk"].duplicated().any()


def test_every_game_has_a_home_plate_umpire(slate):
    games, umpires, _ = slate
    with_hp = set(umpires[umpires["is_home_plate"]]["game_pk"])
    assert set(games["game_pk"]) <= with_hp


def test_every_game_has_two_nine_slot_lineups(slate):
    games, _, lineups = slate
    per_game = lineups.groupby("game_pk").size()
    assert set(games["game_pk"]) <= set(per_game.index)
    assert (per_game.loc[list(games["game_pk"])] == 18).all()


def test_boxscore_yields_all_three_grains(slate):
    from mlb_pred.fetch_data.statsapi.boxscore import (
        fetch_game_grains,
        game_meta_from_schedule,
    )

    games, _, _ = slate
    parsed = fetch_game_grains(game_meta_from_schedule(games)[0])

    assert len(parsed["team_games"]) == 2
    assert len(parsed["batter_games"]) >= 18
    assert len(parsed["pitcher_appearances"]) >= 2
    assert any(row["is_starter"] for row in parsed["pitcher_appearances"])


def test_team_game_runs_reconcile_with_the_schedule_score(slate):
    from mlb_pred.fetch_data.statsapi.boxscore import (
        fetch_game_grains,
        game_meta_from_schedule,
    )

    games, _, _ = slate
    game = games.iloc[0]
    parsed = fetch_game_grains(game_meta_from_schedule(games)[0])

    by_side = {row["home"]: row for row in parsed["team_games"]}
    assert by_side[True]["runs_scored"] == game["home_score"]
    assert by_side[False]["runs_scored"] == game["away_score"]


def test_venues_expose_park_geometry_and_orientation():
    from mlb_pred.fetch_data.statsapi.venues import fetch_venues

    venues = fetch_venues(2025)
    coors = venues[venues["venue_name"] == "Coors Field"].iloc[0]

    # Elevation is the whole reason Coors plays the way it does.
    assert float(coors["elevation_ft"]) > 5000
    # Without the azimuth a wind direction is uninterpretable across parks.
    assert pd.notna(coors["azimuth_angle"])
    assert coors["center"] > 0


def test_transaction_feed_still_publishes_both_dates():
    from mlb_pred.fetch_data.statsapi.transactions import fetch_transactions

    transactions = fetch_transactions("2025-08-20", "2025-08-27")
    assert not transactions.empty
    assert transactions["known_date"].notna().all()

    # If this stops being true, the backdating guard is unenforceable and the
    # availability features become genuinely leaky.
    assert transactions["effective_date"].notna().any()
    assert transactions["is_backdated"].any()
