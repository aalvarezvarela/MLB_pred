"""The name map is the single most fragile part of any multi-source pipeline.

These tests exist because a silent miss here does not fail loudly -- it
produces a training set quietly missing a team for a season.
"""

import pytest

from mlb_pred.config import constants as c


def test_every_canonical_name_has_id_league_division_and_abbreviation():
    assert set(c.TEAM_ID_MAP) == set(c.TEAM_NAME_LEAGUE_MAP)
    assert set(c.TEAM_ID_MAP) == set(c.TEAM_NAME_DIVISION_MAP)
    assert set(c.TEAM_ID_MAP) == set(c.TEAM_ABBREVIATION_MAP)
    assert len(c.TEAM_ID_MAP) == 30


def test_team_ids_are_unique_and_textual():
    ids = list(c.TEAM_ID_MAP.values())
    assert len(set(ids)) == len(ids)
    assert all(isinstance(i, str) for i in ids)


def test_every_alias_resolves_to_a_canonical_name():
    for alias, canonical in c.TEAM_NAME_STANDARDIZATION.items():
        assert canonical in c.TEAM_ID_MAP, f"{alias!r} -> unknown {canonical!r}"


def test_standardization_is_idempotent():
    for canonical in c.TEAM_ID_MAP:
        assert c.standardize_team_name(canonical) == canonical


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("LAD", "Los Angeles Dodgers"),
        ("L.A. Dodgers", "Los Angeles Dodgers"),
        ("Cleveland Indians", "Cleveland Guardians"),
        ("Oakland Athletics", "Athletics"),
        ("Tampa Bay Devil Rays", "Tampa Bay Rays"),
        ("  Boston Red Sox  ", "Boston Red Sox"),
    ],
)
def test_known_aliases(alias, expected):
    assert c.standardize_team_name(alias) == expected


def test_unknown_name_raises_rather_than_returning_none():
    # A silent drop costs a season of results; a loud failure costs an hour.
    with pytest.raises(c.UnknownTeamNameError):
        c.standardize_team_name("Portland Beavers")
    with pytest.raises(c.UnknownTeamNameError):
        c.standardize_team_name(None)


def test_relocated_franchises_keep_one_id():
    # Ids never move; only display names do.
    assert c.team_id_for_name("Oakland Athletics") == c.team_id_for_name("Athletics")
    assert c.team_id_for_name("Cleveland Indians") == c.team_id_for_name(
        "Cleveland Guardians"
    )


def test_ingested_game_types_exclude_non_competitive_games():
    for excluded in ("S", "E", "I", "A"):
        assert excluded not in c.INGESTED_GAME_TYPES
    assert "R" in c.INGESTED_GAME_TYPES
    assert c.POSTSEASON_GAME_TYPES <= c.INGESTED_GAME_TYPES


@pytest.mark.parametrize(
    ("provider_name", "expected"),
    [
        # SportsbookReview spellings, collected across 2019-2025 slates.
        ("Athletics Athletics", "Athletics"),
        ("Oakland Athletics", "Athletics"),
        ("Cleveland Guardians", "Cleveland Guardians"),
        ("St. Louis Cardinals", "St. Louis Cardinals"),
    ],
)
def test_sportsbookreview_team_names_resolve(provider_name, expected):
    """SBR builds fullName as "<location> <nickname>".

    That produces "Athletics Athletics" now the club has no city -- caught by
    the raise-on-unknown guard on the very first odds scrape rather than by a
    season of quietly missing games.
    """
    assert c.standardize_team_name(provider_name) == expected
