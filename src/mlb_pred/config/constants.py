"""
MLB Predictor - Constants Module

Canonical entity maps and tuning constants. Mirrors
``nba_ou.config.constants`` in the NBA repository.

The MLB Stats API (``statsapi.mlb.com``) is the primary provider, and its ids
are adopted as canonical throughout the project:

    game    -> ``gamePk``      (globally unique; doubleheaders get distinct pks)
    team    -> ``team.id``     (stable across relocations and rebrands)
    player  -> ``person.id``   (MLBAM id; the id every other baseball source
                                can be crosswalked to)
    umpire  -> ``official.id`` (an MLBAM person id like any other)
    venue   -> ``venue.id``

Names are mapped onto canonical names; ids are never remapped.
"""

from __future__ import annotations

# ==============================================================================
# TEAM MAPPINGS
# ==============================================================================

# Canonical full team name -> MLB Stats API team id (stored as TEXT, matching
# the NBA repo's convention of keeping all provider ids as strings so that a
# missing value never silently coerces an id column to float).
TEAM_ID_MAP: dict[str, str] = {
    "Arizona Diamondbacks": "109",
    "Athletics": "133",
    "Atlanta Braves": "144",
    "Baltimore Orioles": "110",
    "Boston Red Sox": "111",
    "Chicago Cubs": "112",
    "Chicago White Sox": "145",
    "Cincinnati Reds": "113",
    "Cleveland Guardians": "114",
    "Colorado Rockies": "115",
    "Detroit Tigers": "116",
    "Houston Astros": "117",
    "Kansas City Royals": "118",
    "Los Angeles Angels": "108",
    "Los Angeles Dodgers": "119",
    "Miami Marlins": "146",
    "Milwaukee Brewers": "158",
    "Minnesota Twins": "142",
    "New York Mets": "121",
    "New York Yankees": "147",
    "Philadelphia Phillies": "143",
    "Pittsburgh Pirates": "134",
    "San Diego Padres": "135",
    "San Francisco Giants": "137",
    "Seattle Mariners": "136",
    "St. Louis Cardinals": "138",
    "Tampa Bay Rays": "139",
    "Texas Rangers": "140",
    "Toronto Blue Jays": "141",
    "Washington Nationals": "120",
}

TEAM_ID_TO_NAME: dict[str, str] = {v: k for k, v in TEAM_ID_MAP.items()}

# Canonical full team name -> the abbreviation the Stats API publishes.
TEAM_ABBREVIATION_MAP: dict[str, str] = {
    "Arizona Diamondbacks": "AZ",
    "Athletics": "ATH",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

# Every spelling any source has ever produced -> one canonical team name.
#
# This is a hand-maintained, versioned data artifact, not an algorithm. An
# unknown name must stop the pipeline (see ``standardize_team_name``): a
# relocation, a rebrand, or an odds page switching "LA Dodgers" to
# "L.A. Dodgers" is a real change that needs a human decision, and silently
# dropping the unmatched rows would leave a training set quietly missing a
# team for a season with nothing to surface it.
#
# Fuzzy matching belongs in a one-off script that *proposes additions here*,
# never in the pipeline itself.
TEAM_NAME_STANDARDIZATION: dict[str, str] = {
    # --- canonical names map to themselves -------------------------------
    **{name: name for name in TEAM_ID_MAP},
    # --- Stats API abbreviations -----------------------------------------
    "AZ": "Arizona Diamondbacks",
    "ARI": "Arizona Diamondbacks",
    "ATH": "Athletics",
    "OAK": "Athletics",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CHN": "Chicago Cubs",
    "CWS": "Chicago White Sox",
    "CHW": "Chicago White Sox",
    "CHA": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC": "Kansas City Royals",
    "KCR": "Kansas City Royals",
    "KCA": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "ANA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "LAN": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "FLA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYN": "New York Mets",
    "NYY": "New York Yankees",
    "NYA": "New York Yankees",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres",
    "SDP": "San Diego Padres",
    "SDN": "San Diego Padres",
    "SF": "San Francisco Giants",
    "SFG": "San Francisco Giants",
    "SFN": "San Francisco Giants",
    "SEA": "Seattle Mariners",
    "STL": "St. Louis Cardinals",
    "SLN": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays",
    "TBR": "Tampa Bay Rays",
    "TBA": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals",
    "WSN": "Washington Nationals",
    "WAS": "Washington Nationals",
    # --- city / location only (odds feeds mostly use these) --------------
    "Arizona": "Arizona Diamondbacks",
    "Phoenix": "Arizona Diamondbacks",
    "Atlanta": "Atlanta Braves",
    "Baltimore": "Baltimore Orioles",
    "Boston": "Boston Red Sox",
    "Cincinnati": "Cincinnati Reds",
    "Cleveland": "Cleveland Guardians",
    "Colorado": "Colorado Rockies",
    "Denver": "Colorado Rockies",
    "Detroit": "Detroit Tigers",
    "Houston": "Houston Astros",
    "Kansas City": "Kansas City Royals",
    "Miami": "Miami Marlins",
    "Milwaukee": "Milwaukee Brewers",
    "Minnesota": "Minnesota Twins",
    "Minneapolis": "Minnesota Twins",
    "Philadelphia": "Philadelphia Phillies",
    "Pittsburgh": "Pittsburgh Pirates",
    "San Diego": "San Diego Padres",
    "San Francisco": "San Francisco Giants",
    "Seattle": "Seattle Mariners",
    "St. Louis": "St. Louis Cardinals",
    "St Louis": "St. Louis Cardinals",
    "Tampa Bay": "Tampa Bay Rays",
    "Tampa": "Tampa Bay Rays",
    "Texas": "Texas Rangers",
    "Toronto": "Toronto Blue Jays",
    "Washington": "Washington Nationals",
    "Sacramento": "Athletics",
    "Oakland": "Athletics",
    # --- nickname only ---------------------------------------------------
    "D-backs": "Arizona Diamondbacks",
    "Diamondbacks": "Arizona Diamondbacks",
    "Braves": "Atlanta Braves",
    "Orioles": "Baltimore Orioles",
    "Red Sox": "Boston Red Sox",
    "Cubs": "Chicago Cubs",
    "White Sox": "Chicago White Sox",
    "Reds": "Cincinnati Reds",
    "Guardians": "Cleveland Guardians",
    "Rockies": "Colorado Rockies",
    "Tigers": "Detroit Tigers",
    "Astros": "Houston Astros",
    "Royals": "Kansas City Royals",
    "Angels": "Los Angeles Angels",
    "Dodgers": "Los Angeles Dodgers",
    "Marlins": "Miami Marlins",
    "Brewers": "Milwaukee Brewers",
    "Twins": "Minnesota Twins",
    "Mets": "New York Mets",
    "Yankees": "New York Yankees",
    "Phillies": "Philadelphia Phillies",
    "Pirates": "Pittsburgh Pirates",
    "Padres": "San Diego Padres",
    "Giants": "San Francisco Giants",
    "Mariners": "Seattle Mariners",
    "Cardinals": "St. Louis Cardinals",
    "Rays": "Tampa Bay Rays",
    "Rangers": "Texas Rangers",
    "Blue Jays": "Toronto Blue Jays",
    "Nationals": "Washington Nationals",
    # --- historical names and relocations --------------------------------
    # ids are stable across all of these; only the display name changed.
    "Cleveland Indians": "Cleveland Guardians",
    "Florida Marlins": "Miami Marlins",
    "Tampa Bay Devil Rays": "Tampa Bay Rays",
    "Anaheim Angels": "Los Angeles Angels",
    "Los Angeles Angels of Anaheim": "Los Angeles Angels",
    "Oakland Athletics": "Athletics",
    "Oakland A's": "Athletics",
    "Sacramento Athletics": "Athletics",
    "A's": "Athletics",
    # SportsbookReview builds fullName as "<location> <nickname>". The club
    # dropped its city in 2025, so both halves are now "Athletics" and SBR
    # emits the doubled string. Found by the raise-on-unknown guard on the
    # first odds scrape, which is exactly what it is for.
    "Athletics Athletics": "Athletics",
    "Montreal Expos": "Washington Nationals",
    # --- alternative spellings odds feeds produce -------------------------
    "LA Angels": "Los Angeles Angels",
    "L.A. Angels": "Los Angeles Angels",
    "LA Dodgers": "Los Angeles Dodgers",
    "L.A. Dodgers": "Los Angeles Dodgers",
    "NY Mets": "New York Mets",
    "N.Y. Mets": "New York Mets",
    "NY Yankees": "New York Yankees",
    "N.Y. Yankees": "New York Yankees",
    "Chi Cubs": "Chicago Cubs",
    "Chi White Sox": "Chicago White Sox",
    "Arizona D-backs": "Arizona Diamondbacks",
}

TEAM_NAME_LEAGUE_MAP: dict[str, str] = {
    "Baltimore Orioles": "AL",
    "Boston Red Sox": "AL",
    "New York Yankees": "AL",
    "Tampa Bay Rays": "AL",
    "Toronto Blue Jays": "AL",
    "Chicago White Sox": "AL",
    "Cleveland Guardians": "AL",
    "Detroit Tigers": "AL",
    "Kansas City Royals": "AL",
    "Minnesota Twins": "AL",
    "Athletics": "AL",
    "Houston Astros": "AL",
    "Los Angeles Angels": "AL",
    "Seattle Mariners": "AL",
    "Texas Rangers": "AL",
    "Atlanta Braves": "NL",
    "Miami Marlins": "NL",
    "New York Mets": "NL",
    "Philadelphia Phillies": "NL",
    "Washington Nationals": "NL",
    "Chicago Cubs": "NL",
    "Cincinnati Reds": "NL",
    "Milwaukee Brewers": "NL",
    "Pittsburgh Pirates": "NL",
    "St. Louis Cardinals": "NL",
    "Arizona Diamondbacks": "NL",
    "Colorado Rockies": "NL",
    "Los Angeles Dodgers": "NL",
    "San Diego Padres": "NL",
    "San Francisco Giants": "NL",
}

TEAM_NAME_DIVISION_MAP: dict[str, str] = {
    "Baltimore Orioles": "AL East",
    "Boston Red Sox": "AL East",
    "New York Yankees": "AL East",
    "Tampa Bay Rays": "AL East",
    "Toronto Blue Jays": "AL East",
    "Chicago White Sox": "AL Central",
    "Cleveland Guardians": "AL Central",
    "Detroit Tigers": "AL Central",
    "Kansas City Royals": "AL Central",
    "Minnesota Twins": "AL Central",
    "Athletics": "AL West",
    "Houston Astros": "AL West",
    "Los Angeles Angels": "AL West",
    "Seattle Mariners": "AL West",
    "Texas Rangers": "AL West",
    "Atlanta Braves": "NL East",
    "Miami Marlins": "NL East",
    "New York Mets": "NL East",
    "Philadelphia Phillies": "NL East",
    "Washington Nationals": "NL East",
    "Chicago Cubs": "NL Central",
    "Cincinnati Reds": "NL Central",
    "Milwaukee Brewers": "NL Central",
    "Pittsburgh Pirates": "NL Central",
    "St. Louis Cardinals": "NL Central",
    "Arizona Diamondbacks": "NL West",
    "Colorado Rockies": "NL West",
    "Los Angeles Dodgers": "NL West",
    "San Diego Padres": "NL West",
    "San Francisco Giants": "NL West",
}


class UnknownTeamNameError(RuntimeError):
    """Raised when a source produces a team name that is not in the map."""


def standardize_team_name(name: str) -> str:
    """Map any known spelling of a team onto its canonical name.

    Raises ``UnknownTeamNameError`` rather than returning ``None`` or dropping
    the row. See the comment on ``TEAM_NAME_STANDARDIZATION`` for why the
    failure has to be loud.
    """
    if name is None:
        raise UnknownTeamNameError("Team name is None.")

    cleaned = str(name).strip()
    if cleaned in TEAM_NAME_STANDARDIZATION:
        return TEAM_NAME_STANDARDIZATION[cleaned]

    raise UnknownTeamNameError(
        f"Unknown team name {cleaned!r}. Add it to "
        "TEAM_NAME_STANDARDIZATION in mlb_pred/config/constants.py. "
        "Do not silently drop the row -- an unmapped name usually means a "
        "rebrand, a relocation, or a provider changing its spelling."
    )


def team_id_for_name(name: str) -> str:
    """Canonical Stats API team id for any known spelling of a team name."""
    return TEAM_ID_MAP[standardize_team_name(name)]


def is_known_team_id(team_id: str | None) -> bool:
    """True when ``team_id`` is one of the 30 franchises.

    The Stats API publishes *bracket placeholders* as if they were teams --
    "AL Wild Card #1", "NL Higher Seed", "Higher Seed League Champion" -- on
    postseason games whose participants are not yet determined. They carry ids
    well outside the franchise range (2710, 4613, 5517, ...).

    Identity is checked on the **id**, not the name: placeholder names vary
    freely and change between rounds, while the id range is structural. That
    also keeps the name map doing exactly one job -- resolving real franchises
    -- instead of accumulating bracket vocabulary.
    """
    return team_id is not None and str(team_id) in TEAM_ID_TO_NAME


# ==============================================================================
# SEASON / GAME TYPE
# ==============================================================================

# ``gameType`` codes as published by /api/v1/gameTypes. The Stats API encodes
# the structural distinction in this code, so filters key off it rather than
# off the human-readable ``seriesDescription`` text (which is maintained by
# hand upstream and drifts -- the same reasoning as the NBA repo deriving
# season type from the game-id prefix instead of the SEASON_TYPE column).
GAME_TYPE_MAP: dict[str, str] = {
    "S": "Spring Training",
    "R": "Regular Season",
    "F": "Wild Card",
    "D": "Division Series",
    "L": "League Championship Series",
    "W": "World Series",
    "C": "Championship",
    "P": "Postseason",
    "A": "All-Star Game",
    "I": "Intrasquad",
    "E": "Exhibition",
}

REGULAR_SEASON_GAME_TYPES: frozenset[str] = frozenset({"R"})
POSTSEASON_GAME_TYPES: frozenset[str] = frozenset({"F", "D", "L", "W", "C", "P"})

# What the pipeline ingests. Spring training, exhibitions, intrasquad games and
# the All-Star Game are excluded: they are not played under competitive
# conditions and would poison every rolling mean built on top.
INGESTED_GAME_TYPES: frozenset[str] = REGULAR_SEASON_GAME_TYPES | POSTSEASON_GAME_TYPES

# First season of the backfill. Statcast (barrel rate, xwOBA, launch speed)
# begins in 2015, so starting here keeps the quality-of-contact feature family
# defined on every training row instead of NULL for a third of them.
FIRST_BACKFILL_SEASON: int = 2015

# ==============================================================================
# PLAUSIBILITY BOUNDS (schema-boundary data-quality guards)
# ==============================================================================

# The NBA repo guards its games table with ``pts INTEGER NOT NULL CHECK
# (pts > 40)`` -- a floor, because a basketball team cannot score 40. Baseball
# needs the mirror image: 0 runs is perfectly legal, so the guard is a ceiling
# plus a plausibility band on innings. These catch a box score fetched
# mid-game, which is the failure mode that silently poisons an insert-only
# table.
MAX_PLAUSIBLE_TEAM_RUNS: int = 40
MAX_PLAUSIBLE_GAME_OUTS: int = 120  # deep extra innings
MIN_REGULATION_INNINGS: float = 5.0  # an official (rain-shortened) game

# An absolute floor on outs is nearly useless in baseball, and a naive one is
# actively wrong. A 24-out floor ("nine innings, home team winning") rejects
# every rain-shortened game and every 2020-21 seven-inning doubleheader --
# 523 legitimate rows, 1% of the store.
#
# The floor below is the true minimum for an *official* game: five innings
# with the home team ahead, so the away staff records only four innings.
MIN_PLAUSIBLE_GAME_OUTS: int = 12

# The check that actually earns its keep is relative to the game's own length:
# each team's outs must fall within one inning of the innings played. Verified
# across all 51,266 team-game rows in the store with zero violations, while
# still catching the failure that matters -- a box score fetched mid-game,
# whose outs sit far below the game's final inning count.

# ==============================================================================
# ROLLING / FEATURE TUNING (mirrors the NBA repo's constants block)
# ==============================================================================

DEFAULT_ROLLING_WINDOW: int = 10
WEIGHTED_ROLLING_WINDOW: int = 20

# A starting pitcher accumulates history on his own clock -- roughly every
# fifth team game -- so his rolling window is counted in *appearances*, not in
# team games.
DEFAULT_STARTER_ROLLING_WINDOW: int = 5

# Bullpen fatigue: MLB's closest analogue to NBA rest/fatigue. Counted in days
# because relief usage recovers on a calendar clock, not a game clock.
BULLPEN_USAGE_LOOKBACK_DAYS: int = 3

# ==============================================================================
# LINEUP / ROSTER STRUCTURE
# ==============================================================================

BATTING_ORDER_SLOTS: int = 9
ACTIVE_ROSTER_SIZE: int = 26

STARTING_PITCHER_POSITION_CODE: str = "1"

# Position codes as the Stats API publishes them.
POSITION_CODE_MAP: dict[str, str] = {
    "1": "P",
    "2": "C",
    "3": "1B",
    "4": "2B",
    "5": "3B",
    "6": "SS",
    "7": "LF",
    "8": "CF",
    "9": "RF",
    "10": "DH",
    "Y": "PH",
    "X": "PR",
}

# ==============================================================================
# UMPIRES
# ==============================================================================

# The home-plate umpire is the only crew position that measurably moves run
# scoring (strike-zone size drives K% and BB%). The other three are stored so
# that a crew-level feature stays possible, but HP is the one that matters.
HOME_PLATE_OFFICIAL_TYPE: str = "Home Plate"
OFFICIAL_TYPES: tuple[str, ...] = (
    "Home Plate",
    "First Base",
    "Second Base",
    "Third Base",
    "Left Field",
    "Right Field",
)

# ==============================================================================
# AVAILABILITY
# ==============================================================================

# Transaction ``typeCode`` values that change whether a player is available.
INJURY_TRANSACTION_MARKERS: tuple[str, ...] = (
    "injured list",
    "bereavement list",
    "paternity list",
    "restricted list",
    "family medical emergency list",
)
