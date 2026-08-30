"""Per-game box score -> the three accumulation grains.

The single most consequential decision in this data layer, inherited from the
NBA repo, is that **history is stored at the grain at which it accumulates,
not at the grain at which predictions are made**. Baseball needs three such
grains rather than two:

``team_games``
    Two rows per game, home/away as a boolean column and never as a column
    suffix. Every team-history feature is then
    ``groupby("team_id")[col].shift(1).rolling(n).mean()`` -- one line,
    obviously leakage-safe, identical code for every statistic.

``batter_games``
    One row per batter per game.

``pitcher_appearances``
    One row per pitcher per game. This grain has no NBA analogue and is the
    reason MLB needs three: a starting pitcher's history is the strongest
    team-independent signal for a run total, and it accumulates on its own
    clock -- roughly every fifth team game, not every game. Treating a starter
    as "a player who happened to start" would bury that history inside a
    player table keyed on team games and make the natural rolling window
    (last 5 *starts*) impossible to express.

The game-level, home/away wide form is produced once, at the end, by a single
merge step -- never here.
"""

from __future__ import annotations

from typing import Any, overload

import pandas as pd

from mlb_pred.config.constants import HOME_PLATE_OFFICIAL_TYPE
from mlb_pred.fetch_data.statsapi.client import statsapi_get
from mlb_pred.utils.general_utils import as_id

SIDES = ("home", "away")


# ---------------------------------------------------------------------------
# Scalar coercion
# ---------------------------------------------------------------------------
@overload
def _int(stats: dict, key: str, default: None) -> int | None: ...


@overload
def _int(stats: dict, key: str, default: int = 0) -> int: ...


def _int(stats: dict, key: str, default: int | None = 0) -> int | None:
    value = stats.get(key)
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _rate(numerator: float | None, denominator: float | None) -> float | None:
    """Scale-free rate, or None when the denominator cannot support one.

    Rates rather than raw counts are what make a statistic comparable across
    opponents and eras, and what lets the tempo and efficiency factors be
    recombined against a specific opponent later.
    """
    if numerator is None or not denominator:
        return None
    return round(float(numerator) / float(denominator), 6)


def innings_pitched_to_outs(innings_pitched: Any) -> int | None:
    """Convert baseball's ``7.1``/``7.2`` innings notation into whole outs.

    ``7.1`` means seven innings and one out, not 7.1 innings. Averaging the
    raw string-as-float is a real and easy mistake -- it treats a third of an
    inning as a tenth -- so outs are the stored quantity and innings are only
    ever derived back out of them.
    """
    if innings_pitched is None or innings_pitched == "":
        return None
    try:
        text = str(innings_pitched)
        whole, _, fraction = text.partition(".")
        outs = int(whole) * 3
        if fraction:
            partial = int(fraction[0])
            if partial not in (0, 1, 2):
                return None
            outs += partial
        return outs
    except (TypeError, ValueError):
        return None


def _batting_order_slot(raw: Any) -> tuple[int | None, int | None]:
    """Split the Stats API ``battingOrder`` code into (slot, substitution index).

    ``"300"`` is the third slot's starter; ``"301"`` is the first player to
    replace him there. ``//100`` is the lineup slot, ``%100`` the depth.
    """
    if raw in (None, ""):
        return None, None
    try:
        code = int(raw)
    except (TypeError, ValueError):
        return None, None
    return code // 100, code % 100


# ---------------------------------------------------------------------------
# Team-game grain
# ---------------------------------------------------------------------------
def _team_game_row(boxscore: dict, game_meta: dict, side: str) -> dict:
    opponent = "away" if side == "home" else "home"
    team = boxscore["teams"][side]
    other = boxscore["teams"][opponent]

    bat = team["teamStats"]["batting"]
    pit = team["teamStats"]["pitching"]
    fld = team["teamStats"].get("fielding", {})
    opp_bat = other["teamStats"]["batting"]

    # ---- offence: volume ------------------------------------------------
    plate_appearances = _int(bat, "plateAppearances")
    at_bats = _int(bat, "atBats")
    hits = _int(bat, "hits")
    doubles = _int(bat, "doubles")
    triples = _int(bat, "triples")
    home_runs = _int(bat, "homeRuns")
    singles = (
        None
        if None in (hits, doubles, triples, home_runs)
        else hits - doubles - triples - home_runs
    )
    walks = _int(bat, "baseOnBalls")
    strikeouts = _int(bat, "strikeOuts")
    hbp = _int(bat, "hitByPitch")
    sac_flies = _int(bat, "sacFlies")
    total_bases = _int(bat, "totalBases")
    runs_scored = _int(bat, "runs")

    # ---- pitching: volume ----------------------------------------------
    outs_recorded: int | None = _int(pit, "outs")
    if not outs_recorded:
        outs_recorded = innings_pitched_to_outs(pit.get("inningsPitched"))
    batters_faced = _int(pit, "battersFaced")
    pitches_thrown = _int(pit, "pitchesThrown") or _int(pit, "numberOfPitches")
    hits_allowed = _int(pit, "hits")
    walks_allowed = _int(pit, "baseOnBalls")
    hbp_allowed = _int(pit, "hitBatsmen")
    strikeouts_thrown = _int(pit, "strikeOuts")
    home_runs_allowed = _int(pit, "homeRuns")
    earned_runs = _int(pit, "earnedRuns")
    runs_allowed = _int(pit, "runs")
    if runs_allowed is None:
        runs_allowed = _int(opp_bat, "runs")

    baserunners_allowed = (
        None
        if None in (hits_allowed, walks_allowed, hbp_allowed)
        else hits_allowed + walks_allowed + hbp_allowed
    )

    # ---- efficiency: on-base / slugging denominators ---------------------
    obp_denominator = (
        None
        if None in (at_bats, walks, hbp, sac_flies)
        else at_bats + walks + hbp + sac_flies
    )
    times_on_base = None if None in (hits, walks, hbp) else hits + walks + hbp
    balls_in_play = (
        None
        if None in (at_bats, strikeouts, home_runs, sac_flies)
        else at_bats - strikeouts - home_runs + sac_flies
    )

    return {
        # --- identity -----------------------------------------------------
        "game_pk": game_meta["game_pk"],
        "team_id": as_id(team["team"]["id"]),
        "season_year": game_meta["season_year"],
        "game_date": game_meta["game_date"],
        "first_pitch_utc": game_meta["first_pitch_utc"],
        "game_type": game_meta["game_type"],
        "venue_id": game_meta["venue_id"],
        # Home/away is a boolean column, never a column suffix. The wide,
        # one-row-per-game form is a final merge step, not a storage format.
        "home": side == "home",
        "opponent_team_id": as_id(other["team"]["id"]),
        # --- outcome (targets, and the input to every rolling feature) ----
        "runs_scored": runs_scored,
        "runs_allowed": runs_allowed,
        "run_margin": (
            None if None in (runs_scored, runs_allowed) else runs_scored - runs_allowed
        ),
        "total_runs": (
            None if None in (runs_scored, runs_allowed) else runs_scored + runs_allowed
        ),
        "win": (
            None if None in (runs_scored, runs_allowed) else runs_scored > runs_allowed
        ),
        # --- volume / tempo -----------------------------------------------
        # A run total is fundamentally `expected_events x expected_value_per
        # _event`. Both factors are stored separately so they can be
        # recombined against a specific opponent, rather than only their
        # product.
        "plate_appearances": plate_appearances,
        "at_bats": at_bats,
        "outs_recorded": outs_recorded,
        "innings_pitched": _rate(outs_recorded, 3),
        "batters_faced": batters_faced,
        "pitches_thrown": pitches_thrown,
        "baserunners_allowed": baserunners_allowed,
        "left_on_base": _int(bat, "leftOnBase"),
        # --- offensive component counts ------------------------------------
        "hits": hits,
        "singles": singles,
        "doubles": doubles,
        "triples": triples,
        "home_runs": home_runs,
        "total_bases": total_bases,
        "walks": walks,
        "intentional_walks": _int(bat, "intentionalWalks"),
        "strikeouts": strikeouts,
        "hit_by_pitch": hbp,
        "sac_flies": sac_flies,
        "sac_bunts": _int(bat, "sacBunts"),
        "stolen_bases": _int(bat, "stolenBases"),
        "caught_stealing": _int(bat, "caughtStealing"),
        "grounded_into_double_play": _int(bat, "groundIntoDoublePlay"),
        "rbi": _int(bat, "rbi"),
        "ground_outs": _int(bat, "groundOuts"),
        "air_outs": _int(bat, "airOuts"),
        "fly_outs": _int(bat, "flyOuts"),
        "line_outs": _int(bat, "lineOuts"),
        "pop_outs": _int(bat, "popOuts"),
        # --- pitching component counts ------------------------------------
        "hits_allowed": hits_allowed,
        "home_runs_allowed": home_runs_allowed,
        "walks_allowed": walks_allowed,
        "intentional_walks_allowed": _int(pit, "intentionalWalks"),
        "strikeouts_thrown": strikeouts_thrown,
        "hit_batsmen": hbp_allowed,
        "earned_runs": earned_runs,
        "wild_pitches": _int(pit, "wildPitches"),
        "balks": _int(pit, "balks"),
        # These are not published on the team aggregate. NULL is honest; zero
        # would incorrectly mean no reliever inherited a runner all game.
        "inherited_runners": _int(pit, "inheritedRunners", None),
        "inherited_runners_scored": _int(pit, "inheritedRunnersScored", None),
        "pitching_ground_outs": _int(pit, "groundOuts"),
        "pitching_air_outs": _int(pit, "airOuts"),
        "strikes": _int(pit, "strikes"),
        "balls": _int(pit, "balls"),
        "errors": _int(fld, "errors"),
        # --- efficiency rates (scale-free) ---------------------------------
        "obp": _rate(times_on_base, obp_denominator),
        "slg": _rate(total_bases, at_bats),
        "iso": (
            None
            if None in (total_bases, hits, at_bats) or not at_bats
            else round((total_bases - hits) / at_bats, 6)
        ),
        "babip": _rate(
            None if None in (hits, home_runs) else hits - home_runs, balls_in_play
        ),
        "runs_per_pa": _rate(runs_scored, plate_appearances),
        "k_pct": _rate(strikeouts, plate_appearances),
        "bb_pct": _rate(walks, plate_appearances),
        "hr_per_pa": _rate(home_runs, plate_appearances),
        "ground_out_air_out_ratio": _rate(
            _int(bat, "groundOuts"), _int(bat, "airOuts")
        ),
        "k_pct_allowed": _rate(strikeouts_thrown, batters_faced),
        "bb_pct_allowed": _rate(walks_allowed, batters_faced),
        "hr_per_bf_allowed": _rate(home_runs_allowed, batters_faced),
        "whip": _rate(
            (
                None
                if None in (hits_allowed, walks_allowed)
                else hits_allowed + walks_allowed
            ),
            _rate(outs_recorded, 3),
        ),
        "pitches_per_batter_faced": _rate(pitches_thrown, batters_faced),
        "strike_pct": _rate(_int(pit, "strikes"), pitches_thrown),
        # --- roster context -------------------------------------------------
        "batters_used": len(team.get("batters") or []),
        "pitchers_used": len(team.get("pitchers") or []),
        "bullpen_size": len(team.get("bullpen") or []),
        "bench_size": len(team.get("bench") or []),
    }


# ---------------------------------------------------------------------------
# Player grains
# ---------------------------------------------------------------------------
def _batter_rows(boxscore: dict, game_meta: dict, side: str) -> list[dict]:
    team = boxscore["teams"][side]
    team_id = as_id(team["team"]["id"])
    opponent = "away" if side == "home" else "home"
    opponent_team_id = as_id(boxscore["teams"][opponent]["team"]["id"])

    rows = []
    for player in team.get("players", {}).values():
        batting = (player.get("stats") or {}).get("batting") or {}
        if not batting:
            continue

        slot, substitution_index = _batting_order_slot(player.get("battingOrder"))
        position = player.get("position") or {}
        at_bats = _int(batting, "atBats")
        hits = _int(batting, "hits")
        doubles = _int(batting, "doubles")
        triples = _int(batting, "triples")
        home_runs = _int(batting, "homeRuns")

        rows.append(
            {
                "game_pk": game_meta["game_pk"],
                "team_id": team_id,
                "player_id": as_id((player.get("person") or {}).get("id")),
                "season_year": game_meta["season_year"],
                "game_date": game_meta["game_date"],
                "opponent_team_id": opponent_team_id,
                "home": side == "home",
                "player_name": (player.get("person") or {}).get("fullName"),
                "position_code": position.get("code"),
                "position_abbrev": position.get("abbreviation"),
                "lineup_slot": slot,
                "substitution_index": substitution_index,
                "is_starter": substitution_index == 0,
                "plate_appearances": _int(batting, "plateAppearances"),
                # Provider row inclusion changed in 2022. Feature windows must
                # count actual plate appearances, not merely rows in this table.
                "had_plate_appearance": bool(
                    (_int(batting, "plateAppearances") or 0) > 0
                ),
                "at_bats": at_bats,
                "runs": _int(batting, "runs"),
                "hits": hits,
                "singles": (
                    None
                    if None in (hits, doubles, triples, home_runs)
                    else hits - doubles - triples - home_runs
                ),
                "doubles": doubles,
                "triples": triples,
                "home_runs": home_runs,
                "rbi": _int(batting, "rbi"),
                "walks": _int(batting, "baseOnBalls"),
                "intentional_walks": _int(batting, "intentionalWalks"),
                "strikeouts": _int(batting, "strikeOuts"),
                "hit_by_pitch": _int(batting, "hitByPitch"),
                "sac_flies": _int(batting, "sacFlies"),
                "sac_bunts": _int(batting, "sacBunts"),
                "total_bases": _int(batting, "totalBases"),
                "stolen_bases": _int(batting, "stolenBases"),
                "caught_stealing": _int(batting, "caughtStealing"),
                "grounded_into_double_play": _int(batting, "groundIntoDoublePlay"),
                "left_on_base": _int(batting, "leftOnBase"),
                "ground_outs": _int(batting, "groundOuts"),
                "air_outs": _int(batting, "airOuts"),
            }
        )
    return rows


def _pitcher_rows(boxscore: dict, game_meta: dict, side: str) -> list[dict]:
    team = boxscore["teams"][side]
    team_id = as_id(team["team"]["id"])
    opponent = "away" if side == "home" else "home"
    opponent_team_id = as_id(boxscore["teams"][opponent]["team"]["id"])

    # ``pitchers`` is ordered by appearance, which is what makes bullpen
    # sequence (opener, long man, closer) recoverable at all.
    appearance_order = {
        as_id(pid): index for index, pid in enumerate(team.get("pitchers") or [], 1)
    }

    rows = []
    for player in team.get("players", {}).values():
        pitching = (player.get("stats") or {}).get("pitching") or {}
        if not pitching:
            continue

        player_id = as_id((player.get("person") or {}).get("id"))
        outs: int | None = _int(pitching, "outs")
        if not outs:
            outs = innings_pitched_to_outs(pitching.get("inningsPitched"))
        batters_faced = _int(pitching, "battersFaced")
        pitches = _int(pitching, "pitchesThrown") or _int(pitching, "numberOfPitches")
        games_started = _int(pitching, "gamesStarted")
        hits_allowed = _int(pitching, "hits")
        walks_allowed = _int(pitching, "baseOnBalls")

        rows.append(
            {
                "game_pk": game_meta["game_pk"],
                "team_id": team_id,
                "player_id": player_id,
                "season_year": game_meta["season_year"],
                "game_date": game_meta["game_date"],
                "first_pitch_utc": game_meta["first_pitch_utc"],
                "opponent_team_id": opponent_team_id,
                "home": side == "home",
                "venue_id": game_meta["venue_id"],
                "player_name": (player.get("person") or {}).get("fullName"),
                # The starter/reliever split is what makes this grain useful:
                # a starter's rolling history is counted in starts, a
                # reliever's workload is counted in days.
                "is_starter": bool(games_started),
                "appearance_number": appearance_order.get(player_id),
                "outs_recorded": outs,
                "innings_pitched": _rate(outs, 3),
                "batters_faced": batters_faced,
                "pitches_thrown": pitches,
                "strikes": _int(pitching, "strikes"),
                "balls": _int(pitching, "balls"),
                "runs_allowed": _int(pitching, "runs"),
                "earned_runs": _int(pitching, "earnedRuns"),
                "hits_allowed": hits_allowed,
                "doubles_allowed": _int(pitching, "doubles"),
                "triples_allowed": _int(pitching, "triples"),
                "home_runs_allowed": _int(pitching, "homeRuns"),
                "walks_allowed": walks_allowed,
                "intentional_walks_allowed": _int(pitching, "intentionalWalks"),
                "strikeouts": _int(pitching, "strikeOuts"),
                "hit_batsmen": _int(pitching, "hitBatsmen"),
                "wild_pitches": _int(pitching, "wildPitches"),
                "balks": _int(pitching, "balks"),
                "ground_outs": _int(pitching, "groundOuts"),
                "air_outs": _int(pitching, "airOuts"),
                "fly_outs": _int(pitching, "flyOuts"),
                "line_outs": _int(pitching, "lineOuts"),
                "pop_outs": _int(pitching, "popOuts"),
                "inherited_runners": _int(pitching, "inheritedRunners"),
                "inherited_runners_scored": _int(pitching, "inheritedRunnersScored"),
                "wins": _int(pitching, "wins"),
                "losses": _int(pitching, "losses"),
                "saves": _int(pitching, "saves"),
                "holds": _int(pitching, "holds"),
                "blown_saves": _int(pitching, "blownSaves"),
                "complete_games": _int(pitching, "completeGames"),
                # --- rates ---------------------------------------------------
                "k_pct": _rate(_int(pitching, "strikeOuts"), batters_faced),
                "bb_pct": _rate(walks_allowed, batters_faced),
                "hr_per_bf": _rate(_int(pitching, "homeRuns"), batters_faced),
                "whip": _rate(
                    (
                        None
                        if None in (hits_allowed, walks_allowed)
                        else hits_allowed + walks_allowed
                    ),
                    _rate(outs, 3),
                ),
                "pitches_per_batter_faced": _rate(pitches, batters_faced),
                "strike_pct": _rate(_int(pitching, "strikes"), pitches),
                "ground_out_air_out_ratio": _rate(
                    _int(pitching, "groundOuts"), _int(pitching, "airOuts")
                ),
            }
        )
    return rows


def _official_rows(boxscore: dict, game_meta: dict) -> list[dict]:
    rows = []
    for official in boxscore.get("officials") or []:
        person = official.get("official") or {}
        official_type = official.get("officialType")
        rows.append(
            {
                "game_pk": game_meta["game_pk"],
                "season_year": game_meta["season_year"],
                "game_date": game_meta["game_date"],
                "official_id": as_id(person.get("id")),
                "official_name": person.get("fullName"),
                "official_type": official_type,
                "is_home_plate": official_type == HOME_PLATE_OFFICIAL_TYPE,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def parse_boxscore(boxscore: dict, game_meta: dict) -> dict[str, list[dict]]:
    """Split one box-score payload into its three grains plus officials.

    ``game_meta`` carries the fields the box score itself does not know
    (``game_pk``, ``season_year``, ``game_date``, ``first_pitch_utc``,
    ``game_type``, ``venue_id``) and normally comes from a row of the schedule
    frame.
    """
    team_games, batters, pitchers = [], [], []
    for side in SIDES:
        team_games.append(_team_game_row(boxscore, game_meta, side))
        batters.extend(_batter_rows(boxscore, game_meta, side))
        pitchers.extend(_pitcher_rows(boxscore, game_meta, side))

    return {
        "team_games": team_games,
        "batter_games": batters,
        "pitcher_appearances": pitchers,
        "umpires": _official_rows(boxscore, game_meta),
    }


def fetch_boxscore(game_pk: str) -> dict:
    """Raw box-score payload for one game."""
    return statsapi_get(f"game/{game_pk}/boxscore")


def fetch_game_grains(game_meta: dict) -> dict[str, list[dict]]:
    """Fetch and parse one game into its three grains."""
    return parse_boxscore(fetch_boxscore(game_meta["game_pk"]), game_meta)


def game_meta_from_schedule(games: pd.DataFrame) -> list[dict]:
    """Project the schedule frame down to what ``parse_boxscore`` needs."""
    columns = [
        "game_pk",
        "season_year",
        "game_date",
        "first_pitch_utc",
        "game_type",
        "venue_id",
    ]
    return games[columns].to_dict("records")
