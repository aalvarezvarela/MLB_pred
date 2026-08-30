"""Resolving an SBR event to this project's ``game_pk``.

Every odds row arrives keyed by the *provider's* event id, which means nothing
to the rest of the system. Resolution is::

    SBR event_id
      -> (game_date, team_home, team_away)   # from the payload
      -> standardise both team names          # TEAM_NAME_STANDARDIZATION
      -> look up game_pk                      # from the games table

Four decisions inside that:

**``game_date`` is the Eastern-time date of first pitch.** A 00:30 UTC first
pitch belongs to the previous day's slate. SBR uses this convention and so does
MLB's own ``officialDate`` -- which is the only reason the two join at all.

**Team names go through one hand-maintained map that raises on an unknown
name.** It earned its keep on the first scrape: SBR emits "Athletics Athletics"
for a club with no city.

**Doubleheaders are disambiguated by first pitch.** ``(date, home, away)`` is
not unique in baseball -- roughly 58 games a season are the second half of a
doubleheader. The skill this follows suggests keying on a game number, but SBR
does not publish one. It publishes something better: a ``startDate`` that
matches MLB's scheduled first pitch to the minute. So candidates are ranked by
``|first_pitch_sbr - first_pitch_mlb|`` and the nearest wins, subject to a
tolerance. The announced starters are carried alongside as an independent
cross-check.

**The provider id is kept.** ``event_id`` is stored on the game dimension next
to ``game_pk``. It cannot be re-derived later -- a date holds many games -- and
it is needed to re-fetch one game.

An unmatched game is **diagnosed, not dropped blindly**: every miss is counted
by a named reason. If the miss rate has no such explanation, that is a
name-mapping bug, not noise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from mlb_pred.config.constants import UnknownTeamNameError, team_id_for_name

#: A doubleheader's two games start hours apart, so anything inside this window
#: is the same game. Wide enough to absorb a first pitch SBR recorded before a
#: rain delay pushed it, narrow enough that it can never reach the other half
#: of a doubleheader.
MATCH_TOLERANCE = timedelta(hours=3)

#: Above this, SBR's first pitch and MLB's disagree enough to be worth
#: reporting. The match still stands -- consistency with the ticks matters more
#: than agreeing with a third party -- but a silent disagreement is not
#: acceptable either.
FIRST_PITCH_WARN_TOLERANCE = timedelta(minutes=15)


@dataclass
class ResolutionStats:
    """Why each event did or did not resolve. Provenance, not logging."""

    matched: int = 0
    matched_doubleheader: int = 0
    unmatched: Counter = field(default_factory=Counter)
    first_pitch_disagreements: list[dict] = field(default_factory=list)
    unknown_team_names: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return self.matched + sum(self.unmatched.values())

    @property
    def match_rate(self) -> float:
        return self.matched / self.total if self.total else 0.0

    def summary(self) -> str:
        lines = [
            f"resolved {self.matched}/{self.total} ({self.match_rate:.1%}), "
            f"{self.matched_doubleheader} via doubleheader disambiguation"
        ]
        for reason, count in self.unmatched.most_common():
            lines.append(f"  unmatched [{reason}]: {count}")
        if self.unknown_team_names:
            lines.append(f"  unknown team names: {dict(self.unknown_team_names)}")
        if self.first_pitch_disagreements:
            lines.append(
                f"  first-pitch disagreements >"
                f"{int(FIRST_PITCH_WARN_TOLERANCE.total_seconds() // 60)}min: "
                f"{len(self.first_pitch_disagreements)}"
            )
        return "\n".join(lines)


def build_game_index(games: pd.DataFrame) -> pd.DataFrame:
    """Project the games table down to what matching needs.

    Kept as a frame rather than a dict because a doubleheader legitimately maps
    one key to several rows, and the disambiguation needs all of them.
    """
    columns = [
        "game_pk",
        "game_date",
        "home_team_id",
        "away_team_id",
        "first_pitch_utc",
        "game_number",
        "doubleheader",
    ]
    available = [c for c in columns if c in games.columns]
    index = games[available].copy()
    index["game_date"] = pd.to_datetime(index["game_date"]).dt.date
    index["first_pitch_utc"] = pd.to_datetime(index["first_pitch_utc"], utc=True)
    return index


def resolve_event(
    scraped,
    game_index: pd.DataFrame,
    stats: ResolutionStats | None = None,
) -> dict | None:
    """Match one scraped SBR game to a ``game_pk``.

    Returns the game-dimension row to store, or ``None`` with a named reason
    recorded on ``stats``.
    """
    stats = stats if stats is not None else ResolutionStats()

    try:
        home_team_id = team_id_for_name(scraped.team_home)
        away_team_id = team_id_for_name(scraped.team_away)
    except UnknownTeamNameError as exc:
        # Loud in the stats, not silently dropped: an unmapped name is a
        # rebrand or a new provider spelling and needs a human decision.
        stats.unmatched["unknown_team_name"] += 1
        stats.unknown_team_names[
            str(exc).split("'")[1] if "'" in str(exc) else "?"
        ] += 1
        return None

    candidates = game_index[
        (game_index["game_date"] == scraped.game_date)
        & (game_index["home_team_id"] == home_team_id)
        & (game_index["away_team_id"] == away_team_id)
    ]

    if candidates.empty:
        # SBR lists a slate the games table may legitimately not carry:
        # spring training, a postponed game that never happened, or a season
        # outside the backfill range. Named so the miss rate stays explainable.
        stats.unmatched["no_game_on_date"] += 1
        return None

    is_doubleheader = len(candidates) > 1
    if is_doubleheader:
        deltas = (candidates["first_pitch_utc"] - scraped.first_pitch_utc).abs()
        best_position = deltas.values.argmin()
        match = candidates.iloc[best_position]
        if deltas.iloc[best_position] > MATCH_TOLERANCE:
            stats.unmatched["doubleheader_no_close_start"] += 1
            return None
        stats.matched_doubleheader += 1
    else:
        match = candidates.iloc[0]

    delta = abs(match["first_pitch_utc"] - scraped.first_pitch_utc)
    if delta > FIRST_PITCH_WARN_TOLERANCE:
        # Reported, not silently resolved either way. The scrape-time first
        # pitch is what the ticks' minutes_to_tip was computed against, so it
        # wins -- but a large disagreement is a fact worth surfacing.
        stats.first_pitch_disagreements.append(
            {
                "game_pk": match["game_pk"],
                "event_id": scraped.event_id,
                "mlb_first_pitch_utc": match["first_pitch_utc"],
                "sbr_first_pitch_utc": scraped.first_pitch_utc,
                "delta_minutes": int(delta.total_seconds() // 60),
            }
        )

    stats.matched += 1
    return {
        "game_pk": str(match["game_pk"]),
        "event_id": int(scraped.event_id),
        "game_date": scraped.game_date,
        "season_year": scraped.season_year,
        # SBR's own first pitch, deliberately stored rather than MLB's: it is
        # what every stored minutes_to_tip was computed against, so the ticks
        # stay internally consistent even if the schedule feed disagrees.
        "first_pitch_utc": scraped.first_pitch_utc,
        "team_home_id": home_team_id,
        "team_away_id": away_team_id,
        "team_home": scraped.team_home,
        "team_away": scraped.team_away,
        "starter_home": scraped.starter_home,
        "starter_away": scraped.starter_away,
        "is_doubleheader": bool(is_doubleheader),
        "game_number": int(match.get("game_number") or 1),
    }


def resolve_events(
    scraped_games, games: pd.DataFrame
) -> tuple[list[dict], ResolutionStats]:
    """Resolve many scraped games. Returns ``(dimension rows, stats)``."""
    index = build_game_index(games)
    stats = ResolutionStats()
    rows = [
        row
        for row in (resolve_event(game, index, stats) for game in scraped_games)
        if row is not None
    ]
    return rows, stats
