"""Transaction feed -- the availability signal, captured point-in-time correct.

Read this module's docstring before building anything on top of it. This is
the part of the data layer with the most real leakage risk, and MLB's version
of the trap is sharper than the NBA's.

**The trap.** An injured-list stint is routinely *backdated*. A player placed
on the IL on 25 August is very often placed "retroactive to 24 August", and
some sources record only the retroactive date. Query such a source after the
fact and it will tell you the player was on the IL on the 24th -- on a day
when nobody, including the club's own beat writer, knew it. That is not
optimism, it is genuine look-ahead leakage: the feature would carry
information that did not exist at prediction time.

**Why this feed is safe.** The Stats API publishes *both* dates. ``date`` is
when the move was announced; ``effectiveDate``/``resolutionDate`` is when it
took effect, which may be earlier. This module therefore treats ``date`` as
the only field a point-in-time feature may filter on, and stores the others as
context. :func:`transactions_known_as_of` enforces that.

**What still cannot be backfilled.** The transaction feed says when a player
*became* unavailable, not who a club expected to play. The daily lineup card,
the probable-pitcher listing and the active roster are forecasts that get
revised, and no retrospective query reproduces what they said at 11am. Those
have to be archived as they are published -- see ``mlb_pred.snapshots``. Start
on day one; a year from now that archive is the only point-in-time-correct
availability history that exists, and nothing can recreate it later.
"""

from __future__ import annotations

import pandas as pd

from mlb_pred.config.constants import INJURY_TRANSACTION_MARKERS
from mlb_pred.config.settings import SETTINGS
from mlb_pred.fetch_data.statsapi.client import statsapi_get
from mlb_pred.utils.general_utils import as_id, get_season_year_from_date

TRANSACTION_COLUMNS = [
    "transaction_id",
    "player_id",
    "player_name",
    "season_year",
    # The announcement date. The ONLY date a point-in-time feature may use.
    "known_date",
    # When the move took effect. Frequently EARLIER than known_date; using it
    # as a filter is look-ahead leakage.
    "effective_date",
    "resolution_date",
    "is_backdated",
    "backdated_days",
    "from_team_id",
    "to_team_id",
    "type_code",
    "type_desc",
    "description",
    "is_injury_related",
]


def _is_injury_related(description: str | None, type_desc: str | None) -> bool:
    haystack = f"{description or ''} {type_desc or ''}".lower()
    return any(marker in haystack for marker in INJURY_TRANSACTION_MARKERS)


def fetch_transactions(start_date, end_date) -> pd.DataFrame:
    """Transactions announced within a date range."""
    payload = statsapi_get(
        "transactions",
        {
            "sportId": SETTINGS.statsapi_sport_id,
            "startDate": pd.to_datetime(start_date).strftime("%Y-%m-%d"),
            "endDate": pd.to_datetime(end_date).strftime("%Y-%m-%d"),
        },
    )

    rows = []
    for transaction in payload.get("transactions", []):
        known_date = pd.to_datetime(transaction.get("date"), errors="coerce")
        effective_date = pd.to_datetime(
            transaction.get("effectiveDate"), errors="coerce"
        )
        if pd.isna(known_date):
            continue

        backdated_days = (
            None
            if pd.isna(effective_date)
            else int((known_date.normalize() - effective_date.normalize()).days)
        )
        description = transaction.get("description")
        type_desc = transaction.get("typeDesc")

        rows.append(
            {
                "transaction_id": as_id(transaction.get("id")),
                "player_id": as_id((transaction.get("person") or {}).get("id")),
                "player_name": (transaction.get("person") or {}).get("fullName"),
                "season_year": get_season_year_from_date(known_date),
                "known_date": known_date.date(),
                "effective_date": (
                    None if pd.isna(effective_date) else effective_date.date()
                ),
                "resolution_date": (
                    pd.to_datetime(
                        transaction.get("resolutionDate"), errors="coerce"
                    ).date()
                    if transaction.get("resolutionDate")
                    else None
                ),
                "is_backdated": bool(backdated_days and backdated_days > 0),
                "backdated_days": backdated_days,
                "from_team_id": as_id((transaction.get("fromTeam") or {}).get("id")),
                "to_team_id": as_id((transaction.get("toTeam") or {}).get("id")),
                "type_code": transaction.get("typeCode"),
                "type_desc": type_desc,
                "description": description,
                "is_injury_related": _is_injury_related(description, type_desc),
            }
        )

    return pd.DataFrame(rows, columns=TRANSACTION_COLUMNS)


def transactions_known_as_of(transactions: pd.DataFrame, as_of_date) -> pd.DataFrame:
    """Transactions a model could legitimately have known about on a date.

    Filters on ``known_date`` -- the announcement -- and never on
    ``effective_date``. Use this rather than a hand-rolled date filter; the
    whole point of the module is that the obvious filter is the wrong one.
    """
    if transactions.empty:
        return transactions
    cutoff = pd.to_datetime(as_of_date).date()
    return transactions[transactions["known_date"] <= cutoff].reset_index(drop=True)


def transactions_strictly_before_date(
    transactions: pd.DataFrame, game_date
) -> pd.DataFrame:
    """Leakage-safe historical fallback when announcement time is unavailable.

    Same-day transactions cannot safely be admitted from the retrospective feed;
    use a captured transaction snapshot when they matter.
    """
    if transactions.empty:
        return transactions
    cutoff = pd.to_datetime(game_date).date()
    return transactions[transactions["known_date"] < cutoff].reset_index(drop=True)
