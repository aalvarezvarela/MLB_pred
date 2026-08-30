"""HTTP access to SportsbookReview, via the page's embedded JSON.

**Never parse the rendered DOM.** SBR draws its line-history table client-side
in the *browser's* local timezone and emits no offset, which makes the same
scraper produce different timestamps on a Madrid laptop and a UTC CI runner
with nothing erroring. The NBA project hit exactly this and had to recover
``Europe/Madrid`` after the fact from daylight-saving steps in the stored data;
two seasons could never be pinned and remain excluded from that store.

The same pages ship a Next.js ``__NEXT_DATA__`` payload whose ``oddsDate``
values carry an explicit UTC offset, and which also contains the game's own
``startDate``. Reading that instead means:

* the timezone question never arises -- and a naive datetime is *refused*
  rather than assumed, because assuming one would reintroduce the exact
  ambiguity this module exists to remove;
* the payload is server-rendered, so a plain ``requests.get`` is enough -- no
  browser, no cookie banner, no clicking through book/market tabs. One request
  returns every sportsbook across all three markets;
* first pitch comes from the same payload as the ticks, so ``minutes_to_tip``
  is internally consistent even if an external schedule feed disagrees.

Verified on MLB pages back to 2015 (the daily slate) and 2019 (tick history).
"""

from __future__ import annotations

import json
import random
import re
import time
from datetime import UTC, date, datetime
from typing import Any

import requests

BASE_URL = "https://www.sportsbookreview.com"
LEAGUE_PATH = "mlb-baseball"

# Polite pacing between requests, matching the NBA scraper's budget.
SLEEP_MIN_S = 0.4
SLEEP_MAX_S = 1.1

DEFAULT_TIMEOUT_S = 40.0
DEFAULT_RETRIES = 3

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json"[^>]*>(.*?)</script>',
    re.DOTALL,
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class SbrFetchError(RuntimeError):
    """A page could not be fetched, or carried no usable payload."""


def new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(_HEADERS)
    return session


def sleep_politely() -> None:
    time.sleep(random.uniform(SLEEP_MIN_S, SLEEP_MAX_S))


def build_daily_odds_url(day: date, market_path: str = "totals/full-game") -> str:
    """The day's slate page. Used for discovery and for the wide opening/current lines."""
    return (
        f"{BASE_URL}/betting-odds/{LEAGUE_PATH}/{market_path}/"
        f"?date={day.isoformat()}"
    )


def build_line_history_url(event_id: int | str) -> str:
    """One game's full tick history: every book, all three markets, one request."""
    return f"{BASE_URL}/betting-odds/{LEAGUE_PATH}/line-history/{event_id}/"


def extract_next_data(html: str) -> dict[str, Any]:
    match = _NEXT_DATA_RE.search(html)
    if not match:
        raise SbrFetchError("No __NEXT_DATA__ payload in response")
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise SbrFetchError(f"Malformed __NEXT_DATA__ payload: {exc}") from exc


def fetch_next_data(
    session: requests.Session,
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    """GET ``url`` and return its embedded payload, retrying transient failures.

    Note the MLB line-history path 308-redirects when given a trailing slash;
    ``requests`` follows that by default, which is why the URL builder keeps
    the slash for consistency with the rest of the site.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        if attempt:
            # Linear backoff; SBR rate-limits bursts.
            time.sleep(2.0 * attempt)
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return extract_next_data(response.text)
        except (requests.RequestException, SbrFetchError) as exc:
            last_error = exc
    raise SbrFetchError(f"{url}: {last_error}")


# ---------------------------------------------------------------------------
# Payload scalar helpers
# ---------------------------------------------------------------------------
def parse_utc(value: Any) -> datetime | None:
    """ISO-8601 *with an explicit offset* -> tz-aware UTC, truncated to the minute.

    A naive value returns ``None`` rather than being localised to anything.
    That refusal is the whole point of reading the payload instead of the DOM.

    Truncation to the minute is deliberate: SBR polls books on a fixed cadence
    and the seconds carry no information, while dropping them makes a re-scrape
    reproduce byte-identical keys -- which is what makes loads idempotent.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC).replace(second=0, microsecond=0)


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def slugify_bookmaker(name: str) -> str:
    """Book display name -> the slug used by the book dimension.

    Kept byte-compatible with the NBA project's slugs ("Fanatics Sportsbook"
    -> ``fanatics_sportsbook``) so the two repos' book tables stay comparable.
    """
    out: list[str] = []
    previous_underscore = False
    for char in name.strip().lower():
        if char.isalnum():
            out.append(char)
            previous_underscore = False
        elif not previous_underscore:
            out.append("_")
            previous_underscore = True
    return "".join(out).strip("_")


def team_full_name(team: dict[str, Any] | None) -> str:
    if not team:
        return ""
    return str(team.get("fullName") or team.get("name") or "").strip()


def starter_name(starter: dict[str, Any] | None) -> str | None:
    """SBR's announced starting pitcher, as ``"First Last"``.

    MLB-specific and worth capturing: a run total is heavily conditioned on the
    announced starters, and books re-hang or void a total when one changes.
    SBR reports the starters as of *scrape time*, not per tick, so this is a
    cross-check and a join handle -- not a per-tick announcement state. That
    has to come from the point-in-time probables archive.
    """
    if not starter:
        return None
    first = str(starter.get("firstName") or "").strip()
    last = str(starter.get("lastName") or "").strip()
    full = f"{first} {last}".strip()
    return full or None
