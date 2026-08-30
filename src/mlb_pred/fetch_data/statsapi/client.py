"""Thin, polite HTTP client for the MLB Stats API.

Every fetcher in ``mlb_pred.fetch_data.statsapi`` goes through
:func:`statsapi_get`. It centralises three things:

* **Self-imposed throttling.** The API is unauthenticated and publishes no
  quota, so we pace ourselves rather than discovering the limit the hard way.
* **The transient / throttled distinction.** A connection reset or a 5xx is
  transient and is retried with backoff. A 429 (or a 403 that reads as a
  block) is *throttling*: it raises :class:`RateLimitedError` immediately
  instead of retrying, because retrying through a rate limit is how you get
  banned. Every updater in this repo is gap-driven and insert-only, so
  stopping early is free -- the next run resumes where this one stopped.
* **Session reuse**, with a reset hook for when a session goes stale.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import requests

from mlb_pred.config.settings import SETTINGS

_USER_AGENT = "mlb_pred/0.1 (personal research project)"

_session: requests.Session | None = None
_session_lock = threading.Lock()
_last_request_at = 0.0


class StatsApiError(RuntimeError):
    """A Stats API request failed after exhausting its retries."""


class RateLimitedError(StatsApiError):
    """The API is throttling us. Stop now and re-run later."""


def get_session() -> requests.Session:
    global _session
    with _session_lock:
        if _session is None:
            _session = requests.Session()
            _session.headers.update(
                {"User-Agent": _USER_AGENT, "Accept": "application/json"}
            )
        return _session


def reset_session() -> None:
    """Drop and rebuild the HTTP session (clears stale keep-alive sockets)."""
    global _session
    with _session_lock:
        if _session is not None:
            _session.close()
        _session = None


def _throttle() -> None:
    global _last_request_at
    interval = SETTINGS.statsapi_min_request_interval
    elapsed = time.monotonic() - _last_request_at
    if elapsed < interval:
        time.sleep(interval - elapsed)
    _last_request_at = time.monotonic()


def statsapi_get(
    path: str,
    params: dict[str, Any] | None = None,
    *,
    api_version: str = "v1",
    max_retries: int | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """GET a Stats API path and return the decoded JSON body.

    Args:
        path: Path below the version, e.g. ``"schedule"`` or ``"game/1/boxscore"``.
        params: Query parameters. ``None`` values are dropped.
        api_version: ``"v1"`` for almost everything; ``"v1.1"`` for the live feed.
        max_retries: Overrides ``[StatsApi] MAX_RETRIES``.
        timeout: Overrides ``[StatsApi] TIMEOUT_SECONDS``.

    Raises:
        RateLimitedError: The API is throttling. Stop the run.
        StatsApiError: A transient failure that survived every retry.
    """
    retries = SETTINGS.statsapi_max_retries if max_retries is None else max_retries
    request_timeout = SETTINGS.statsapi_timeout if timeout is None else timeout

    url = f"{SETTINGS.statsapi_base_url.rstrip('/')}/{api_version}/{path.lstrip('/')}"
    clean_params = {k: v for k, v in (params or {}).items() if v is not None}

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        _throttle()
        try:
            response = get_session().get(
                url, params=clean_params, timeout=request_timeout
            )
        except requests.RequestException as exc:  # connection reset, timeout, DNS
            last_error = exc
            reset_session()
            if attempt < retries:
                time.sleep(2.0 * attempt)
                continue
            break

        # Throttled: stop the whole run rather than retrying into a ban.
        if response.status_code in (429, 403):
            raise RateLimitedError(
                f"Stats API returned {response.status_code} for {url}. "
                "This looks like throttling, not a transient failure. "
                "Stopping now -- re-run later and the gap-driven updaters will "
                "resume from where this run stopped."
            )

        if response.status_code >= 500:
            last_error = StatsApiError(f"HTTP {response.status_code} for {url}")
            if attempt < retries:
                time.sleep(2.0 * attempt)
                continue
            break

        if response.status_code == 404:
            raise StatsApiError(f"Stats API 404 for {url} (params={clean_params}).")

        response.raise_for_status()
        try:
            return response.json()
        except ValueError as exc:
            last_error = exc
            if attempt < retries:
                reset_session()
                time.sleep(2.0 * attempt)
                continue
            break

    raise StatsApiError(
        f"Stats API request failed after {retries} attempts: {url} "
        f"(params={clean_params}). Last error: {last_error}"
    ) from last_error
