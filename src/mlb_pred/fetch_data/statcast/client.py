"""Polite access to Baseball Savant's documented Statcast CSV download."""

from __future__ import annotations

import io
import threading
import time
from typing import Any

import pandas as pd
import requests

from mlb_pred.config.settings import SETTINGS
from mlb_pred.utils.general_utils import as_id

STATCAST_COLUMNS = (
    "game_pk",
    "game_date",
    "game_year",
    "game_type",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "pitch_type",
    "pitch_name",
    "description",
    "events",
    "batter_id",
    "pitcher_id",
    "stand",
    "p_throws",
    "balls",
    "strikes",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "release_speed",
    "effective_speed",
    "release_spin_rate",
    "spin_axis",
    "release_extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "zone",
    "sz_top",
    "sz_bot",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "launch_speed_angle",
    "bb_type",
    "estimated_ba_using_speedangle",
    "estimated_slg_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "woba_denom",
    "babip_value",
    "iso_value",
    "hc_x",
    "hc_y",
    "hit_location",
    "if_fielding_alignment",
    "of_fielding_alignment",
    "home_score",
    "away_score",
    "bat_score",
    "fld_score",
    "post_home_score",
    "post_away_score",
    "delta_home_win_exp",
    "delta_run_exp",
    "n_thruorder_pitcher",
    "n_priorpa_thisgame_player_at_bat",
    "arm_angle",
    "bat_speed",
    "swing_length",
    "season_year",
)


class StatcastError(RuntimeError):
    pass


_session: requests.Session | None = None
_lock = threading.Lock()
_last_request_at = 0.0


def _get_session() -> requests.Session:
    global _session
    with _lock:
        if _session is None:
            _session = requests.Session()
            _session.headers.update(
                {
                    "User-Agent": "mlb_pred/0.1 (personal research project)",
                    "Accept": "text/csv,*/*",
                }
            )
        return _session


def _throttle() -> None:
    global _last_request_at
    interval = SETTINGS.statcast_min_request_interval
    with _lock:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < interval:
            time.sleep(interval - elapsed)
        _last_request_at = time.monotonic()


def _request_csv(params: dict[str, Any]) -> str:
    url = f"{SETTINGS.statcast_base_url.rstrip('/')}/statcast_search/csv"
    last_error: Exception | None = None
    for attempt in range(1, SETTINGS.statcast_max_retries + 1):
        _throttle()
        try:
            response = _get_session().get(
                url, params=params, timeout=SETTINGS.statcast_timeout
            )
            if response.status_code in (403, 429):
                raise StatcastError(
                    f"Baseball Savant returned {response.status_code}; stop and retry later"
                )
            response.raise_for_status()
            text = response.text
            if text.lstrip().startswith("<"):
                raise StatcastError("Baseball Savant returned HTML instead of CSV")
            return text
        except (requests.RequestException, StatcastError) as exc:
            last_error = exc
            if isinstance(exc, StatcastError) and "stop and retry" in str(exc):
                raise
            if attempt < SETTINGS.statcast_max_retries:
                time.sleep(2.0 * attempt)
    raise StatcastError(f"Statcast request failed: {last_error}")


def normalize_statcast_frame(
    raw: pd.DataFrame, *, game_pk: str, season_year: int
) -> pd.DataFrame:
    """Normalize provider names and freeze a stable pitch-level schema."""
    if raw.empty:
        return pd.DataFrame(columns=STATCAST_COLUMNS)
    frame = raw.rename(columns={"batter": "batter_id", "pitcher": "pitcher_id"}).copy()
    if "game_pk" not in frame:
        frame["game_pk"] = game_pk
    frame["game_pk"] = frame["game_pk"].map(as_id)
    frame["batter_id"] = frame.get("batter_id", pd.Series(index=frame.index)).map(as_id)
    frame["pitcher_id"] = frame.get("pitcher_id", pd.Series(index=frame.index)).map(as_id)
    frame["season_year"] = season_year
    for column in STATCAST_COLUMNS:
        if column not in frame:
            frame[column] = None
    frame = frame.dropna(subset=["at_bat_number", "pitch_number"])
    frame["at_bat_number"] = pd.to_numeric(frame["at_bat_number"], errors="raise").astype(int)
    frame["pitch_number"] = pd.to_numeric(frame["pitch_number"], errors="raise").astype(int)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.date
    return frame[list(STATCAST_COLUMNS)].drop_duplicates(
        ["game_pk", "at_bat_number", "pitch_number"], keep="last"
    )


def fetch_statcast_game(game_pk: str, season_year: int) -> pd.DataFrame:
    """Fetch every Statcast pitch for one MLB game."""
    text = _request_csv({"all": "true", "type": "details", "game_pk": game_pk})
    try:
        raw = pd.read_csv(io.StringIO(text), low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=STATCAST_COLUMNS)
    except pd.errors.ParserError as exc:
        raise StatcastError(f"game {game_pk}: malformed CSV: {exc}") from exc
    if "error" in raw.columns:
        raise StatcastError(f"game {game_pk}: {raw['error'].iloc[0]}")
    return normalize_statcast_frame(raw, game_pk=str(game_pk), season_year=season_year)
