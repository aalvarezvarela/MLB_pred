"""
MLB Predictor - Settings Module

Mirrors ``nba_ou.config.settings``: a declarative schema maps property names
onto ``(section, key, type, default)``, and ``SETTINGS.<name>`` resolves
lazily from ``config.ini`` overlaid with the gitignored
``config.secrets.ini``.

ADDING NEW SETTINGS
-------------------
Add one line to ``CONFIG_SCHEMA``::

    "property_name": ("Section", "CONFIG_KEY", type, default_value)

``type`` is ``str``, ``bool``, ``int``, ``float`` or the literal ``"secret"``.
A ``"secret"`` resolves from the environment first, then the secrets file, and
raises if it is required and missing.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

CONFIG_FILE = PACKAGE_ROOT / "config.ini"
SECRETS_FILE = PACKAGE_ROOT / "config.secrets.ini"
REPO_ROOT_SECRETS_FILE = PROJECT_ROOT / "config.secrets.ini"


CONFIG_SCHEMA: dict[str, tuple[str, str, Any, Any]] = {
    # Database
    "db_env": ("Database", "DB_ENV", str, "local"),
    "db_name": ("Database", "DB_NAME", str, "mlb"),
    "schema_name_games": ("Database", "SCHEMA_NAME_GAMES", str, "mlb_games"),
    "schema_name_team_games": (
        "Database",
        "SCHEMA_NAME_TEAM_GAMES",
        str,
        "mlb_team_games",
    ),
    "schema_name_batters": ("Database", "SCHEMA_NAME_BATTERS", str, "mlb_batter_games"),
    "schema_name_pitchers": (
        "Database",
        "SCHEMA_NAME_PITCHERS",
        str,
        "mlb_pitcher_appearances",
    ),
    "schema_name_umpires": ("Database", "SCHEMA_NAME_UMPIRES", str, "mlb_umpires"),
    "schema_name_lineups": ("Database", "SCHEMA_NAME_LINEUPS", str, "mlb_lineups"),
    "schema_name_venues": ("Database", "SCHEMA_NAME_VENUES", str, "mlb_venues"),
    "schema_name_transactions": (
        "Database",
        "SCHEMA_NAME_TRANSACTIONS",
        str,
        "mlb_transactions",
    ),
    "schema_name_statcast_games": (
        "Database",
        "SCHEMA_NAME_STATCAST_GAMES",
        str,
        "mlb_statcast",
    ),
    "schema_name_statcast_pitches": (
        "Database",
        "SCHEMA_NAME_STATCAST_PITCHES",
        str,
        "mlb_statcast",
    ),
    "schema_name_snapshots": (
        "Database",
        "SCHEMA_NAME_SNAPSHOTS",
        str,
        "mlb_snapshots",
    ),
    "schema_name_predictions": (
        "Database",
        "SCHEMA_NAME_PREDICTIONS",
        str,
        "mlb_predictions",
    ),
    # Stats API
    "statsapi_base_url": ("StatsApi", "BASE_URL", str, "https://statsapi.mlb.com/api"),
    "statsapi_sport_id": ("StatsApi", "SPORT_ID", int, 1),
    "statsapi_min_request_interval": (
        "StatsApi",
        "MIN_REQUEST_INTERVAL_SECONDS",
        float,
        0.12,
    ),
    "statsapi_timeout": ("StatsApi", "TIMEOUT_SECONDS", int, 30),
    "statsapi_max_retries": ("StatsApi", "MAX_RETRIES", int, 3),
    # Statcast
    "statcast_base_url": (
        "Statcast",
        "BASE_URL",
        str,
        "https://baseballsavant.mlb.com",
    ),
    "statcast_timeout": ("Statcast", "TIMEOUT_SECONDS", int, 120),
    "statcast_max_retries": ("Statcast", "MAX_RETRIES", int, 3),
    "statcast_min_request_interval": (
        "Statcast",
        "MIN_REQUEST_INTERVAL_SECONDS",
        float,
        0.5,
    ),
    # Paths
    "raw_data_dir": ("Paths", "RAW_DATA_DIR", str, "./data/raw"),
    "snapshot_dir": ("Paths", "SNAPSHOT_DIR", str, "./data/snapshots"),
    # Backfill
    "first_season": ("Backfill", "FIRST_SEASON", int, 2015),
    # Scraping
    "headless": ("Scraping", "HEADLESS", bool, True),
}


class Settings:
    def __init__(
        self, config_path: str | None = None, secrets_path: str | None = None
    ) -> None:
        self.config = configparser.ConfigParser()

        main_path = Path(config_path) if config_path else CONFIG_FILE
        if not main_path.exists():
            raise FileNotFoundError(
                f"Config file not found at {main_path}. "
                "Ensure config.ini exists inside the mlb_pred package."
            )
        self.config.read(main_path)

        # Overlay optional gitignored secrets, package-local last so it wins.
        if REPO_ROOT_SECRETS_FILE.exists():
            self.config.read(REPO_ROOT_SECRETS_FILE)
        overlay = Path(secrets_path) if secrets_path else SECRETS_FILE
        if overlay.exists():
            self.config.read(overlay)

        self._cache: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name in ("config", "_cache"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        if name in self._cache:
            return self._cache[name]
        if name not in CONFIG_SCHEMA:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        section, key, value_type, default = CONFIG_SCHEMA[name]

        value: Any
        if value_type == "secret":
            value = self._get_secret(
                section, key, key, required=default is None, fallback=default
            )
        elif value_type is bool:
            value = self.config.getboolean(section, key, fallback=default)
        elif value_type is int:
            value = self.config.getint(section, key, fallback=default)
        elif value_type is float:
            value = self.config.getfloat(section, key, fallback=default)
        else:
            value = self.config.get(section, key, fallback=default)

        self._cache[name] = value
        return value

    def _get_secret(
        self,
        section: str,
        key: str,
        env_var: str,
        *,
        required: bool = True,
        fallback: str | None = None,
    ) -> str:
        """Resolve a secret: environment variable -> config file -> fallback."""
        v = os.getenv(env_var)
        if v:
            return v
        if self.config.has_option(section, key):
            v = self.config.get(section, key).strip()
            if v:
                return v
        if fallback is not None:
            return fallback
        if required:
            raise ValueError(
                f"Missing required secret [{section}] {key}. "
                f"Set env var {env_var} or define it in config.secrets.ini."
            )
        return ""

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def get_absolute_path(self, relative_path: str) -> Path:
        if os.path.isabs(relative_path):
            return Path(relative_path)
        return PROJECT_ROOT / relative_path

    @property
    def raw_data_path(self) -> Path:
        return self.get_absolute_path(self.raw_data_dir)

    @property
    def snapshot_path(self) -> Path:
        return self.get_absolute_path(self.snapshot_dir)

    def ensure_directories_exist(self) -> None:
        for directory in (self.raw_data_path, self.snapshot_path):
            directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"Settings(db_env={self.db_env!r}, "
            f"raw_data_dir={self.raw_data_dir!r}, "
            f"first_season={self.first_season})"
        )


SETTINGS = Settings()
