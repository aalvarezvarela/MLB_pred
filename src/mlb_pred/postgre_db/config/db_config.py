"""Centralised database configuration.

Mirrors ``nba_ou.postgre_db.config.db_config``. Two environments are
supported: ``local`` (discrete host/user/port settings) and ``aiven``
(configured by a full connection URI). ``[Database] DB_ENV`` selects; the
``DB_ENV`` environment variable overrides.

Nothing writes to Aiven until ``DB_ENV`` is switched -- the shipped default is
``local``.
"""

from __future__ import annotations

import os
from typing import Any

import psycopg
from psycopg import sql

from mlb_pred.config.settings import SETTINGS

# Environments configured by a connection URI, as (section, key, env var).
_DSN_SOURCES: dict[str, tuple[str, str, str]] = {
    "aiven": ("DatabaseAiven", "AIVEN_DB_URL", "AIVEN_DB_URL"),
}

_CREDENTIAL_SECTIONS: dict[str, str] = {
    "aiven": "DatabaseAiven",
    "local": "DatabaseLocal",
}

_PASSWORD_ENV_VARS: dict[str, str] = {
    "aiven": "AIVEN_DB_PASSWORD",
    "local": "LOCAL_DB_PASSWORD",
}


def get_db_env() -> str:
    """Current database environment: ``local`` or ``aiven``."""
    return os.getenv("DB_ENV", SETTINGS.db_env).strip().lower()


def _resolve(section: str, key: str, env_var: str, *, required: bool = True) -> str:
    value = os.getenv(env_var)
    if value:
        return value
    if SETTINGS.config.has_option(section, key):
        value = SETTINGS.config.get(section, key).strip()
        if value:
            return value
    if required:
        raise ValueError(
            f"Missing required setting [{section}] {key}. "
            f"Set env var {env_var} or define it in config.secrets.ini."
        )
    return ""


def get_db_dsn(env: str | None = None) -> str:
    """Connection URI for ``env``, or ``""`` when it is not URI-configured."""
    resolved = (env or get_db_env()).strip().lower()
    spec = _DSN_SOURCES.get(resolved)
    if spec is None:
        return ""
    return _resolve(*spec, required=False)


def get_db_credentials(env: str | None = None) -> dict[str, Any]:
    """Discrete connection settings for ``env``."""
    db_env = (env or get_db_env()).strip().lower()
    section = _CREDENTIAL_SECTIONS.get(db_env, "DatabaseLocal")
    config = SETTINGS.config

    dbname = config.get(section, "DB_NAME", fallback=SETTINGS.db_name)
    return {
        "env": db_env,
        "user": config.get(section, "DB_USER"),
        "password": _resolve(
            section, "DB_PASSWORD", _PASSWORD_ENV_VARS.get(db_env, "LOCAL_DB_PASSWORD")
        ),
        "host": config.get(section, "DB_HOST"),
        "port": config.get(section, "DB_PORT"),
        "dbname": dbname.strip().strip('"').strip("'"),
        "sslmode": config.get(section, "DB_SSLMODE", fallback="").strip() or None,
    }


def connect_mlb_db(env: str | None = None) -> psycopg.Connection:
    """Connect to the project database."""
    dsn = get_db_dsn(env)
    if dsn:
        return psycopg.connect(dsn)

    credentials = get_db_credentials(env)
    return psycopg.connect(
        user=credentials["user"],
        password=credentials["password"],
        host=credentials["host"],
        port=credentials["port"],
        dbname=credentials["dbname"],
        sslmode=credentials["sslmode"],
    )


def connect_postgres_db(env: str | None = None) -> psycopg.Connection:
    """Connect to the maintenance ``postgres`` database (for CREATE DATABASE)."""
    credentials = get_db_credentials(env)
    conn = psycopg.connect(
        user=credentials["user"],
        password=credentials["password"],
        host=credentials["host"],
        port=credentials["port"],
        dbname="postgres",
        sslmode=credentials["sslmode"],
    )
    conn.autocommit = True
    return conn


def create_schema_if_not_exists(conn: psycopg.Connection, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
    conn.commit()
