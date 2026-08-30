# MLB run-line / totals predictor

## Project overview

Predicts two MLB betting markets:

1. **Total runs** (over/under) — combined runs scored.
2. **Run line** (spread) — home minus away, stored as `run_line_margin`.

The architecture is a deliberate port of
`/home/adrian_alvarez/Projects/NBA_over_under_predictor/`. **Prefer reuse and
consistency with that repo over introducing new patterns.** Where baseball
forced a divergence, the reason is recorded in
`docs/mlb-sports-data-architecture.md` — read that before changing the data
layer.

Current state: the **sports-data, Statcast, and SportsbookReview odds layers
are built**. Feature engineering, modelling, prediction and settlement are not
started. (The NBA repo's Yahoo odds scraper is deliberately not ported.)

## Reference material

- `.claude/skills/sports-data-architecture` — entities, identity,
  availability, context data. Copied unchanged from the NBA repo; it is the
  *reference*, not a description of this repo.
- `.claude/skills/odds-data-architecture` — line ingestion, tick storage,
  snapshots.
- `.claude/skills/feature-engineering` — temporal rules, feature families.
- `docs/mlb-sports-data-architecture.md` — what this repo actually built, and
  every deliberate divergence.
- `docs/mlb-odds-architecture.md` — the odds layer, and every number that was
  *measured* (first usable season, quote increment, left/right orientation,
  per-market sigma) rather than assumed.

## Repo map

- `src/mlb_pred/` — installable Poetry package (`from mlb_pred...`).
  - `config/constants.py` — team id map, `TEAM_NAME_STANDARDIZATION`
    (**raises** on unknown), game-type codes, plausibility bounds.
  - `config/settings.py` — `SETTINGS`, reads `config.ini` overlaid with the
    gitignored `config.secrets.ini`.
  - `fetch_data/statsapi/` — `client.py` (throttle + retry semantics),
    `schedule.py`, `boxscore.py`, `venues.py`, `transactions.py`.
  - `fetch_data/statcast/` — Baseball Savant CSV client and stable pitch schema.
  - `local_store/` — `tables.py` (the natural-key registry),
    `parquet_store.py`, `updaters.py` (gap-driven).
  - `snapshots/archive.py` — point-in-time capture. **Cannot be backfilled.**
  - `fetch_data/sbr/` — SportsbookReview: `client.py` (embedded-JSON access),
    `line_history.py` (tick parsing, the `left`/`right` convention).
  - `odds/` — `encoding.py` (×2 SMALLINT lines, devig), `identity.py`
    (SBR event -> `game_pk`, doubleheaders), `ingest.py` (repairs + drop
    counters), `planner.py` (what to fetch next).
  - `postgre_db/` — `config/db_config.py`, `schema.py` (DDL), `load.py`.
- `scripts/` — thin CLI entry points:
  - `fetch_data/backfill_mlb_data.py`
  - `update_databases/update_mlb_data.py`
  - `daily/capture_snapshots.py`
  - `create_databases/create_mlb_databases.py`
  - `fetch_data/backfill_odds.py`, `update_databases/update_odds.py`
  - `fetch_data/backfill_statcast.py`, `update_databases/update_statcast.py`
- `data/` — local raw copy, **gitignored and regenerable**. Parquet,
  partitioned by season.
- `tests/` — pytest, one file per concern.

## Data-layer conventions that matter

- **The team-game is the atomic row.** Two rows per game; `home` is a boolean
  column, never a column suffix. The wide `_HOME` / `_AWAY` form is produced
  once, as a final merge, and never stored.
- **Three accumulation grains**: `team_games`, `batter_games`,
  `pitcher_appearances`. The third has no NBA equivalent and exists because a
  starter's history accumulates every ~5 team games, not every game.
- **All provider ids are `TEXT`.** `gamePk`, `team.id`, MLBAM `person.id`.
  Never remap an id; map names onto ids.
- **`standardize_team_name` raises on an unknown name.** Do not add a fallback.
  A silent drop costs a season of results. The one sanctioned exception is an
  unplayed postseason **bracket placeholder** (`AL Wild Card #1`), detected by
  its non-franchise team *id*; an unknown id on a regular-season or completed
  game still raises, because that reading is an expansion team.
- **Structure comes from the code, not the label.** Filter on `game_type`
  (`R`, `W`, …), never on `series_description`.
- **`game_date` joins; `first_pitch_utc` orders.** Never make one do both jobs.
- **`season_year` is the calendar year** — use `get_season_year_from_date`.
- **Innings are stored as `outs_recorded`.** `7.1` IP means seven innings and
  one out. Derive innings from outs; never average the notation.
- **Statcast pitch identity is** `(game_pk, at_bat_number, pitch_number)`.
  Write pitch facts before `statcast_games` completion metadata.
- **`_BEFORE` marks a leakage-safe, pre-game feature** (same convention as the
  NBA repo). Feature work must respect it.

## Odds-layer conventions

- **The tick is the source of truth.** An odds record is
  `(game, sportsbook, market, side, line, price, timestamp)`. Closing lines,
  opening lines and consensus are *views* over it — never stored shapes.
- **`left`/`right`, not over/under/home/away.** totals: left=OVER, right=UNDER.
  run_line and money_line: left=AWAY, right=HOME. **Measured against realised
  outcomes** and pinned by `tests/test_odds_orientation.py`. Do not change it
  without re-measuring.
- **The market is `run_line`, not `point_spread`.** SBR's cross-sport template
  calls it a spread; baseball does not.
- **Lines are stored doubled as SMALLINT** (`8.5 -> 17`). `encode_line` raises
  rather than rounding a value it cannot represent exactly.
- **Never parse SBR's rendered DOM** — it carries no timezone offset. Read the
  `__NEXT_DATA__` payload. `parse_utc` refuses a naive datetime by design.
- **Odds writes are insert-only.** The source can lose data; a delete-then-reload
  refresh would destroy rows that can never be refetched.
- **Odds exist only from 2019** (`FIRST_ODDS_SEASON`) — measured, not assumed.

## Leakage rules

Two are enforced in code, not just documented:

1. **Availability filters on `known_date`, never `effective_date`.** IL stints
   are routinely backdated; `effective_date` can precede the announcement by
   days. Use `transactions_known_as_of()`.
2. **Backtests read `latest_snapshot_before(...)`, not the last capture of a
   day.** A lineup revised at 15:40 must not reach a model predicting from
   11:00 information.

Never ingest an in-progress game — a mid-game box score looks complete and
poisons an insert-only table permanently.

For odds, the equivalent hazard is that **SBR records in-play ticks with the
same shape as pre-game ones**. `mins_to_tip` and `is_pregame` are stored NOT
NULL because they are the only thing separating a feature from target leakage;
3.2% of recent ticks land at or after first pitch. Never rely on a runtime
filter for this.

## Ingestion

Every updater is **gap-driven and insert-only**: it asks the store what is
missing rather than being told a date range. Re-running is always safe and
cheap; interrupting costs nothing.

```bash
poetry run python scripts/fetch_data/backfill_mlb_data.py            # 2015..now
poetry run python scripts/update_databases/update_mlb_data.py        # daily
poetry run python scripts/daily/capture_snapshots.py                 # several times daily
poetry run python scripts/daily/capture_snapshots.py --coverage      # check for gaps
```

```bash
poetry run python scripts/fetch_data/validate_store.py   # exits 1 on any finding

# Odds. The planner decides what to fetch; --dry-run shows it without fetching.
poetry run python scripts/fetch_data/backfill_odds.py
poetry run python scripts/update_databases/update_odds.py --dry-run
poetry run python scripts/update_databases/update_odds.py
```

The snapshot job is the one whose missed runs are **permanent**.

## Storage

Local Parquet under `data/raw/` is the working store. Postgres mirrors the
same schemas, one schema per domain, and is switched on by setting
`[Database] DB_ENV = aiven` (or the `DB_ENV` env var) plus credentials in
`config.secrets.ini`. The shipped default is `local`, so nothing reaches the
shared database until that is changed deliberately.

```bash
poetry run python scripts/create_databases/create_mlb_databases.py
DB_ENV=aiven poetry run python scripts/create_databases/create_mlb_databases.py --sync
```

## Environment / tooling

- Poetry, Python 3.11–3.13. `ruff` / `black` / `isort` / `mypy`
  (line-length 88, py311 target). Tests via `pytest`.
- `config.secrets.ini` holds DB credentials — never print or quote it.
- The MLB Stats API is unauthenticated with no published quota. Throttling is
  self-imposed in `fetch_data/statsapi/client.py`; a 429/403 raises
  `RateLimitedError` and **stops the run** rather than retrying into a ban.
