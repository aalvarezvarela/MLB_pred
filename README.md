# MLB_pred

Run-line and total-runs prediction for MLB. Architecture ported from
[`NBA_over_under_predictor`](../NBA_over_under_predictor).

**Status:** sports-data, Statcast, and SportsbookReview odds layers complete.
Closing-market, rolling team-form, matchup, market-regime and contextual
pregame features are built; modelling is not started.

## Quick start

```bash
poetry install

# Backfill 2015..present into the local Parquet store (~90 min, resumable --
# safe to interrupt; the next run picks up exactly where it stopped).
poetry run python scripts/fetch_data/backfill_mlb_data.py

# Start the point-in-time snapshot archive. Run this several times a day --
# missed runs cannot be backfilled.
poetry run python scripts/daily/capture_snapshots.py

# Daily refresh of the last few finished slates.
poetry run python scripts/update_databases/update_mlb_data.py

# Check the store's internal consistency (exits 1 on any finding).
poetry run python scripts/fetch_data/validate_store.py

# Betting lines from SportsbookReview (2019+; earlier seasons carry no prices).
poetry run python scripts/fetch_data/backfill_odds.py
poetry run python scripts/update_databases/update_odds.py --dry-run

# Pitch-level Statcast from Baseball Savant (2015+; resumable by game_pk).
poetry run python scripts/fetch_data/backfill_statcast.py
poetry run python scripts/update_databases/update_statcast.py
```

Build the first leakage-safe feature family, closing totals/run lines/moneylines,
from the local odds parquet store:

```bash
poetry run python scripts/create_features/create_closing_lines.py
```

Generated partitions are written under `data/features/closing_lines/`. Columns
are grouped by `GAME_*`, `ODDS_TOTAL_*`, `ODDS_RUN_LINE_*`,
`ODDS_MONEY_LINE_*`, and `ODDS_DERIVED_*`; realised outcomes are deliberately
absent.

Build the closing features together with NBA-shaped rolling team and market
history (use `--seasons 2025` to write only selected output seasons while still
retaining all earlier seasons as rolling context):

```bash
poetry run python scripts/create_features/create_pregame_features.py
```

The output under `data/features/pregame/` contains the NBA 1/2/3/5/10-game,
WMA-5, season mean/std and 5/10 trend families, plus MLB 20/30-game windows.
It also includes stable home/away team one-hot columns, win counts/ratios and
streaks, rest and schedule density, series-aware travel, matchup/style
crossings, total and spread market regimes, odds interactions, prior
head-to-head context, calendar/competition context, and extra-inning history.
Columns are grouped under `TEAM_*`, `SCHEDULE_*`, `TRAVEL_*`, and `ODDS_*`;
every advanced engineered feature contains `_BEFORE`, and final-game source
statistics never leave the builders.

Data lands in `data/` (gitignored, regenerable) except `data/snapshots/`,
which is not regenerable and should be backed up.

## Where things are

| | |
|---|---|
| Design rationale and every NBA divergence | [`docs/mlb-sports-data-architecture.md`](docs/mlb-sports-data-architecture.md) |
| Odds layer, and what was measured vs assumed | [`docs/mlb-odds-architecture.md`](docs/mlb-odds-architecture.md) |
| Rolling families and leakage contract | [`docs/mlb-feature-engineering.md`](docs/mlb-feature-engineering.md) |
| Repo conventions and leakage rules | [`CLAUDE.md`](CLAUDE.md) |
| Portable architecture skills | [`.claude/skills/`](.claude/skills/) |

## Data source

Everything comes from the MLB Stats API (`statsapi.mlb.com`), unauthenticated:
one request returns a whole season's schedule with umpire crews, announced
lineups, probable starters and first-pitch weather; one request per game
returns the box score. Ballparks, IL transactions and rosters have their own
endpoints. Baseball Savant supplies the pitch-level Statcast enrichment; raw
pitches are keyed by `(game_pk, at_bat_number, pitch_number)` and a separate
completion table prevents failed or empty responses from closing a gap.

Betting lines come from SportsbookReview, read out of each page's embedded
`__NEXT_DATA__` payload rather than the rendered DOM — the rendered table
carries no timezone offset, which makes a DOM scraper silently
machine-dependent. One request per game returns every sportsbook across
totals, run line and moneyline.

## PostgreSQL mirror and S3 archive

Local Parquet is the working store. Selected seasons can be mirrored to Aiven
PostgreSQL using the domain schemas defined by the project:

```bash
DB_ENV=aiven poetry run python \
  scripts/create_databases/create_mlb_databases.py --seasons 2026 --sync
```

The command is idempotent. Do not pass ``--drop-existing`` during a normal
refresh.

Historical Parquet archival is dry-run-first and never deletes local files:

```bash
# Manifest every downloaded Parquet partition (bucket/profile/region come from
# [S3] in src/mlb_pred/config.ini unless overridden by CLI or environment).
poetry run python scripts/archive_historical_parquet.py --all
poetry run python scripts/archive_historical_parquet.py --all --execute

# Or archive only historical seasons.
poetry run python scripts/archive_historical_parquet.py --before-season 2025

MLB_ARCHIVE_S3_BUCKET=YOUR_BUCKET poetry run python \
  scripts/archive_historical_parquet.py --before-season 2025 --execute
```

Each archive writes a JSON manifest containing row counts, byte sizes and
SHA-256 checksums. Uploads store the checksum as S3 object metadata and read it
back with the object size before reporting success.
