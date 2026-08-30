# MLB_pred

Run-line and total-runs prediction for MLB. Architecture ported from
[`NBA_over_under_predictor`](../NBA_over_under_predictor).

**Status:** sports-data, Statcast, and SportsbookReview odds layers complete.
Features and modelling not started.

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

Data lands in `data/` (gitignored, regenerable) except `data/snapshots/`,
which is not regenerable and should be backed up.

## Where things are

| | |
|---|---|
| Design rationale and every NBA divergence | [`docs/mlb-sports-data-architecture.md`](docs/mlb-sports-data-architecture.md) |
| Odds layer, and what was measured vs assumed | [`docs/mlb-odds-architecture.md`](docs/mlb-odds-architecture.md) |
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
