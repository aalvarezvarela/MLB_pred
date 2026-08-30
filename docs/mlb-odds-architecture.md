# MLB odds architecture (SportsbookReview)

What this repo built for betting-market data, why, and every number that was
**measured** rather than assumed.

The reference is `.claude/skills/odds-data-architecture` (copied unchanged from
the NBA repo). Companion: [`mlb-sports-data-architecture.md`](mlb-sports-data-architecture.md),
which must be built first — odds have no useful identity until they can be
joined to a `game_pk`.

Scope: **SportsbookReview only.** The NBA repo also carries a Yahoo scraper;
that is deliberately not ported.

---

## The one sentence

> An odds record is **(game, sportsbook, market, side, line, price, timestamp)**.
> Everything else — books as columns, closing lines, opening lines, consensus —
> is a *view* over that record.

This repo builds the tick as the source of truth and derives the views. The NBA
project built a wide, one-row-per-game table first and says plainly that if you
are starting fresh you should not repeat that: adding a book to a wide table is
a schema migration, and a book that only exists in recent seasons becomes a
season indicator the model can exploit.

---

## Source: the embedded JSON, never the DOM

SBR renders its line-history table client-side in the **browser's** local
timezone with no offset. A scraper reading that produces different timestamps
on a Madrid laptop than on a UTC CI runner, and nothing errors. The NBA project
hit exactly this, had to recover `Europe/Madrid` after the fact from
daylight-saving steps in the stored data, and still has two seasons it could
never pin.

The same pages ship a Next.js `__NEXT_DATA__` payload whose `oddsDate` values
carry an explicit UTC offset. Reading that instead means:

* **the timezone question never arises** — and `parse_utc` *refuses* a naive
  datetime rather than assuming one, because assuming would reintroduce the
  exact ambiguity the module exists to remove;
* **no browser** — one plain `requests.get` returns every sportsbook across all
  three markets. No Playwright, no cookie banner, no clicking book/market tabs;
* **first pitch comes from the same payload as the ticks**, so `mins_to_tip` is
  internally consistent even when the schedule feed disagrees.

Timestamps are truncated to the minute. The seconds are a per-game polling
offset (`:23`, `:00`, `:14` … vary by game) and carry no information, while
dropping them makes a re-scrape reproduce byte-identical keys — which is what
makes loads idempotent. Within a minute the latest quote wins, so truncation
can never produce two rows on one key.

### Endpoints

| | URL | Cost |
|---|---|---|
| Slate discovery | `/betting-odds/mlb-baseball/totals/full-game/?date=YYYY-MM-DD` | 1 per day |
| Tick history | `/betting-odds/mlb-baseball/line-history/{event_id}/` | 1 per game |

Discovery and detail are separate requests on purpose: a slate listing tells
you what exists, and detail URLs are never built from a guessed id range.

MLB costs ~15 games/day over ~186 days — roughly 2,700 requests per season,
about 2.5× the NBA budget, which is why the planner below matters more here.

---

## What was measured

The skill is emphatic that several things must be measured on real data before
being committed to. All of these were.

### 1. First usable season: **2019**

| season | slate listed | tick history |
|---|---|---|
| 2015 | 14 games | **0 ticks** |
| 2018 | 15 games | **0 ticks** |
| 2019 | 15 games | 116 ticks, 3 books |
| 2021 | 15 games | 107 ticks, 4 books |
| 2022–2025 | full | 6–7 books |

SBR lists MLB slates back to 2015 and even carries starters, but **no prices
before 2019**. `FIRST_ODDS_SEASON = 2019` is that measurement, not a guess.

**This bounds the trainable range.** The sports-data layer backfills from 2015;
odds exist only from 2019, so ~7 seasons is the real overlap.

### 2. Quote increment: half-points, so the ×2 encoding is safe

Surveyed 32 games across the 2025 season, 6,923 ticks:

```
totals    {4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5}
run line  {-3.5, -2.5, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5, 3.5}
```

Every value a multiple of 0.5, **including the alternate run lines** — which
was the specific risk the skill flagged for MLB. So lines are stored doubled as
`SMALLINT` (`8.5 → 17`): exact, 2 bytes against ~12 for `NUMERIC`.

A 32-game sample is evidence, not proof, so `encode_line` **raises** on a value
it cannot represent exactly rather than rounding it away. If SBR ever quotes a
quarter-point, the load fails loudly on the first one and the scale changes by
one constant — instead of silently storing wrong numbers in millions of rows.

### 3. The `left`/`right` convention, verified against realised outcomes

| market | `left` | `right` | invariant |
|---|---|---|---|
| totals | OVER | UNDER | `left_line == right_line` |
| run_line | AWAY | HOME | `left_line == -right_line` |
| money_line | AWAY | HOME | both line columns NULL by nature |

Measured on 196 MLB games across 13 slates of 2025, 36,015 ticks, 100%
resolved, closing quote per (game, market, book):

```
RUN LINE   right = HOME  residual std 4.719
           right = AWAY  residual std 4.908      <- worse, as required
           raw home-margin std 4.588
           corr(-right_line, home_margin) = +0.068

MONEYLINE  mean devigged fair_right (home) = 0.5409
           realised home win rate          = 0.5495     <- tracks it
           mean overround                  = 0.0438

TOTALS     mean devigged fair_left (OVER)  = 0.5012
           corr(fair_left, went_over)      = +0.042
```

**The moneyline is the decisive test for MLB, not the run line — and that is a
genuine departure from the NBA.** A baseball run line is a near-constant ±1.5
handicap, so it carries almost no information about margin: its residual
(4.719) is *larger* than the raw margin std (4.588), and the two orientations
separate only weakly. The NBA spread genuinely forecasts margin (13.46 against
a raw 15.57) and discriminates sharply. In MLB the information sits in the
**price**, not the line. Anyone re-verifying this convention should lean on the
moneyline, where a flip would read 0.459 against a realised 0.550.

Pinned by `tests/test_odds_orientation.py`, structurally offline and
statistically under `-m live`.

### 4. Per-market sigma

```
totals   sigma = 4.221     (std of realised total − closing line)
run line sigma = 4.719     (std of home margin + closing home line)
```

Recorded because any line/price normalisation needs a per-market
outcome-distribution parameter, and using one market's for the other is a real
error — here they differ by half a run. (Normalisation itself is feature-layer
work and is not built yet.)

### 5. In-play share: **3.2%**

Of 5,876 ticks on two real slates, 188 were quoted at or after first pitch —
inside the 2.7–3.7% the NBA project measured.

---

## Identity: the hard part

```
SBR event_id
  -> (game_date, team_home, team_away)   # from the payload
  -> standardise both team names          # TEAM_NAME_STANDARDIZATION
  -> look up game_pk                      # from the games table
```

**`game_date` is the Eastern-time date of first pitch.** A 02:10 UTC first
pitch is a 22:10 Eastern game on the *previous* calendar day. SBR files it
there and so does MLB's own `officialDate` — which is the only reason the two
join at all.

This is not a rare edge case. Measured on this repo's store: **21.5% of
resolved games (23 of 107) have a UTC date that differs from their slate
date.** Keying on the UTC date instead would silently fail to match every one
of them — the "late third of every slate" the skill warns about, quantified.

**The name map earned its keep on the first scrape.** SBR builds `fullName` as
`"<location> <nickname>"`, and the Athletics no longer have a city, so SBR
emits **`"Athletics Athletics"`**. The raise-on-unknown guard caught it
immediately instead of a season quietly missing a team.

### Doubleheaders: solved better than the skill expected

The skill says to key on `(date, home, away, game_number)` "from day one"
because `(date, home, away)` is not unique — ~58 MLB games a season are the
second half of a doubleheader. SBR publishes no game number, but it publishes
something better: **a `startDate` that matches MLB's scheduled first pitch to
the minute.**

```
2025-04-06, St. Louis @ Boston
  SBR 341365 @ 17:35Z  ->  game_pk 778443  (game 1)
  SBR 341380 @ 23:10Z  ->  game_pk 778432  (game 2)
```

Candidates are ranked by `|first_pitch_sbr − first_pitch_mlb|` and the nearest
wins within a 3-hour tolerance; beyond it the event is left unmatched with a
named reason rather than attached to the wrong half. The **announced starters**
(Pallante/Newcomb vs Mikolas/Dobbins) are stored as an independent cross-check.

Measured resolution rate on real slates: **100%** (196/196 games, 6 via
doubleheader disambiguation).

**The provider id is kept.** `event_id` sits on the game dimension next to
`game_pk`: it cannot be re-derived later (a date holds many games) and is
needed to re-fetch one game. That mapping is a stored dimension row, never a
join to redo.

**Misses are diagnosed, not dropped.** Every unmatched event is counted under a
named reason (`no_game_on_date`, `unknown_team_name`,
`doubleheader_no_close_start`). A miss rate with no named explanation is a
mapping bug, not noise.

### First-pitch disagreement is reported, not silently resolved

SBR's own start time is what every stored `mins_to_tip` was computed against,
so it wins — but a disagreement above 15 minutes with the schedule feed is
recorded on `ResolutionStats.first_pitch_disagreements`.

---

## Storage

### Local (working store)

| table | grain | primary key |
|---|---|---|
| `odds_games` | one row per game | `game_pk` |
| `odds_ticks` | one row per (game, market, book, minute) | `(game_pk, market, book_slug, line_ts)` |

Parquet, season-partitioned, gitignored. `book_slug` stays a readable string
here; Postgres resolves it to a surrogate id at the boundary.

### Postgres star schema — `odds_line_history`

```
lh_book       (book_id SMALLINT PK, slug UNIQUE, name)
lh_market     (market_id SMALLINT PK, code UNIQUE)   -- totals | run_line | money_line
lh_game       (game_pk TEXT PK, event_id, game_date, season_year, first_pitch_utc,
               team_home_id, team_away_id, team_home, team_away,
               starter_home, starter_away, is_doubleheader, game_number)
lh_line       (game_pk, season_year, market_id, book_id, line_ts,
               mins_to_tip, is_pregame, is_opener,
               left_line, left_price, right_line, right_price,
               PRIMARY KEY (game_pk, market_id, book_id, line_ts, season_year))
              PARTITION BY LIST (season_year)
lh_load_meta  (season_year PK, confidence, source_ticks, loaded_ticks,
               games_seen, games_loaded, dropped_rows JSONB, resolution JSONB, loaded_at)
```

Every column is a join key, a filter, or a price. The market is called
**`run_line`**, not `point_spread`: SBR reuses a cross-sport page template that
calls it a spread, but in baseball it is the run line.

`season_year` appears **last** in the primary key only because Postgres
requires the partition key inside a unique constraint; `game_pk` stays leading
so "all lines for game X" still uses the index prefix.

`starter_home` / `starter_away` are MLB-specific additions to the NBA schema.

### `mins_to_tip` and `is_pregame` are NOT NULL, and are not conveniences

SBR records in-play ticks with **exactly the same shape** as pre-game ones.
Nothing else in the row separates a legitimate feature from direct target
leakage. So the distance to first pitch is computed at load, stored, and
enforced non-null. A runtime filter can be forgotten at one call site; a NOT
NULL column cannot.

### Insert-only, everywhere

`ON CONFLICT DO NOTHING`. Never upsert a tick, never replace a game's rows
wholesale. **The source can lose data** — SBR dropped Caesars from its NBA
pages and ~270k historical rows became unrefetchable; a "delete then reload"
refresh would have destroyed them permanently. Insert-only makes re-fetching
cost one HTTP request and nothing else, which is what makes it safe to re-fetch
aggressively.

The game *dimension* is insert-only too, not an upsert: every stored tick's
`mins_to_tip` was computed against the `first_pitch_utc` on that row, so
silently moving it would desynchronise rows the run is not touching.

Bulk path: `COPY` into an `UNLOGGED` staging table, then one
`INSERT … SELECT … ON CONFLICT DO NOTHING` into the partitioned target, staged
one season at a time.

---

## Load-time repairs, counted by reason

Preferring **structural invariants** over magnitude heuristics wherever an
invariant exists:

| check | reason code | rationale |
|---|---|---|
| run line not mirrored | `run_line_not_mirrored` | a genuine spread satisfies `left == -right`; a complementary *price* pair landing in the line fields is the known failure mode |
| totals sides disagree | `totals_sides_disagree` | a total quotes one number |
| pre-game line out of band | `line_out_of_bounds_cleared` | a dropped decimal turns 8.5 into 85 |
| line not a half-point | `line_not_half_point` | the encoding's precondition |
| all four fields empty | `empty_quote` | nothing to store |
| off-the-board sentinel | (nulled in `encode_price`) | `-100000` survives into every mean, std and devig |

**In-play rows are exempt from the bounds.** A live run line legitimately blows
out during a rout; nulling it would be destroying data, not repairing it.
And the bad **field** is cleared, never the row — the prices are still real.

Every drop and repair is counted into `IngestStats` and persisted to
`lh_load_meta`, so "we lost 4% of season 2023" is answerable.

---

## The update planner

"What should I fetch?" is a query against our own store, not a date range
someone types. Three independent reasons, unioned:

**1. Refresh window (unconditional).** Dates with games in the last 3 days,
re-fetched whether or not they are already stored. *Presence is not finality*:
a game fetched on the morning it is played is present but its lines keep moving
until first pitch, so a presence-based gap check would never bring it back and
the store would hold a truncated history forever.

**2. Gaps.** Games the games table knows about that the odds store does not.
Bounded below by the store's own earliest game — below that there is no history
to be *missing*, only history never collected.

**3. Partial coverage.** Stored games missing a book their slate-mates carry.
Two guards, both learned the hard way in the NBA project:

* **Discontinued books are excluded outright** (`DISCONTINUED_BOOKS`, currently
  empty for MLB — Caesars is still present here). Otherwise those games are
  re-fetched forever waiting for data the source no longer has.
* **A book counts as "expected" on a date only if it priced ≥50% of that date's
  games.** Without the threshold, a book's launch day marks every other game on
  it partial, permanently.

`--dry-run` reports the whole plan and stops before the first request.

**A scrape failure never ends a backfill.** `on_error="warn"` prints and
continues (`"raise"` is the strict mode for tests). Observed in practice: a 502
on one event during the orientation run warned and carried on. Partial success
is a first-class outcome the caller reads, not an exception.

---

## Leakage rules specific to odds

1. **In-play ticks are indistinguishable from pre-game ones except by
   `mins_to_tip`.** Enforced as a NOT NULL stored column. 3.2% of recent-season
   rows land at or after first pitch.
2. **`mins_to_tip` is negative pre-game** as stored. Any read layer must flip it
   so the sign convention never reaches feature code, and should refuse
   anything inside a safety margin of first pitch.
3. **Closing lines must be physically separated** into a scoring sidecar when
   the snapshot dataset is built — not filtered behind a prefix. The NBA
   project tried the prefix approach and the feature selector did not drop
   them, because it drops only *configured* exclusions.
4. **The opening line is safe at every horizon** and is deliberately not swept
   up with the closing columns — it is the baseline the betting evaluation
   compares against.
5. **Column availability can encode the season.** Books appear over time (3 in
   2019, 6–7 by 2022). A model can recover the year from which columns are
   non-null. Audit availability-by-season for every odds column before
   training.

---

## MLB-specific: the starting pitcher

The skill flags this as the thing that should change for baseball, and it is
the sharpest open item.

MLB totals are heavily conditioned on the **announced starting pitchers**,
which land at very different times, and books void or re-hang a total when a
starter changes. A snapshot grid anchored on hours-before-first-pitch alone
mixes "pitchers known" and "pitchers unknown" into the same horizon.

What is built: SBR's announced starters are captured on `lh_game`
(`starter_home`, `starter_away`), giving a cross-check and a join handle.

What is **not** built, and cannot be: SBR reports the starters as of *scrape
time*, not per tick. A per-tick "were the pitchers announced yet?" flag has to
come from the point-in-time probables archive in
`src/mlb_pred/snapshots/archive.py` — which is another reason to run that job
from day one.

---

## Limitations versus the NBA implementation

1. **Odds start in 2019**, three seasons after the sports-data floor of 2015.
2. **The run line is nearly uninformative about margin** (±1.5 almost always).
   Run-line modelling has to lean on the price, and orientation checks have to
   lean on the moneyline.
3. **Book coverage grows over the history** — 3 books in 2019 against 6–7 from
   2022. A real season-encoding leak; audit before training.
4. **Rain delays** are not yet handled. A tick 20 minutes "after" a scheduled
   first pitch may still be pre-game. `lh_game` stores SBR's scheduled start;
   distinguishing scheduled from actual would need the StatsAPI actual start.
5. **No normalisation layer yet** (devig-to-−110 equivalent, centered lines).
   The sigmas are measured; the transform is feature-layer work.
6. **No snapshot panel yet.** The grid should be chosen after measuring the
   coverage cliff on real MLB data, and should account for the
   pitchers-announced state rather than hours alone.
7. **No wide/closing-line table.** Deliberate — the tick store is the source of
   truth and the wide form is a derived view, per the skill's own advice for a
   fresh build.

---

## Build order status

Against the skill's checklist:

| # | Step | Status |
|---|---|---|
| 1 | Games table with stable ids, doubleheaders settled | Done |
| 2 | First-pitch times | Done |
| 3 | Provider's embedded JSON with explicit offsets | Done — verified |
| 4 | `left`/`right` per market, **measured**, pinned by a test | Done |
| 5 | Verify the quote increment before the integer encoding | Done — half-points |
| 6 | Star schema, insert-only, `mins_to_tip`/`is_pregame` NOT NULL | Done |
| 7 | Load-time repairs and per-reason drop counters | Done |
| 8 | Day-by-day backfill, polite sleeps, retries, warn-not-raise | Done |
| 9 | Update planner (refresh ∪ gaps ∪ partial) with dry-run | Done |
| 10 | Calibrate per-market sigma before normalising | Sigmas measured; transform not built |
| 11 | Measure snapshot coverage by horizon, then choose the grid | Not started |
| 12 | Separate closing lines into a scoring sidecar | Not started |

Steps 10–12 are feature-layer work and belong with
`.claude/skills/feature-engineering`.
