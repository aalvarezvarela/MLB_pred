# MLB sports-data architecture

What this repo built, why, and where it deliberately diverges from the NBA
system it is modelled on.

The reference is `.claude/skills/sports-data-architecture` (copied unchanged
from the NBA repo). This document is the MLB counterpart: it answers the same
questions with baseball's answers and records the decisions that had no NBA
equivalent.

---

## 1. What the NBA layer does, in one paragraph

One provider (the league's own API) supplies games, box scores, inactives and
officials; its ids are adopted as canonical so those four sources join with no
mapping layer. History is stored at the grain at which it *accumulates* —
one row per team per game, home/away as a boolean — and pivoted to the
prediction grain only as a final merge. `season_year` is the season's start
year everywhere; `game_date` is the Eastern-time date of tipoff and joins
across sources, while `tipoff_utc` is a real instant for anything temporal.
Every updater is gap-driven: it asks the database what is missing rather than
being told a date range, which makes it idempotent and free to re-run.
Standings are not stored — they are derived from the event log, so they are
automatically point-in-time correct.

Every one of those decisions transfers. What follows is what changed.

---

## 2. Data sources

| Need | Source | Cost | Verified back to |
|---|---|---|---|
| Games, schedule, scores, status | MLB Stats API `/v1/schedule` | **1 request per season** | 2000 |
| Umpire crew | same request (`hydrate=officials`) | free | 2000 |
| Announced lineups | same request (`hydrate=lineups`) | free | 2000 |
| Probable starters | same request (`hydrate=probablePitcher`) | free | ~2008 |
| First-pitch weather | same request (`hydrate=weather`) | free | 2000 |
| Team / batter / pitcher box lines | `/v1/game/{gamePk}/boxscore` | 1 per game (~2,430/season) | 2000 |
| Ballpark geometry, elevation, roof, orientation | `/v1/venues?hydrate=location,fieldInfo,timezone` | 1 per season | — |
| IL moves and roster churn | `/v1/transactions` | 1 per season | — |
| Active rosters | `/v1/teams/{id}/roster` | 30 per capture | — |
| Quality of contact (barrels, xwOBA, launch speed) | Baseball Savant CSV export | 1 per finished game | 2015 |
| Cross-provider player id crosswalk | Chadwick Bureau register | 16 CSV shards | — |

Everything in the first block comes from **one** provider, which is why the
NBA repo's "adopt the provider's ids as canonical" decision transfers intact.
The Stats API is unauthenticated, has no published quota, and returns a whole
season's schedule in under three seconds.

**Not collected, on purpose:** published park factors. A park factor is a
full-season snapshot, so joining one onto a mid-season game leaks the rest of
that season backwards. Park effects are derived from this repo's own
team-game log with an as-of cut, for the same reason the NBA repo derives
standings from its event log instead of scraping them.

---

## 3. What transfers directly, and what does not

### Transfers unchanged

| NBA decision | MLB status |
|---|---|
| Team-game is the atomic row; home/away is a boolean | Unchanged |
| Provider ids are canonical; names mapped, ids never | Unchanged (`gamePk`, `team.id`, MLBAM `person.id`) |
| One hand-maintained name map that **raises** on unknown | Unchanged — `TEAM_NAME_STANDARDIZATION`, 163 aliases |
| Structure comes from the id/code, not the text label | Unchanged — filters key off `gameType`, not `seriesDescription` |
| Local slate date **and** a UTC instant, never one doing both jobs | Unchanged — `game_date` + `first_pitch_utc` |
| Plausibility CHECKs at the schema boundary | Adapted (see below) |
| Gap-driven, insert-only updaters | Unchanged |
| Never ingest an in-progress event | Unchanged |
| Retry transient, **stop** on throttling | Unchanged |
| Derive point-in-time state from the event log, not snapshots | Unchanged — no standings table |
| One schema per domain | Unchanged |

### Baseball-specific adaptations

**A third accumulation grain: `pitcher_appearances`.** The NBA has two grains
(team-game, player-game). MLB needs three. A starting pitcher's history is the
strongest team-independent signal for a run total and it accumulates on its
own clock — roughly every fifth team game. Burying it in a player table keyed
on team games makes the natural window ("last 5 starts") inexpressible.
`is_starter` and `appearance_number` are first-class columns so that a
starter's history can be counted in starts while a reliever's workload is
counted in days.

**`season_year` is trivially the calendar year.** An MLB season does not
straddle New Year. The named helper is kept anyway so the concept stays one
function.

**Doubleheaders are a non-issue.** The skill flags "settle doubleheader keying
now" as step 1. `gamePk` is globally unique, so both games of a doubleheader
already have distinct ids; `game_number` and `doubleHeader` are stored as
context, not identity. No compound key is needed.

**Plausibility guards are inverted, and the useful one is relative.** The NBA
guards `pts > 40` — a floor, because a basketball team cannot score 40. Zero
runs is legal in baseball, so the run guard is a **ceiling**
(`runs BETWEEN 0 AND 40`).

The outs guard took two attempts. An absolute floor of 24 ("nine innings, home
team winning") flagged 523 rows — 1% of the store — every one of them a
legitimate rain-shortened game or a 2020-21 seven-inning doubleheader. Any
floor high enough to catch a partial nine-inning box score also catches a real
short game, so an absolute floor cannot do this job.

The check that works is **relative to the game's own length**: each team's outs
must fall within one inning of `innings_played`. Verified across all 51,266
team-game rows with zero violations, while still catching the failure that
matters — a mid-game fetch, whose outs sit far below the game's final inning
count. The absolute band survives only as a cheap outer bound (12 outs, the
true minimum for an official five-inning game).

**Innings are stored as outs.** `7.1` innings means seven innings and one
out, not 7.1 innings. Every table stores `outs_recorded` as the integer of
record and derives `innings_pitched` from it. Averaging the raw notation
treats a third of an inning as a tenth.

**Umpires replace referees, and only one position matters.** The home-plate
umpire's strike zone drives K% and BB% and therefore run totals. The full crew
is stored (`official_type`, `is_home_plate`) so a crew-level feature stays
possible, but HP is the analogue the NBA officials table was built for.

**The ballpark is a first-class entity, not a lat/lon lookup.** The NBA keeps
`CITY_TO_LATLON` and `CITY_TO_TIMEZONE` because travel is all an arena
contributes. Park effects dominate MLB run totals, so `venues` carries
outfield dimensions at seven points, elevation, roof type, turf, and
`azimuth_angle` — the compass bearing from home plate to centre field, without
which a wind reading like "12 mph, Out To RF" is uninterpretable across parks.
Venues are keyed `(venue_id, season)` because dimensions get changed.

**Bullpen state has no NBA equivalent and is genuinely new work.** The
`pitcher_appearances` grain is the substrate: relief innings thrown in the
last three days is MLB's closest analogue to NBA fatigue. The data layer
stores the appearances; the feature is downstream work.

### Enrichment layer

- **Statcast enrichment** stores Baseball Savant pitch details at
  `(game_pk, at_bat_number, pitch_number)`: pitch mix/velocity/movement,
  location, contact quality, xBA/xwOBA, batted-ball shape, and score/base-out
  context. It is gap-driven through `statcast_games`; facts are committed
  before completion metadata so interruption causes a safe re-fetch.

### Deliberately deferred
- **Chadwick crosswalk.** Not needed while the Stats API is the only source —
  MLBAM ids are already canonical. It becomes necessary the moment a FanGraphs
  or Baseball-Reference feed is added, and the register is live (16 shards).
- **An external importance signal** (projected WAR / Steamer / ZiPS), the
  analogue of the NBA's All-Star voting scrape. The reusable point is not
  "all-star votes" but *having one importance signal that is not a function of
  your own rolling stats*, so that "this team's best player is out" means
  something in April when the rolling stats are empty.
- **Odds.** See `.claude/skills/odds-data-architecture`.

---

## 4. Schema

One Postgres database, one schema per domain, mirroring the NBA layout.

| Schema.table | Grain | Primary key |
|---|---|---|
| `mlb_games.games` | one row per game | `game_pk` |
| `mlb_team_games.team_games` | **two rows per game** | `(game_pk, team_id)` |
| `mlb_batter_games.batter_games` | one row per batter per game | `(game_pk, team_id, player_id)` |
| `mlb_pitcher_appearances.pitcher_appearances` | one row per pitcher per game | `(game_pk, team_id, player_id)` |
| `mlb_umpires.umpires` | one row per official per game | `(game_pk, official_id, official_type)` |
| `mlb_lineups.lineups` | one row per listed player per game | `(game_pk, team_id, player_id)` |
| `mlb_venues.venues` | one row per park per season | `(venue_id, season)` |
| `mlb_transactions.transactions` | one row per announced move | `transaction_id` |
| `mlb_statcast.statcast_games` | one fetch-completion row per game | `game_pk` |
| `mlb_statcast.statcast_pitches` | one row per pitch | `(game_pk, at_bat_number, pitch_number)` |

All ids are `TEXT`. They are opaque identifiers, never arithmetic operands,
and keeping them textual stops a missing value from coercing an id column to
float and silently reformatting every id in it.

`first_pitch_utc` is MLB's scheduled start instant, including schedule
revisions. It is useful for ordering and odds alignment, but should not be
misread as the observed first pitch when a game starts after a delay.

**One naming trap worth stating.** `home_runs` means two different things
in a baseball schema: runs scored by the home team, and home runs hit. The
games table therefore uses `home_score` / `away_score`, and `home_runs` is
reserved for the batted-ball meaning in `team_games` and `batter_games`. The
collision was not theoretical — it broke the store validator's join before
the rename.

`team_games` follows the skill's category list rather than a stat dump:

- **Outcome** — `runs_scored`, `runs_allowed`, `run_margin`, `total_runs`, `win`
- **Volume / tempo** — `plate_appearances`, `batters_faced`, `outs_recorded`,
  `pitches_thrown`, `baserunners_allowed`
- **Efficiency rates** — `obp`, `slg`, `iso`, `babip`, `k_pct`, `bb_pct`,
  `whip`, `strike_pct`, and the allowed-side mirrors
- **Component counts** — the hit types, batted-ball mix, walks, strikeouts
- **Context** — `home`, `venue_id`, `game_type`, `extra_innings`

The tempo/efficiency split is the transferable idea: a run total is
`expected_events × expected_value_per_event`, and storing both factors as
rates lets them be recombined against a specific opponent rather than only
their product.

---

## 5. Ingestion and keeping current

**Backfill** (`scripts/fetch_data/backfill_mlb_data.py`) walks seasons from
2015. Per season: one schedule request populates games, umpires and lineups;
one venues request; one transactions request; then one box-score request per
missing game at roughly 300 games per minute — about eight minutes per
season, ninety minutes for the full 2015-present range. Results flush to the
store every 250 games, so an interrupted run keeps what it fetched.

**Gap-driven.** Each updater loads what the games table says should exist,
asks the target table which ids it already has, and fetches only the
difference. A game counts as done only when it is present in *every*
box-score-derived table, so a run that died between two writes is repaired on
the next pass rather than leaving a game half-ingested. Interrupting a
backfill costs nothing.

**Daily** (`scripts/update_databases/update_mlb_data.py`) refreshes the last
three finished slates rather than only yesterday's — games get suspended and
resumed, and scores get corrected. Today's slate is excluded: a box score
fetched mid-game is a *partial* box score that looks complete, and written
into an insert-only table it poisons that game permanently.

**Validated, not assumed.** `scripts/fetch_data/validate_store.py` runs the
same plausibility bounds Postgres enforces as CHECK constraints, plus the
relational invariants no per-row bound can see: exactly two team rows per
game, one starting pitcher per team, pitcher outs summing to the team-game
total, runs reciprocal between the two sides, and no orphans. That last group
is what catches a *partial* ingest, which is the failure mode a mid-game fetch
produces.

**Throttling stops the run; transient failure retries.** The client raises
`RateLimitedError` on a 429/403 instead of retrying into a ban. Because
everything is gap-driven and insert-only, stopping early is free.

### Bracket placeholders

The Stats API publishes an **unplayed postseason bracket** with placeholder
"teams" — `AL Wild Card #1`, `NL Higher Seed`, `Higher Seed League Champion` —
53 of them for 2026. The raise-on-unknown name map stopped the backfill on the
first one, which is exactly what it is for; the fix is to recognise them
rather than to weaken the guard.

They are detected by **team id, not name**: every placeholder carries an id
well outside the 30-franchise range (2710, 4613, 5517 …), while placeholder
*names* vary freely between rounds. Keying on the name would mean accumulating
bracket vocabulary in a map whose only job is resolving real franchises.

The escape hatch is deliberately narrow. The other reading of an unknown team
id is an **expansion franchise**, which must never be silently skipped — so a
placeholder is only accepted as such on a postseason game that has *not been
played*. An unknown id on a regular-season game, or on any game with a final
score, still falls through to the name map and raises.

### The schedule feed's double-listing

The feed lists a game once per slot it has occupied. In 2025 that produced 34
duplicated `gamePk`s from two causes — postponed-then-replayed (the original
slot survives with no score) and suspended-then-resumed (both rows Final, same
score, different first pitch). Left alone this silently doubles a team's game
count for those dates. `deduplicate_games` prefers a row that actually
finished and, among finished rows, keeps the earliest first pitch. Games that
never reached a final score are dropped rather than stored with NULL scores,
because a NULL-score row reaches a rolling mean as a real observation.

---

## 6. Availability, and the leakage line

This is the subtlest part of the model and where MLB's trap is **sharper than
the NBA's**.

An injured-list stint is routinely *backdated*: a player placed on the IL on
25 August is placed "retroactive to 24 August". Query a source that records
only the retroactive date and it will tell you the player was on the IL on the
24th — on a day when nobody knew. That is not optimism, it is genuine
look-ahead leakage.

**The Stats API publishes both dates**, so this one is fixable rather than
merely documentable. `known_date` is the announcement; `effective_date` is
when it took effect and may be earlier. `transactions_known_as_of()` filters
on `known_date` only, and `is_backdated` / `backdated_days` make the exposure
measurable. In one sample week, 23 of 305 transactions were backdated by up to
three days.

The same trap applies to `/v1/teams/{id}/roster?date=`. It looks like it makes
point-in-time rosters free. It does not — that query is reconstructed after
the fact and reflects backdated placements.

### What cannot be backfilled

The transaction feed says when a player *became* unavailable. It does not say
who a club *expected* to play. The lineup card, the probable-pitcher listing
and the active roster are forecasts that get revised, and no retrospective
query reproduces what they said at 11am.

`src/mlb_pred/snapshots/archive.py` archives them, timestamped at fetch,
append-only — a lineup revised at 15:40 does not replace the 11:00 version,
because the revision is itself signal. `latest_snapshot_before()` is the
accessor a backtest must use; reading the last capture of a day would hand the
model a lineup revised after the information it is predicting from.

**This is the one job whose missed runs cannot be repaired.** Run it from day
one. `--coverage` reports gaps, and they are permanent.

---

## 7. Known limitations versus the NBA implementation

1. **Backtest availability features are an upper bound until the snapshot
   archive matures.** Today's backtest reads the settled transaction feed
   while production will read a forecast. The archive is what eventually
   closes the gap *and* measures it; until then, treat availability features
   as optimistic. This is the NBA repo's documented, unmeasured skew — here it
   is at least instrumented.
2. **Probable starters are not guaranteed.** A listed starter can be scratched
   an hour before first pitch. `probable_announced` records whether one was
   named at all, which is itself signal (a TBD often means a bullpen game).
3. **Weather from the schedule feed is the recorded condition for finished
   games and a forecast for scheduled ones.** Same field, two semantics.
   Anything sensitive to it needs the snapshot archive, not the games table.
4. **Statcast starts in 2015.** This set the backfill floor. Extending earlier
   is possible — the Stats API reaches 2000 — but quality-of-contact features
   would be NULL for those rows.
5. **Rule-change discontinuities have no NBA analogue at this scale.** The
   2020 60-game season, the 2022 universal DH, and the 2023 pitch clock plus
   shift ban each moved the run environment. Rolling windows do not span them
   cleanly, and `season_year` is the handle for excluding or flagging them.
6. **No park factors, by design.** See §2.
7. **Fielding is thin.** The box score gives errors and little else. Defensive
   run value would need a separate source.

---

## 8. Build order, and where this repo stands

Following the skill's ordered checklist:

| # | Step | Status |
|---|---|---|
| 1 | Games table, plausibility CHECKs, doubleheader keying | Done |
| 2 | Team-game table, two rows per game | Done |
| 3 | Name standardisation map that raises | Done |
| 4 | Player-game table **and** pitcher-appearance grain | Done |
| 5 | Schedule feed for start times | Done |
| 6 | **Daily point-in-time snapshots** | Done — start running it now |
| 7 | Officials / umpires | Done |
| 8 | Odds | Done — SportsbookReview tick history (2019+) |
| 9 | Venue geography and park context | Done |
| 10 | One external importance signal | Not started |
| 11 | Gap-driven updaters, daily orchestrator | Done |
| 12 | Predictions table and settlement job | Not started |

Steps 10 and 12 are the remaining sports-data-adjacent work. Statcast is also
implemented as a pitch-level enrichment. Feature engineering and modelling
are deliberately untouched; no additional weather enrichment was added in
this work.
