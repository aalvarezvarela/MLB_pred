# Intermediate-line dataset — a port of the NBA design

*Third revision. The first draft planned the right thing; the second
misread the request and planned a line-forecasting dataset. This one is the
faithful port that was asked for.*

> **The design.** One row per (game, snapshot). At each snapshot the market line
> **as of that moment** replaces the closing line as the anchor, and the target
> is the same residual against it that the main dataset measures against the
> close. The real closing line goes to a scoring sidecar and measures CLV. The
> snapshot time is carried as a column, for filtering.

This is `nba_ou/create_training_data/create_intermediate_line_df.py` and the five
modules under `nba_ou/data_processing/line_history/`. The MLB port keeps their
structure, their column semantics, and their leakage gate.

---

## 1. The shape, stated once

| | main dataset (`training_data_2_0`) | this dataset |
|---|---|---|
| grain | one row per game | one row per **(game, snapshot)** |
| anchor line | closing total / run line | the total / run line **as of the snapshot** |
| target | `LINE_ERROR = TOTAL_RUNS − closing line` | `LINE_ERROR = TOTAL_RUNS − snapshot line` |
| closing line | is the anchor | **scoring sidecar only**, to measure CLV |
| time column | — | `TIME_TO_MATCH_MIN` |

The anchor column keeps its ordinary name and holds the snapshot value. NBA
states the reason and it applies unchanged: the target is derived against the
anchor column and a bet settles into the same number, so target and settlement
price must be the same quantity. The genuine close is carried separately under
`ODDS_CLOSING_*` and never reaches the feature matrix.

`TIME_TO_MATCH_MIN` uses NBA's exact column name so downstream filtering and
per-horizon reporting can be lifted across too.

---

## 2. Is it worth building? Measured, and the answer shapes the grid

**The anchor barely degrades with lead time.** What the model is handed at each
snapshot, over 17,588 games:

| snapshot | 12h | 8h | 6h | 4h | 3h | 2h | 1h | close |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coverage | 92.1% | 96.2% | 97.6% | 99.0% | 99.7% | 99.9% | 100.0% | 100.0% |
| corr(line, TOTAL_RUNS) | 0.2001 | 0.2106 | 0.2114 | 0.2140 | 0.2144 | 0.2163 | 0.2171 | 0.2174 |
| RMSE | 4.466 | 4.460 | 4.459 | 4.455 | 4.453 | 4.452 | 4.452 | 4.452 |
| over-rate at the line | 49.27% | 49.30% | 49.37% | 49.20% | 49.26% | 49.33% | 49.29% | 49.36% |

Every snapshot line is nearly as sharp as the close, and every one of them is
efficient in the same way — the over-rate sits at 49.3% throughout, so there is
no horizon at which the market is simply mispricing.

**So the edge is not in beating the snapshot line outright. It is in being early
to the move the market itself will make.** Two prior measurements say that move
is real information: it happens in 49.2% of games between 12h and close with mean
absolute size 0.235 runs, and regressing the eventual residual on it gives
**beta +1.02** — the move is fully justified by what happens.

**The ceiling, and the reason to cut the grid short.** Betting the side the
market is about to move toward, at the snapshot price — i.e. perfect foresight of
the line's own path, nothing about baseball:

| decision point | bets | win rate | vs 52.38% breakeven | ROI @ −110 |
|---|---:|---:|---:|---:|
| **12h** | 7,666 | 54.70% | +2.32 pts | **+4.42%** |
| **8h** | 6,314 | 53.98% | +1.60 pts | **+3.04%** |
| **6h** | 5,655 | 54.24% | +1.86 pts | **+3.54%** |
| **4h** | 4,866 | 53.51% | +1.13 pts | **+2.16%** |
| 3h | 4,249 | 52.29% | −0.09 pts | −0.16% |
| 2h | 3,219 | 51.38% | −1.00 pts | −1.91% |
| 1h | 2,090 | 50.43% | −1.95 pts | −3.72% |

Restricted to moves of at least half a run, the same ceiling runs +6.2% at 12h,
+7.1% at 6h, +5.9% at 4h, +4.4% at 3h, +2.2% at 2h.

**Inside three hours the game is over before it starts.** Not because the model
would be bad, but because the remaining movement (0.04-0.05 runs mean absolute)
cannot cover the vig even if predicted perfectly. The exploitable window is
**12h to 4h**.

### Grid recommendation

```python
SNAPSHOT_GRID = (0, 60, 120, 180, 240, 360, 480, 720)   # minutes
MODELLING_HORIZONS = (240, 360, 480, 720)               # 4h, 6h, 8h, 12h
```

Keep all eight as **rows** — snapshots are rows, so dropping one later is a
filter while adding one back is a rebuild, and the short horizons are needed
anyway to measure CLV and to serve as the reference the long horizons are scored
against. But **train and report on 4h-12h**, and treat 3h/2h/1h as diagnostics.

Not adding 18h (88.2% coverage) or 24h (70.9%): the missing games are the ones
whose books opened late, so a deeper horizon buys a biased sample rather than
lead time.

---

## 3. Module map — what to port, what already exists

| NBA | MLB | status |
|---|---|---|
| `line_history/snapshots.py` | `features/line_snapshots.py` | **new**, direct port |
| `line_history/movement_features.py` | `features/line_movement.py` | **new**, direct port |
| `line_history/cross_book.py` | `features/line_cross_book.py` | **new**, direct port |
| `line_history/history_features.py` | — | **already exists** as `ODDS_HISTORY_*` in `rolling_features.py` (89 columns of prior games' market error and level) |
| `line_history/normalization.py` | — | **already exists** as `features/market_normalization.py` |
| `line_history/book_merge.py` | — | **not needed**; `STABLE_BOOKS` already handles MLB's book drift |
| `create_intermediate_line_df.py` | `create_training_data/intermediate_frame.py` | **new** |
| `select_intermediate_columns.py` | `create_training_data/select_intermediate_columns.py` | **new**, direct port |

Two thirds of the machinery is already in the repo. The new work is the snapshot
panel, the movement layer over it, and the gate.

**Anchor book: bet365.** The only book quoting essentially every game in every
season (2019: 2,445/2,445 … 2026: 2,077/2,079). Same choice NBA makes.

---

## 4. Temporal contract

`mtt` = minutes before scheduled first pitch = `−mins_to_tip` (the store holds
`mins_to_tip` **negative** pre-game).

At snapshot **T**, a tick is admissible iff `is_pregame` **and** `mtt ≥ T`. The
snapshot value is the **most recent admissible tick** — the smallest `mtt` that
is still `≥ T` — carried forward per `(game_pk, market, book_slug)`. `≥` is
inclusive at the boundary only: a tick exactly at the horizon was observable, one
minute later was not.

`LINE_AGE_MIN = mtt(tick) − T` rides beside every quote. NBA emits it because the
carried line is often substantially stale, and staleness is itself information.

> **Sign hazard, pinned by test T1.** `mtt` counts backwards, so the largest
> admissible value is the **earliest** tick, not the latest. Inverting it returns
> the opener at every snapshot, which does not read as a bug — it reads as a
> perfectly stable market. It produced two wrong measurements while preparing an
> earlier draft (movement at 12h reported as 2.7%; the true figure is 49.2%).

Non-odds features must be computable from games strictly earlier by **date**.
That is already what the existing builders enforce, and it is conservative at
every horizon here.

---

## 5. Feature families

### 5.1 Reused unchanged — 583 columns

All read strictly-prior completed games, so they are valid at every snapshot.

`TEAM_ROLLING_*` (299) · `ODDS_HISTORY_*` (89) · `ODDS_MARKET_REGIME_*` (62) ·
`STATCAST_*` (60) · `BULLPEN_*` (32) · `TEAM_MATCHUP_*` (26) · `SCHEDULE_*` (25) ·
`TRAVEL_*` (22) · `TEAM_EXTRA_INNINGS_*` (15) · `PARK_*` (11) ·
`TEAM_HISTORICAL_MATCHUP_*` (10) · `GAME_CALENDAR_*` / `GAME_COMPETITION_*`

`ODDS_HISTORY_*` deserves a note: it rolls *prior* games' closing lines and
market errors. Those games closed before the snapshot, so their closing lines
were genuinely known. This is NBA's `history_features.py`, and MLB already has
it — the one place a closing line is legitimately readable.

### 5.2 Rebuilt as-of the snapshot — replaces 190 closing-derived columns

`ODDS_TOTAL_*` (59), `ODDS_RUN_LINE_*` (67), `ODDS_MONEY_LINE_*` (41),
`ODDS_INTERACTION_*` (21) and `ODDS_DERIVED_*` (2) all hold the **closing**
quote. They become `ODDS_SNAP_*` at T, and the closing versions move to the
sidecar under `ODDS_CLOSING_*`.

### 5.3 Excluded entirely — 163 columns

| family | cols | reason |
|---|---:|---|
| `TEAM_AVAILABILITY_*` | 138 | Built from the final lineup and final IL state. We hold the outcome of those processes, not their publication times, so the state at a 12h snapshot cannot be reconstructed. |
| `ODDS_MOVEMENT_*` | 20 | Every column measures open-to-**close**. Direct leakage at every T > 0; superseded by §5.4. |
| `UMPIRE_*` | 5 | Same reconstruction problem — the `umpires` table is the final recorded crew, with no publication time. |

Cost on the record: `TEAM_AVAILABILITY_STARTING_PITCHER_*` goes out with its
prefix, and the starting pitcher is the largest single pre-game determinant of an
MLB total. The exclusion is right — probables are announced days ahead but "days
ahead" is not reconstructable, and late scratches are exactly what moves a line.
`snapshots/archive.py` already captures daily probables, so this family is
reinstatable **prospectively** once that archive has depth.

### 5.4 New — the snapshot market block

**`ODDS_SNAP_*`, the quote at T.** Anchor book in full, consensus over the stable
books, anchor-minus-consensus:

```
RAW_LINE · NORM_LINE · NORM_MINUS_RAW · PRICE_OVER_RAW · PRICE_UNDER_RAW
FAIR_PROB_OVER · OVERROUND · HAS_QUOTE · LINE_AGE_MIN
CONSENSUS_{MEDIAN,STD,RANGE,BOOK_COUNT} · ANCHOR_MINUS_CONSENSUS
```

`NORM_MINUS_RAW` is the half-tick a book has priced but not yet taken. Measured
here, it is the strongest single market-internal predictor of where the line goes
next (r ≈ +0.13, beta up to +0.67), and `center_total_lines` already computes it.

**`ODDS_SNAP_PATH_*`, the path up to T** — this is the "línea 1h antes" family:

```
OPEN_LINE · MINUTES_SINCE_OPEN · MOVE_FROM_OPEN · ABS_MOVE_FROM_OPEN
FIRST_MOVE_DIRECTION · N_TICKS · N_MOVES · N_PRICE_ONLY_TICKS
N_DISTINCT_LEVELS · N_REVERSALS · LINE_MIN · LINE_MAX · LINE_RANGE
POSITION_IN_RANGE · N_MOVES_PER_HOUR
MOVE_LAST_{w} · ABS_MOVE_LAST_{w} · VELOCITY_LAST_{w} · HAS_WINDOW_{w}
```

for `w ∈ (60, 120, 240, 480)`. `MOVE_LAST_60` **is** "the line one hour before
this snapshot", expressed as a delta. NBA answers these with a **second as-of
read at `T + w`** rather than separate code, so both come from one implementation
with one leakage surface — port that. `HAS_WINDOW_{w}` separates "did not move"
from "no history"; without it the long horizons, systematically the ones lacking
history, are the rows a NaN-count cleaner deletes.

NBA's 15- and 30-minute windows are dropped: only 3.1% of MLB line moves land
inside 30 minutes of first pitch.

**`ODDS_SNAP_XBOOK_*`, cross-book at T.** Consensus median, dispersion, book
count, median/max line age, and steam over a pinned 60-minute window
(`STEAM_BOOKS_UP/DOWN/NET`, `STEAM_AGREEMENT`), plus each book's
`DEVIATION_FROM_CONSENSUS` and `DEVIATION_Z`.

---

## 6. Schema

**Grain:** one row per `(GAME_ID, TIME_TO_MATCH_MIN)`. 17,588 games × 8 snapshots
≈ 140,000 rows.

**Identifiers.** `GAME_ID`, `GAME_DATE`, `SEASON_YEAR`, `TIME_TO_MATCH_MIN`.
`FIRST_PITCH_UTC` and `SNAPSHOT_TS_UTC` are sidecar-only — raw timestamps let a
model pin individual games.

**Anchor columns** (hold the snapshot value, survive the gate under their own
names):

```
ODDS_TOTAL_BET365_LINE_NORMALIZED          the totals anchor
ODDS_RUN_LINE_BET365_HOME_HANDICAP_NORMALIZED   the run-line anchor
```

**Targets**, derived against those anchors exactly as the main dataset derives
them against the close:

```
TOTAL_RUNS · RUN_LINE_MARGIN            recomputed from the scores
LINE_ERROR   = TOTAL_RUNS      − totals anchor at T
SPREAD_ERROR = RUN_LINE_MARGIN − run-line anchor at T
```

The run-line target carries a caveat: 88% of MLB run-line quotes are pinned at
±1.5, so `SPREAD_ERROR` is coarse and most of its variation is the game, not the
line. Worth building — NBA has both markets — but expect the totals target to
carry the project.

**Features:** §5.1 + §5.4.

**Sidecar**, keyed `(GAME_ID, TIME_TO_MATCH_MIN)`, physically separate file:

```
FIRST_PITCH_UTC · SNAPSHOT_TS_UTC
ODDS_CLOSING_*              the whole closing quote, both markets
HOME_SCORE · AWAY_SCORE
CLV_TOTAL   = closing total    − snapshot total
CLV_RUN_LINE = closing handicap − snapshot handicap
ODDS_SNAP_MONEY_LINE_*      moneyline block, pending §7
```

Physical separation, not a prefix convention: a prefix relies on every consumer
remembering to filter, and that is exactly how NBA's closing lines reached X.

---

## 7. The path block — "velocity y esas cosas"

Every snapshot carries the game's **own earlier timepoints**, summarised. Three
encodings are possible; NBA uses two of them and deliberately refuses the third.

**(a) Aggregates over the whole path up to T** — everything the market has done
on this game so far:

```
OPEN_LINE · MINUTES_SINCE_OPEN · MOVE_FROM_OPEN · ABS_MOVE_FROM_OPEN
PCT_MOVE_FROM_OPEN · FIRST_MOVE_DIRECTION · OPPOSES_OPENING_DIRECTION
N_TICKS · N_MOVES · N_PRICE_ONLY_TICKS · N_DISTINCT_LEVELS · N_REVERSALS
LINE_MIN · LINE_MAX · LINE_RANGE · LINE_STD · POSITION_IN_RANGE
N_MOVES_PER_HOUR
```

`N_MOVES_PER_HOUR` exists because `N_MOVES` grows as T approaches first pitch and
is therefore partly a proxy for elapsed time; the rate separates "busy market"
from "more hours elapsed".

**(b) Windowed deltas and velocity** — the shape of the recent path:

```
MOVE_LAST_{w} · ABS_MOVE_LAST_{w} · VELOCITY_LAST_{w} · HAS_WINDOW_{w}
PROB_MOVE_LAST_{w}
MOVE_ACCELERATION = MOVE_LAST_60 - MOVE_LAST_180 / 3
```

for `w ∈ (60, 120, 240, 480)`. `VELOCITY_LAST_{w} = MOVE_LAST_{w} / (w/60)` —
runs per hour, so windows are comparable to each other. `MOVE_LAST_60` **is**
"the line one hour before this snapshot", expressed as a delta.

These are answered by a **second as-of read at `T + w`** over an extended grid
(`{T} ∪ {T+w}`), not by separate code. One implementation, one leakage surface —
port that structure, not just the columns. `HAS_WINDOW_{w}` separates "did not
move" from "no history"; without it the long horizons, systematically the ones
lacking history, are exactly the rows a NaN-count cleaner deletes.

**(c) Raw lagged levels** — the line at T+60 as its own column. **Not emitted.**
It correlates above 0.98 with the current level (measured: adjacent horizons give
R² 0.976-0.986), so it is a near-duplicate that splits permutation importance
between twins and adds no information the delta does not carry. The delta is the
correct encoding of a lag.

NBA's 15- and 30-minute windows are dropped: only 3.1% of MLB line moves land
inside 30 minutes of first pitch.

### Do they earn their place? Measured

At T=12h, against the totals target `TOTAL_RUNS − line(12h)` (n = 15,007, residual
std 4.452):

| feature | r with residual |
|---|---:|
| `NORM_MINUS_RAW` (half-tick priced, not taken) | **+0.0320** |
| `MOVE_LAST_480` | +0.0094 |
| `MOVE_LAST_240` / `VELOCITY_LAST_240` | +0.0092 |
| `LINE_RANGE` | −0.0095 |
| `MOVE_LAST_60` | +0.0052 |
| `N_MOVES`, `N_TICKS`, `MOVES_PER_HOUR`, `POSITION_IN_RANGE` | +0.003 or less |

Joint OLS R² = **0.00130**, roughly nine times the 1.4e-4 the mechanical chain
(path → move R²≈0.02 → outcome beta≈1.0) would predict.

Read honestly: on their own these would pick winners at about 51.1%, below the
52.38% breakeven. They are a contributing block among ~700 columns, not the edge,
and `NORM_MINUS_RAW` is the one carrying most of it. Worth building — the
correlation is ~3.9 SE from zero at this sample size — but not worth expecting
much from alone.

---

## 7b. A third dataset for the spread — possible, but the spread has nothing in it

**Technically: yes, and cheaply.** The snapshot panel, the path layer, the
cross-book layer and the gate are all market-agnostic. A run-line dataset is the
same code with `market="run_line"` and the level read as the devigged home-cover
probability rather than the handicap.

**Empirically: the MLB run line does not reward being early.** Three
measurements, all on 15,000-16,000 games.

*The handicap barely moves; the price moves constantly.* Share of games whose
value at T differs from the close:

| T | 12h | 8h | 6h | 4h | 3h | 2h | 1h |
|---|---:|---:|---:|---:|---:|---:|---:|
| handicap changed | 8.1% | 6.0% | 5.5% | 4.7% | 3.9% | 2.8% | 1.8% |
| **cover probability changed** | **95.0%** | 92.2% | 90.2% | 87.4% | 83.8% | 74.6% | 58.7% |
| mean abs Δ probability | 2.60 pts | — | 1.77 pts | 1.48 pts | — | — | — |

So the level must be the probability, not the handicap — the ±1.5 pinning means
the number is nearly constant while the market re-prices underneath it.

*But that movement is uninformative.* Regressing the realised cover residual on
the probability move:

| T | 12h | 8h | 6h | 4h | 3h | 2h | 1h |
|---|---:|---:|---:|---:|---:|---:|---:|
| beta | −0.055 | −0.043 | +0.078 | +0.044 | +0.026 | −0.024 | −0.006 |
| r | −0.006 | −0.004 | +0.007 | +0.003 | +0.002 | −0.002 | −0.000 |

Zero at every horizon. Following the move wins **49.5-50.1%** — a coin flip.
Compare the totals, where the same test gives beta **+1.02** and a +4.42% ROI
ceiling at 12h.

*And decisively: the closing run line is no sharper than the 12h run line.*
Like-for-like on 16,172 games quoted at every horizon:

| T | 12h | 8h | 6h | 4h | 3h | 2h | 1h | close |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Brier | **0.24157** | 0.24175 | 0.24212 | 0.24224 | 0.24198 | 0.24201 | 0.24199 | 0.24204 |
| corr(p, covered) | +0.1799 | +0.1777 | +0.1733 | +0.1719 | +0.1750 | +0.1741 | +0.1745 | +0.1740 |

Flat, and if anything the 12h line is marginally *better*. There is no
information accumulating between 12h and first pitch on this market.

For contrast, the **moneyline** does accumulate information, monotonically —
Brier 0.24039 at 12h to 0.23986 at 1h, corr +0.1861 to +0.1917 — but the
improvement is 0.2%, an order of magnitude below the totals.

### What this means, and the recommendation

The negative result has a positive reading. Because the run-line market does not
sharpen, **a run-line bet placed 12 hours out is anchored on a line as good as the
close.** There is no "get in early" edge, but there is also no cost to being
early. That is a liquidity and timing argument, not an information one, and it is
worth stating because it is the actual reason to build the file.

**Recommendation: one builder, parameterised by market, emitting per-market
files.**

```
create_intermediate_line_data.py --market totals    -> intermediate_lines_totals_*.csv
create_intermediate_line_data.py --market run_line  -> intermediate_lines_run_line_*.csv
                                                    -> ..._scoring.csv for each
```

You get the third dataset as an artifact without a third codebase. NBA puts both
markets in one file; separate files suit MLB better here, because the two markets
need different levels (line vs probability), different targets and different
column blocks, and each training run then reads a focused CSV.

**Build the totals file first.** It is where the measured edge is. The run-line
file costs little once the totals one exists — the same modules with a market
argument — but it should be built with the expectation set by the table above,
not with the totals result in mind.

**The moneyline** stays in the sidecar for now. Promote it to features only if a
`line only` vs `line + moneyline` walk-forward comparison shows incremental
value, the same rule the consolidation work used.

---

## 8. Pipeline

One script, one `--market` argument, one file per market (§7b):

```
scripts/create_train_data/create_intermediate_line_data.py --market totals
  -> data/train_data/intermediate_totals_{VERSION}_{maxdate}.csv
  -> data/train_data/intermediate_totals_{VERSION}_{maxdate}_scoring.csv

scripts/create_train_data/create_intermediate_line_data.py --market run_line
  -> data/train_data/intermediate_run_line_{VERSION}_{maxdate}.csv
  -> data/train_data/intermediate_run_line_{VERSION}_{maxdate}_scoring.csv
```

The market argument selects the anchor, the target and the level definition
(raw line for totals, devigged home-cover probability for the run line). Every
other module is market-agnostic and takes it as a parameter.

Add `INTERMEDIATE_LINE_SCHEMA_VERSION` to `config/dataset_versions.py`, versioned
independently of `TRAINING_DATA_SCHEMA_VERSION`. Both markets share the version:
they are generated by one builder and a schema change touches both.

Horizon selection at the cleaning step, with the duplicate check as an assertion:

```python
def select_horizon(frame, minutes):
    rows = frame.loc[frame["TIME_TO_MATCH_MIN"] == minutes]
    if rows.empty:
        raise ValueError(f"No rows at TIME_TO_MATCH_MIN={minutes}")
    if rows["GAME_ID"].duplicated().any():
        raise ValueError("Horizon filter must leave exactly one row per game.")
    return rows
```

Filtering to one horizon is what makes each model's rows independent. NBA pooled
horizons and needed a per-row weight to correct the overstated evidence — a
weight that was emitted and read by nothing, later replaced by per-horizon
reporting. Filtering first avoids the problem rather than patching it.

---

## 9. Phases and acceptance tests

### Phase 1 — `features/line_snapshots.py` — **DONE**

Shipped as `src/mlb_pred/features/line_snapshots.py` with 23 tests in
`tests/test_line_snapshots.py` (20 unit, 3 behind a new `store` marker that
reads `data/`, deselected by default like `live`).

Built on the real store: **4,303,770 panel rows** over 17,588 games, 7 books and
3 markets in 45 seconds. Grid as shipped:

```python
DEFAULT_SNAPSHOT_GRID = (0, 15, 30, 45, 60, 90, 120, 180, 240, 300, 360,
                         480, 600, 720, 900, 1080)
```

Denser than the four modelling horizons of §2 by design — horizons are rows, so
narrowing later is a filter. Measured totals coverage: 100% out to 90 minutes,
99.0% at 240, 97.6% at 360, 92.2% at 720, 88.3% at 1080.

Two findings from building it:

* **The shipped consensus is over `STABLE_BOOKS`, not every book.** Per-book
  closing columns exist only for those four, so the consensus median aggregates
  them. The panel defaults to *all* books, since an absent book should lower a
  book count rather than vanish, and choosing which books earn their own columns
  belongs to the pivot step. T1 therefore restricts explicitly.
* **T4 is answered, and it was real** — see below.

Mutation-tested. Reversing the as-of read to take the earliest eligible tick
drops T1 from 99.994% to **42.31%** and fails four unit tests; flipping the
eligibility comparison fails twelve; making the boundary exclusive fails exactly
the boundary test; reading the run-line level as the handicap fails exactly the
two market-orientation tests.

#### T4, answered: 0.49% of games mislabel their own history

`mins_to_tip` is computed at ingest against the first pitch known *then*. When a
game is postponed the earlier ticks keep the old reference, so a quote posted six
hours out is stored as "forty minutes before first pitch".

Found by T1 rather than by inspection: game 824913 disagreed with the shipped
close because quotes recorded around 17:15 UTC were labelled against an 18:00
start while the game began at 23:15. Measured store-wide, comparing each tick's
implied first pitch (`line_ts + minutes_before_start`) against the stored one:

| | ticks | games |
|---|---:|---:|
| drifting by more than 2 minutes | 13,934 of 3,432,529 (0.406%) | 86 of 17,588 (**0.49%**) |

Median worst drift on an affected game is 175 minutes. By season: 2020 is worst
at 3.30%, which fits its rescheduling; every other season is under 0.9%.

Two consequences, both shipped:

1. **Ordering follows `line_ts`, not the derived minutes.** The two disagree
   exactly on these games, and the wall clock is the observed truth — it is also
   what `select_closing_quotes` orders by, which is what lets horizon zero equal
   the shipped close.
2. **`game_start_is_reliable` is emitted per game**, derived from the ticks alone
   so no caller can forget to supply a second table. Flagged rather than dropped:
   the affected horizons are still recoverable from `line_ts` if a later stage
   wants them.

#### Tests as shipped

* **T1 — line reconstruction.** The snapshot at T=0 must reproduce
  `ODDS_TOTAL_CONSENSUS_LINE_RAW_MEDIAN` from the shipped closing lines: ≥ 99.9%
  exact agreement. Measured on the corrected read: **99.99%, corr 0.999925**.
  This is the test that catches the §4 sign inversion, which no name check sees.
* **T2 — no post-snapshot ticks.** Fixture with ticks at 800, 400 and 100 minutes
  out: the value at T=360 must equal the 400-minute tick and must not change when
  the 100-minute tick is edited.
* **T3 — boundary.** A tick at exactly `mtt == T` is admissible; at `T − 1` it is
  not.
* **T4 — doubleheaders and delays.** Quantified above; `game_start_is_reliable`
  is asserted to stay under 2% of games in the real store, so a change in the
  odds ingest that reintroduced the problem would fail rather than pass quietly.

### Phase 2 — `features/line_movement.py`, `features/line_cross_book.py` — **DONE**

Shipped with 29 tests in `tests/test_line_movement.py`. Full suite 326 passing.

Built on the real store: the movement panel is **4,303,770 rows x 83 columns in
202 seconds** (47 movement columns on top of the snapshot panel's 20); the
cross-book layer reduces it to **821,732 consensus rows in 3 seconds**, and the
per-book deviation join adds 4 columns in 2.

Three design decisions that diverge from the NBA source, each for a measured
reason.

**The path is accumulated per tick, not re-aggregated per horizon.** NBA loops
the horizons and re-aggregates the eligible prefix each time. Here every "so
far" quantity is a cumulative statistic in chronological order, so the as-of read
picks up the state for free. It is O(ticks) instead of O(horizons x ticks), and
— the reason that matters — the eligibility filter then exists in exactly one
place, `line_snapshots.as_of_ticks`, rather than in two.

This required a small Phase 1 refactor: `prepare_snapshot_ticks` now does all
per-tick arithmetic (decoded lines, devig, centering, `level`) *before* any
horizon is considered, and `stack_horizons` is shared. The movement layer
therefore reads the same numbers the panel does. Computing `level` twice, once
per consumer, is how a windowed feature ends up measured against a different
definition of the line than the snapshot it is attached to.

**Two move counts, because in baseball they are two different events.**
`n_moves_so_far` counts changes in `level`; `n_line_moves_so_far` counts changes
in the number on the board. For totals they coincide. For the run line, measured
at horizon zero over the whole store:

| | mean per game |
|---|---:|
| `n_moves_so_far` (price) | **11.42** |
| `n_line_moves_so_far` (handicap) | **0.92** |

A 12:1 ratio. Reading the run line's level as its handicap — the obvious
choice — would have discarded 92% of the market's observed activity. This is the
strongest confirmation of the §7b level definition.

**`n_distinct_levels` dropped.** It is `n_moves + 1` on any path that does not
revisit a level, and `level_range_so_far` already describes the spread. Same
near-duplicate reasoning the feature consolidation used.

Also measured, on totals: books re-price the same number more than twice as often
as they move it (7.06 price-only ticks against 3.19 moves by horizon zero), which
is the half-tick mechanism of §7 quantified.

#### Mutation-tested, and one test was found wanting

Six mutations; five caught immediately:

| mutation | result |
|---|---|
| window reads `T − w` instead of `T + w` (look-ahead) | 4 tests fail |
| opening direction becomes the *latest* move | 1 test fails |
| path counters read the whole series, not the prefix | 2 tests fail |
| run line stops distinguishing a handicap move | 2 tests fail |
| consensus becomes a mean | 2 tests fail |
| **steam window unpinned** | **passed — test was wrong** |

The steam test configured windows `(60, 120, 480)`, where the pinned window and
the shortest window are the same column, so unpinning it changed nothing. Fixed
to configure a genuinely shorter window `(30, 60, 480)` and to assert that steam
reports identical figures with and without it — which is the actual NBA failure
mode. The mutation now fails.

#### Performance note

The cross-book aggregation first ran in 149 seconds; the cost was a
`groupby.apply` building a Series per group over 821,732 groups. Rewritten as
grouped sums it takes 3 seconds, and the vectorised standard deviation, range
and steam counts were checked to match the per-group definitions exactly on a
400-game sample before the old code was removed.

#### Original plan for this phase

* **T5 — window direction.** `MOVE_LAST_{w}` must equal `level(T) − level(T + w)`,
  and a non-positive `w` must raise: a negative window reads the panel at a
  *later* moment, a look-ahead that passes every column-name check.
* **T6 — steam window pinned.** Steam is measured over a fixed 60 minutes, not
  "the shortest configured window" — otherwise adding a window silently redefines
  an existing feature, which is what happened in NBA.

### Phase 3 — `create_training_data/intermediate_frame.py` — **DONE**

Shipped with `scripts/create_train_data/create_intermediate_line_data.py` and 26
tests in `tests/test_intermediate_frame.py`. Full suite 352 passing, lint clean
across the whole repo.

Both datasets built:

| | rows | games x horizons | columns | features | sidecar |
|---|---:|---|---:|---:|---:|
| `intermediate_totals_1_0` | 268,576 | 17,538 x 16 | 813 | 800 | 203 |
| `intermediate_run_line_1_0` | 268,103 | 17,535 x 16 | 814 | 800 | 203 |

Verified on the built files rather than only in fixtures:

* `LINE_ERROR + anchor == TOTAL_RUNS` exactly, over all 268,576 rows;
  `SPREAD_ERROR + SPREAD_LINE_HOME == RUN_LINE_MARGIN` likewise.
* The anchor is the snapshot: it differs from the close on 12.3% of games at
  one hour, 27.6% at four, **45.2% at twelve**.
* No forbidden prefix and no timestamp column in either training file; 190
  `ODDS_CLOSING_*` columns and the anchor moneyline prices in each sidecar.
* `select_horizon` leaves exactly one row per game at every horizon.

**Structure.** One builder, one `--market`, one pair of files per market. The
anchor market gets the full block (anchor book, consensus, non-anchor books'
deviation); the other price market gets a compact consensus block, since it
prices the same game but does not need the anchor's treatment; the moneyline
rides in the sidecar with its raw prices, promotable only once a `line only`
versus `line + moneyline` walk-forward comparison earns it.

**One deliberate divergence from NBA.** There the anchor keeps the closing
column's own name so an established training pipeline needs no change, at the
cost of a column named "closing" whose contents are not. MLB has no such
pipeline yet, so the anchor is named for what it is:
`ODDS_SNAP_ANCHOR_TOTAL_LINE_{RAW,NORMALIZED}_BEFORE`.

#### A leak the tests did not catch, and the rule that replaced them

The first build put **`GAME_FIRST_PITCH_UTC` in the training file**. It matched
neither `EXCLUDED_PREFIXES` nor `CLOSING_PREFIXES`; the twelve per-book
`*_CLOSE_TIMESTAMP_UTC` columns were only caught because they happen to start
with an odds prefix. Every fixture-based test passed.

Fixed by adding a third routing rule that is **by dtype, not by name**: any
timestamp column other than `GAME_DATE` goes to the sidecar. A name-based list
would have to be kept correct by hand forever, and it had already failed once.
The test now asserts over the frame's dtypes rather than against a list of
names, and disabling the rule fails two tests.

`feature_columns` had masked the impact — it filters by numeric dtype, so the
column was never trainable — but presence in the file is what matters, not
presence in a helper's output.

#### Mutation-tested

| mutation | result |
|---|---|
| anchor on the closing quote instead of the snapshot | 3 tests fail |
| closing quotes not routed to the sidecar | 3 fail |
| excluded families not excluded | 1 fails |
| run-line sign inverted | 1 fails |
| horizon filter stops checking duplicates | 1 fails |
| raw timestamps left in the training frame | 1 fails |

Two of these first *passed*, because the mutation prepended a sentinel to the
prefix tuple instead of replacing it — the real prefixes were still there. Re-run
correctly, both fail. A mutation that does not mutate proves nothing.

#### Original plan for this phase

* **T7 — anchor is the snapshot.** For sampled games and horizons, the anchor
  column recomputed independently from the raw ticks must match exactly, and must
  differ from the closing line on the ~49% of games that moved.
* **T8 — target consistency.** `LINE_ERROR + anchor == TOTAL_RUNS` exactly, and
  `TOTAL_RUNS` recomputed from `home_score + away_score` must match the carried
  column — the same reproducibility check `training_frame.py` already performs.
* **T9 — physical separation.** No `ODDS_CLOSING_*`, no realised score beyond the
  declared targets, no raw timestamp in the training file. Checked against the
  file's own columns, not a helper's output.

### Phase 4 — the gate

* **T10 — excluded families absent.** No `TEAM_AVAILABILITY_`, `UMPIRE_` or
  `ODDS_MOVEMENT_` column survives. Prefix match on raw columns.
* **T11 — reconstruction audit.** Port `audit_closing_line_reconstruction`:
  correlate each kept feature against the closing line, flag > 0.999, then search
  *pairs* of high-correlation candidates for an exact additive rebuild. It
  matters more here than in NBA — the path block carries an opener, a
  move-from-open and a current line, and opener + movement rebuilds the line
  exactly.
* **T12 — one row per game per horizon.** `select_horizon` leaves no duplicate
  `GAME_ID`, for every horizon.

### Phase 5 — first model

Walk-forward by season, trained and reported **per horizon**. Two baselines
beside every result:

1. the anchor alone at that horizon, and
2. the §2 ceiling — what perfect foresight of the market's own move would have
   earned (+4.42% at 12h, +3.54% at 6h, +2.16% at 4h). A model's ROI should be
   read as a fraction of that, because it is what the horizon has to give.

ROI simulated against the **raw** anchor quote, not the normalised one: the
normalised line is comparable across books and snapshots but is not a number
anyone can bet.

---

## 10. Decisions needing sign-off

1. Grid `(0, 60, 120, 180, 240, 360, 480, 720)` as rows; **model only 4h-12h**,
   because inside 3h even perfect foresight loses to the vig. §2
2. Anchor book **bet365**. §3
3. `UMPIRE_*` excluded alongside availability, by the same principle. §5.3
4. Moneyline deferred to the sidecar pending a measured `line` vs
   `line + moneyline` comparison. It is the one non-totals market where sharpness
   does improve monotonically toward first pitch, but only by 0.2%. §7b
5. **The run line gets its own file, not a second anchor in the totals file** —
   one builder with `--market`, two outputs. §7b
6. **Build the totals file first.** The run line shows no information
   accumulating between 12h and first pitch (Brier flat at 0.242, move beta ≈ 0,
   following the move wins 49.5-50.1%), so its file is worth having for the
   timing benefit but should not be expected to carry an edge. §7b
7. Path features (`velocity`, move counts, reversals, position-in-range) emitted
   as aggregates and windowed **deltas**, never as raw lagged levels. §7
