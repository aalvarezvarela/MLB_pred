# MLB feature consolidation plan

Status: **implemented**. Phases 1-4 are done; measured outcomes are in
section 8. Sections 1-7 are the original analysis and are left as written so
the plan and the result can be compared.

The modelling premise throughout: **at prediction time the sportsbook line is
known and is given to the model.** Features are judged by what they add
*conditional on the line*, never by whether they can reproduce or beat it
alone. Every number below was measured on this repo's data, not assumed.

Measurement base: `data/features/pregame/*.parquet` (2019-2026) joined to
`data/raw/games` -> 17,638 completed games x 5,617 columns.

---

## 1. Current feature families

| Family | Cols | Main problem |
|---|---:|---|
| `TEAM_ROLLING_*` / `TEAM_RATIO_*` | 2,160 | Full 20-variant window template applied to ~95 metrics |
| `ODDS_*` per book (7 books) | 1,995 | 3 of 7 books do not exist before 2022/2025 |
| `TEAM_AVAILABILITY_*` | 468 | 432 of them are one hitter-availability template |
| `ODDS_HISTORY_*` | 252 | Team-level market-error form, 15 windows x 3 roles |
| `ODDS_*_CONSENSUS_*` | 238 | 7 summary stats per quantity, 4 of them near-duplicates |
| `ODDS_MARKET_REGIME_*` | 204 | 8 windows x 10 stats x 2 markets |
| `TEAM_IDENTITY_*` | 60 | 60 one-hots for 30 teams |
| `TEAM_RECORD_*` | 51 | `WINS_LAST_N` and `WIN_RATIO_LAST_N` are the same column |
| `SCHEDULE_*`, `TRAVEL_*`, `GAME_*`, matchup | ~140 | Reasonable size |

**Effective rank of the whole matrix: 80.5.** 468 principal components carry
90% of the variance; 1,266 carry 99%. 5,617 columns encode roughly 80
independent directions.

### Objective redundancy (measured)

- **243 constant columns.** Includes 28 `*_PRICE_*_NORMALIZED` that are always
  exactly `-110` (that is what normalisation means) and the
  `*_LAST_5_OBSERVATIONS_BEFORE` counters.
- **190 of 190 `_DIFF_BEFORE` columns equal `HOME - AWAY` exactly.** Each
  triplet is rank 2 stored as rank 3.
- **`TEAM_RECORD_WINS_LAST_N` vs `WIN_RATIO_LAST_N`: r = 1.000000** for
  N = 5/10/20/30, because `GAMES_LAST_N` is constant.
- **Consensus summary stats collapse**: `STD`~`RANGE` r = 0.99;
  `MEAN`~`MEDIAN` r = 0.99; `MAD` is zero in 92.4% of rows; `IQR` in 71.0%.
  Seven stats carry about three.
- **Raw vs normalized total line**: r = 0.996, identical in 92.5% of rows.

### Book coverage is a structural break, not just sparsity

% of games with a closing total quote:

| Season | bet365 | caesars | draftkings | fanduel | betmgm | betrivers | fanatics |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 100.0 | 72.1 | 1.4 | 92.8 | 7.2 | 0.0 | 0.0 |
| 2021 | 99.1 | 99.0 | 96.6 | 97.3 | 0.2 | 0.0 | 0.0 |
| 2023 | 99.8 | 99.9 | 100.0 | 99.9 | 99.9 | 92.9 | 0.0 |
| 2026 | 99.8 | 100.0 | 100.0 | 100.0 | 99.9 | 100.0 | 100.0 |

Only **bet365, caesars, draftkings, fanduel** exist in every season. Per-book
columns for the other three switch on mid-history, exactly across the split a
walk-forward model trains over.

### Implementation bugs (flagged, not part of the feature redesign)

These belong to the later `X/y` filtering step, not to feature generation:

1. `SPREAD_LINE_HOME` is written into `outcome_order`
   ([training_frame.py:367](../src/mlb_pred/create_training_data/training_frame.py#L367))
   but is absent from `OUTCOME_ONLY_COLUMNS`, so `feature_columns()` returns
   it and `assert_no_outcome_features` does not catch it. It is the exact
   quantity `SPREAD_ERROR` is defined against.
2. 21 tz-aware `*_CLOSE_TIMESTAMP_UTC` columns reach the feature allow-list.

Neither is a wrongly-constructed feature; both are allow-list gaps.

---

## 2. Keep / consolidate / remove

### Base metrics that deserve temporal treatment

**Tier 1 - full template** (run-scoring drivers, 8 metrics):
`RUNS_SCORED`, `RUNS_ALLOWED`, `TOTAL_RUNS`, `RUN_MARGIN`,
`OFFENSE_RUNS_PER_PA`, `OFFENSE_OBP`, `OFFENSE_SLG`, `PITCHING_ERA_PER_9`.

**Tier 2 - window 5 + window 10 + season mean** (16 metrics):
`OFFENSE_K_PCT`, `OFFENSE_BB_PCT`, `OFFENSE_HR_PER_PA`, `OFFENSE_ISO`,
`OFFENSE_TOTAL_BASES_PER_PA`, `OFFENSE_BASERUNNERS_PER_PA`,
`PITCHING_K_PCT`, `PITCHING_BB_PCT`, `PITCHING_HR_PER_BF`, `PITCHING_WHIP`,
`PITCHING_BASERUNNERS_PER_BF`, `PITCHING_STRIKE_PCT`, `WIN_RATE`,
`PLATE_APPEARANCES`, `PITCHES_THROWN`, `OFFENSE_BABIP`.

**Tier 3 - season mean only** (context, ~10): `BULLPEN_SIZE`, `BENCH_SIZE`,
`PITCHERS_USED`, `BATTERS_USED`, `FIELDING_ERRORS_PER_OUT`,
`GROUNDED_INTO_DOUBLE_PLAY`, `STOLEN_BASES`, `LEFT_ON_BASE`,
`OFFENSE_GROUND_OUT_RATE`, `PITCHING_GROUND_OUT_RATE`.

**Remove entirely from the rolling engine** (counting stats already implied by
the rates above, and never independently informative): `AT_BATS`,
`OUTS_RECORDED`, `BATTERS_FACED`, `HITS`, `DOUBLES`, `TRIPLES`, `HOME_RUNS`,
`TOTAL_BASES`, `WALKS`, `STRIKEOUTS`, `HIT_BY_PITCH`, `HITS_ALLOWED`,
`HOME_RUNS_ALLOWED`, `WALKS_ALLOWED`, `STRIKEOUTS_THROWN`, `HIT_BATSMEN`,
`EARNED_RUNS`, `WILD_PITCHES`, `BASERUNNERS_ALLOWED`, `FIELDING_ERRORS`,
`OFFENSE_HITS_PER_PA`, `OFFENSE_EXTRA_BASE_HITS_PER_PA`,
`OFFENSE_SB_ATTEMPTS_PER_PA`, `OFFENSE_GROUND_OUT_AIR_OUT_RATIO`,
`PITCHING_PITCHES_PER_BF`, `PITCHING_PITCHES_PER_OUT`.

### Representations to remove

| Remove | Why |
|---|---|
| All 190 `_DIFF_BEFORE` columns | Exactly `HOME - AWAY`; a tree recovers it |
| Windows 1, 2, 3 outside Tier 1 | A 1-game team rate in MLB is opponent noise |
| Windows 20 and 30 | NBA has neither; keep season-to-date instead |
| `TEAM_RECORD_WINS_LAST_N` | r = 1.000000 with `WIN_RATIO_LAST_N` |
| `TEAM_RECORD_GAMES_LAST_N` | Constant |
| `*_OBSERVATIONS_BEFORE` | Constant |
| `*_PRICE_*_NORMALIZED` | Always `-110` |
| Consensus `MEAN`, `RANGE`, `IQR`, `MAD` | Keep `MEDIAN`, `STD`, `BOOK_COUNT` |
| `TEAM_IDENTITY_*` (60) | 60 one-hots over 17.6k rows |
| `TEAM_RATIO_*` prefix | Same metric under a second family prefix; fold into `TEAM_ROLLING_*` |

---

## 3. NBA alignment

Read from `NBA_over_under_predictor/src/nba_ou/data_processing/team/rolling.py`
and its emitted `training_data_2_1_20260828.csv` (1,802 cols, 11,544 rows).

NBA is a **pyramid**: a broad set of metrics gets window 5 + season average,
and only a hand-picked few get the full ladder. `COLS_FOR_SHORT_WINDOWS` has
**4 entries** (`PTS`, `PTS_PER_40`, `DIFF_FROM_LINE_bet365`, the total line).

MLB has the same idea in `EXTENDED_TEAM_LABELS` but the list grew to **24**,
and the non-extended tier is far richer than NBA's.

| | NBA | MLB | 
|---|---:|---:|
| Metrics rolled | ~86 | ~95 |
| Get window 10 | ~35 | 59 |
| Get short windows 1/2/3 | **~14** | **59** |
| Windows 20 / 30 | none | 59 each |
| `TREND_SLOPE` columns | 198 | 708 |
| `MINUS_LAST` columns | 52 | 708 |
| `WMA` columns | 93 | 354 |
| Home-minus-away DIFF columns | **3**, hand-picked | **190**, automatic |
| Total columns / rows | 1,802 / 11,544 | 5,617 / 17,638 |

NBA emits only three explicit home-minus-away diffs
(`PTS_TREND_SLOPE`, `REST_DAYS`, `STAR_PTS_PCT`). MLB emits 190.

### Recommended MLB template

```
Tier 1 (8 metrics):   L1, L3, L5, L10, season avg, season std,
                      WMA-5, home/away split-5, trend slope 5 and 10
Tier 2 (16 metrics):  L5, L10, season avg
Tier 3 (10 metrics):  season avg
All tiers:            emit _TEAM_HOME and _TEAM_AWAY only
Hand-picked DIFF:     ~5 columns (run margin form, rest days, SP quality,
                      bullpen fatigue, season win rate)
```

Roughly `8x10 + 16x3 + 10x1 = 138` metric-variants x 2 roles = **~276**, plus
5 diffs. Against 2,160 today.

---

## 4. New features from existing data

Everything below was **built and tested** during this analysis. Sources are
tables already on disk.

| Family | Source | Status |
|---|---|---|
| Market movement | `data/raw/odds_ticks` | Built, 20 cols, 17,588 games |
| Park / venue | `data/raw/venues` + `games` | Built, 16 cols |
| Umpire (home plate) | `data/raw/umpires` | Built, 4 cols |
| Statcast form | `data/raw/statcast_pitches` | Built, 84 cols |
| Bullpen | `data/raw/pitcher_appearances` | Built, 32 cols |

**None of these five sources is referenced by any module in
`src/mlb_pred/features/` today.** `venues` is used only for travel distance;
`statcast_pitches` and `umpires` are ingested and never read.

- **Market movement.** 100% opener coverage on all three markets, ~1M pregame
  ticks/season, median tick 11.5h before first pitch. Mean absolute open-to-close
  total movement is **0.28 runs**. Must be computed **per book then aggregated
  across the four all-season books** - pooling ticks across books makes tick
  counts track book count, which drifts 4 -> 7 across the history.
- **Park.** `venues` carries `elevation_ft`, `roof_type`, `turf_type` and seven
  outfield distances - all unused. Plus a leakage-safe expanding venue run
  factor.
- **Umpire.** Expanding pre-game mean total runs per home-plate official,
  shifted so the current game never enters its own history.
- **Statcast.** `estimated_woba_using_speedangle`, `launch_speed`,
  `delta_run_exp` -> team xwOBA for/allowed, hard-hit and barrel rates on
  NBA-style L5/L10/season windows.
- **Bullpen.** Relief pitches and outs over the previous 1/3/5 team games,
  relievers used, starter outs last game, plus L10/L30 relief ERA9 and K%.

---

## 5. Proposed compact architecture

| Family | Cols |
|---|---:|
| Market: consensus line, dispersion, book count, 4 stable books | 30 |
| Market movement: open, open-to-close, abs, late-180m, disagreement | 20 |
| Team rolling (tiered template above) | 280 |
| Starting pitcher | 24 |
| Bullpen workload + quality | 32 |
| Statcast form | 48 |
| Availability / injury (compacted from 468) | 60 |
| Park / venue | 16 |
| Umpire | 4 |
| Schedule / rest / travel | 30 |
| Matchup, game context, record | 60 |
| **Total** | **~600** |

From 5,617. Above the 80-dimension effective rank with room for genuine
interactions, and roughly 1 column per 30 training rows.

---

## 6. Incremental validation - results

Ridge, expanding-window walk-forward, test seasons 2022-2026 individually,
pooled out-of-sample R². **The line is in every row.** Each family is also
compared against random-noise columns of the same count, so "worse than
baseline" can be separated from "worse than adding nothing".

### Target `TOTAL_RUNS`, baseline = closing total line (R² = 0.04205)

| line + compact family | cols | R² | delta | noise@same n | verdict |
|---|---:|---:|---:|---:|---|
| schedule / rest | 3 | 0.04186 | -0.00018 | -0.00012 | at noise |
| starting pitcher | 12 | 0.04154 | -0.00051 | -0.00185 | above noise |
| umpire | 3 | 0.04145 | -0.00060 | -0.00012 | at noise |
| bullpen | 10 | 0.04071 | -0.00134 | -0.00158 | above noise |
| park / venue | 5 | 0.04080 | -0.00125 | -0.00105 | at noise |
| market movement | 9 | 0.04054 | -0.00150 | -0.00152 | above noise |
| team run form | 16 | 0.04007 | -0.00198 | -0.00186 | at noise |
| Statcast | 16 | 0.03977 | -0.00228 | -0.00186 | at noise |
| **combined compact** | 74 | 0.03253 | -0.00951 | -0.00748 | |

### Target `RUN_LINE_MARGIN`, baseline = closing handicap (R² = 0.04407)

| line + compact family | cols | R² | delta | noise@same n | verdict |
|---|---:|---:|---:|---:|---|
| **market movement / dispersion** | 8 | **0.04423** | **+0.00016** | -0.00060 | **ADDS** |
| team run form | 16 | 0.04380 | -0.00026 | -0.00192 | above noise |
| bullpen | 10 | 0.04362 | -0.00045 | -0.00057 | above noise |
| park / venue | 5 | 0.04392 | -0.00014 | -0.00030 | above noise |
| umpire | 3 | 0.04401 | -0.00005 | -0.00023 | above noise |
| Statcast | 16 | 0.04212 | -0.00195 | -0.00192 | at noise |
| **combined compact** | 73 | 0.04242 | -0.00164 | -0.00783 | above noise |

### Interactions with the line, and classification

The hypothesis that an 8.5 behaves differently under different conditions was
tested directly, by adding `line x context` product terms:

```
line only                             R2 = 0.04205
line + context (additive)             R2 = 0.03928   (-0.00277)
line + context + line*context         R2 = 0.03454   (-0.00750)
```

Over / Under classification, pushes dropped, n = 16,869:

```
base rate over                = 0.4934  (95% CI 0.4858 - 0.5009)
line only                     logloss 0.69315   acc 0.5012
line + context                logloss 0.69323   acc 0.5068
line + context + interactions logloss 0.69318   acc 0.5021
intercept-only                logloss 0.69302
```

Every variant is *worse* than predicting the base rate.

### What this means, and what it does not

**It is not a power problem.** With n = 16,869 the standard error on the over
rate is 0.0038. A -110 breakeven edge (52.38%) would sit **6.2 SE** from a coin
flip. If any of these families carried a betting edge on totals, this sample
would show it. It does not.

**Above noise is a real signal.** Starting pitcher, bullpen, market movement
and team form all beat random columns of the same count. They carry
information - just less than the variance they cost a linear model. That is a
reason to keep them small and well-shrunk, not a reason to delete them.

**Honest limits of this test.** Ridge is linear; the interaction test covered
only `line x context` products, not the full interaction space a GBDT would
search. The combined compact set on the run line lands far above its noise
control (-0.00164 vs -0.00783), which is what a diluted-but-real signal looks
like. A gradient-boosted model on the ~600-column set is the fair next test,
and Phase 1-3 are what make that test cheap enough to run properly.

---

## 7. Implementation priorities

**Phase 1 - mechanical redundancy and NBA-aligned windows.**
Drop the 243 constants, the 190 DIFF columns, the duplicate consensus stats,
`WINS_*`/`GAMES_*`, and normalized prices. Retier the rolling template.
Restrict per-book columns to the four all-season books. Fold `TEAM_RATIO_*`
into `TEAM_ROLLING_*`. 5,617 -> ~900. Pure deletion; no new code paths.

**Phase 2 - select the meaningful base metrics.**
Apply the Tier 1/2/3 split. Compact `TEAM_AVAILABILITY_*` from 468 to ~60.
~900 -> ~500.

**Phase 3 - build the five unused-source families.**
Market movement, Statcast, bullpen, park, umpire. ~500 -> ~600. Prototypes
from this analysis are in the scratchpad and can seed the modules.

**Phase 4 - validate.**
Re-run the incremental protocol above with a gradient-boosted model on the
~600-column set, per family and combined, walk-forward, with the size-matched
noise controls retained. The controls are what make a negative result
interpretable.

Fix the two allow-list bugs at any point; they are independent of the above.


---

## 8. Implemented outcome

### Size and redundancy, measured on the rebuilt partitions

| | Before | After |
|---|---:|---:|
| Columns (17,638 games) | 5,617 | **1,051** |
| Model features after the allow-list | 5,589 | **1,031** |
| Constant / all-null columns | 243 | **13** |
| `_DIFF_BEFORE` columns | 1,655 | **24** |
| Effective rank | 80.5 | 65.2 |
| PCs for 90% variance | 468 | 246 |
| Pairs at \|r\|>0.99 | 2,173 | 87 |
| Pairs at \|r\|>0.95 | 9,058 | 248 |
| Columns per training row | 0.318 | **0.058** |

Final family sizes: team rolling 299, availability 138, per-book odds 128,
team market history 89, market regime 62, Statcast 60, consensus 39, bullpen
32, matchup 36, schedule 25, game context 24, travel 22, odds interaction 21,
market movement 20, team record 20, park 11, umpire 5.

This lands at 1,051 rather than the ~600 section 5 estimated. The estimate
under-counted two families that were left larger on purpose: the per-book
block (128) keeps the raw executable quotes needed to settle a bet at the
price it could have been struck at, and availability (138) keeps three
distinct absence states that a single count cannot express.

### What was cut, and on what evidence

* 243 -> 13 constants. The `*_PRICE_*_NORMALIZED` columns were `-110` by
  definition of the restatement; `*_LAST_5_OBSERVATIONS_BEFORE` was always 5.
  The 13 that remain (lineup and transaction coverage, `N_LINEUP_STARTERS`)
  are constant only on completed games -- they are the guards that go below 1
  when a lineup has not posted, which is exactly the prediction-time case.
* 1,655 -> 24 diffs. All 190 rolling triplets, and all 104 that survived the
  first pass, were verified to equal `HOME - AWAY` exactly.
* `TEAM_AVAILABILITY_HITTER_MAX_*` was dropped after verifying it is identical
  to `TOP1` across **all 24 column pairs over 17,638 games**.
* `TEAM_RECORD_WINS_LAST_N` dropped (r = 1.000000 with the ratio).
* Consensus statistics cut from seven to three (MEDIAN, STD, BOOK_COUNT).
* Per-book columns confined to the four books quoting in every season.
* The 30-team one-hot block removed. **Reinstated afterwards** as
  `TEAM_IDENTITY_*` in `features/context_features.py`: it was cut here as
  redundant width, never measured to cost anything, and the NBA project
  keeps the equivalent block. Pass `include_team_identity=False` to
  `build_context_features` for a run without it.
* Market regime cut from 204 to 62: the calendar-day windows duplicated the
  game windows, and three EWM spans and three tail thresholds became one each.

### Incremental validation, conditional on the line

Ridge, expanding-window walk-forward, test seasons 2022-2026, size-matched
noise controls. The line is in every row.

`TOTAL_RUNS`, baseline = closing total line, R² = 0.04177:

| line + family | cols | R² | delta | noise@n | verdict |
|---|---:|---:|---:|---:|---|
| umpire (NEW) | 5 | 0.04138 | -0.00039 | -0.00093 | above noise |
| travel | 22 | 0.04120 | -0.00057 | -0.00244 | above noise |
| starting pitcher | 23 | 0.04045 | -0.00133 | -0.00247 | above noise |
| team record | 20 | 0.04005 | -0.00172 | -0.00225 | above noise |
| bullpen (NEW) | 32 | 0.03926 | -0.00251 | -0.00346 | above noise |
| Statcast (NEW) | 60 | 0.03688 | -0.00490 | -0.00562 | above noise |
| market movement (NEW) | 20 | 0.03587 | -0.00590 | -0.00225 | at/below noise |
| team rolling form | 299 | 0.03069 | -0.01108 | -0.01946 | above noise |
| ODDS per-book | 139 | 0.01617 | -0.02560 | -0.01199 | at/below noise |
| ALL FEATURES | 1029 | 0.03159 | -0.01018 | -0.01944 | above noise |

`RUN_LINE_MARGIN`, baseline = closing home handicap, R² = 0.04466:

| line + family | cols | R² | delta | noise@n | verdict |
|---|---:|---:|---:|---:|---|
| **ODDS consensus/dispersion** | 37 | **0.04487** | **+0.00021** | -0.00502 | **ADDS** |
| umpire (NEW) | 5 | 0.04461 | -0.00005 | -0.00030 | above noise |
| schedule/rest | 25 | 0.04427 | -0.00040 | -0.00274 | above noise |
| bullpen (NEW) | 32 | 0.04291 | -0.00175 | -0.00411 | above noise |
| Statcast (NEW) | 60 | 0.04158 | -0.00308 | -0.00735 | above noise |
| team rolling form | 299 | 0.03697 | -0.00769 | -0.02104 | above noise |
| ALL FEATURES | 1029 | 0.04318 | -0.00148 | -0.02098 | above noise |

The picture is unchanged by the consolidation, which is itself the useful
result: **cutting 81% of the columns cost nothing**. Almost every family now
sits above its size-matched noise control -- they carry real information -- and
the full set on the run line is now only -0.00148 against a -0.02098 noise
floor, where before the same comparison was -0.00857.

Two families remain at or below their noise control and are the first
candidates for the next pass: the per-book block on totals, and market
movement. Market movement being below noise while the consensus dispersion
family is the one family that *adds* suggests the useful market-context signal
is cross-book disagreement rather than the time path.

### What Phase 4 still cannot answer

Everything above is linear. The combined set now sits far above its noise
control on both targets, which is the shape of a real-but-diluted signal, and
that is precisely what a gradient-boosted model is for. The consolidation is
what makes that test cheap: 1,031 features over 17,638 rows is a tractable
GBDT problem in a way 5,589 was not.
