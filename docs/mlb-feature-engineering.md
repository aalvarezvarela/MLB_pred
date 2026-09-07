# MLB feature engineering

This layer follows the implementation on the NBA repository's
`dev/fix-training-pipeline` branch, translated to the team-game grain used by
MLB. It contains closing markets, rolling team/market history, contextual
schedule features, game-level matchup combinations, total/spread market
regimes, strictly historical matchup/extra-inning context, and player
availability.

## Column contract

- `GAME_*`: identifiers and schedule fields known before first pitch.
- `ODDS_*`: current closing markets or strictly historical market aggregates.
- `TEAM_ROLLING_*`: strictly historical team form.
- `TEAM_RATIO_*`: ratios of historical estimates.
- `TEAM_IDENTITY_*`: stable home/away team one-hot indicators.
- `TEAM_RECORD_*`: prior wins, win ratios, and consecutive-win streaks.
- `TEAM_AVAILABILITY_*`: final-lineup availability, observed absences,
  transaction-confirmed IL absences, starting-pitcher quality, and historical
  player effects.
- `SCHEDULE_*`: series position, rest, game density, and day-after-night flags.
- `TRAVEL_*`: series travel, recent distance load, and timezone disruption.
- `_TEAM_HOME` / `_TEAM_AWAY`: the side after the final team-game pivot.
- `_DIFF_BEFORE`: home minus away.
- `_BEFORE`: computable before the target game's first pitch. For historical
  aggregates this also means the target game, and every game on the same MLB
  calendar date, is excluded.

The final builder raises if an unlabelled column survives. Raw scores, results,
cover outcomes and source box-score fields remain inside the rolling context
and are never written to the feature partitions.

It also raises when any engineered `TEAM_*`, `SCHEDULE_*`, `TRAVEL_*`,
`GAME_*`, or `ODDS_*` column lacks `_BEFORE`. The only exceptions are explicit
`GAME_*` metadata and current closing-market columns supplied by the audited
closing feature frame. Outcome and target columns are not exceptions and
belong in a separate scoring sidecar.

## Rolling family

The NBA windows are preserved: prior 1/2/3/5/10 games, WMA over five games,
last-five minus last-ten, season-to-date mean/std, home/away minus overall, and
trend slopes over 5/10 games. Baseball also receives 20/30-game means and
last-five minus those longer estimates.

Rates cover offensive volume/efficiency, pitching volume/efficiency and
fielding. Counts are retained where they describe game volume, while additional
rate sources are derived before rolling (for example hits per PA, baserunners
per BF, pitches per out and ERA per nine).

Every rolling operation sees prior valid observations only. A scheduled row
has null source statistics, is allowed to receive history, and never consumes a
rolling-window slot. Season estimates fall back to the same home/away quantity
from the previous **regular** season and then to a documented neutral zero.

MLB doubleheaders receive an extra date gate. Both games use only the history
available before that calendar date because the local store does not carry a
trustworthy result-publication timestamp for game one. This is conservative and
prevents game two from learning a result that may not be known at feature time.

## Market history

Historical total lines use the equal-pay normalized total. Run lines are
oriented from each team's side before rolling, and moneyline history uses fair
devigged win probability. Prior-game total error and run-line cover margin are
legal history sources only after the same shift and date gate; their unshifted
values never leave the builder.

Build from local Parquet:

```bash
poetry run python scripts/create_features/create_closing_lines.py
poetry run python scripts/create_features/create_pregame_features.py
```

Passing `--seasons` limits written rows, not historical context. This is
required so season openers and historical odds windows remain warm.

## Identity, record, rest, and travel

Team identity is emitted as 60 stable binary columns: one 30-team block for the
home side and another for the away side. Unknown ids raise rather than silently
producing an all-zero row. Raw `GAME_HOME_TEAM_ID` / `GAME_AWAY_TEAM_ID` remain
metadata and should be excluded from the numeric training matrix.

Win form includes season wins/games/win ratio, a season-history flag, wins and
win ratios over the last 5/10/20/30 completed games, and the current consecutive
win streak. Results are applied only after every game on a calendar date has
received its features, so neither game of a doubleheader can read the other's
outcome. A season-opening win ratio falls back to the previous regular season,
then to the neutral value `0.5`.

Schedule features include actual off-days, the calendar gap, consecutive game
days, games played over the prior 3/7/14 days, series position, series
opener/getaway-day flags, no-rest flags, and day-game-after-night-game flags.
The first game of a season uses the NBA-compatible seven-day rest default.

Travel is computed at the MLB series grain. The trip from the previous series is
attached to every game in the current series, while cumulative 1/2/5/7/14-day
distance counts the transition only once. Distances use season-specific venue
coordinates and `log1p`; timezone changes use IANA zones on the game date and
are neutralized after four days of adaptation. Missing coordinate/timezone
flags distinguish a genuine zero trip from missing geography.

## Advanced game-level families

All columns in these families contain `_BEFORE`, in addition to a selectable
family label:

- `TEAM_MATCHUP_*`: home offence crossed with away pitching and vice versa.
  It includes recent/season run expectations, form z-scores, expected plate
  appearances, and expected K, BB, HR and baserunner rates/event counts.
- `ODDS_MARKET_REGIME_TOTAL_*` and `ODDS_MARKET_REGIME_SPREAD_*`: league-wide
  bias, MAE, error volatility, direction/push rates, line and realised levels,
  tail misses, EWMs and short-versus-long regime changes. All date windows use
  strictly earlier dates.
- `ODDS_INTERACTION_*`: total/spread probability skew, line disagreement times
  overround, total times favourite magnitude, moneyline favourite strength,
  and implied-team-run ratios/gaps.
- `TEAM_HISTORICAL_MATCHUP_*`: prior meetings for the unordered team pair,
  including totals over 3/5/10 meetings, current-home-oriented margins and a
  history/missingness indicator.
- `GAME_CALENDAR_*`, `GAME_COMPETITION_*`, and `TEAM_COMPETITION_*`: month,
  weekend, federal holiday, league/division/interleague flags, postseason round
  and prior-season postseason games.
- `TEAM_EXTRA_INNINGS_*`: last-game and 5/10/season extra-inning frequency by
  side plus home-minus-away differences.

The historical matchup, extra-innings and global market builders apply each
date's completed results only after every row on that date has received its
features. This is the same conservative doubleheader rule used by rolling team
form and records.

## Player availability and the final-lineup proxy

Historical availability uses the final nine-player lineup stored by the MLB
Stats API as a proxy for the lineup known before first pitch. The repository
does not retain the historical publication time of that lineup. This proxy can
therefore make a backtest optimistic and must not be treated as equivalent to
the timestamped snapshots used for live prediction. A production evaluation
must compare this family with the stricter transaction-only variant; an uplift
that exists only with final lineups is not deployable without an archived,
timestamped lineup source.

The three hitter states have deliberately different meanings:

- `available` means the player appears in the final starting lineup;
- `observed_absent` means a high-role candidate neither started nor appeared
  in the game, but no known IL state explains the absence; and
- `injured` means the same kind of relevant absence is supported by an open IL
  transaction state.

An observed absence is not an injury label. It may reflect rest, a coaching
decision, a minor-league option, or another cause. IL state is reconstructed in
`known_date` and transaction-id order: placement opens it, transfer maintains
it, and activation or reinstatement closes it. The direction comes from
`type_desc` and `description`, because activation text also contains the words
"injured list". As specified for this retrospective family, a transaction is
admitted when `known_date <= game_date`; the missing intraday timestamp remains
a documented limitation.

Hitter quality uses a ten-appearance half-life EWMA for PA, OBP, SLG, HR/PA,
BB%, K%, total bases/PA, and baserunners/PA. Starting pitchers use a separate
EWMA path for outs, batters faced, ERA/9, WHIP, K%, BB%, HR/BF, and pitches/BF.
Both paths fall back from current-season history to the immediately preceding
regular season and then to fixed neutral values. Hitter neutrals are 0 PA,
.320 OBP, .400 SLG, .030 HR/PA, .080 BB%, .220 K%, .360 total bases/PA, and
.320 baserunners/PA. Pitcher neutrals are 0 outs/BF, 4.50 ERA/9, 1.30 WHIP,
.220 K%, .080 BB%, .030 HR/BF, and 3.90 pitches/BF. The target game and all
games on its date are excluded.

All eight hitter metrics reach the team output, in two tiers. PA, OBP, SLG and
HR/PA are primary: they carry totals, means, maxima and per-rank top-N columns,
because a single star dominates them. BB%, K%, total bases/PA and baserunners/PA
are secondary and carry only means and maxima — a sum over rate statistics has
no interpretation, and a per-rank ordering of them is not a quality ordering.

Expected hitters are the top nine recent team candidates, ordered primarily by
PA EWMA and then by starting-lineup share over the prior 20 team games. Player
quality follows the player across a team change, while membership follows the
latest prior appearance and known transactions. Pitchers on the IL contribute
a depth count, but a reliever who simply does not pitch is never classified as
absent.

No starter-scratch flag is emitted. The only leakage-safe evidence would be a
change between an archived pre-game probable and the announced starter, and the
snapshot archive does not reach back over the backfill; comparing the schedule's
probable against who actually pitched is post-game information. The column is
therefore absent rather than present and constantly zero.

Playing roles are accumulated as independent flags — has batted, has pitched,
has started on the mound — never as one exclusive label. A two-way player is
genuinely both, and an exclusive label forced him to flip: he became a pitcher
on the date he started on the mound and only reverted on his next batting date,
which erased the best hitter on the roster from the game in between. The lineup
slot marked `P` is the pitcher taking his own turn at bat, which is the
2015-2021 National League default and not a hitter, so `N_AVAILABLE` is
legitimately eight rather than nine for those games. An established batting role,
evidenced only from strictly earlier dates, overrides that slot, which is how a
two-way player listed at `P` while batting stays a hitter.

Absence is inferred by differencing the expected hitters against the starters,
so it requires a lineup to exist. When one does not, no absence is emitted at
all and `LINEUP_COVERAGE` carries the fact; differencing against an empty
starter set would otherwise report every expected hitter as absent, which reads
as a squad-wide injury crisis rather than as missing data.

Empirical hitter effects compare earlier team/game outcomes when each relevant
player was available versus absent. They use only the current and previous
season and shrink by `n_eff / (n_eff + 10)`, where `n_eff` is the smaller of
the available and absent samples. Effect values are zero when either side has
no evidence, while sample-size columns retain that distinction.

## Training-data boundary

Pregame partitions remain physically free of final scores. The historical
training builder in `mlb_pred.create_training_data.training_frame` is the one
explicit boundary that joins those partitions to the `games` scoring sidecar.
It recomputes total runs and home margin from the two team scores, checks them
against the stored game outcomes, and emits `TOTAL_RUNS`, `RUN_LINE_MARGIN`,
the NBA-compatible alias `HOME_MARGIN`, `LINE_ERROR`, and `SPREAD_ERROR`.

`LINE_ERROR` is actual total runs minus the normalized consensus total.
`SPREAD_LINE_HOME` is the market-implied home margin, obtained by negating the
normalized consensus home handicap, and `SPREAD_ERROR` is actual home margin
minus that implied margin. Positive spread error means the home team covered;
zero is a push and remains a valid regression target. Rows without one market
retain a null residual for that market so they remain usable by the other
targets.

Both residuals default to the normalized close, and `--raw-lines` measures them
against the raw executable close instead. This is where the NBA script's
`--no-normalize-total-lines` and `--no-normalize-spread-lines` switches land in
this repo: MLB normalizes upstream in `market_normalization.py` and carries both
shapes into the pregame partition, so the choice here is which stored quote the
target is measured against rather than whether to normalize at all.

The output also retains final home/away runs and the extra-innings indicator for
settlement and filtering. These, and every derived target, are listed in
`OUTCOME_ONLY_COLUMNS`; `feature_columns()` excludes them and raw game metadata
from the model allow-list.
