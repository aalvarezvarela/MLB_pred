# MLB feature engineering

This layer follows the implementation on the NBA repository's
`dev/fix-training-pipeline` branch, translated to the team-game grain used by
MLB. It contains closing markets, rolling team/market history, contextual
schedule features, game-level matchup combinations, total/spread market
regimes, and strictly historical matchup/extra-inning context. Injuries are
deliberately excluded from this iteration.

## Column contract

- `GAME_*`: identifiers and schedule fields known before first pitch.
- `ODDS_*`: current closing markets or strictly historical market aggregates.
- `TEAM_ROLLING_*`: strictly historical team form.
- `TEAM_RATIO_*`: ratios of historical estimates.
- `TEAM_IDENTITY_*`: stable home/away team one-hot indicators.
- `TEAM_RECORD_*`: prior wins, win ratios, and consecutive-win streaks.
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
