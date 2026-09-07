# Implementation plan: player injuries and availability

This feature family is directly inspired by the NBA implementation, adapted to
MLB-specific constraints. Its purpose is to represent both the best players
who participate and the best players who are unavailable, without
automatically equating a routine rest day with an injury.

## Scope decision

Three distinct states will be used:

- `injured`: a player on an injury absence or injured list (IL) identified from
  transactions.
- `observed_absent`: a relevant player who was not in the starting lineup and
  did not make a game appearance. This can reflect an injury, rest, a coaching
  decision, a minor-league option, or another reason.
- `available`: a player in the game's starting lineup.

For historical data, the final lineup is deliberately accepted as a proxy for
information available before the game, even though the repository does not
retain its exact publication time. This can make the backtest somewhat
optimistic. Documentation and source names must make that limitation explicit.

## 1. Create the availability module

Create `src/mlb_pred/features/availability_features.py`.

The module will receive `games`, `lineups`, `batter_games`,
`pitcher_appearances`, `transactions`, and closing lines. It will return one
row per `GAME_ID`, ready to merge with the existing feature set.

## 2. Build player history before each game

For hitters, calculate an EWMA with a 10-appearance half-life, falling back to
the prior regular season, for:

- PA per game as an importance and usage measure;
- OBP, SLG, HR/PA, BB%, K%, total bases/PA, and baserunners/PA.

For pitchers, retain a separate path with outs, BF, ERA/9, WHIP, K%, BB%,
HR/BF, and pitches/BF.

Every player statistic will exclude the target game and every game on the same
date, including doubleheaders. The fallback will follow the existing feature
contract:

```text
current season to date
  -> prior regular season
  -> documented neutral value
```

Player values may retain their history through a team change, but membership on
a team for a game will be resolved from the player's latest prior appearance
and available transactions.

## 3. Reconstruct hitter availability

- `lineups` provides the historical nine starting players per team and game.
- `expected` candidates will be hitters with the largest prior role, primarily
  based on PA EWMA and recent starting-lineup share.
- A relevant candidate missing from the lineup will be marked
  `observed_absent`.
- If a transaction also places that player on the IL on that date, the player
  will be marked `injured`.

Transactions will be treated as a state machine:

- an IL placement or transfer opens or maintains the absence;
- an IL activation closes it;
- records will be processed by player, team, `known_date`, and transaction id;
- the feature uses `known_date <= game_date`, accepting that intraday timing is
  unavailable.

`is_injury_related` alone is insufficient: an activation from the IL can also
contain the text "injured list". The classifier must interpret the direction of
the transaction using `type_desc` and `description`.

## 4. Emit quality and depth features

The output will follow the `TEAM_AVAILABILITY_*_BEFORE` convention, with home,
away, and home-minus-away views where appropriate.

Initial feature families:

- count of starters, observed absences, and confirmed injured players;
- sum, mean, and maximum of PA, OBP, SLG, and HR/PA among starters;
- sum, mean, and maximum of the same measures among observed absences and
  confirmed injured players;
- top 1--5 starters and top 1--4 absences by each metric;
- absence streak, measured in team games, for relevant players;
- coverage flags and counts that distinguish zero absences from insufficient
  coverage.

Example columns:

```text
TEAM_AVAILABILITY_HITTER_N_AVAILABLE_BEFORE_TEAM_HOME
TEAM_AVAILABILITY_HITTER_N_OBSERVED_ABSENT_BEFORE_TEAM_HOME
TEAM_AVAILABILITY_HITTER_N_INJURED_BEFORE_TEAM_HOME
TEAM_AVAILABILITY_HITTER_TOP1_AVAILABLE_OBP_BEFORE_TEAM_AWAY
TEAM_AVAILABILITY_HITTER_TOTAL_INJURED_PA_BEFORE_DIFF_BEFORE
```

Player ids and names will exist only during feature calculation. They will not
be written to feature partitions or passed into the model.

## 5. Empirical availability effects

For each relevant player, calculate the following over strictly earlier games
from the current and previous season:

```text
effect = mean(outcome | player available)
         - mean(outcome | player absent)

n_eff = min(n_available, n_absent)
shrunk_effect = effect * n_eff / (n_eff + 10)
```

Outcomes will be:

- runs scored by the player's team;
- total game runs;
- total-line error, where market data is available (2019+).

For top available and absent players, aggregate effects as `MEAN`, `MAX_ABS`,
and sample size. Where no evidence exists, the effect is filled with zero while
sample sizes are retained to distinguish "no effect" from "no data".

## 6. Treat pitchers separately

A pitcher will not be marked absent or injured merely for not pitching on a
given day: that is normal for both rotations and bullpens.

- The starting pitcher will be its own feature family, with historical quality
  and a change or scratch flag when supported by a reliable source.
- IL absences for pitchers will enter depth counts, but not an aggregate of
  "best pitchers who did not play today".
- Bullpen workload will be implemented later as a rolling state family based on
  recent appearances, rather than as a binary injury feature.

## 7. Integrate with the pipeline

Update `scripts/create_features/create_pregame_features.py` to:

1. load the required tables;
2. build the availability feature block;
3. merge it with context before calling `build_pregame_features`;
4. preserve validation that rejects columns without a family prefix or required
   temporal suffix.

Update `docs/mlb-feature-engineering.md` to document the final-lineup proxy and
the distinction between confirmed injury and observed absence.

## 8. Required tests

- EWMA excludes the target game and the first game of a doubleheader.
- Prior regular-season fallback.
- Team changes and players with few appearances.
- IL placement, transfer, and activation.
- A regular starter resting: `observed_absent`, not `injured`.
- A relief pitcher who does not pitch: never marked absent or injured for that
  reason.
- No player id, name, or raw outcome reaches the feature output.
- Full integration and stable schema across seasons.

## 9. Validation and rollout

Run a walk-forward evaluation beginning in 2019 with:

1. the current baseline;
2. baseline plus observed availability.

To quantify proxy optimism, also compare against a stricter variant that uses
only IL/transactions and available historical snapshots. If improvement only
appears with the final lineup, this family cannot be deployed to production
without an archived, timestamped lineup source.

## Existing data

The MVP requires no new external source:

- `lineups` contains starters for each game;
- `batter_games` and `pitcher_appearances` provide historical player statistics;
- `transactions` contains IL placements, transfers, and activations;
- closing lines allow effects against market error from 2019 onward.
