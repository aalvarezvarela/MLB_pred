# Skills

These three skills were distilled from the NBA over/under system
(`/home/adrian_alvarez/Projects/NBA_over_under_predictor/`) and copied here
unchanged, as that repo's own skills README instructs. They are sport-agnostic
decision records with the NBA implementation as evidence and an explicit MLB
translation in each section:

- `sports-data-architecture` — entities, identity, availability, context data
- `odds-data-architecture` — line ingestion, tick storage, snapshots
- `feature-engineering` — temporal rules and the eleven feature families

Do not edit them to describe MLB. They are the *reference*; what this repo
actually built, and where it deliberately diverged, is recorded in
[`docs/mlb-sports-data-architecture.md`](../../docs/mlb-sports-data-architecture.md).

The NBA repo's fourth skill, `experiments`, is repo-specific and should be
ported once this repo has a training pipeline.
