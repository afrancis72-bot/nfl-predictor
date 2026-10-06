# NFL Predictor Pro — DFS Engine V2

Built from the V3.1.7 Classic + V1.8 Showdown codebase.

## V2 architecture
- Full-slate correlated NFL scenario simulation for Classic.
- Showdown correlated game scripts with explicit slow/neutral/shootout/tail regimes.
- Heavy-tailed player variance plus separate model-uncertainty shifts.
- Scenario-based candidate scoring (mean, median, P90, P95, optimal rate).
- Portfolio selection rewards marginal scenario coverage and penalizes redundant overlap.
- Rare legal constructions are no longer universally banned just because they are unusual.
- Showdown projection-integrity gate makes weak weekly-model coverage explicit.
- Simulation Validation tab reports mean, median, binned mode, P10/P25/P75/P90/P95 and supports post-event Actual + Actual Percentile grading.

## Important data principle
The app does not solve stale weekly research by pretending DraftKings PPG is a true weekly projection. Showdown now raises a visible integrity warning when weekly-model coverage is weak. Refresh the weekly projection inputs before trusting automated entries.

## Deployment
Replace the existing app.py with this V2 app.py in the same repository that already contains the `data/` directory used by V3.1.7. The ZIP intentionally does not duplicate weekly data files.
