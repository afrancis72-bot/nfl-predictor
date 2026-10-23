# NFL engine integration audit — 2026-10-22

Status: source inspection only; not a runtime, simulation-calibration, or deployment test.

## Confirmed reusable components

- `app.py` imports `dfs_stats_engine.simulate_stat_driven_dfs` and SciPy MILP optimization, and parses current DraftKings salary files. It also contains Week 4 hardcoded model data paths. Do not reuse its default slate or copy its UI into the unified app.
- `dfs_stats_engine.py` generates football-stat outcomes first, then fantasy points, and models shared game scenarios and opposing DST. Its module docstring explicitly says **game-level, not play-by-play**.
- `props_lab.py` has an independent observed-stat-input team-volume Monte Carlo, with target and carry allocation; this is not the complete DFS scoring engine.
- `props_matchup.py` has pregame opponent adjustments with evidence thresholds and shrinkage; preserve provenance and fail-closed behavior.
- `ai2_nfl_inputs.py`, `ai2_nfl_eligibility.py`, `ai2_nfl_lineup.py`, and `ai2_nfl_export.py` provide slate ingestion, eligibility, legal roster validation, and review CSV.

## Critical integration blockers

1. **Independence requirement:** `dfs_stats_engine._team_scoring_budget` explicitly weights sportsbook-implied points at 70%. This violates the agreed requirement to freeze independently simulated football statistics before comparing against markets. Replace this with a historical, non-market team scoring baseline before using that engine for official TEAM AI² projections. No market-implied totals, spreads, or prop prices may enter the simulation.
2. **Current slate:** legacy `app.py` includes Week 4 data paths; replace hardcoded week data with the verified current-week slate, schedule, injuries, rates, and defensive inputs.
3. **Player identity:** verify exact current-team matches and report unmatched players; role proxies are not independently observed player rates and must be visibly marked or excluded under the agreed opportunity policy.
4. **Calibration:** correlated game-level Monte Carlo is not snap-by-snap simulation and has not been validated against the current contest's outcomes or ownership. Don't label it validated or predict contest profitability.
5. **Optimization:** existing MILP logic should be extracted behind a data-only interface and tested against the 20-lineup portfolio rules, 35% exposure, no same-team RBs, no QB vs opposing DST, and DK export template.
6. **Cloud persistence:** SQLite on Streamlit Community Cloud's ephemeral local filesystem is not durable across redeployments; do not promise uploaded slates will survive rebuilds without external persistent storage.
7. **Navigation:** legacy multipage links visible in Streamlit's sidebar are not integrated tabs. Audit their scripts and remove from the preview navigation only after their functionality is accounted for.

## Safe sequence

1. Write independent team scoring inputs and tests that prove no market fields are consumed.
2. Add adapter from verified weekly salary/rate inputs into the existing game-level engine; fail closed on missing required inputs.
3. Run historical holdout tests and inspect full per-player football stat distributions, not just mean DK scores.
4. Wire simulated fantasy distributions to portfolio optimizer, test 20 legal lineups, and validate CSV against an actual contest-entry template.
5. Configure genuinely durable storage before asking the user to upload data to the hosted preview.

No real-money release approval is implied by this audit.
