# NFL DFS Engine V2.0.4.2 — Active Simulation Slate Fix

- Simulation tab now uses only the active uploaded DraftKings slate.
- Bundled historical Week 4 Monte Carlo rows are blocked from the current-slate Simulation view.
- Simulation requires both verified current game environment and current-week projection integrity.
- Current slate game list is displayed before simulation.
- Supports 10k / 25k / 50k / 100k coherent full-slate scenarios (50k default).
- Simulation results show Mean, Median, binned Mode, P10/P25/P75/P90/P95 and current player/game metadata.
- Cached simulation output is slate-signed; results from another slate are blocked automatically.
- Current-slate simulation baseline is stored for the Simulation Validation tab and later actual-results grading.

## V2.0.5 — Viable Simulation Pool
- Separates the complete DraftKings salary pool from the simulation/optimizer pool.
- Emergency/depth players no longer force projection-integrity failures merely because DK priced them.
- Viability is determined conservatively from current-slate team/position salary rank plus demonstrated DK production; no fake projection is created to satisfy the gate.
- The projection-integrity gate now evaluates only simulation-viable, non-blocked players.
- `simulation_viable` and transparent role labels remain available for audit.


## V2.0.6 distribution audit repair
- Replaces additive-noise-plus-zero-clipping for RB/WR/TE with a smooth positive right-skewed distribution.
- Allows DST simulations to produce negative DraftKings scores and QB rare negative outcomes.
- Adds simulation-derived portfolio relevance: skill players need Mean >= 4 or P90 >= 10; QBs need Mean >= 10. This preserves real tail punts while excluding buried depth players from portfolio generation.
- Keeps the broad simulation pool for calibration while narrowing lineup eligibility downstream.

## V2.0.7 — bounded tail generator
- Repairs low-mean/high-variance lognormal numerical explosions found in the V2.0.6 full-slate audit.
- Bounds log-space dispersion and latent shocks for RB/WR/TE while preserving right-skewed distributions.
- Adds adaptive position-aware simulation guardrails; these scale with projection/variance rather than imposing one universal fantasy ceiling.
- Adds fail-closed numerical sanity assertions before optimizer use.

## V2.0.8 — Scenario Portfolio Selection
- Freezes the V2.0.7 projection and bounded-tail simulation layers.
- Scores Classic candidates as complete lineups against the same coherent full-slate simulations.
- Adds exact-optimal and within-5-DK-points near-optimal scenario rates.
- Portfolio selection rewards NEW near-optimal scenario coverage from each additional lineup.
- Adds a smooth concentration cost for repeatedly using the same player core before the safety exposure ceiling is reached.
- Keeps exposure as a broad safety ceiling; no player-specific exposure caps were introduced.
- Keeps legality, salary, minimum-unique and projection-quality constraints unchanged.

## V2.0.9 — Portfolio Attribution Diagnostics
- Diagnostic-only update; V2.0.8 projection, simulation, candidate scoring, and selection logic are unchanged.
- Adds lineup Portfolio Win %, Within-5-of-bank-best %, unique marginal scenario coverage, cumulative coverage, winner score, and regret.
- Adds player exposure versus presence in portfolio-winning scenarios and the exposure-minus-winning-presence gap.
- Adds team presence attribution for game-environment concentration review.
- Adds downloadable lineup, player, and team attribution CSVs.
