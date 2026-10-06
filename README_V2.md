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
