# NFL Predictor Pro — DFS Engine V2.0.2

## Current-slate lock fix

V2.0.2 removes the misleading **Use built-in weekly model slate** action. The bundled Week 4 research files are historical/demo inputs, not a live current-week feed.

Weekly workflow:
1. Upload the current DraftKings Classic salary CSV.
2. That upload becomes the locked active slate and cannot silently revert to bundled prior-week data.
3. Verify/enter game totals for every active game and click **Activate current game environment**.
4. Generate simulations/portfolios only after the current-slate environment gate passes.

The only reset action is now **Clear uploaded slate**, explicitly labeled as returning to historical/demo data. Clearing also marks the environment unverified so historical game totals cannot be mistaken for current verified inputs.
