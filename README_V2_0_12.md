# NFL Predictor V2.0.12 — RB Opportunity & Backfield Portfolio Controls

Adds two transparent controls without hard-coding any player or team:

1. **RB Opportunity Overrides** on Slate Setup. Verified role/news changes can be represented by a bounded 0.75–1.35 multiplier. The multiplier applies to projection and tournament tails without compounding, is shown in the RB audit, and invalidates stale simulations/portfolios when changed.
2. **Same-team RB portfolio guardrails** in the V2 scenario selector. Default combined same-team RB selections are capped at 55% of lineup count, and at most one lineup may roster two RBs from the same team.
3. **RB Opportunity Audit** in Lineup Builder shows projection, ceiling, P95, team implied points, DK PPG, opportunity multiplier/note, projection source and value.

No player is forced into a lineup. This allows a clean A/B test: baseline portfolio vs verified opportunity-adjusted portfolio.
