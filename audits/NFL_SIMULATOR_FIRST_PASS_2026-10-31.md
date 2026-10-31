# NFL simulator first-pass inspection — 2026-10-31
## Scope
This inspection concerns ONLY the actual single Team AI² Pro Streamlit application launched from `unified_app.py` on branch `feature/unified-ai2-platform`, via the NFL section and `ai2_nfl_oneclick.py`. No runtime changes are made by this note.

## Verified current behavior (code inspected)
- Salary file + seed + count (15/20/50/150) through unified UI; projected-player stat inputs are automatically sourced.
- `props_data.make_rates`: past weeks only, most recent eight appearances, exp decay 0.87 per game; includes prior season as available. Recent in-season sample checks reject certain stale roles.
- `ai2_nfl_auto_intelligence`: nflverse weekly box statistics and official NFL injury page; opposing defensive allowances are researched but do NOT drive scored simulation.
- `ai2_nfl_model_pool`: filters OUT/IR/DOUBTFUL; also excludes all QUESTIONABLE players pending review (not proof of nonparticipation). RB >= 4500, WR >= 4000; 6+ RB carry+target, 3.5+ WR targets, 2.3+ TE targets based on observed/adjusted averages.
- `dfs_stats_engine`: scenario-shared plays, pass/rush allocation, targets/carries from team totals; touchdown capacity, QB-DST event relationship, OUT-player redistribution; generic priors include 32 passing attempts, 65 plays and capped scoring tail, not calibrated to week-specific team conditions. 10,000 game-level simulations use seeded draws. Some components have conservative bounds.
- `ai2_nfl_candidate_builder`: generate salary-legal lineup candidates via random sampling plus iterative swaps, retaining QB diversity. `ai2_nfl_sim_portfolio`: rank against common simulated worlds; 35% exposure, mostly mean-focused plus 25% P99-focused, season/week-specific special premium exclusion in Week 8, preserve approval. P90/P99 are lineup outcomes, not contest cash probability.
- Source/UI explicitly acknowledges no verified late breaking news, no full depth chart, ownership, reliable corner coverage matchups, independent DST model, or calibrated win/ROI model. Major already-known limitations documented by current code.

## Primary gaps to test BEFORE production projection change
1. *Current weekly role* — Last-eight-game averages, sample cutoffs and heuristics are not equivalent to recent snaps, routes, healthy starter/backup context, changing scheme or actual target share. Backtest low-usage/later-season roles and verify each opportunity flag source.
2. *Game and matchup conditioning* — Team passes and plays use a generic 32/65 prior; no independently validated team-specific projected snaps/plays, market total and spread, defensive-adjusted passing/rushing efficiency, or weather effects in scoring. Evaluate data-feed freshness/coverage before fit.
3. *Availability* — Blanket Q exclusion avoids risk but loses viable active high-opportunity players. Require official confirmation plus controlled eligibility treatment; distinguish expected workload from mere active designation.
4. *Defenses* — Simulated turnovers/points provide useful correlation, but the defense team strength prior is neutral or weak, and the opponent report is display-only. Verify DST versus actual points, sacks, turnover rates before attaching multipliers.
5. *Uncertainty and probabilities* — Percentile generation is mechanically sound but not validated against held-out outcomes. Walk-forward audit by player/position and game for mean errors, coverage of P10/P50/P90/P99, low-volume false positives, plus lineup outcomes and portfolio return in actual contest fields.
6. *Ownership and concentration* — No validated ownership, field duplication or payout-distribution objective. 35% exposure protects diversification; it does not establish actual GPP ROI.
7. *Classic and Showdown* — Current unified entry point routes NFL Classic one-click. Audit Showdown separately and DO NOT import Classic-specific QB-vs-opponent-DST rule into Showdown.

## Safe sequencing
A. Establish versioned pre-lock NFL evidence snapshots and independent historical score data. Do not contaminate test predictions with post-lock data.
B. Produce read-only walk-forward calibration per player, position, matchup, and slate comparing current v3.11 versus naive historical baseline.
C. Validate availability/opportunity and team-pace components individually by ablation, not blanket arbitrary bonuses.
D. Add review screen: player mean/P90, projected touches/targets, recent role trend, opponent/condition evidence, ownership if available, risks and human veto, documented data freshness; old optimizer preserved.
E. Compare side-by-side pre/post versions on historical slates and real contest outcomes; promote changes only after measurable improvements.
F. Later inspect NFL Showdown separately, including captain viability.

## Guardrails
One canonical app, no parallel sports app, no reboot until cloud deploy confirmed. For Sunday 2026-11-01 use existing approved lineups and official inactives; no untested model coefficient changes before contest lock. No claims of predictive calibration or profit until observed.
