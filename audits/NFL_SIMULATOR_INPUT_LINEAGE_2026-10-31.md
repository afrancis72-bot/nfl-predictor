# NFL Classic: input-to-prediction lineage audit (read only, 2026-10-31)

**Canonical application:** `unified_app.py`, NFL section, `ai2_nfl_oneclick.build_oneclick_portfolio` on `feature/unified-ai2-platform`.
**Scope:** What affects *simulated DK points* versus what affects only eligibility/selection/display.
**Important:** Source-code route verification is not evidence of forecast calibration; no model coefficient or approved lineup changes in this audit.

| Input or intelligence | Acquired / computed | Used to score simulation? | Source / evidence |
| --- | --- | --- | --- |
| DK salary, position, team, game, ID | CSV | NO (used in roster/pool/optimizer) | `ai2_nfl_model_pool`, `ai2_nfl_candidate_builder` |
| NFL player weekly historical pass attempts, completions, pass yards/TD/INT and sacks suffered | nflverse, last up to 8, .87 decay | YES (QB and team pass volume / sacks / TD model) | `props_data.make_rates`, `dfs_stats_engine` |
| Historical carries, rush yards/TD, targets, catches, receiving yards/TD | nflverse | YES (simulated shares, efficiency, TD scoring) | `dfs_stats_engine` |
| Current injury official OUT/IR/etc | Official NFL injury report and DK | YES indirect (exclude; redistribute absent role's historical volume) | `ai2_nfl_auto_intelligence`, `ai2_nfl_model_pool`, `dfs_stats_engine` |
| QUESTIONABLE tag | Official NFL injury report and DK | NO numerical workload adjustment; all Q excluded from pool | `ai2_nfl_model_pool` |
| Opponent positional fantasy points allowed | Historical `opponent_team` matchups | **NO** — only presented in research; never passed to scored simulator | `ai2_nfl_matchup_diagnostics.defensive_allowance`, `ai2_nfl_opponent_intelligence`, call chain in `ai2_nfl_oneclick` |
| Opposing defensive injury status, front/coverage role | Official NFL practice/injury report | **NO** — display-only; fixed `Applied simulation boost: 1.0` | `ai2_nfl_opponent_intelligence.make_opponent_report` |
| Offensive practice participation & injury descriptions | NFL injury page | Only status gate; NO usage/minutes/snap adjustment | `ai2_nfl_oneclick` pregame report |
| Market spread/total or implied points | Not collected in NFL Classic pipeline | NO | `ai2_nfl_oneclick` input and `dfs_stats_engine` |
| Current weather/wind | Not collected in NFL Classic pipeline | NO | same |
| Current snap counts, routes run, teammate lineup/depth chart | Not in complete collected model, historical boxscore proxies only | NO directly | `props_data.COUNT_COLS`, `ai2_nfl_model_pool` |
| Ownership/duplicated GPP lineup probability | Not validated/connected | NO | `ai2_nfl_sim_portfolio` |
| Team season pace, independent pace source | Not available as an independent team-game input | NO — generic 65-plays/32-pass-attempt priors blended with historical individual usage, scenario randomness | `dfs_stats_engine` |
| DST team strength | Placeholder `proj=6.5` for DST (neutral), while QB sack tendency and game-scoring events change DST results | PARTLY, but no calibrated opponent-specific team defense | `ai2_nfl_model_pool`, `dfs_stats_engine` |
| Simulated DK mean/P90/P99 (same-game correlated football events) | Engine computed | YES for portfolio selection, not independent source | `ai2_nfl_sim_portfolio` |

## High-priority semantic issue
`ai2_nfl_model_pool.create_observed_model_pool` sets `weekly_projection_valid=True` and `opportunity_verified=True` after **historical match, min-games, recency and volume filters**. These flags mean a *valid historical and heuristic evidence gate*, not a confirmed current-week starter/snap/route forecast. Rename in future versions, or introduce distinct `historical_rate_valid` and `current_role_verified` flags. Do not treat the current labels as comprehensive independent validation.

## Independent verification needed
1. Add automated source lineage report per player and per forecast feature: fetched, transformed, applied, diagnostic-only, missing, with last-observed date.
2. Add a non-production counterfactual check: changing opponent allowance or defender status should currently leave offensive scoring draws unchanged (given identical rates/seed). This demonstrates unused input; it is NOT a recommendation to automatically assign naive boosts.
3. Walk-forward accuracy audit for DK points, targets, carries, passing volume, P10/P90 calibration and opportunity changes; compare current model to historical baseline with no future data leakage.
4. Introduce opponent/pace/weather/ownership factors **only** after complete coverage + calibrated out-of-sample evidence; monitor ablations.
5. Preserve approved live lineups, and never deploy experimental scoring coefficient changes before Sunday lock without explicit verification.

**Summary:** There are demonstrable collected-but-not-scored inputs. NFL Classic is a conditional historical-stat scenario engine, *not* yet an all-relevant-data predictor. No proof that any proposed multiplier improves accuracy until tested.
