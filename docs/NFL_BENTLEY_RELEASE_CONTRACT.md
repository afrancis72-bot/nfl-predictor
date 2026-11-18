# Team AI² Pro: NFL Classic “Bentley” acceptance contract

**Sole production app:** `unified_app.py` on `feature/unified-ai2-platform`. NFL, PGA,
NBA and NASCAR remain within this same app. Do not deploy or revise an NFL
standalone app. **Showdown is the secondary NFL track** with its own single-game
Captain/FLEX checks; it is not a substitute for finishing Classic.

## Binding release rule

**NFL CLASSIC IS NOT READY FOR ENTRY.** This is not a subjective quality label.
`ai2_nfl_release_gate.nfl_classic_release_status()` is the application-visible
source of truth. If ANY mandatory acceptance test is not independently
verified, the NFL tab must disable **BUILD MY LINEUPS**, must not display
stale session results as current entry-ready output, and must not permit a
new 20/40-entry approval. It may show previously saved, approved roster
snapshots for archival review, without altering them. Other sports remain
operational.

A green unit-test run, a new seed, a GitHub commit, an NFL.com article or a
plausible-looking projected mean cannot mark the model ready by itself.
The protected user-approved selections are always separate from model results.

## Acceptance: *actual scoring-path* requirements

| ID | Pass criterion |
| --- | --- |
| POSSESSION_WORLD | One correlated chronological game with consistent shared possessions, attempts, targets, carries, scoring, sacks/turnovers, DST and game script; no fabricated plays; 10,000+ stat scenario worlds; tested on real heldout games |
| ROLE_SPECIFIC_DEFENSE | Historical opposing QB1/RB1/RB2/WR1/WR2/WR3/TE1/TE2 separated **by their pregame roles**, compared to *their own normal pregame production*, correcting for sample and opponent quality; credible differences actively adjust appropriate per-stat channels, not just a summary table; zero lookahead; blind A/B improvement |
| CURRENT_PLAYER_ROLES | Weekly teams/transactions, real snaps/routes/target roles, team passing/run totals, returning/injured starters, replacement players and ambiguous/Q tags; documented source timestamps; appropriate WR1–WR3 historical eligibility; absolutely no phantom zero-opportunity TD punts |
| LIVE_AVAILABILITY | Official source verification for matchups and player status, honest active/OUT/uncertain distinctions; prekickoff source bytes and timestamps, critical gaps surfaced rather than silently filled; actual real-slate weekend validation |
| STAT_OUTCOME_CALIBRATION | Mean/median/P90 for actual passing attempts/yards/TD/INT, rushing carries/yards/TD, targets/receptions/receiving yards/TD, DST; scoring reconciliation and correlation checks; real-slate position-level bias and P90 empirical coverage **near nominal 90%** |
| OUT_OF_SAMPLE_VALIDATION | Same seeded common random worlds, leakage-free retrospective and prospective freeze; compare each proposed adjustment to the old engine and simple pregame baseline; do not activate regressions (e.g. naive teammate-universe or simple defensive multiplier) that worsened MAE/bias/P90 |
| CONTEST_PORTFOLIO | 9 unique DK positions / <=$50,000, prefer >=$49,600, 35% exposure final roster set, no two same-team RBs, no QB opposing DST in Classic, actual verified opportunity, no illegal stacks; diversified QB+WR/TE + viable bring-back, controlled duplication; support the user's **four $1 + sixteen $0.50** tier plan instead of hard-coded $3; split robust cash-credible/first-place-upside (about 70–80%/20–30%) |
| PROVENANCE_AND_AUDIT | Exact deployed commit and source freshness in each CSV/XLSX/JSON run; immutable pregame stats/lineups/means/P90 with seed, no postgame leakage; independent postgame score, ranking, calibration and player stat audit; preserve human-approved lineup selections and any swaps |
| FULL_REGRESSION_AND_DEPLOYMENT | Critical tests green across NFL and all four unified sport interfaces, no unresolved known failures, real DraftKings salary CSV to full lineup/export same-app test, timestamped real-game status and roles, exact Streamlit SHA verified |
 
For each ID, record dated source data, test case, test run/commit, holdout
dates, statistics, result, and who reviewed the decision. **A contract can only
change to PASS after the behavior is integrated in the production
`build_oneclick_portfolio` path and the relevant evidence is examined.**
Successful code plumbing without scoring or calibration is NOT PASS.

Known failures at initiation: position-specific defensive history appears in
reports but `defense_weight=0.0` disables its impact; possession research is
not integrated into the one-click scoring path; the role-based defense and WR
role candidate experiments have not earned strong independent improvement;
full live-weekend status coverage and slate-level holdout have not passed;
the existing 40 lineup contest-tier assumptions do not match the user's
current 4 × $1 + 16 × $0.50 plan. Earlier P90 coverage was below its 90%
nominal target, and earlier full-suite failures still need a verified clean
regression, beyond targeted suites.

## Development and review sequence

1. Inventory **every requirement**, trace to actual entrypoint function, state
   whether ACTIVE / RESEARCH ONLY / DISABLED / UNKNOWN, capture proof.
2. Complete shared possession/game simulation and role-specific defense
   stat channel integration **in the existing app**, not another app.
3. Compare to baseline with prekickoff evidence and real heldout weeks, fail
   effects that make forecasts worse. Verify P90 coverage and correlation.
4. Validate live official coverage, all optimizer safeguards and the precise
   contest tiers with an authentic DraftKings slate, and freeze a pregame
   prediction manifest.
5. Run the **full** test suite and app deployment acceptance, not just targeted
   tests; review warnings and ownership/availability gaps.
6. Update the contract only after each pass is documented with evidence, then
   unlock new Classic builds **with final human veto**.
7. Only after Classic passes, build the separate single-game Showdown flow
   inside NFL using the same game simulation and intelligence, with Captain
   1.5× scoring/salary, FLEX rules and the user's Showdown-specific allowance
   of QB against opposing DST.

**Never tell the user “try running some lineups” or recommend entering them
while the gate is red.** Research backtests are permitted in automated tests
and isolated diagnostic code, but no app build/export should be labeled ready
for actual DraftKings entries before acceptance. No changes to PGA/NBA/NASCAR
workflow should be made as part of this release lock.
