# PGA V11.0 — Golf-first DFS integration (experimental)

## Deploy
1. Back up the current GitHub repository or create a branch.
2. Upload **all top-level .py files** and requirements.txt from this ZIP into the existing Streamlit repo root, overwriting matching filenames. Do not upload the enclosing folder as a folder.
3. Streamlit should restart after the commit. Check logs for import errors.
4. In Setup / Inputs, upload the usual DK and OTIS files, choose **No cut** or **Standard cut**, confirm course par, and preferably provide an 18-hole `course_holes.csv` with a `par` column.
5. Click Load inputs & build tournament model, then Tournament DFS > Build Lineups. Rankings display mean, median, p75, p90, p95 simulated DK points.

## Scope
- Replaces V10 synthetic DK proxy in Tournament DFS with hole-by-hole simulated DraftKings Classic scoring. DFS optimizer now prioritizes simulated DK mean/ceiling objectives.
- Preserves separate Round Parlay V10.9.8 and Round Showdown modules; does NOT change their model logic.
- Preserves existing setup/navigation, course DNA, ingestion.
- Cut event simulation: top N including ties after R2; missed-cut players score only R1/R2 DK points, no finish/4-round bonus. Real event-specific cut rules can vary, so configure explicitly.
- No-cut events: all four rounds, including Baycurrent.
- Uses the V10 pre-tournament skill estimates as the golfer-skill input.

## IMPORTANT model limitations
- Hole outcome probabilities are provisional and **not** calibrated to actual player/course hole-level history. This is an experimental release, not a verified predictive upgrade.
- If 18 hole pars are not supplied, a generic par-70/71/72/73 layout is substituted and labeled provisional.
- Simulation results are estimates, not historical tournament outcomes. The prior Baycurrent A/B test was retrospective and does not prove repeatability.
- Scoring assumes four-round DraftKings PGA Classic. Does not support alternative formats or withdraws/DQs.
- Shared golfer/round shocks are simulated, but the portfolio optimizer scores candidates using additive individual golfer fantasy distributions; it does not yet optimize full lineup-level covariance or simulated top-1% ROI.
- The cut policy uses top N with ties; tournaments using a different rule must be treated separately.
- Set tournament format manually every week. No-cut detection is NOT automated.

## Smoke checks run
- All Python files compiled successfully.
- Simulated Baycurrent saved projections with both no-cut and cut modes.
- Portfolio optimizer produced 3 valid lineups from V11 simulated DK projections.

## Original source
V10.9.4 base + latest uploaded app.py + V10.9.8 round_parlay.py + V10.9.5 round_showdown.py + V11 golf_first_v11.py and V11 optimizer integration.
