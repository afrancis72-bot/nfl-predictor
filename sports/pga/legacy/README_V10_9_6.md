# PGA Predictor V10.9.6 — Showdown Distribution Upgrade

This is an isolated Showdown-layer update. Tournament DFS and Round Parlays are unchanged.

## Changes
- Replaces the fixed `showdown_proxy + 14` ceiling with a 25,000-draw player-specific single-round proxy distribution.
- Uses only data already available to the model: validated round rating, completed-round adjustment, Course DNA, and OTIS APP/OTT/ARG/PUTT component percentiles when present.
- Missing component data is neutral (50), never zero.
- Adds transparent `upside_signal`, `stability_signal`, mean, SD, P75, P90, P95, GPP score, and value fields to the golfer audit.
- Portfolio optimization now ranks lineups by a GPP distribution score (55% mean / 30% P90 / 15% P95) while retaining salary, exposure and diversification controls.

## Important limitation
This build does **not** claim to have hole-level birdie/bogey data. The fantasy outputs remain DK-style ranking proxies, not official DraftKings point projections. The purpose is to stop treating every golfer as having the same fixed ceiling and to make single-round upside/risk player-specific using only supported inputs.

## Deploy
Replace `app.py` and `round_showdown.py` only. Do not replace `round_parlay.py`.
