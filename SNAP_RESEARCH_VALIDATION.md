# Snap Research Lab — engineering acceptance checklist

**Branch:** `dev/showdown-game-script-v1` (production is not modified).

## Running validation
```bash
python -m pip install numpy pandas streamlit
python -m py_compile snap_level_research.py historical_usage.py defensive_priors_research.py backtest_research.py historical_replay_research.py
python -m unittest discover -s tests -p 'test_*research.py' -v
streamlit run app.py
```

## Historical replay workflow
1. Select an **already completed** game, season and week.
2. Upload a DraftKings-style FLEX player pool for that historical matchup. Verify
   the historical starters, injury exclusions and personnel using independently
   timestamped pregame sources. Current rosters must not be substituted.
3. Load weekly nflverse player statistics. Only games with week strictly below
   the selected week are eligible for usage priors.
4. Upload historical nflverse play-by-play with columns `season,week,defteam,
   posteam,play_type,yards_gained`; optional `complete_pass,season_type`.
   The defensive-prior pipeline rejects missing data and excludes the target
   week and later weeks.
5. Upload actuals with columns `player,stat,actual`, and optionally `baseline`
   containing a **genuine pregame** forecast from an independently frozen
   baseline. The observed values are joined after simulation, never used to
   produce the simulated games.
6. Enter a unique historical matchup ID and run replay. Export per-player errors
   and calibration report. Repeat over many independent historical matchups.
7. Compare MAE, signed bias, 80/90% interval coverage and baseline MAE. Do not
   tune parameters on the same held-out games used for the final evaluation.

## Current limitations (not sportsbook-calibrated)
- Snap/route personnel probabilities are provisional unless `snap_share`
  priors are supplied. They are **not** inferred from actual on-field tracking.
- Pressure, penalty, punt, interception, fumble and fourth-down parameters are
  prototype assumptions; measured play-by-play calibration is required.
- No explicit quarter/halftime, kickoffs, touchbacks, timeouts, two-point
  conversion, overtime, punt return or defense/special-teams scoring.
- Defensive profiles use same-season prior games and league-relative rates, not
  full opponent-adjusted schedule strength.
- Verified pregame historical player pools are not auto-generated; the replay
  interface requires supplied snapshots.
- GitHub commits and a CI workflow are not evidence that CI ran or passed.
- A single historical replay does not establish predictive calibration.

## Definition of done
- CI actually green, with artifacts or logs inspected.
- Independently verified pregame historical datasets, including starter status.
- Measured and calibrated possession/pressure/penalty/personnel distributions.
- Repeatable holdout backtest with sufficient independent matchups.
- Benchmark improvement over a documented frozen baseline, with interval
  calibration and error by position, team and game script.
- Streamlit smoke test and documented reproducible deployment.
