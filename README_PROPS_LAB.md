# Player Props Lab — experimental add-on

This ZIP is based on the supplied `nfl-predictor-main (3).zip`. Existing Classic and Showdown simulation/optimizer functions have not been modified. Added `props_lab.py` and a new sidebar page in `app.py`.

## Start

1. Replace your repository with the contents of this ZIP, or copy `props_lab.py` and the modified `app.py` into your repo root. Existing `requirements.txt` is unchanged.
2. Launch Streamlit as usual: `streamlit run app.py`.
3. Select **Player Props Lab** in the sidebar.
4. Upload a **current, observed** weekly player-stat CSV. Use the included `props_input_schema_example.csv` only as a column schema, not as verified current statistics. Required columns: `player,team,position`; add per-game rates such as `pass_attempts_pg`, `carries_pg`, `targets_pg`, `receptions_pg`, `receiving_yards_pg`, `rushing_yards_pg`. Optional `games_played` and `game`.
5. Run simulations and download summary/raw player scenarios.
6. Upload an odds CSV with `player,market,line,side,price` and optional `bookmaker`; or supply your own The Odds API key and click Fetch. Compare and export ranked estimated EV.

## Data caveats / limitations

- The bundled 2026 Week 4 projections are NOT automatically fed into the props model. The module explicitly requires uploaded football-stat rates.
- The underlying generator is an **experimental** team-volume/usage model, not a calibrated full play-by-play engine. It links teammates' targets/carries through shared team volumes and links QB passing production to simulated receiving production, but is not yet validated for real betting use.
- A complete QB receiving corps is needed for accurate passing-yard simulation. Unmodeled receivers are approximated; insufficient receiver coverage biases projections.
- Missing inputs are NOT silently replaced with historical observations. Some conditional assumptions (catch rate 65%, YPC 10, rush YPC 4) are provisional and must be calibrated; don't treat those as actual stats.
- The odds API uses one request for events and one per event; player-prop market coverage varies by sportsbook, provider subscription, and time. API calls can incur usage charges.
- EV assumes prices in American odds and handles pushes for integer lines. Positive EV estimates do not establish a real edge until holdout backtesting and calibration.
- Player matching is exact name (case-insensitive); ambiguous names are excluded. Inactive/uncertain roles must be excluded or adjusted upstream.
- Historical backtesting, true injury/role adjustments, opponent defense and automatic NFL stats ingestion are **not yet implemented**. This is a functional first stage, not a finished predictive betting product.
