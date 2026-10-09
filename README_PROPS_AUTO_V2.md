# Player Props Lab V2 — automated weekly stats + baseline audit

## Install
Upload `app.py`, `props_lab.py`, `props_data.py`, and this README to your existing GitHub repository root. Keep the existing `data/` folder and all other files unchanged. Existing `requirements.txt` already supports this addition.

## Run
1. Launch Streamlit; choose Player Props Lab.
2. Select NFL season, target week, recent-games window.
3. Click **Load NFL statistics automatically**. The app downloads weekly player stats for the selected season and preceding season from nflverse (internet required), filters to observations **before** the target week, and constructs recency-weighted per-game averages.
4. Review the resulting players and run football-stat simulations.
5. Optional: use your own The Odds API key and click **Fetch NFL player props from API** (calls may consume credits). The existing market matching, EV, and CSV export functions remain available.
6. Open **Historical accuracy — walk-forward baseline audit** to evaluate the previous season's recency-weighted stat forecasts against subsequent actual outcomes. CSV export includes each prediction/actual.

## Important limitations
- The new data feed and backtest have been checked with synthetic fixture data, **not** a live 2026 download in this build environment.
- Stats are official-derived nflverse weekly player statistics. Missing games are not treated as zero; injury/role/active-status verification is NOT automated.
- Stats are aggregated pregame baselines; opponent, pace, game scripts, and role uncertainty are not fully calibrated. The 100K simulation is experimental, not independently validated for betting.
- The walk-forward audit tests **baseline means**, not full distribution calibration or prop betting profitability.
- Live odds require an external API key and supported provider coverage. API integration was retained from V1 and not live-tested here.
- Avoid using projected positive EV for real-money decisions until full out-of-sample calibration and injury checks are implemented.
