# NFL Predictor Pro — DFS Engine V2.0.3

## Automatic current-slate market environment

V2.0.3 keeps the V2.0.2 current-slate lock and adds an automatic weekly market-data step.

Weekly workflow:
1. Upload the current DraftKings Classic salary CSV. The DK file supplies the active roster, games, IDs, salaries, and slate date.
2. Click **Auto-fetch current market totals & spreads**. The app queries the current NFL scoreboard market feed for the DK slate date and matches only the active games.
3. Review the returned total/spread/source/status table. If any game cannot be resolved, that game remains unverified and may be entered manually.
4. Click **Activate reviewed game environment** only after every active game has a valid total.
5. V2 simulation/portfolio generation remains blocked until full current-slate coverage is verified.

Safety behavior: no unresolved current game is ever filled from the bundled historical Week 4 environment. The automatic feed is a convenience source, not a silent fallback; unresolved or failed fetches remain visible.
