# TEAM AI² Pro — unified development app

## Run the NFL development version

Install requirements and launch `streamlit run unified_app.py`.
In Streamlit Community Cloud, select repository `afrancis72-bot/nfl-predictor`,
branch `feature/unified-ai2-platform`, and main file `unified_app.py`.
This configuration is **not** deployed automatically by a GitHub commit.
The repository's original `main/app.py` remains a separate entry point.

## NFL Classic one-button workflow

1. Open NFL from the home screen and upload the DraftKings salaries CSV.
2. Optionally set the simulation seed (default 42).
3. Click **BUILD MY 20 LINEUPS**.

The experimental model fetches nflverse observed player history and official NFL
game-status designations, builds a game-level 10,000-scenario simulation, and
generates 20 checked NFL Classic roster combinations. No projection CSV needed.

The app displays **human-readable names, team, position and scores** and has
separate downloadable named Excel, named CSV, and DraftKings roster-slot ID CSV.
The roster-ID CSV is **not a verified contest-entry upload template**: match
it against your DraftKings contest entry requirements before submitting.

## News and injury handling

The same build retrieves the official NFL injury table, including practice
participation, injury note, official game designation, source URL and UTC
retrieval time where available. Current-week questionable and otherwise
unverified players are conservatively excluded. Confirmed unavailable
players have their observed workloads redistributed in the simulation
once. Practice participation **is not** a final game designation,
and active status alone does not confirm workload.

There is **no independently verified, comprehensive breaking-player-news
or depth-chart feed yet**. The app explicitly warns about this gap.
It is necessary to recheck late-breaking news and final Sunday inactives.

## Human review and approvals

The five $3 lineups and fifteen $0.50 lineups can be reviewed; any proposed
contest-tier swap must comply with QB/team diversification and
current-slate premium-only exclusions. Swaps change **contest assignments only**
and cannot alter rosters. A one-click "Approve and save" action stores a
separate JSON lineup snapshot, without overwriting it when new simulations
are run. JSON snapshots are downloadable for backup. Previously approved
DK entries can also be imported and saved separately. Historical
vetoes are **not** automatically carried forward to new slates.

## Data storage

SQLite upload storage lives under `AI2_DATA_DIR` or
`~/TEAM_AI2_Pro_Data/slates.sqlite3`. Replacement archives prior
versions. Streamlit Community Cloud storage may be ephemeral; download
portable backups, or use a durable mounted volume, before relying
on saved files through redeployments. Nothing uploaded in ChatGPT is
automatically copied into Streamlit.

## Release limits

The NFL engine passed controlled and real 2026 Week 8 data tests and
performs legality and exposure checks. It is **experimental**, not
backtested for profitability or independently calibrated for current
depth charts, player news, ownership, defense, or fantasy-scoring distributions.
Browser verification of the deployed Streamlit URL remains a separate acceptance
step. The other sports tabs retain their prior research/preview behavior.
