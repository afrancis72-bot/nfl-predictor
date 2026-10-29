# TEAM AI² Pro — launch the unified NFL development app

## Streamlit Community Cloud

Choose repository `afrancis72-bot/nfl-predictor`, branch
`feature/unified-ai2-platform`, and entry point `unified_app.py`
in your Streamlit app settings. Save/redeploy these settings.
The existing app configured to use `main/app.py` will not automatically
run this branch. The development app was tested in GitHub's headless
Streamlit runtime; a browser acceptance test is still needed.

## Local Windows launch

From the repository's development-branch checkout:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run unified_app.py
```

Open NFL. Upload a DraftKings Classic salary CSV, optionally set the
simulation seed, and click **BUILD MY 20 LINEUPS**. No projections file
is necessary. You will see named lineups, a downloadable named Excel,
named CSV, a roster-ID CSV, official injury/practice details with
retrieval time, and explicit notices about unverified breaking news.

Review and adjust the $3/$0.50 contest assignments without changing
the nine players in a lineup. Click **APPROVE AND SAVE THESE 20 LINEUPS**
to protect the exact roster and contest tiers independently of
future model builds. The protected approvals section also lets you
download/restore portable approval JSON, and separately import existing
DraftKings entry CSVs such as your prior 15 approved lineups.

## Storage precautions

Data is stored at `%USERPROFILE%\\TEAM_AI2_Pro_Data\\slates.sqlite3` on
Windows unless `AI2_DATA_DIR` is configured. Community Cloud file
storage is often ephemeral. **Download backup JSON/CSV files before
restarting or redeploying** if there is no durable storage mount.
ChatGPT uploads are not automatically present in Streamlit.

## Model qualification

The generated 20-lineup portfolios passed technical real-slate and
Streamlit smoke checks, but are **not verified to have positive expected
contest ROI**. Official NFL game designations and published practice
information are incorporated; Sunday inactives, coaching/news changes
and depth-chart claims still require a manual check. The one-click
model never submits a wager or DraftKings entry itself.
