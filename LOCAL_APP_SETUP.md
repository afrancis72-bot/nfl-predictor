# TEAM AI² Pro — local home-computer launch

Run from the repository directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run unified_app.py
```

The app opens in your browser. By default, uploaded files persist in
`%USERPROFILE%\\TEAM_AI2_Pro_Data\\slates.sqlite3` on Windows. This location
is independent of the Git checkout and survives normal app restarts.

To use a different storage location, set `AI2_DATA_DIR` before starting
Streamlit. **Keep this directory private and back it up**: it contains uploaded
data and previous versions. Do not put it in Git or a public synced folder.

On the NFL **My Slate** tab, select the file type, upload a file, and click
**Save / replace file**. Returning later does not require another upload.
Replacement automatically archives the prior version. Archived versions can
be downloaded from the same tab.

Important limitations:
- This is local persistence, not automatic cross-computer synchronization.
- Files shared in ChatGPT are not automatically copied into the app.
- A local directory can still be lost if the computer or drive fails.
- A saved slate is not necessarily a fresh or verified slate.
- NFL lineup building requires verified current-week projection fields.
- Correlated simulations, live data certification, DraftKings-ready export,
  and end-to-end production validation remain unfinished.
