"""PGA within the unified Team AI² Pro Streamlit application.

The original PGA model remains in sports/pga/legacy.
"""
from pathlib import Path
import runpy
import sys

import pandas as pd
import streamlit as st

from sports.pga.dk_entries import fill_entries_template

ROOT = Path(__file__).resolve().parents[1] / "sports" / "pga" / "legacy"
REQUIRED = (
    "app.py", "pga_predictor_pro.py", "golf_first_v11.py",
    "portfolio_optimizer.py", "round_parlay.py",
    "international_ingestion.py", "round_showdown.py",
    "course_fit_simulation.py", "course_fit_ceiling.py", "data_ingestion.py",
)
missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
if missing:
    st.error("PGA is not completely installed in this deployment.")
    st.caption("Missing original modules: " + ", ".join(missing))
    st.stop()

st.caption("PGA V11 golf-first simulator · Team AI² Pro")
st.info(
    "PGA source is installed. Current-week data and deployment validation "
    "are still required before using entries in a contest."
)

# Prevent an old tournament's portfolio being reused after PGA slate changes.
bundle = st.session_state.get("pga_bundle")
if bundle:
    players = bundle.get("players")
    if isinstance(players, pd.DataFrame) and {"player", "salary"}.issubset(players.columns):
        cols = [x for x in ("player", "salary", "ID") if x in players.columns]
        roster_signature = tuple(
            tuple(str(v) for v in row)
            for row in players[cols].sort_values("player").itertuples(index=False, name=None)
        )
        signature = (
            str(bundle.get("tournament_name")),
            str(bundle.get("tournament_start")),
            roster_signature,
        )
        previous = st.session_state.get("_pga_portfolio_slate_signature")
        if previous is not None and previous != signature:
            st.session_state.pop("dfs_portfolio", None)
        st.session_state["_pga_portfolio_slate_signature"] = signature

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
context = runpy.run_path(str(ROOT / "app.py"), run_name="__main__")

# The standalone PGA app produces a research long-form CSV. Offer a separate,
# validated DK Entries-file export only with real user-supplied entry IDs.
bundle = st.session_state.get("pga_bundle")
portfolio = st.session_state.get("dfs_portfolio")
if context.get("page") == "🏆 Tournament DFS" and bundle and portfolio:
    with st.expander("DraftKings contest-entry export (optional)", expanded=False):
        st.caption(
            "Upload your actual PGA DraftKings Entries CSV to fill six G slots "
            "using golfer IDs from this week's DraftKings salary file. "
            "Contest IDs and entry IDs are preserved."
        )
        template_file = st.file_uploader(
            "DraftKings PGA Entries CSV", type=["csv"], key="pga_dk_entries_template"
        )
        if template_file is not None:
            try:
                template = pd.read_csv(template_file, dtype=str)
                completed = fill_entries_template(
                    portfolio["portfolio"], bundle["players"], template
                )
                st.success(f"Validated {len(portfolio['summary'])} lineups for entry export.")
                st.download_button(
                    "Download completed DraftKings Entries CSV",
                    completed.to_csv(index=False),
                    "team_ai2_pga_entries.csv",
                    "text/csv",
                    key="pga_dk_entries_download",
                )
            except Exception as exc:
                st.error(f"DraftKings export blocked: {exc}")
