"""Team AI² Pro PGA entry export (separate from the original PGA simulator).

Streamlit's original PGA dashboard calls st.stop() on Setup, so this workflow
must be its own page instead of being appended after runpy.run_path().
"""
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from sports.pga.dk_entries import fill_entries_template
from sports.pga.dk_template import parse_dk_entries

st.title("⛳ PGA · DraftKings Entry Export")
st.caption("Team AI² Pro · Uses the PGA tournament model already built this session")

bundle = st.session_state.get("pga_bundle")
portfolio = st.session_state.get("dfs_portfolio")
if bundle is None or portfolio is None:
    st.info("Build a PGA tournament model and generate lineups on the PGA page first.")
    st.page_link("pages/2_PGA.py", label="Back to PGA model")
    st.stop()

players = bundle.get("players")
if not isinstance(players, pd.DataFrame) or not {"player", "salary"}.issubset(players.columns):
    st.error("Current PGA player pool is unavailable. Rebuild the tournament setup.")
    st.stop()

cols = [c for c in ("player", "salary", "ID") if c in players]
roster_signature = tuple(
    tuple(str(v) for v in row)
    for row in players[cols].sort_values("player").itertuples(index=False, name=None)
)
signature = (
    str(bundle.get("tournament_name")), str(bundle.get("tournament_start")),
    roster_signature,
)
if st.session_state.get("_pga_portfolio_slate_signature") != signature:
    st.session_state.pop("dfs_portfolio", None)
    st.error(
        "The saved lineups have not been validated against this tournament setup. "
        "Return to PGA, generate a new portfolio, then open Entry Export."
    )
    st.stop()

st.write(
    f"**{bundle.get('tournament_name', 'Tournament')}** — "
    f"{len(portfolio['summary'])} saved lineups"
)
st.caption(
    "Upload your actual DraftKings PGA Entries CSV. We preserve each entry's "
    "Entry ID and contest details and replace only six G roster slots. "
    "Lineups are checked against the current DraftKings salary file."
)

template_file = st.file_uploader(
    "DraftKings PGA Entries CSV", type=["csv"], key="pga_dk_entries_template"
)
if template_file is None:
    st.stop()

try:
    template = parse_dk_entries(template_file.getvalue())
    completed = fill_entries_template(
        portfolio["portfolio"], players, template
    )
    st.success(
        f"Validated {len(portfolio['summary'])} PGA lineups "
        f"for {len(template)} available entries."
    )
    st.dataframe(completed.iloc[:len(portfolio["summary"])], hide_index=True, use_container_width=True)
    st.download_button(
        "Download completed DraftKings Entries CSV",
        data=completed.to_csv(index=False),
        file_name="team_ai2_pga_entries.csv",
        mime="text/csv",
        key="pga_dk_entries_download",
    )
except (ValueError, KeyError, TypeError) as exc:
    st.error(f"DraftKings export blocked: {exc}")
