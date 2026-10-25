"""PGA within the unified Team AI² Pro Streamlit application.

The original PGA model remains in sports/pga/legacy.
"""
from pathlib import Path
import runpy
import sys

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

ROOT = APP_ROOT / "sports" / "pga" / "legacy"
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
st.page_link("pages/3_PGA_Entries.py", label="DraftKings PGA Entries export", icon="📄")
st.info(
    "PGA source is installed. Current-week data and deployment validation "
    "are still required before using entries in a contest."
)

# Streamlit preserves session_state objects between pages. A fresh tournament
# setup replaces the bundle, so immediately discard portfolios from prior setups
# even if golfer names and salaries happen to match.
bundle = st.session_state.get("pga_bundle")
if bundle is not None:
    bundle_id = id(bundle)
    prior_id = st.session_state.get("_pga_active_bundle_id")
    if prior_id != bundle_id:
        st.session_state.pop("dfs_portfolio", None)
        st.session_state.pop("_pga_portfolio_origin", None)
        st.session_state["_pga_active_bundle_id"] = bundle_id

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
context = runpy.run_path(str(ROOT / "app.py"), run_name="__main__")

# Reached only after the DFS page finishes. Setup intentionally calls st.stop().
# Stamp a newly-generated portfolio to the exact in-memory setup it used.
bundle = st.session_state.get("pga_bundle")
portfolio = st.session_state.get("dfs_portfolio")
if context.get("page") == "🏆 Tournament DFS" and bundle is not None and portfolio is not None:
    st.session_state["_pga_portfolio_origin"] = (id(bundle), id(portfolio))

