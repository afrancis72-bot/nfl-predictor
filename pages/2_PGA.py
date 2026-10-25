"""PGA within the Team AI² Pro unified Streamlit application.

This page executes the original PGA dashboard with its unchanged engine
modules. Model parity and current-week data freshness must be validated
separately; copying an engine is not a betting or DFS performance guarantee.
"""
from pathlib import Path
import runpy
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1] / "sports" / "pga" / "legacy"
REQUIRED = (
    "app.py",
    "pga_predictor_pro.py",
    "golf_first_v11.py",
    "portfolio_optimizer.py",
    "round_parlay.py",
    "international_ingestion.py",
    "round_showdown.py",
    "course_fit_simulation.py",
    "course_fit_ceiling.py",
    "data_ingestion.py",
)
missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
if missing:
    st.error("PGA is not completely installed in this deployment.")
    st.caption("Missing original modules: " + ", ".join(missing))
    st.stop()

st.caption("PGA V11 golf-first simulator · Team AI² Pro")
st.info(
    "PGA migration validation: engine files are installed. "
    "Use current tournament data; deployment and simulation parity checks "
    "must pass before treating lineup exports as validated."
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
