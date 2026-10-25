"""PGA page for the unified Team AI² Pro Streamlit app.

The legacy PGA engine must be installed under sports/pga/legacy.
Keep its original calculations intact until source-vs-unified parity passes.
"""
from pathlib import Path
import runpy
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[1] / "sports" / "pga" / "legacy"
if not (ROOT / "app.py").is_file():
    st.error("PGA source files have not been installed in this deployment.")
    st.stop()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
