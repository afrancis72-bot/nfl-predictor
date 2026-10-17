"""Unified NFL entry point for the experimental Team AI² workflow.

Keeps the existing Classic production page and research pages intact.
"""
import streamlit as st

st.title("🏈 Team AI² | NFL Command Center")
st.caption("Experimental unified workflow — the production NFL predictor remains available.")

st.warning("Research status: Snap Showdown exports are not yet DraftKings entry-ready; verify roster and availability. Classic DST and ownership estimates are not calibrated.")
st.markdown("### Choose your contest")
left,right=st.columns(2)
with left:
    st.subheader("Showdown")
    st.write("Single game: verify starters and historical usage, simulate game scripts, review player outcomes and research lineups.")
    if st.button("Open Showdown Research Lab",use_container_width=True):
        st.switch_page("pages/9_Snap_Research_Lab.py")
with right:
    st.subheader("Main Slate")
    st.write("Full slate: simulate matchups, estimate ownership and DST, build diversified nine-player portfolios.")
    if st.button("Open Main Slate Simulator",use_container_width=True):
        st.switch_page("pages/10_Main_Slate_Simulator.py")

st.divider()
st.subheader("Shared NFL workflow")
st.markdown("""
1. **Load the DraftKings salary file** for the correct contest.
2. **Verify team, player, starter and injury status** before simulation.
3. **Simulate and audit** player outcomes; flag missing history and implausible game volume.
4. **Build and compare** lineups for projection, upside, salary and exposure.
5. **Export only validated entries**; use DraftKings' original entry template to retain contest IDs.

**Current integration boundary:** Both builders already live in this repository and share snap-research modules. They still have separate page state and incomplete export/roster-validation integration. This hub does not claim those gaps are solved.
""")
