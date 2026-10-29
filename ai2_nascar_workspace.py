"""NASCAR race sim + DraftKings lineups: single screen for Team AI² Pro."""
from __future__ import annotations
from io import BytesIO
import hashlib
import secrets

import pandas as pd
import streamlit as st

from ai2_nascar_oneclick import (
    track_catalog, load_dk_slate, run_oneclick,
    reference_signals, PRACTICE_COLS,
)

st.subheader("NASCAR · One-Click Race Simulator & Lineup Builder")
st.caption(
    "Select track → upload the POST-qualifying DraftKings salary CSV → "
    "Run race simulations + build your portfolio. "
    "The starting grid comes from DK; practice is optional."
)

catalog=track_catalog()
track_names=catalog["track"].astype(str).tolist()
col_track,col_dk=st.columns([1.15,2.0])
with col_track:
    selected_track=st.selectbox("1. Race track",track_names,key="nascar_track")
    default_laps=int(catalog.loc[catalog["track"]==selected_track,"laps"].iloc[0])
    laps=int(st.number_input(
        "Scheduled laps (adjust if race distance differs)",min_value=25,max_value=700,
        value=default_laps,step=1,key=f"nascar_laps_{selected_track}",
    ))
with col_dk:
    salary_file=st.file_uploader(
        "2. DraftKings NASCAR salary slate — AFTER qualifying",
        type=["csv"],key="nascar_oneclick_dk",
        help="Must contain driver Name, unique ID, Salary, and complete Starting positions.",
    )
    st.caption(
        "DraftKings' **Starting** column must contain the final qualifying grid. "
        "We never invent qualifying positions, practice speeds, or driver identities."
    )

dk=None
if salary_file is not None:
    try:
        dk=load_dk_slate(BytesIO(salary_file.getvalue()))
        st.success(f"{len(dk)} real drivers and all {len(dk)} qualifying positions verified.")
        with st.expander("Preview imported starting grid"):
            st.dataframe(dk[["Name","Starting","Salary","AvgPointsPerGame"]],
                         hide_index=True,use_container_width=True)
    except Exception as exc:
        st.error(f"DraftKings slate isn't race-ready: {exc}")

with st.expander("Optional practice data and model controls",expanded=False):
    st.caption(
        "Practice improves driver speed estimates, especially long-run 10/15/20-lap pace. "
        "Leave it empty for normal one-file use; missing practice is disclosed and not invented."
    )
    practice_file=st.file_uploader(
        "Optional practice CSV: driver name + actual speed or rank columns",
        type=["csv"],key="nascar_practice_optional",
    )
    excluded=st.multiselect(
        "Driver exclusions / human veto (optional)",
        dk["Name"].tolist() if dk is not None else [],
        key="nascar_exclusions",
        help="Excluded drivers still participate in the simulated race but cannot appear in your fantasy portfolio.",
    )
    template=pd.DataFrame(columns=["Name"]+list(PRACTICE_COLS))
    st.download_button("Practice template (optional)",template.to_csv(index=False).encode(),
                       "nascar_optional_practice.csv",mime="text/csv")
    st.caption(
        "Recognized columns include single_lap_rank, avg5_rank, avg10_rank, avg15_rank, "
        "avg20_rank—or corresponding actual mph speed columns. Speed converts to ranks automatically."
    )

controls=st.columns(4)
with controls[0]:
    sims=int(st.selectbox("Race simulations",[10000,25000,50000,100000],
                          index=1,key="nascar_sim_count"))
with controls[1]:
    lineup_count=int(st.number_input("Number of lineups",min_value=1,max_value=50,
                                     value=10,step=1,key="nascar_lineup_count"))
with controls[2]:
    exposure_pct=int(st.slider("Driver exposure cap (%)",15,100,35,5,
                               key="nascar_driver_exposure"))
with controls[3]:
    overlap=int(st.slider("Max shared drivers",0,5,4,1,key="nascar_overlap"))

a,b,c=st.columns([1,1.2,1])
with a:
    seed_mode=st.selectbox("Seed mode",["Fixed","Fibonacci preset","Fresh random"],
                           key="nascar_seed_mode")
with b:
    if seed_mode=="Fixed":
        chosen_seed=int(st.number_input("Reproducible seed",min_value=0,
                              max_value=2147483647,value=42,step=1,
                              key="nascar_seed_fixed"))
    elif seed_mode=="Fibonacci preset":
        chosen_seed=int(st.selectbox("Fibonacci seed",
                    [1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181],
                    index=7,key="nascar_fibonacci_seed"))
    else:
        chosen_seed=None
        st.caption("A fresh seed is picked each time you run.")
with c:
    salary_floor=int(st.number_input(
        "Portfolio min salary ($)",min_value=0,max_value=50000,
        value=47000,step=100,key="nascar_salary_floor",
    ))

st.caption(
    "Simulates coherent finishing order, laps led, fastest laps, and DraftKings scores, "
    "then ranks six-driver lineups against the same race scenarios. "
    "No player is chosen based only on an individual P90."
)
st.caption(
    "Automatic driver form and comparable-track information is attempted from "
    "NASCAR Reference. If that source is unavailable, only the DK salary/FPPG baseline, "
    "track profile, and verified grid are available; missing information is flagged."
)

@st.cache_data(ttl=21600,show_spinner=False)
def cached_nascar_reference(records,track):
    return reference_signals(pd.DataFrame(records, columns=["Name","ID","Salary","AvgPointsPerGame"]),track)

current_signature = (
    hashlib.sha256(salary_file.getvalue()).hexdigest() if salary_file else None,
    selected_track, laps, sims, lineup_count, exposure_pct, overlap,
    chosen_seed, seed_mode, salary_floor, tuple(sorted(excluded)),
    hashlib.sha256(practice_file.getvalue()).hexdigest() if practice_file else None,
)

run_requested=st.button(
    "🏁 Run race simulation + build DraftKings lineups",
    type="primary",use_container_width=True,disabled=dk is None,
    key="nascar_run_oneclick",
)
if run_requested:
    try:
        practice=pd.read_csv(BytesIO(practice_file.getvalue())) if practice_file is not None else None
        run_seed=int(chosen_seed) if chosen_seed is not None else secrets.randbelow(2147483647)
        with st.spinner("Checking published driver form and similar-track NASCAR history…"):
            try:
                ref_data=cached_nascar_reference(tuple(
                    tuple(row) for row in dk[["Name","ID","Salary","AvgPointsPerGame"]].itertuples(index=False,name=None)
                ), selected_track)
            except Exception as ex:
                ref_data=None
                st.warning(
                    "NASCAR driver-history source is unavailable. "
                    "Continuing in lower-confidence market-baseline research mode; "
                    "real practice speeds will not be fabricated."
                )
        with st.spinner("Running coherent race simulations and building risk-controlled lineups…"):
            result=run_oneclick(
                dk,selected_track,laps,sims=sims,seed=run_seed,
                portfolio_size=lineup_count,max_exposure=exposure_pct/100,
                max_overlap=overlap,salary_floor=salary_floor,
                practice=practice,reference=ref_data,
                excluded_drivers=excluded,
            )
        st.session_state["nascar_unified_result"]=result
        st.session_state["nascar_result_signature"]=current_signature
        st.success(
            f"Built {len(result['lineups'])} valid DraftKings lineups from "
            f"{sims:,} simulated races. Seed {run_seed}."
        )
    except Exception as exc:
        st.session_state.pop("nascar_unified_result",None)
        st.session_state.pop("nascar_result_signature",None)
        st.error(f"NASCAR build was blocked: {exc}")
        st.caption("No partial, unverified or illegal lineups will be exported.")

result=st.session_state.get("nascar_unified_result")
if result is not None and st.session_state.get("nascar_result_signature") != current_signature:
    result=None
    st.info("Race, slate, seed, or portfolio settings changed. Click Run again to build for the current configuration.")
if result:
    meta=result["metadata"]
    # Prevent stale outputs from different track, slate or controls being mistaken
    # for an active race build. A new upload requires a new button click.
    st.divider()
    st.subheader("Completed race simulation")
    st.caption(
        f"Race: {meta['track']} · {meta['laps']} laps · "
        f"{meta['drivers']} drivers · {meta['sims']:,} scenarios · "
        f"seed {meta['seed']} · built {meta['built_at']}"
    )
    if not meta["reference_coverage_drivers"]:
        st.warning(
            "No scored NASCAR Reference driver history was available. "
            "These lineups are a lower-confidence research baseline using actual DK "
            "salaries, DK FPPG, the verified grid, and track assumptions; "
            "they are not validated winning projections."
        )
    else:
        st.caption(
            f"NASCAR Reference parsed: {meta['reference_drivers']}/{meta['drivers']} drivers; "
            f"meaningful scored history: {meta['reference_coverage_drivers']}/{meta['drivers']}."
        )
    if not meta["practice_drivers"]:
        st.info("No measured practice speeds supplied. The model does NOT assume or invent practice pace.")
    else:
        st.caption(f"Actual practice data used for {meta['practice_drivers']} drivers.")

    tab_lineups,tab_drivers,tab_scripts=st.tabs([
        "🏆 DraftKings lineups","🏎️ Driver race projections","🎲 Race scripts / diagnostics"
    ])
    with tab_lineups:
        st.write("All entries passed driver IDs, salary, overlap, and exposure-cap validation.")
        st.dataframe(result["lineups"],use_container_width=True,hide_index=True)
        st.download_button("Download lineup review CSV",
            result["lineups"].to_csv(index=False).encode("utf-8-sig"),
            "team_ai2_nascar_lineups_review.csv",mime="text/csv")
        st.download_button("Download 6-driver DK roster IDs",
            result["roster_csv"],"team_ai2_nascar_dk_rosters.csv",mime="text/csv",
            help="Contains six D columns with DraftKings driver IDs. Compare headers/IDs with your actual DK entry template before uploading.")
        st.caption(
            "The six D columns provide real DraftKings driver IDs; DK contest-entry templates "
            "can vary. Verify the template or share your DKEntries CSV for a tailored entry upload."
        )
        st.subheader("Exposure")
        st.dataframe(result["exposure"],hide_index=True,use_container_width=True)
        st.download_button("Download driver exposure",
            result["exposure"].to_csv(index=False).encode("utf-8-sig"),
            "team_ai2_nascar_exposure.csv",mime="text/csv")
    with tab_drivers:
        stats=["Name","ID","Salary","projected_start","proj_dk",
               "median_p50","floor_p20","ceiling_p90","ceiling_p95",
               "sim_top6_pct","win_pct","top5_pct","top10_pct",
               "exp_finish","exp_laps_led","exp_fastest_laps",
               "data_coverage","model_confidence"]
        shown=result["projections"][[c for c in stats if c in result["projections"]]]
        st.dataframe(shown,hide_index=True,use_container_width=True)
        st.download_button("Download race projections",
            result["projections"].to_csv(index=False).encode("utf-8-sig"),
            "team_ai2_nascar_driver_simulations.csv",mime="text/csv")
        st.caption("Model confidence reflects input coverage; it has not been calibrated against real-race win rates.")
    with tab_scripts:
        st.dataframe(result["scripts"],hide_index=True,use_container_width=True)
        st.caption(
            "Primary dominator, split dominator, and chaos outcomes are "
            "model-generated scenarios, not verified future race probabilities."
        )
elif salary_file is None:
    st.info("Choose the race track and upload the DraftKings salary file after qualifying to enable one-click NASCAR building.")
