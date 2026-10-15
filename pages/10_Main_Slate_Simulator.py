"""Team AI² experimental NFL classic main-slate snap research page.

The page deliberately fails closed on unverified usage and unknown starters.
"""
import io
import pandas as pd
import streamlit as st
from classic_slate_research import prepare_classic_slate
from classic_snap_runner import run_classic_snap_slate
from historical_usage import download_weekly_stats, build_usage_priors, apply_usage_priors, usage_audit

st.title("Main Slate Simulator — Research")
st.caption("Experimental snap-by-snap research. No calibrated DST projection or entry-ready optimizer.")
season=st.number_input("Season",min_value=1999,max_value=2100,value=2026,step=1)
week=st.number_input("NFL week",min_value=1,max_value=22,value=6,step=1)
n_games=st.number_input("Simulated games per matchup",min_value=1,max_value=100000,value=10,step=10)
salary_file=st.file_uploader("DraftKings main slate salary CSV",type="csv",key="classic_salaries")
weekly_file=st.file_uploader("Historical weekly player stats CSV (optional)",type="csv",key="classic_history")
exclusions=st.text_area("Excluded players (one full name per line)",value="Jalen Hurts\nTanner McKee\nCole Payton")
if salary_file is not None:
    try:
        salary=pd.read_csv(salary_file)
        salary_pool,offense,audit,excluded=prepare_classic_slate(salary, ())
        st.subheader("Matchup and roster audit")
        st.dataframe(audit,use_container_width=True)
        st.caption("Select the actual starter even when excluded from DFS. Exclusions affect lineup eligibility, not simulation.")
        starters={}
        for matchup in sorted(offense.matchup.unique()):
            st.markdown(f"**{matchup}**")
            teams=matchup.split("@")
            cols=st.columns(2)
            for col,team in zip(cols,teams):
                options=["— Select confirmed starter —"]+sorted(
                    salary_pool.loc[(salary_pool.matchup==matchup)&(salary_pool.TeamAbbrev==team)&
                                (salary_pool.Position=="QB"),"Name"].unique().tolist())
                with col:
                    chosen=st.selectbox(f"{team} starting QB",options,key=f"starter_{matchup}_{team}")
                    if chosen!=options[0]: starters[team]=chosen
        ready=len(starters)==2*len(audit)
        if not ready: st.warning("Select one confirmed starting QB for every team before running.")
        if st.button("Run main slate snap simulations",disabled=not ready):
            progress=st.progress(0,text="Preparing historical data...")
            def report_progress(done,total,matchup,status):
                progress.progress(done/total,text=f"{done}/{total} matchups finished — {matchup} {status}")
            with st.spinner("Running snap-level simulations..."):
                if weekly_file is not None:
                    weekly=pd.read_csv(weekly_file)
                else:
                    weekly=download_weekly_stats(int(season),"/tmp/teamai2_classic_weekly.csv")
                priors=build_usage_priors(weekly,as_of_week=int(week),season=int(season))
                diagnostic=usage_audit(apply_usage_priors(offense,priors))
                st.session_state["classic_usage_audit"]=diagnostic
                projections,stats,game_audit,excluded_players,omitted_players=run_classic_snap_slate(
                    salary,weekly,season=int(season),week=int(week),starting_qbs=starters,
                    excluded_names=[s.strip() for s in exclusions.splitlines() if s.strip()],
                    n_games=int(n_games),progress_callback=report_progress)
            progress.progress(1.0,text="All matchups completed")
            st.success("Research simulation completed — incomplete player coverage; not entry-ready")
            if not omitted_players.empty:
                st.warning(f"{len(omitted_players)} players omitted due to insufficient matched history. This can include important players; do not treat results as calibrated.")
                st.dataframe(omitted_players,use_container_width=True)
                st.download_button("Download omitted-player audit",omitted_players.to_csv(index=False),file_name="teamai2_omitted_players.csv",mime="text/csv")
            st.dataframe(projections,use_container_width=True)
            st.download_button("Download offensive projections CSV",projections.to_csv(index=False),
                               file_name="teamai2_classic_snap_projections.csv",mime="text/csv")
            st.download_button("Download snap-level outcomes CSV",stats.to_csv(index=False),
                               file_name="teamai2_classic_snap_outcomes.csv",mime="text/csv")
    except Exception as exc:
        st.error(f"Validation stopped the run: {exc}")
    if "classic_usage_audit" in st.session_state:
        st.subheader("Historical usage matching diagnostic")
        diag=st.session_state["classic_usage_audit"]
        st.metric("Players matched with 2+ games",int(diag["usage_verified"].sum()))
        st.dataframe(diag,use_container_width=True)
        st.download_button("Download usage diagnostic",diag.to_csv(index=False),file_name="teamai2_usage_diagnostic.csv",mime="text/csv")
