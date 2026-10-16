"""Team AI² experimental NFL classic main-slate snap research page.

The page deliberately fails closed on unverified usage and unknown starters.
"""
import io
import time
import secrets
import pandas as pd
import streamlit as st
from classic_lineup_builder import build_classic_lineups
from classic_slate_research import prepare_classic_slate
from classic_snap_runner import run_classic_snap_slate
from historical_usage import download_weekly_stats, build_usage_priors, apply_usage_priors, usage_audit

st.title("Main Slate Simulator — Research")
st.caption("Experimental snap-by-snap research. No calibrated DST projection or entry-ready optimizer.")
season=st.number_input("Season",min_value=1999,max_value=2100,value=2026,step=1)
week=st.number_input("NFL week",min_value=1,max_value=22,value=6,step=1)
n_games=st.number_input("Simulated games per matchup",min_value=1,max_value=100000,value=10,step=10)
seed_mode=st.radio("Simulation randomness",["Reproducible","Fresh random sample"],horizontal=True)
st.caption("A seed only controls the random sequence. Any whole number works; choose Fresh random sample to let the app pick automatically.")
fixed_seed=st.number_input("Seed (editable in Reproducible mode)",min_value=0,max_value=2147483647,value=5100,step=1,disabled=seed_mode!="Reproducible",key="classic_sim_seed")
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
            started=time.perf_counter()
            run_seed=int(fixed_seed) if seed_mode=='Reproducible' else secrets.randbelow(2147483647)
            st.caption(f'Simulation seed: {run_seed}')
            stage=st.empty()
            progress=st.progress(0,text="Preparing historical data...")
            stage.info("Stage 1: loading historical data")
            def report_progress(done,total,matchup,status):
                progress.progress(done/total,text=f"{done}/{total} matchups finished — {matchup} {status}")
                stage.info(f"Stage 3: {matchup} {status}; elapsed {time.perf_counter()-started:.1f}s")
            with st.spinner("Running snap-level simulations..."):
                if weekly_file is not None:
                    weekly=pd.read_csv(weekly_file)
                else:
                    weekly=download_weekly_stats(int(season),"/tmp/teamai2_classic_weekly.csv")
                stage.info(f"Stage 2: validating historical usage; elapsed {time.perf_counter()-started:.1f}s")
                priors=build_usage_priors(weekly,as_of_week=int(week),season=int(season))
                diagnostic=usage_audit(apply_usage_priors(offense,priors))
                st.session_state["classic_usage_audit"]=diagnostic
                projections,stats,game_audit,excluded_players,omitted_players=run_classic_snap_slate(
                    salary,weekly,season=int(season),week=int(week),starting_qbs=starters,
                    excluded_names=[s.strip() for s in exclusions.splitlines() if s.strip()],
                    n_games=int(n_games),seed=run_seed,progress_callback=report_progress)
            st.session_state["classic_last_run"]={
                "projections":projections,"stats":stats,"omitted_players":omitted_players,
                "seed":run_seed,"n_games":int(n_games),"season":int(season),"week":int(week),
                "elapsed":time.perf_counter()-started}
            progress.progress(1.0,text="All matchups completed")
            stage.success(f"Finished in {time.perf_counter()-started:.1f} seconds")
    except Exception as exc:
        st.error(f"Validation stopped the run: {exc}")
    if "classic_last_run" in st.session_state:
        result=st.session_state["classic_last_run"]
        projections=result["projections"]
        stats=result["stats"]
        omitted_players=result["omitted_players"]
        st.success(f"Saved simulation: {result['n_games']} games/matchup, seed {result['seed']}, {result['elapsed']:.1f}s. Downloads will not rerun the simulation.")
        if not omitted_players.empty:
            st.warning(f"{len(omitted_players)} players omitted due to insufficient matched history. Research output is not entry-ready.")
            st.dataframe(omitted_players,use_container_width=True)
            st.download_button("Download omitted-player audit",omitted_players.to_csv(index=False),file_name="teamai2_omitted_players.csv",mime="text/csv",key="classic_download_omitted")
        team_games=stats.groupby(["matchup","team","game"],as_index=False)[["pass_att","rush_att","targets"]].sum()
        team_volume=team_games.groupby(["matchup","team"],as_index=False)[["pass_att","rush_att","targets"]].mean()
        team_volume["offensive_plays"]=team_volume["pass_att"]+team_volume["rush_att"]
        team_volume["pass_rate"]=team_volume["pass_att"]/team_volume["offensive_plays"].replace(0,float("nan"))
        team_volume["volume_warning"]=(team_volume["offensive_plays"]<40)|(team_volume["offensive_plays"]>85)|(team_volume["pass_rate"]<.30)|(team_volume["pass_rate"]>.75)
        st.subheader("Experimental Classic Lineup Builder")
        st.caption("Ownership is optional. Upload CSV with columns ID and ownership_pct (0–100). These are external estimates, not model-derived ownership.")
        ownership_file=st.file_uploader("Projected ownership CSV (optional)",type="csv",key="classic_ownership")
        ownership_data=pd.read_csv(ownership_file) if ownership_file is not None else None
        leverage_weight=st.slider("Ownership leverage strength (every fifth lineup)",0.0,10.0,4.0,0.5,disabled=ownership_data is None)
        st.caption("Nine-player DraftKings lineups with DST. DST projections below are YOUR assumptions, not snap-simulated predictions. Check injuries, eligibility and projections before entry.")
        dst_pool=salary_pool[salary_pool.Position.isin(["DST","D"])].copy()
        if dst_pool.empty:
            st.warning("No DST entries found in the uploaded DraftKings salary file.")
        else:
            with st.expander("DST point assumptions (editable)",expanded=True):
                dst_scores={}
                for _,d in dst_pool.sort_values("TeamAbbrev").iterrows():
                    dst_scores[d["TeamAbbrev"]]=st.number_input(
                        f'{d["TeamAbbrev"]} DST — ${int(d["Salary"]):,}',
                        min_value=-10.0,max_value=40.0,value=5.0,step=0.5,
                        key=f'classic_dst_{d["TeamAbbrev"]}')
            dst_options=["No DST lock"]+sorted(dst_scores)
            dst_lock=st.selectbox("Lock a defense (optional)",dst_options,
                                  index=dst_options.index("CAR") if "CAR" in dst_options else 0)
            lineup_count=st.number_input("Number of lineups",min_value=1,max_value=20,value=5)
            floor=st.number_input("Minimum total salary",min_value=0,max_value=50000,value=47500,step=100)
            exposure=st.slider("Maximum player exposure (%)",min_value=20,max_value=100,value=40,step=5,help="A locked DST is exempt. For five lineups, 40% means at most two appearances.")
            upside=st.slider("Tournament upside weight",min_value=0.0,max_value=1.0,value=0.15,step=0.05,help="Boosts 90th-percentile upside in every fourth lineup.")
            variance=st.slider("Balanced-lineup variability penalty",min_value=0.0,max_value=0.5,value=0.05,step=0.05,help="Penalizes the difference between mean and 90th-percentile points in other lineups.")
            if st.button("Build experimental nine-player lineups"):
                try:
                    picks=build_classic_lineups(
                        salary,projections,dst_points=dst_scores,count=int(lineup_count),
                        salary_floor=int(floor),ownership=ownership_data,leverage_weight=float(leverage_weight),max_exposure=exposure/100,
                        upside_weight=upside,variance_penalty=variance,
                        locked_dst=None if dst_lock=="No DST lock" else dst_lock,
                        excluded_names=[s.strip() for s in exclusions.splitlines() if s.strip()])
                    st.session_state["classic_lineups"]=picks
                except Exception as exc:
                    st.error(f"Lineup builder: {exc}")
            if "classic_lineups" in st.session_state:
                picks=st.session_state["classic_lineups"]
                st.dataframe(picks,use_container_width=True)
                exposures=picks.groupby(["ID","Name","team"],as_index=False).agg(lineups=("lineup","nunique"))
                exposures["exposure_pct"]=100*exposures["lineups"]/picks["lineup"].nunique()
                st.subheader("Player exposure audit")
                st.dataframe(exposures.sort_values(["lineups","Name"],ascending=[False,True]),use_container_width=True)
                st.download_button("Download exposure audit",exposures.to_csv(index=False),file_name="teamai2_classic_exposure.csv",mime="text/csv",key="classic_download_exposure")
                st.dataframe(picks.groupby("lineup").agg(salary=("salary","sum"),
                    projected_points=("projected_points","sum"),ownership_sum=("ownership_pct","sum")),use_container_width=True)
                exposure=picks.groupby(["ID","Name"],as_index=False).agg(lineups=("lineup","nunique"),projected_points=("projected_points","mean"),ownership_pct=("ownership_pct","mean"))
                exposure["exposure_pct"]=100*exposure["lineups"]/picks["lineup"].nunique()
                st.subheader("Portfolio exposure")
                st.dataframe(exposure.sort_values(["lineups","projected_points"],ascending=False),use_container_width=True)
                st.download_button("Download portfolio exposure CSV",exposure.to_csv(index=False),file_name="teamai2_classic_exposure.csv",mime="text/csv",key="classic_download_exposure")
                if ownership_data is not None:
                    matched=picks["ownership_pct"].notna().sum()
                    st.caption(f"Ownership coverage: {matched}/{len(picks)} lineup slots. Ownership sum is descriptive, not a calibrated duplication probability.")
                st.download_button("Download nine-player lineup CSV",
                    picks.to_csv(index=False),file_name="teamai2_experimental_classic_lineups.csv",
                    mime="text/csv",key="classic_download_lineups")
        st.subheader("Team opportunity sanity audit")
        st.dataframe(team_volume,use_container_width=True)
        st.download_button("Download team volume audit",team_volume.to_csv(index=False),file_name="teamai2_team_volume_audit.csv",mime="text/csv",key="classic_download_volume")
        if team_volume["volume_warning"].any():
            st.error("Unrealistic team opportunity volume detected. Projections are NOT entry-ready.")
        st.dataframe(projections,use_container_width=True)
        st.download_button("Download offensive projections CSV",projections.to_csv(index=False),file_name="teamai2_classic_snap_projections.csv",mime="text/csv",key="classic_download_projections")
        st.download_button("Download snap-level outcomes CSV",stats.to_csv(index=False),file_name="teamai2_classic_snap_outcomes.csv",mime="text/csv",key="classic_download_outcomes")
    if "classic_usage_audit" in st.session_state:
        st.subheader("Historical usage matching diagnostic")
        diag=st.session_state["classic_usage_audit"]
        st.metric("Players matched with 2+ games",int(diag["usage_verified"].sum()))
        st.dataframe(diag,use_container_width=True)
        st.download_button("Download usage diagnostic",diag.to_csv(index=False),file_name="teamai2_usage_diagnostic.csv",mime="text/csv")
