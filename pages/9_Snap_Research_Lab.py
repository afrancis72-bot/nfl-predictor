"""Experimental Streamlit page. Production app remains untouched."""
import io
import pandas as pd
import numpy as np
import streamlit as st
from historical_usage import download_weekly_stats, build_usage_priors, apply_usage_priors, usage_audit
from snap_level_research import simulate_snaps, summarize_player_props
from snap_portfolio_research import build_research_portfolio
from defensive_priors_research import build_defense_profiles
from historical_replay_research import replay_game

def normalize_showdown_pool(raw):
    """Accept native DK Showdown salary exports or existing FLEX-only research pools."""
    df=raw.copy()
    df.columns=df.columns.astype(str).str.strip()
    if "TeamAbbrev" not in df and "Team" in df:
        df=df.rename(columns={"Team":"TeamAbbrev"})
    if "Name" not in df and "Name + ID" in df:
        df["Name"]=df["Name + ID"].astype(str).str.replace(r"\\s*\\(\\d+\\)\\s*$","",regex=True)
    if "Name" not in df:
        raise ValueError("Missing player Name (or Name + ID) in salary file")
    if "Position" not in df:
        raise ValueError("Missing Position in salary file")
    if "TeamAbbrev" not in df:
        raise ValueError("Missing TeamAbbrev in salary file")
    if "Roster Position" in df:
        roster=df["Roster Position"].astype(str).str.upper().str.strip()
        flex=roster.str.split("/").apply(lambda slots:"FLEX" in slots)
        df=df.loc[flex].copy()
        if df.empty:
            raise ValueError("No FLEX salary rows found in DK Showdown export")
    if "FLEX Salary" not in df:
        if "Salary" not in df:
            raise ValueError("Missing Salary or FLEX Salary column")
        df["FLEX Salary"]=pd.to_numeric(df["Salary"],errors="coerce")
    df["FLEX Salary"]=pd.to_numeric(df["FLEX Salary"],errors="coerce")
    if df["FLEX Salary"].isna().any():
        raise ValueError("Some FLEX salaries are missing or invalid")
    df["Name"]=df["Name"].astype(str).str.strip()
    df["TeamAbbrev"]=df["TeamAbbrev"].astype(str).str.upper().str.strip()
    df["Position"]=df["Position"].astype(str).str.upper().str.strip().replace({"D":"DST","DEF":"DST"})
    df=df.drop_duplicates(["Name","TeamAbbrev"]).copy()
    if df["TeamAbbrev"].nunique()!=2:
        raise ValueError("Showdown pool must contain exactly two teams")
    return df

st.title("Snap-by-snap Research Lab")
st.caption("Research build: 2026-10-14 / personnel-v1 / historical-replay-v1")
st.warning("Experimental: uncalibrated simulation. Do not use for wagering or live entries.")
st.caption("Historical data is restricted to weeks before the selected matchup. Upload the original DraftKings Showdown salary CSV directly (CPT/FLEX rows accepted), or a FLEX-only research pool.")
salary_file=st.file_uploader("Showdown FLEX player pool CSV",type=["csv"],key="snap_pool")
a,b,c=st.columns(3)
season=a.number_input("Season",2020,2030,2026,1)
week=b.number_input("Matchup week",1,22,6,1)
games=c.number_input("Research simulations",100,5000,1000,100)
source=st.radio("Historical data source",["Download nflverse weekly stats","Upload saved nflverse weekly CSV"],horizontal=True)
weekly_file=st.file_uploader("nflverse weekly CSV",type=["csv"],key="snap_weekly") if source.startswith("Upload") else None
if st.button("Load historical usage and audit",type="primary"):
    if salary_file is None:
        st.error("Upload the matchup FLEX pool first.")
    else:
        try:
            pool=normalize_showdown_pool(pd.read_csv(salary_file))
            if weekly_file is not None:
                weekly=pd.read_csv(weekly_file)
            else:
                weekly=download_weekly_stats(int(season),f"/tmp/nflverse_stats_player_week_{int(season)}.csv")
            priors=build_usage_priors(weekly,int(week),int(season))
            merged=apply_usage_priors(pool,priors)
            st.session_state.pop("snap_summary",None)
            st.session_state.pop("snap_results",None)
            st.session_state["snap_prepared_players"]=merged
            st.session_state["snap_weekly_audit"]=usage_audit(merged)
            st.session_state["snap_meta"]=(int(season),int(week))
        except Exception as exc:
            st.error(f"Data ingestion failed: {exc}")
st.subheader("Optional historical defense data")
st.caption("Upload nflverse play-by-play with season, week, defteam, posteam, play_type, yards_gained. Only earlier weeks are used. No sportsbook data enters the simulator.")
defense_file=st.file_uploader("Pregame play-by-play CSV for defensive priors",type=["csv"],key="snap_defense_pbp")
if defense_file is not None:
    try:
        profiles,def_audit=build_defense_profiles(pd.read_csv(defense_file),int(season),int(week))
        st.session_state["snap_defense_profiles"]=profiles
        st.session_state["snap_defense_meta"]=(int(season),int(week))
        st.dataframe(def_audit,hide_index=True,use_container_width=True)
        st.download_button("Download defense audit",def_audit.to_csv(index=False),"snap_defense_audit.csv","text/csv")
    except Exception as exc:
        st.session_state.pop("snap_defense_profiles",None)
        st.error(f"Defensive priors unavailable: {exc}")
if "snap_weekly_audit" in st.session_state:
    audit=st.session_state["snap_weekly_audit"]
    st.subheader("Historical usage audit")
    st.dataframe(audit,use_container_width=True)
    st.download_button("Download usage audit",audit.to_csv(index=False),"snap_usage_audit.csv","text/csv")
    st.caption("K/DST are not yet modeled. Historical usage does not establish active status or starting roles. Verify starters and inactive players independently.")
    prepared=st.session_state["snap_prepared_players"].copy()
    offensive=prepared[prepared["Position"].isin(["QB","RB","WR","TE"])].copy()
    missing=offensive[~offensive["usage_verified"].fillna(False)]
    if len(missing):
        st.warning(f"{len(missing)} offensive players lack two matched historical games. They are excluded by default, not treated as inactive.")
        st.dataframe(missing[["Name","TeamAbbrev","Position","games"]],hide_index=True)
    verified=offensive[offensive["usage_verified"].fillna(False)].copy()
    teams=sorted(offensive["TeamAbbrev"].unique())
    st.subheader("Confirm starters and eligible players")
    st.warning("Choose quarterbacks from independently verified starting-lineup information. This page cannot confirm today's starters or injuries.")
    selected_qbs={}
    for team in teams:
        qb_options=verified[(verified["TeamAbbrev"]==team)&(verified["Position"]=="QB")]["Name"].tolist()
        if not qb_options:
            st.error(f"No verified quarterback for {team}; simulation is blocked.")
            continue
        chosen_qb=st.selectbox(f"{team} confirmed starting quarterback",["Select a starter"]+qb_options,index=0,key=f"snap_qb_{team}")
        if chosen_qb!="Select a starter":
            selected_qbs[team]=chosen_qb
    other=verified[verified["Position"]!="QB"].copy()
    labels={f"{row.Name} | {row.TeamAbbrev} | {row.Position}":(row.Name,row.TeamAbbrev) for row in other.itertuples(index=False)}
    excluded=st.multiselect("Exclude confirmed OUT, IR, inactive, or otherwise ineligible players",
                            options=list(labels),default=[],key="snap_inactive_exclusions")
    excluded_ids={labels[k] for k in excluded}
    selected=verified[
        ((verified["Position"]!="QB") & ~verified.apply(lambda x:(x["Name"],x["TeamAbbrev"]) in excluded_ids,axis=1))
        | verified.apply(lambda x:x["Position"]=="QB" and selected_qbs.get(x["TeamAbbrev"])==x["Name"],axis=1)
    ].copy()
    st.caption(f"Simulation pool: {len(selected)} historically verified players after exclusions; {len(missing)} unverified offensive players excluded.")
    if len(selected):
        st.dataframe(selected[["Name","TeamAbbrev","Position","games","carries_per_game","targets_per_game"]],hide_index=True,use_container_width=True)
        st.download_button("Download selected player pool",selected.to_csv(index=False),"snap_selected_player_pool.csv","text/csv")
    can_run=len(selected_qbs)==2 and selected["TeamAbbrev"].nunique()==2
    if not can_run:
        st.info("Select one confirmed starting quarterback for each team to enable simulation.")
    if st.button("Run snap simulation and generate 5 research lineups",disabled=not can_run):
        try:
            for team in teams:
                t=selected[selected["TeamAbbrev"]==team]
                if not (t["rush_weight"]>0).any() or not (t["target_weight"]>0).any():
                    raise ValueError(f"{team}: no verified rushing or receiving opportunity after exclusions")
            profiles=st.session_state.get("snap_defense_profiles",{})
            if st.session_state.get("snap_defense_meta")!=(int(season),int(week)):
                profiles={}
            missing_def=[t for t in teams if t not in profiles]
            if missing_def:
                st.warning("Neutral defensive priors for: "+", ".join(missing_def)+". Upload historical play-by-play for measured matchup effects.")
            pts,stats=simulate_snaps(selected,teams,n_games=int(games),seed=316,defense_profiles=profiles)
            st.session_state["snap_summary"]=summarize_player_props(stats)
            st.session_state["snap_results"]=(selected.reset_index(drop=True),pts)
            st.session_state["snap_results_metadata"]={"season":int(season),"week":int(week),"teams":teams}
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
if "snap_summary" in st.session_state:
    summary=st.session_state["snap_summary"]
    st.subheader("QB passing validation")
    qb_pool=st.session_state["snap_results"][0]
    rows=[]
    for _,qb in qb_pool[qb_pool["Position"]=="QB"].iterrows():
        match=summary[(summary["Name"]==qb["Name"]) & (summary["Team"]==qb["TeamAbbrev"])]
        if match.empty: continue
        v=match.iloc[0]
        rows.append({"QB":qb["Name"],"Team":qb["TeamAbbrev"],
                     "Historical attempts/game":round(float(qb.get("attempts_per_game",0)),1),
                     "Historical pass yards/game":round(float(qb.get("passing_yards_per_game",0)),1),
                     "Simulated attempts/game":v["pass_att_mean"],
                     "Simulated pass yards/game":v["pass_yd_mean"]})
    if rows: st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.caption("Sportsbook lines are comparison benchmarks, not simulation inputs.")
    st.subheader("Simulated player statistics")
    st.dataframe(summary,use_container_width=True)
    st.download_button("Download player means, medians and P90",summary.to_csv(index=False),"snap_player_statistics.csv","text/csv")
    pool,pts=st.session_state["snap_results"]
    try:
        lineups=build_research_portfolio(pool,pts,count=5,seed=316)
        chosen=lineups.to_dict("records")
    except Exception as exc:
        st.error(f"Research portfolio construction failed: {exc}")
        chosen=[]
    st.subheader("Five research lineups")
    st.caption("Portfolio engine: scenario-sampled v2; export includes Portfolio research score.")
    if len(chosen)<5:
        st.warning(f"Only {len(chosen)} feasible diversified research lineups found. K/DST not modeled; no fabricated players added.")
    if chosen:
        lineups=pd.DataFrame(chosen)
        st.dataframe(lineups,use_container_width=True)
        st.download_button("Download research lineups",lineups.to_csv(index=False),"snap_five_research_lineups.csv","text/csv")
    st.info("Research-only: candidate lineups are sampled, not globally optimized. Scenario scores are not contest ROI or winning probabilities. Real-time depth charts, injuries, measured opponent defense profiles, and historical calibration remain missing. DK IDs are not included, so this is NOT a DK entry upload file.")

st.divider()
st.subheader("Historical game replay — held-out evaluation")
st.caption("Use a previously played matchup. Load its verified pregame roster above, and upload play-by-play plus observed player statistics. Observations are not passed into the simulator.")
actual_file=st.file_uploader("Held-out actual player stats (player, stat, actual)",type=["csv"],key="snap_replay_actual")
matchup_id=st.text_input("Historical matchup ID",value="",placeholder="2025_06_SEA_DEN")
if st.button("Run historical replay",disabled=actual_file is None or "snap_prepared_players" not in st.session_state):
    try:
        if defense_file is None:
            raise ValueError("Historical play-by-play CSV required for replay")
        if st.session_state.get("snap_meta")!=(int(season),int(week)):
            raise ValueError("Reload pregame player usage for the selected season/week")
        if len(selected_qbs)!=2:
            raise ValueError("Confirm both historical starting quarterbacks")
        if not matchup_id.strip():
            raise ValueError("Provide a unique historical matchup ID")
        if len(selected)==0 or not selected["usage_verified"].fillna(False).all():
            raise ValueError("Pregame player pool must be verified")
        report=replay_game(matchup_id.strip(),int(season),int(week),teams,selected,
                           pd.read_csv(defense_file),pd.read_csv(actual_file),
                           n_games=int(games),seed=316)
        st.session_state["snap_replay_report"]=report
    except Exception as exc:
        st.error(f"Historical replay failed: {exc}")
if "snap_replay_report" in st.session_state:
    report=st.session_state["snap_replay_report"]
    st.dataframe(report["metrics"],hide_index=True,use_container_width=True)
    st.download_button("Download replay calibration report",
        report["metrics"].to_csv(index=False),"snap_replay_calibration.csv","text/csv")
    st.download_button("Download replay player errors",
        report["details"].to_csv(index=False),"snap_replay_player_errors.csv","text/csv")
    st.caption("A single matchup is diagnostic, not a sufficient calibration sample. Use multiple held-out games before drawing predictive conclusions.")
