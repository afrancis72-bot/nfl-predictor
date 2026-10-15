"""Experimental Streamlit page. Production app remains untouched."""
import io
import pandas as pd
import numpy as np
import streamlit as st
from historical_usage import download_weekly_stats, build_usage_priors, apply_usage_priors, usage_audit
from snap_level_research import simulate_snaps, summarize_player_props

st.title("Snap-by-snap Research Lab")
st.warning("Experimental: uncalibrated simulation. Do not use for wagering or live entries.")
st.caption("Historical data is restricted to weeks before the selected matchup. Upload a pre-lock salary pool with one row per player: Name, TeamAbbrev, Position, FLEX Salary. For DK showdown exports, prepare one FLEX row per player.")
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
            pool=pd.read_csv(salary_file)
            required={"Name","TeamAbbrev","Position","FLEX Salary"}
            if not required.issubset(pool):
                raise ValueError("Pool must contain: "+", ".join(sorted(required)))
            pool=pool.drop_duplicates(["Name","TeamAbbrev"]).copy()
            pool["TeamAbbrev"]=pool["TeamAbbrev"].astype(str).str.upper().str.strip()
            pool["Position"]=pool["Position"].astype(str).str.upper().str.strip()
            if pool["TeamAbbrev"].nunique()!=2: raise ValueError("Pool must contain exactly two teams")
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
            pts,stats=simulate_snaps(selected,teams,n_games=int(games),seed=316)
            st.session_state["snap_summary"]=summarize_player_props(stats)
            st.session_state["snap_results"]=(selected.reset_index(drop=True),pts)
            st.session_state["snap_results_metadata"]={"season":int(season),"week":int(week),"teams":teams}
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
if "snap_summary" in st.session_state:
    summary=st.session_state["snap_summary"]
    st.subheader("Simulated player statistics")
    st.dataframe(summary,use_container_width=True)
    st.download_button("Download player means, medians and P90",summary.to_csv(index=False),"snap_player_statistics.csv","text/csv")
    pool,pts=st.session_state["snap_results"]
    salaries=pd.to_numeric(pool["FLEX Salary"],errors="coerce").to_numpy()
    eligible=np.where(salaries>=2000)[0]
    mean=pts.mean(axis=0)
    p90=np.quantile(pts,.90,axis=0)
    candidates=[]
    # Enumerate captain seeds and diversify greedily. This is a research heuristic,
    # not the production contest-ROI portfolio optimizer.
    for cap in sorted(eligible,key=lambda i:-p90[i])[:min(20,len(eligible))]:
        cap_cost=int(round(salaries[cap]*1.5))
        if cap_cost>50000: continue
        others=sorted([i for i in eligible if i!=cap],key=lambda i:-(.7*mean[i]+.3*p90[i]))
        selected=[cap];spent=cap_cost
        for idx in others:
            if len(selected)==6:break
            if spent+salaries[idx]<=50000:
                selected.append(idx);spent+=salaries[idx]
        if len(selected)!=6:continue
        if len(set(pool.iloc[selected]["TeamAbbrev"]))!=2:continue
        score=float(1.5*mean[cap]+sum(mean[i] for i in selected[1:]))
        candidates.append((score,selected,int(spent)))
    chosen=[];captains=set()
    for score,ids,cost in sorted(candidates,reverse=True):
        if ids[0] in captains:continue
        chosen.append({"CPT":pool.iloc[ids[0]]["Name"],**{f"FLEX{i}":pool.iloc[j]["Name"] for i,j in enumerate(ids[1:],1)},"Salary":cost,"Simulated mean DK":round(score,2)})
        captains.add(ids[0])
        if len(chosen)==5:break
    st.subheader("Five research lineups")
    if len(chosen)<5:
        st.warning(f"Only {len(chosen)} feasible captain-diversified research lineups found. K/DST not modeled; no fabricated players added.")
    if chosen:
        lineups=pd.DataFrame(chosen)
        st.dataframe(lineups,use_container_width=True)
        st.download_button("Download research lineups",lineups.to_csv(index=False),"snap_five_research_lineups.csv","text/csv")
    st.info("Research-only: missing real-time depth chart, injuries, defensive stats, and historical calibration. DK IDs are not included, so this is NOT a DK entry upload file.")
