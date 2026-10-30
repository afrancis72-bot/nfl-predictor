"""PGA in the same Team AI² Pro app as NFL, NBA and NASCAR."""
from __future__ import annotations
from datetime import date
from io import BytesIO
import hashlib
import secrets

import pandas as pd
import streamlit as st

from ai2_pga_oneclick import (
    parse_dk, run_tournament,dk_roster_ids, evaluate_groups,
    build_round_showdown,normalize_groupings_upload,normalize_live_results,
)
from round_parlay import fetch_groupings,fetch_official_prior_rounds
from international_ingestion import geocode_course, location_detail, fetch_weather

st.subheader("PGA · Tournament DFS, Round Showdown & 3-Ball Parlays")
st.caption(
    "One unified golf workspace. Tournament DFS uses the existing V11 "
    "hole-by-hole scoring simulator. Live round props use published groupings "
    "and actual completed-round evidence."
)

tab_tournament,tab_parlay,tab_showdown,tab_audit=st.tabs([
    "🏆 Tournament DFS · One Click",
    "🎯 Round 3-Balls & 6-Leg Research",
    "⛳ Round Showdown",
    "🧬 Course DNA & Data Quality",
])
with tab_tournament:
    top=st.columns([1.1,1.2,.8])
    with top[0]:
        tournament=st.text_input("1. Tournament",placeholder="Current PGA TOUR tournament",key="ai2_pga_tournament")
    with top[1]:
        course=st.text_input("Course / venue",placeholder="Actual tournament course",key="ai2_pga_course")
    with top[2]:
        start_date=st.date_input("Tournament start",value=date.today(),key="ai2_pga_start_date")
    c1,c2,c3=st.columns(3)
    with c1:
        cut_type=st.selectbox("Tournament format",["Standard cut after Round 2","No cut (all four rounds)"],key="pga_cut")
    with c2:
        course_par=int(st.selectbox("Course par",[70,71,72,73],index=2,key="pga_par"))
    with c3:
        sims=int(st.selectbox("Tournament simulations",[1000,3000,5000],index=1,key="pga_sims"))
    dk_file=st.file_uploader("2. DraftKings golf salary slate — current event",
            type=["csv"],key="ai2_pga_dk")
    dk=None
    if dk_file is not None:
        try:
            dk=parse_dk(BytesIO(dk_file.getvalue()))
            st.success(f"{len(dk)} current DraftKings golfers loaded; salaries checked.")
        except Exception as exc:
            st.error(f"DraftKings field check: {exc}")
    with st.expander("Optional golfer research, course pars and advanced portfolio settings"):
        independent_file=st.file_uploader(
            "RickRunGood / independent SG Stats Bank CSV (preferred measured golfer input)",
            type=["csv"],key="ai2_pga_independent")
        rrg_fit_file=st.file_uploader(
            "RickRunGood Course Fit / Player Fit CSV (optional, separate from SG)",
            type=["csv"],key="ai2_pga_rrg_fit")
        st.caption(
            "RRG course-fit exports: Player plus FIT, or the OTT / APP / ARG / PUTT / "
            "SCORING skill-fit columns. When components exist, Team AI² Pro uses "
            "its own category weights (APP 35%, OTT 25%, scoring 18%, putting 14%, "
            "ARG 8%) rather than copying the site's overall weighting. "
            "The adjustment is deliberately small and shown per golfer."
        )
        otis_file=st.file_uploader("OTIS Advanced Course-Fit CSV (optional comparison and fallback)",
                                type=["csv"],key="ai2_pga_otis")
        st.caption(
            "Measured RRG strokes gained takes priority over OTIS in the independent golfer model. "
            "OTIS is optional; without measured golfer skill/form/course fit, "
            "the one-button DK-only simulator is a clearly labeled "
            "salary-market research baseline, NOT an independently calibrated forecast. "
            "OTIS ranking/model columns are audit-only, not forecast inputs."
        )
        holes_file=st.file_uploader("Verified 18-hole par CSV (optional)",
                                 type=["csv"],key="ai2_pga_holes")
        st.caption("If hole pars are missing, the V11 engine uses a generic par-matched layout. It is provisional.")
        st.markdown("#### Optional automatic course weather")
        country_hint=st.text_input("Course country / region hint",key="pga_country_hint",
                                   placeholder="Mexico, Japan, Florida, etc.")
        @st.cache_data(ttl=21600,show_spinner=False)
        def _cached_course_search(q,hint):
            return geocode_course(q,hint,limit=5)
        @st.cache_data(ttl=3600,show_spinner=False)
        def _cached_verified_weather(selected,tournament_name,tournament_date):
            return fetch_weather(selected,tournament_name,tournament_date,days=4)
        if st.button("Find course / weather coordinates",key="pga_find_course",
                     disabled=not course.strip()):
            try:
                with st.spinner("Checking course location matches…"):
                    matches=_cached_course_search(course,country_hint)
                st.session_state["ai2_pga_course_matches"]=matches
                st.session_state["ai2_pga_course_matches_query"]=(course,country_hint)
                if not matches:
                    st.warning("Could not verify the course location. Weather will remain unavailable.")
            except Exception as exc:
                st.session_state["ai2_pga_course_matches"]=[]
                st.warning("Course geocoding unavailable: "+str(exc))
        matches=(st.session_state.get("ai2_pga_course_matches",[])
                 if st.session_state.get("ai2_pga_course_matches_query")==(course,country_hint)
                 else [])
        selected_location=None
        if matches:
            selected_index=int(st.selectbox(
                "Verify actual course location before using forecast",
                list(range(len(matches))),
                format_func=lambda n:(str(matches[n].get("display_name")
                                         or matches[n].get("name") or "Unknown course")),
                key="pga_course_location_choice"))
            selected_location=location_detail(matches[selected_index])
            st.caption("Wind/temperature come from Open-Meteo for this explicitly selected location. "
                       "Geocoded locations and weather are not proof of actual tournament tee waves.")

        lineup_count=int(st.number_input("Number of DFS lineups",1,30,10,step=1,key="pga_lineup_count"))
        exposure_pct=int(st.slider("Maximum golfer exposure (%)",10,100,35,5,key="pga_exp"))
        min_unique=int(st.slider("Minimum unique golfers between lineups",1,5,3,key="pga_unique"))
        salary_floor=int(st.number_input("Minimum total lineup salary",0,50000,46500,100,key="pga_floor"))
        strategy=st.selectbox("Portfolio objective",
                    ["Balanced / Single Entry","GPP Ceiling","Cut Equity"],
                    key="pga_strategy")
        cut_size=int(st.number_input("Standard cut: approximate position (ties included)",
                    1,100,65,key="pga_cut_size",disabled=cut_type.startswith("No cut")))
        if dk is not None:
            golfer_names=dk["player"].tolist()
            locks=st.multiselect("Lock golfers (optional)",golfer_names,key="pga_locks")
            exclusions=st.multiselect("Exclude golfers (optional)",
                                      [x for x in golfer_names if x not in locks],key="pga_exclusions")
        else:
            locks=[]; exclusions=[]
        seed_mode=st.selectbox("Seed mode",["Fixed","Fibonacci preset","Fresh random"],
                          key="pga_seed_mode")
        if seed_mode=="Fixed":
            chosen_seed=int(st.number_input("Simulation seed",0,2147483647,42,step=1,key="pga_seed"))
        elif seed_mode=="Fibonacci preset":
            chosen_seed=int(st.selectbox("Fibonacci seed",
                [1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181],
                index=7,key="pga_fib"))
        else:
            chosen_seed=None
            st.caption("A new random seed is selected at each run.")
    setup_signature=(
        hashlib.sha256(dk_file.getvalue()).hexdigest() if dk_file else None,
        tournament,course,str(start_date),cut_type,course_par,sims,lineup_count,
        exposure_pct,min_unique,salary_floor,strategy,cut_size,
        tuple(locks),tuple(exclusions),seed_mode,chosen_seed,
        hashlib.sha256(otis_file.getvalue()).hexdigest() if otis_file else None,
        hashlib.sha256(independent_file.getvalue()).hexdigest() if independent_file else None,
        hashlib.sha256(rrg_fit_file.getvalue()).hexdigest() if rrg_fit_file else None,
        hashlib.sha256(holes_file.getvalue()).hexdigest() if holes_file else None,
        tuple(sorted(selected_location.items())) if selected_location else None,
    )
    do_run=st.button("⛳ Run PGA simulation + build DFS lineups",
                     type="primary",use_container_width=True,disabled=dk is None,
                     key="pga_build")
    if do_run:
        if not tournament.strip() or not course.strip():
            st.error("Enter the real tournament and course first.")
        else:
            try:
                otis_raw=pd.read_csv(BytesIO(otis_file.getvalue())) if otis_file else None
                independent_raw=pd.read_csv(BytesIO(independent_file.getvalue())) if independent_file else None
                rrg_fit_raw=pd.read_csv(BytesIO(rrg_fit_file.getvalue())) if rrg_fit_file else None
                holes=pd.read_csv(BytesIO(holes_file.getvalue())) if holes_file else None
                seed=int(chosen_seed) if chosen_seed is not None else secrets.randbelow(2147483647)
                forecast=pd.DataFrame()
                if selected_location is not None:
                    try:
                        with st.spinner("Retrieving the selected course's Open-Meteo weather forecast…"):
                            forecast=_cached_verified_weather(selected_location,tournament,start_date)
                    except Exception as weather_error:
                        st.warning("Verified course weather unavailable; treating weather effect as unknown: "
                                   +str(weather_error))
                with st.spinner(f"Simulating {sims:,} 72-hole tournament worlds and optimizing DFS portfolio…"):
                    result=run_tournament(dk,tournament,course_par,
                        no_cut=cut_type.startswith("No cut"),
                        cut_size=cut_size,hole_frame=holes,otis_raw=otis_raw,
                        independent_raw=independent_raw,course_fit_raw=rrg_fit_raw,
                        weather_raw=forecast,
                        sims=sims,seed=seed,lineup_count=lineup_count,
                        max_exposure=exposure_pct/100,
                        salary_floor=salary_floor,min_unique=min_unique,
                        strategy=strategy,locks=locks,excludes=exclusions)
                result["metadata"]["course"]=course
                result["metadata"]["start_date"]=str(start_date)
                st.session_state["ai2_pga_result"]=result
                st.session_state["ai2_pga_signature"]=setup_signature
                st.session_state.pop("ai2_pga_3ball",None)
                st.session_state.pop("ai2_pga_showdown",None)
                st.success(f"Built {len(result['summary'])} lineups using seed {seed}.")
            except Exception as exc:
                st.session_state.pop("ai2_pga_result",None)
                st.session_state.pop("ai2_pga_signature",None)
                st.error(f"PGA builder blocked: {exc}")
                st.caption("No incomplete, illegally priced, or duplicate-field portfolios are exported.")
    result=st.session_state.get("ai2_pga_result")
    if result is not None and st.session_state.get("ai2_pga_signature")!=setup_signature:
        result=None
        st.info("The tournament, field, or settings have changed. Re-run to refresh the results.")
    if result:
        meta=result["metadata"]
        st.divider()
        st.subheader("Built portfolio — human review before entering")
        st.caption(f"{meta['tournament']} · {meta['course']} · {meta['simulations']:,} simulations · seed {meta['seed']}")
        if meta["otis_match"]==0 and meta["independent_sg_match"]==0:
            st.warning(
                "LOW-CONFIDENCE RESEARCH: golfer quality is inferred only from DraftKings salary. "
                "Independent measured SG, injuries and ownership are NOT available in this run. "
                "An optional RRG course-fit tilt alone does not establish golfer skill. "
                "Do not treat the modeled probabilities as verified edge."
            )
        else:
            st.caption(
                f"Observed independent SG: {meta['independent_sg_match']}/{meta['field_size']} golfers. "
                f"OTIS course-fit available: {meta['otis_match']}/{meta['field_size']}. "
                f"RRG course fit: {meta['rrg_fit_match']}/{meta['field_size']} "
                f"({meta['rrg_fit_components']} with skill categories). "
                "V11 scoring probabilities remain experimental and have not been calibrated."
            )
        if meta["rrg_fit_match"]:
            st.caption(
                "RickRunGood course-fit tilt is a *provisional* capped adjustment "
                "to per-round scoring expectation, not an independently validated prediction. "
                "Its FIT-only fallback receives reduced weight."
            )
        if not meta["hole_pars_verified"]:
            st.info("Generic provisional hole pars were used; verify actual course layout to strengthen the model.")
        if meta["weather_hours"]:
            st.caption(f"Selected-course Open-Meteo forecast loaded: {meta['weather_hours']} hourly readings; "
                       f"mean wind {meta['weather_wind_mph']} mph. "
                       "The original V10 model can use aggregate weather only when measured golfer inputs are present. "
                       "No individual tee-wave assignments were inferred.")
        else:
            st.caption("No verified location-specific current weather was loaded.")
        st.dataframe(result["summary"],hide_index=True,use_container_width=True)
        st.dataframe(result["lineups"],hide_index=True,use_container_width=True)
        st.download_button("Download lineup review CSV",
            result["lineups"].to_csv(index=False).encode("utf-8-sig"),
            "team_ai2_pga_lineup_review.csv",mime="text/csv")
        try:
            roster_bytes=dk_roster_ids(result["lineups"],result["projections"])
            st.download_button("Download six DraftKings golfer IDs",
                roster_bytes,"team_ai2_pga_dk_rosters.csv",mime="text/csv")
            st.caption("Six G columns of real DK golfer IDs; verify your contest's DKEntries template before uploading.")
        except ValueError as err:
            st.caption("DraftKings entry-ID export unavailable: "+str(err))
        st.subheader("Exposure")
        st.dataframe(result["exposure"],hide_index=True,use_container_width=True)
        st.download_button("Download golfer projections",
            result["projections"].to_csv(index=False).encode("utf-8-sig"),
            "team_ai2_pga_simulated_projections.csv",mime="text/csv")

with tab_parlay:
    st.subheader("PGA TOUR 3-Ball Matchups · 6-Leg Ticket Research")
    st.caption(
        "Official round groupings and completed-round scores are attempted automatically. "
        "Parlay win/push probabilities are research estimates, not sportsbook prices. "
        "No automatic betting or guaranteed edges."
    )
    cols=st.columns([1,1,1,1])
    round_no=int(cols[0].selectbox("Round",[1,2,3,4],index=1,key="ai2_pga_group_round"))
    group_sims=int(cols[1].selectbox("Group simulations",[10000,25000,50000],index=1,key="pga_group_sims"))
    ticket_count=int(cols[2].number_input("6-leg ticket candidates",1,10,3,key="pga_ticket_count"))
    run_seed=int(cols[3].number_input("Group seed",0,2147483647,42,key="pga_group_seed"))
    with st.expander("Optional source fallbacks (only when official data fail)"):
        tournament_id=st.text_input("PGA TOUR tournament ID (if known)",key="pga_tid")
        groups_upload=st.file_uploader("Verified tee groupings CSV",type=["csv"],key="pga_group_csv")
        results_upload=st.file_uploader("Completed earlier rounds CSV",type=["csv"],key="pga_prior_results_csv")
    group_signature=(result["metadata"]["created_utc"] if result else None,
                     round_no,group_sims,ticket_count,run_seed,tournament,tournament_id,
                     hashlib.sha256(groups_upload.getvalue()).hexdigest() if groups_upload else None,
                     hashlib.sha256(results_upload.getvalue()).hexdigest() if results_upload else None)
    do_groups=st.button("Research this round's 3-balls",type="primary",
                        key="pga_research_group",use_container_width=True,
                        disabled=result is None)
    if do_groups:
        try:
            if groups_upload is not None:
                groupings=normalize_groupings_upload(pd.read_csv(BytesIO(groups_upload.getvalue())),round_no)
                source="Verified grouping CSV upload"
            else:
                with st.spinner("Retrieving official PGA TOUR round groupings…"):
                    groupings,source,note=fetch_groupings(tournament,int(start_date.year),round_no,
                                                         tournament_id=tournament_id)
            if groupings.empty:
                raise ValueError("No valid three-golfer round groups available. Upload verified tee times as a fallback.")
            live=pd.DataFrame()
            live_source="Round 1: pre-tournament information only"
            if round_no>1:
                if results_upload:
                    live=normalize_live_results(pd.read_csv(BytesIO(results_upload.getvalue())))
                    live_source="User-uploaded completed-round scores"
                else:
                    with st.spinner("Retrieving official PGA TOUR completed round scoring…"):
                        live,live_source=fetch_official_prior_rounds(
                            tournament,int(start_date.year),round_no,tournament_id)
                if live.empty:
                    raise ValueError("No verified completed-round scores for this round. Cannot present a live-informed matchup prediction.")
            analyzed=evaluate_groups(groupings,result,int(round_no),live=live,
                                     seed=run_seed,sims=group_sims,tickets=ticket_count)
            analyzed["metadata"]={
                "round":round_no,"groupings_source":source,
                "completed_rounds_source":live_source,
                "tournament":tournament,"seed":run_seed,
            }
            analyzed["live"]=live
            analyzed["signature"]=group_signature
            st.session_state["ai2_pga_3ball"]=analyzed
            st.success(f"Evaluated {len(groupings)} groups for Round {round_no}.")
        except Exception as exc:
            st.session_state.pop("ai2_pga_3ball",None)
            st.error(str(exc))
    grp=st.session_state.get("ai2_pga_3ball")
    if grp and result and grp.get("signature")==group_signature:
        st.caption(f"Groupings: {grp['metadata']['groupings_source']} · "
                   f"Prior scores: {grp['metadata']['completed_rounds_source']} · "
                   f"Ratings: {grp['source']}")
        st.subheader("Individual three-ball probabilities")
        st.dataframe(grp["probabilities"],use_container_width=True,hide_index=True)
        st.download_button("Download 3-ball estimates",
            grp["probabilities"].to_csv(index=False).encode("utf-8-sig"),
            f"team_ai2_pga_r{round_no}_groups.csv",mime="text/csv")
        st.subheader("Six-leg candidate research")
        if grp["ticket_legs"].empty:
            st.info("Fewer than six fully matched three-ball groups. No forced six-leg ticket.")
        else:
            st.dataframe(grp["ticket_summary"],hide_index=True,use_container_width=True)
            st.dataframe(grp["ticket_legs"],hide_index=True,use_container_width=True)
            st.caption("The displayed full-ticket probabilities assume approximate independence, and do not incorporate sportsbook odds. Ties in 3-balls can push a leg.")
            st.download_button("Download researched 6-leg candidates",
                grp["ticket_legs"].to_csv(index=False).encode("utf-8-sig"),
                f"team_ai2_pga_r{round_no}_six_legs.csv",mime="text/csv")
    elif result is None:
        st.info("Run Tournament DFS once to establish the current field and player strength before round research.")

with tab_showdown:
    st.subheader("Single-Round DraftKings Showdown")
    st.caption("Uses the round-specific DraftKings salary slate plus published prior-round scoring. Not a calibrated round DK point distribution.")
    if not result:
        st.info("Build the main tournament model first, then use Showdown.")
    else:
        sd_round=int(st.selectbox("Showdown round",[2,3,4],key="pga_showdown_round"))
        sd_file=st.file_uploader("DraftKings round-specific Showdown salary CSV",
                                type=["csv"],key="pga_showdown_file")
        sd_count=int(st.number_input("Showdown lineups",1,20,4,key="pga_showdown_count"))
        sd_exp=float(st.slider("Showdown max exposure (%)",10,100,50,5,key="pga_showdown_exposure"))/100
        sd_floor=int(st.number_input("Showdown minimum salary",0,50000,46500,100,key="pga_showdown_floor"))
        sd_prior=st.file_uploader("Optional verified completed-round scores CSV",
                                 type=["csv"],key="pga_showdown_live")
        sd_signature=(result["metadata"]["created_utc"],sd_round,sd_count,sd_exp,sd_floor,
                      hashlib.sha256(sd_file.getvalue()).hexdigest() if sd_file else None,
                      hashlib.sha256(sd_prior.getvalue()).hexdigest() if sd_prior else None)
        if st.button("Build round Showdown",type="primary",disabled=sd_file is None,key="pga_run_showdown"):
            try:
                if sd_prior is not None:
                    live=normalize_live_results(pd.read_csv(BytesIO(sd_prior.getvalue())))
                elif (st.session_state.get("ai2_pga_3ball") or {}).get("metadata",{}).get("round")==sd_round:
                    live=st.session_state["ai2_pga_3ball"]["live"]
                else:
                    with st.spinner("Retrieving real completed rounds…"):
                        live,_=fetch_official_prior_rounds(
                            tournament,int(start_date.year),sd_round,"")
                if live is None or live.empty:
                    raise ValueError("Completed prior-round scores unavailable. Upload genuine data; no simulated substitute.")
                sd=build_round_showdown(
                    pd.read_csv(BytesIO(sd_file.getvalue())),result,sd_round,live,
                    count=sd_count,exposure=sd_exp,salary_floor=sd_floor,
                )
                st.session_state["ai2_pga_showdown"]=sd
                st.session_state["ai2_pga_showdown_round"]=sd_round
                st.session_state["ai2_pga_showdown_signature"]=sd_signature
                st.success(f"Built {len(sd['lineups'])} round-{sd_round} lineups.")
            except Exception as exc:
                st.session_state.pop("ai2_pga_showdown",None)
                st.error(str(exc))
        sd=st.session_state.get("ai2_pga_showdown")
        if sd is not None and st.session_state.get("ai2_pga_showdown_signature")==sd_signature:
            st.dataframe(sd["lineups"],hide_index=True,use_container_width=True)
            st.dataframe(sd["exposure"],hide_index=True,use_container_width=True)
            st.download_button("Download Showdown lineup review",
                sd["lineups"].to_csv(index=False).encode("utf-8-sig"),
                f"team_ai2_pga_r{sd_round}_showdown.csv",mime="text/csv")

with tab_audit:
    st.subheader("Course DNA & Data Quality")
    if not result:
        st.info("Run a tournament simulation to see the actual driver-independent golfer baselines and data provenance.")
    else:
        meta=result["metadata"]
        st.write(f"Model source: **{meta['source_mode']}**")
        st.write(f"Verified 18 hole pars: **{'Yes' if meta['hole_pars_verified'] else 'No — generic provisional layout'}**")
        st.write(f"Selected-course Open-Meteo weather: **{meta['weather_hours']} hourly observations**; "
                 "individual tee wave impacts: **not inferred**.")
        st.write("Observed independent golfer strokes gained: "
                 f"**{meta['independent_sg_match']}/{meta['field_size']} matched golfers**. "
                 "Measured input overrides OTIS model scores; no invented SG.")
        st.write(
            f"Optional RickRunGood Course Fit: **{meta['rrg_fit_match']}/{meta['field_size']} matched** "
            f"({meta['rrg_fit_components']} component-based, "
            f"{meta['rrg_fit_overall_only']} FIT-only fallback)."
        )
        if meta["rrg_fit_match"]:
            st.caption("Independent Team AI² Pro category weights: APP 35%, OTT 25%, "
                       "SCORING 18%, PUTT 14%, ARG 8%; small capped per-round tilts.")
            fitcols=[c for c in ["player","rrg_fit_app","rrg_fit_ott",
                  "rrg_fit_arg","rrg_fit_putt","rrg_fit_scoring","rrg_fit_overall",
                  "rrg_fit_signal_z","rrg_fit_adjustment_strokes",
                  "rrg_fit_evidence","expected_round_before_rrg_fit",
                  "expected_round"] if c in result["projections"].columns]
            st.dataframe(result["projections"][fitcols],
                         use_container_width=True,hide_index=True)
            st.download_button(
                "Download RRG course-fit impact audit",
                result["projections"][fitcols].to_csv(index=False).encode("utf-8-sig"),
                "team_ai2_pga_rrg_course_fit_impact.csv",mime="text/csv"
            )
        show=["player","salary","expected_round","round_sd","v11_mean_dk",
              "v11_median_dk","v11_p90_dk","v11_p95_dk",
              "v11_make_cut_pct","otis_true_skill","otis_form","otis_course_fit"]
        available=[x for x in show if x in result["projections"].columns]
        st.dataframe(result["projections"][available],hide_index=True,use_container_width=True)
        st.caption("Existing V11 scoring probabilities and lineup optimization are experimental. "
                   "The app does not yet use measured hole-level SG-by-distance, fully correlated lineup optimization, "
                   "current ownership estimates or verified contest ROI.")
