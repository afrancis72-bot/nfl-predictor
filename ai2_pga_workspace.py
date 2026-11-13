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
    bookmaker_filtered_tickets,
)
from round_parlay import fetch_groupings,fetch_official_prior_rounds
from international_ingestion import geocode_course, location_detail, fetch_weather
from ai2_pga_review_ui import render_review_page
from ai2_pga_salary_review import rebuild_reviewed
from four_source_blend import SOURCE_DEFAULTS, CATEGORY_DEFAULTS, ADVANCED_DEFAULTS
from pga_official_season_stats import retrieve_official_season_stats

@st.cache_data(ttl=21600,show_spinner=False)
def _cached_official_pga_season(player_names,as_of,metric_names):
    return retrieve_official_season_stats(
        pd.DataFrame({"player":list(player_names)}),as_of,selected_stats=metric_names)

st.subheader("PGA · Tournament DFS, Round Showdown & Group-Matchup Parlays")
st.caption(
    "One unified golf workspace. Tournament DFS uses the existing V11 "
    "hole-by-hole scoring simulator. Live round props use published groupings "
    "and actual completed-round evidence."
)
st.caption("PGA DFS release 2026-11-03 • JOINT-SCENARIO PORTFOLIO OPTIMIZER • same-world six-of-six and ceiling audit")

tab_tournament,tab_review,tab_parlay,tab_showdown,tab_audit=st.tabs([
    "🏆 Tournament DFS · One Click",
    "🔎 Salary-Tier Review & Vetoes",
    "🎯 Round Matchups · 2 & 3 Balls",
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
        st.markdown("#### AI² Course-Weighted PGA Model")
        weighted_mode=st.checkbox("Enable adjustable PGA source and category weights",
            value=False,key="pga_four_source_active_v1",
            help="The existing PGA model remains unchanged when this is off. The new weights affect V11 scoring and the joint lineup optimizer.")
        weighted_file=st.file_uploader("RunGood weighted strokes gained CSV (SG OTT/APP/ARG/PUTT (w))",
            type=["csv"],key="pga_weighted_csv_v1",disabled=not weighted_mode)
        if weighted_mode:
            st.caption("Source weights (relative percentages; normalized automatically). With no RunGood Course Fit, set its share to zero.")
            src_col=st.columns(5)
            source_weights={
                "weighted_sg":float(src_col[0].number_input("Weighted SG",0.,100.,20.,1.,key="pga_src_weighted_v1")),
                "standard_sg":float(src_col[1].number_input("Standard SG",0.,100.,20.,1.,key="pga_src_standard_v1")),
                "otis_course_fit":float(src_col[2].number_input("OTIS fit",0.,100.,30.,1.,key="pga_src_otis_v1")),
                "rrg_course_fit":float(src_col[3].number_input("RunGood fit",0.,100.,
                    30. if rrg_fit_file is not None else 0.,1.,key="pga_src_rrg_v1")),
                "advanced_stats":float(src_col[4].number_input("Advanced stats",0.,100.,0.,1.,key="pga_src_advanced_v1")),
            }
            st.caption("Both SG groups use these category weights; native vendor Course Fit scores retain each provider's own internal weights.")
            cat_col=st.columns(4)
            category_weights={
                "sg_ott":float(cat_col[0].number_input("Off tee",0.,100.,25.,1.,key="pga_stat_ott_v1")),
                "sg_approach":float(cat_col[1].number_input("Approach",0.,100.,25.,1.,key="pga_stat_app_v1")),
                "sg_arg":float(cat_col[2].number_input("Around green",0.,100.,25.,1.,key="pga_stat_arg_v1")),
                "sg_putting":float(cat_col[3].number_input("Putting",0.,100.,25.,1.,key="pga_stat_putt_v1")),
            }
            advanced_labels={
                "birdie_rate":"Birdie or better %","driving_distance":"Driving distance",
                "driving_accuracy":"Driving accuracy %","gir_rate":"Greens in regulation %",
                "scrambling":"Scrambling %","bogey_avoidance":"Bogey avoidance %",
                "scoring_before_cut":"Average scoring before cut (Rounds 1-2 proxy)",
                "par3_scoring":"Par-3 scoring","par4_scoring":"Par-4 scoring",
                "par5_scoring":"Par-5 scoring",
                "approach_175_200":"Approach proximity 175-200y",
                "approach_200_225":"Approach proximity 200-225y",
            }
            with st.expander("Advanced PGA Tour stats (optional fifth data source)"):
                stats_grid=pd.DataFrame([{"Statistic":advanced_labels[k],"Key":k,"Weight":v}
                    for k,v in ADVANCED_DEFAULTS.items()])
                edited=st.data_editor(stats_grid,hide_index=True,num_rows="fixed",
                    disabled=["Statistic","Key"],key="pga_stats_grid_v1",
                    column_config={"Weight":st.column_config.NumberColumn(
                        "Relative weight",min_value=0.,max_value=100.,step=1.)})
                advanced_weights={str(z.Key):float(z.Weight) for z in edited.itertuples(index=False)}
                st.caption("Choose only statistics justified by the course. Early-round/Par scoring and proximity are lower-is-better. All missing data remains missing.")
            official_stats_enabled=st.checkbox("Automatically fill missing stats from PGA TOUR",
                value=True,key="pga_auto_tour_stats_v1",
                help="Unofficial PGA TOUR StatDetails client. Refuses historical-slate lookahead. RunGood supplied values take precedence.")
            sw_total=sum(source_weights.values())
            if sw_total:
                st.caption("Effective source shares: "+", ".join(
                    f"{key.replace('_',' ').title()} {100*value/sw_total:.1f}%"
                    for key,value in source_weights.items() if value))
            else:st.error("Source weights must sum above zero.")
        else:
            source_weights=SOURCE_DEFAULTS.copy()
            category_weights=CATEGORY_DEFAULTS.copy()
            advanced_weights=ADVANCED_DEFAULTS.copy()
            official_stats_enabled=False

        st.caption(
            "Measured RRG strokes gained takes priority over OTIS in the independent golfer model. "
            "OTIS is optional; without measured golfer skill/form/course fit, "
            "the one-button DK-only simulator is a clearly labeled "
            "salary-market research baseline, NOT an independently calibrated forecast. "
            "OTIS ranking/model columns are audit-only, not forecast inputs. " 
            "The RRG Stats Bank also contains Recent -1S to -8S and Hist YEAR finishes. "
            "Recent results receive only a SMALL uncalibrated adjustment; historical "
            "event results are shown as evidence, NOT automatically counted as SG."
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
        exposure_pct=int(st.slider("Maximum golfer exposure (%)",10,100,20,5,
                                    key="pga_exp_20261031_unified_v2",
                                    help="20% of 10 lineups = maximum two appearances per golfer. This control has a new key to clear the former 35% setting."))
        allowed=max(1,int((lineup_count*exposure_pct/100)//1))
        required_distinct=int((lineup_count*6+allowed-1)//allowed)
        st.caption(f"Portfolio exposure limit: no more than {allowed}/{lineup_count} lineups per golfer; "
                   f"at least {required_distinct} distinct golfers needed for 6-player lineups.")
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
        weighted_mode,
        hashlib.sha256(weighted_file.getvalue()).hexdigest() if weighted_file else None,
        tuple(source_weights.items()),tuple(category_weights.items()),
        tuple(advanced_weights.items()),bool(official_stats_enabled),
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
                weighted_sg_raw=(pd.read_csv(BytesIO(weighted_file.getvalue()))
                                 if weighted_file is not None and weighted_mode else None)
                official_season=pd.DataFrame()
                official_stat_audit=pd.DataFrame()
                if weighted_mode:
                    if sum(source_weights.values())<=0 or sum(category_weights.values())<=0:
                        raise ValueError("Source and SG-category totals must be positive.")
                    if source_weights["weighted_sg"]>0 and weighted_file is None:
                        raise ValueError("Upload weighted SG or set Weighted SG share to zero.")
                    if source_weights["rrg_course_fit"]>0 and rrg_fit_file is None:
                        raise ValueError("Upload RunGood Course Fit or set its share to zero.")
                    if source_weights["advanced_stats"]>0 and sum(advanced_weights.values())<=0:
                        raise ValueError("Assign at least one advanced statistic a nonzero weight.")
                    if official_stats_enabled:
                        requested=[]
                        if source_weights["standard_sg"]>0:
                            requested.extend(["sg_ott","sg_approach","sg_arg","sg_putting"])
                        if source_weights["advanced_stats"]>0:
                            requested.extend(k for k,v in advanced_weights.items() if v>0)
                        if requested:
                            with st.spinner("Retrieving PGA TOUR season statistics…"):
                                official_season,official_stat_audit=_cached_official_pga_season(
                                    tuple(dk.player.astype(str)),str(start_date),
                                    tuple(dict.fromkeys(requested)))
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
                        weather_raw=forecast,event_year=start_date.year,
                        sims=sims,seed=seed,lineup_count=lineup_count,
                        max_exposure=exposure_pct/100,
                        salary_floor=salary_floor,min_unique=min_unique,
                        strategy=strategy,locks=locks,excludes=exclusions,
                        four_source_mode=weighted_mode,weighted_sg_raw=weighted_sg_raw,
                        source_weights=source_weights,category_weights=category_weights,
                        advanced_weights=advanced_weights,official_stats=official_season,
                        official_stats_audit=official_stat_audit)
                result["metadata"]["course"]=course
                result["metadata"]["start_date"]=str(start_date)
                st.session_state["ai2_pga_result"]=result
                st.session_state["ai2_pga_signature"]=setup_signature
                st.session_state.pop("ai2_pga_review_portfolio",None)
                st.session_state.pop("ai2_pga_objectives_result",None)
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
        if meta.get("portfolio_selection")!="joint-scenario":
            st.warning(
                "This saved lineup run predates the joint-scenario portfolio "
                "optimizer. Re-run the PGA simulation to see newly selected "
                "cut/GPP/balanced lineups. Approved entries were not changed.")
        st.divider()
        st.subheader("Built portfolio — human review before entering")
        st.info("For salary-tier recommendations and veto decisions, open **🔎 Salary-Tier Review & Vetoes** above before entering lineups.")
        st.caption(f"{meta['tournament']} · {meta['course']} · {meta['simulations']:,} simulations · seed {meta['seed']}")
        if meta.get("four_source_enabled"):
            st.caption("PGA weights ACTIVE: they affect V11 simulated scoring and the joint lineup optimizer.")
            evidence=result.get("four_source",{})
            coverage=evidence.get("blend",{})
            if coverage:
                st.caption(f"Eligible golfers: {coverage.get('eligible_golfers',0)}/{coverage.get('total_golfers',0)}. "
                           f"Partial coverage: {coverage.get('partial_golfers',0)}. "
                           f"Max scoring adjustment: {coverage.get('max_abs_delta_strokes',0):.2f} strokes/round (uncalibrated).")
            with st.expander("Source, golfer and advanced PGA TOUR statistics audit"):
                st.json({k:v for k,v in evidence.items() if k!="official_stats_audit"})
                if evidence.get("official_stats_audit"):
                    st.dataframe(pd.DataFrame(evidence["official_stats_audit"]),
                        hide_index=True,use_container_width=True)
                selected=[v for v in ["player","salary","four_source_rank",
                    "four_source_source_coverage","four_source_status",
                    "four_source_weighted_sg_z","four_source_standard_sg_z",
                    "four_source_otis_course_fit_z","four_source_rrg_course_fit_z",
                    "four_source_advanced_stats_z","expected_round_pre_four_source",
                    "four_source_delta_strokes","expected_round","v11_mean_dk",
                    "v11_p90_dk","v11_make_cut_pct","optimization_eligible"]
                    if v in result["projections"]]
                st.dataframe(result["projections"][selected],hide_index=True,use_container_width=True)
                st.download_button("Download PGA source contribution audit",
                    result["projections"].to_csv(index=False).encode("utf-8-sig"),
                    "team_ai2_pga_four_source_evidence.csv",mime="text/csv",
                    key="pga_four_source_evidence_dl")
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
        if meta.get("recent_form_matched",0):
            st.caption(
                f"RunGood recent-result evidence used for {meta['recent_form_matched']}/"
                f"{meta['field_size']} golfers (up to eight previous starts); "
                f"event-history results available for {meta['historical_event_matched']}/"
                f"{meta['field_size']}. The per-round recent-form tilt is bounded "
                "to 0.12 strokes and is NOT historically calibrated. "
                "Tour levels and dates of individual starts were not verified. "
                "Event-history results are audit-only, not an assumed course-fit bonus."
            )
        elif meta["independent_sg_match"]:
            st.warning(
                "RunGood SG loaded, but no usable recent finishing history was identified. "
                "Recent five-to-eight-event results, cut trends and event-history records "
                "were NOT applied to golfer projections."
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
        st.caption(
            "Joint-scenario portfolio selection: Cut Equity seeks additional 6/6 "
            "tournament worlds; Balanced combines cut coverage, simulated mean "
            "and best-score upside; GPP emphasizes the portfolio's highest-score "
            "tail. These are experimental objectives, not calibrated cash rates."
        )
        pa=result.get("portfolio_audit")
        if pa:
            m1,m2,m3=st.columns(3)
            m1.metric("At least one 6/6 lineup (modeled)",
                      f"{pa['sim_portfolio_at_least_one_6of6_pct']:.2f}%")
            m2.metric("Portfolio best-score P90",
                      f"{pa['sim_portfolio_best_score_p90']:.2f}")
            m3.metric("Portfolio best-score P99",
                      f"{pa['sim_portfolio_best_score_p99']:.2f}")
            st.download_button(
                "Download PGA portfolio-level simulation audit",
                pd.DataFrame([pa]).to_csv(index=False).encode("utf-8-sig"),
                "team_ai2_pga_portfolio_level_audit.csv",mime="text/csv",
                key="ai2_pga_joint_portfolio_audit")
        with st.expander("Compare all three objectives using this exact simulation"):
            st.caption(
                "Reuses the current golfer worlds and rules; no additional input "
                "upload or new 72-hole simulation. The approved lineups are never "
                "replaced. Differences can legitimately be small."
            )
            if st.button("Compare Balanced / GPP / Cut Equity",
                         key="ai2_pga_compare_objectives"):
                try:
                    compared=[]
                    for objective in ("Balanced / Single Entry","GPP Ceiling","Cut Equity"):
                        if objective==strategy:
                            audit=pa
                        else:
                            other=rebuild_reviewed(
                                result,vetoes=exclusions,locks=locks,
                                lineup_count=lineup_count,max_exposure=exposure_pct/100,
                                salary_floor=salary_floor,min_unique=min_unique,
                                strategy=objective)
                            audit=other.get("portfolio_audit")
                        if audit is None:
                            raise ValueError("Required joint simulation audit missing.")
                        compared.append(dict(audit))
                    st.session_state["ai2_pga_objectives_result"]={
                        "signature":setup_signature,
                        "rows":compared,
                    }
                except Exception as exc:
                    st.session_state.pop("ai2_pga_objectives_result",None)
                    st.error("PGA objective comparison blocked: "+str(exc))
            compared=st.session_state.get("ai2_pga_objectives_result")
            if compared is not None and compared.get("signature")==setup_signature:
                comparison=pd.DataFrame(compared["rows"])
                selected=[
                    "strategy","sim_portfolio_at_least_one_6of6_pct",
                    "sim_portfolio_expected_6of6_count",
                    "sim_portfolio_best_score_mean",
                    "sim_portfolio_best_score_p90",
                    "sim_portfolio_best_score_p95",
                    "sim_portfolio_best_score_p99",
                ]
                st.dataframe(comparison[selected],hide_index=True,use_container_width=True)
                st.download_button(
                    "Download three-strategy joint simulation comparison",
                    comparison.to_csv(index=False).encode("utf-8-sig"),
                    "team_ai2_pga_three_strategy_comparison.csv",
                    mime="text/csv",key="ai2_pga_three_way_audit")

        st.dataframe(result["summary"],hide_index=True,use_container_width=True)
        st.download_button(
            "Download six-of-six and lineup simulation audit",
            result["summary"].to_csv(index=False).encode("utf-8-sig"),
            "team_ai2_pga_lineup_simulation_audit.csv",mime="text/csv",
            key="ai2_pga_lineup_joint_audit",
        )
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

with tab_review:
    render_review_page(
        result,signature=setup_signature,
        lineup_count=lineup_count,max_exposure=exposure_pct/100,
        salary_floor=salary_floor,min_unique=min_unique,
        strategy=strategy,locks=locks,base_exclusions=exclusions,
    )

with tab_parlay:
    st.subheader("PGA TOUR 2-Ball & 3-Ball Matchups · 6-Leg Research")
    st.caption(
        "Official round groupings and completed-round scores are attempted automatically. "
        "Parlay win/push probabilities are research estimates, not sportsbook prices. "
        "Weekend groups of two are supported, including separate tie-outcome markets. "
        "No automatic betting or guaranteed edges."
    )
    cols=st.columns([1,1,1,1])
    round_no=int(cols[0].selectbox("Round",[1,2,3,4],index=1,key="ai2_pga_group_round"))
    group_sims=int(cols[1].selectbox("Group simulations",[10000,25000,50000],index=1,key="pga_group_sims"))
    ticket_count=int(cols[2].number_input("6-leg ticket candidates",1,10,3,key="pga_ticket_count"))
    run_seed=int(cols[3].number_input("Group seed",0,2147483647,42,key="pga_group_seed"))
    two_ball_rule=st.radio(
        "Two-player matchup settlement (check your sportsbook)",
        ["3-way: separate Tie outcome (golfer loses on a tie)",
         "2-way: tied scores push the golfer selection"],
        index=0,horizontal=True,key="pga_two_ball_rule")
    tie_rule="loss" if two_ball_rule.startswith("3-way") else "push"
    st.caption("Three-ball ties are modeled as pushes. For two-ball markets, "
               "select whether a tie loses your golfer pick (3-way) or pushes it (2-way). "
               "A separate 'Tie' outcome is displayed only for 3-way markets.")
    with st.expander("Optional source fallbacks (only when official data fail)"):
        tournament_id=st.text_input("PGA TOUR tournament ID (if known)",key="pga_tid")
        groups_upload=st.file_uploader("Verified tee groupings CSV",type=["csv"],key="pga_group_csv")
        results_upload=st.file_uploader("Completed earlier rounds CSV",type=["csv"],key="pga_prior_results_csv")
    group_signature=(result["metadata"]["created_utc"] if result else None,
                     round_no,group_sims,ticket_count,run_seed,tournament,tournament_id,tie_rule,
                     hashlib.sha256(groups_upload.getvalue()).hexdigest() if groups_upload else None,
                     hashlib.sha256(results_upload.getvalue()).hexdigest() if results_upload else None)
    do_groups=st.button("Research this round's group matchups",type="primary",
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
                raise ValueError("No valid two- or three-golfer round groups available. Upload verified tee times as a fallback.")
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
                                     seed=run_seed,sims=group_sims,tickets=ticket_count,
                                     two_ball_tie_rule=tie_rule)
            analyzed["metadata"]={
                "round":round_no,"groupings_source":source,
                "completed_rounds_source":live_source,
                "tournament":tournament,"seed":run_seed,
            }
            analyzed["live"]=live
            analyzed["groups"]=groupings
            analyzed["signature"]=group_signature
            st.session_state["ai2_pga_3ball"]=analyzed
            pair_count=int((groupings["group_size"]==2).sum())
            trio_count=int((groupings["group_size"]==3).sum())
            st.success(
                f"Evaluated {len(groupings)} groups for Round {round_no}: "
                f"{pair_count} two-balls and {trio_count} three-balls."
            )
        except Exception as exc:
            st.session_state.pop("ai2_pga_3ball",None)
            st.error(str(exc))
    grp=st.session_state.get("ai2_pga_3ball")
    if grp and result and grp.get("signature")==group_signature:
        st.caption(f"Groupings: {grp['metadata']['groupings_source']} · "
                   f"Prior scores: {grp['metadata']['completed_rounds_source']} · "
                   f"Ratings: {grp['source']}")
        st.subheader("Only bookable Hard Rock matchups belong in the parlay")
        st.caption(
            "Official PGA tee groups are NOT the same as the group-matchup markets "
            "Hard Rock actually offers. Select only groups visibly listed in your sportsbook. "
            "The simulator can research every tee group, but won't generate a bettable "
            "six-leg shortlist from unconfirmed groups."
        )
        all_groups=grp["groups"]
        choices={int(g.group):(
            f"R{round_no} · {g.tee_time} · "
            f"{g.player1} / {g.player2}"
            + (f" / {g.player3}" if int(g.group_size)==3 else "")
            + f" ({int(g.group_size)}-ball)"
        ) for g in all_groups.itertuples(index=False)}
        unique_market_key=hashlib.sha256(repr(group_signature).encode()).hexdigest()[:15]
        offered_ids=st.multiselect(
            "Hard Rock group markets currently offered",
            options=list(choices),
            default=[],
            format_func=lambda x:choices[x],
            key=f"pga_offered_market_ids_{unique_market_key}",
            help="Select the GROUPS shown in Hard Rock. Changing this list "
                 "immediately recalculates the six-leg suggestions without new simulations."
        )
        restricted_probs,restricted_summary,restricted_legs=bookmaker_filtered_tickets(
            grp["probabilities"],all_groups,offered_ids,ticket_count=ticket_count
        )
        st.caption(
            f"{len(offered_ids)} sportsbook-listed groups selected out of "
            f"{len(all_groups)} official tee-time groups."
        )
        with st.expander("All official-group research (includes unlisted/unbettable markets)"):
            st.dataframe(grp["probabilities"],use_container_width=True,hide_index=True)
            st.download_button("Download all theoretical group probabilities",
                grp["probabilities"].to_csv(index=False).encode("utf-8-sig"),
                f"team_ai2_pga_r{round_no}_all_official_groups.csv",mime="text/csv")
        st.subheader("Hard Rock-listed matchup probabilities")
        if restricted_probs.empty:
            st.info("Choose the groups that Hard Rock actually lists before building a parlay.")
        else:
            st.dataframe(restricted_probs,use_container_width=True,hide_index=True)
            st.download_button("Download Hard Rock-listed matchup estimates",
                restricted_probs.to_csv(index=False).encode("utf-8-sig"),
                f"team_ai2_pga_r{round_no}_hard_rock_groups.csv",mime="text/csv")
        st.subheader("Six-leg candidates from offered groups only")
        if restricted_legs.empty:
            st.info(
                "No six-leg ticket generated. Select at least six ACTUALLY OFFERED, "
                "fully matched sportsbook group markets; six weak or unavailable "
                "legs should not be forced into a parlay."
            )
        else:
            st.dataframe(restricted_summary,hide_index=True,use_container_width=True)
            st.dataframe(restricted_legs,hide_index=True,use_container_width=True)
            st.caption(
                "Selection is restricted to your confirmed Hard Rock group markets. "
                "PGA tee-time status does not confirm sportsbook availability. "
                "Probabilities are provisional and assume approximate independence. "
                "Compare actual sportsbook prices and settlement rules before wagering."
            )
            st.download_button("Download offered-market six-leg candidates",
                restricted_legs.to_csv(index=False).encode("utf-8-sig"),
                f"team_ai2_pga_r{round_no}_hard_rock_six_legs.csv",mime="text/csv")
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
