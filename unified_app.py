"""TEAM AI² Pro unified platform prototype.

Run: streamlit run unified_app.py
Persistent uploads use SQLite on a durable volume configured by AI2_DATA_DIR.
This prototype does not claim any of the sport engines are validated.
"""
import os
import base64
import html
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
import streamlit as st
from ai2_settings import load_setting, save_setting
from ai2_local_storage import open_store, save_file, load_file, list_history
from ai2_nfl_preflight import validate_pool
from ai2_nfl_inputs import inspect_saved_nfl_inputs
from ai2_nfl_eligibility import eligible_nfl_rows
from ai2_nfl_feasibility import find_classic_lineup
from ai2_nfl_candidate_builder import build_classic_candidates
from ai2_nfl_portfolio_store import slate_fingerprint, save_portfolio, load_latest_portfolio
from ai2_nfl_export import export_classic_candidates
from ai2_market_challenge import freeze_simulation, compare_frozen_to_props
from ai2_market_identity import parse_identity_csv
from ai2_market_provider import retrieve_market_props

SPORTS = {
    "PGA": {"icon": "⛳", "description": "Classic · Showdown · 3-ball matchups"},
    "NFL": {"icon": "🏈", "description": "Classic · Showdown · Player props"},
    "NASCAR": {"icon": "🏁", "description": "Race simulations · DFS"},
    "NBA": {"icon": "🏀", "description": "DFS · Player props · Parlays"},
}
DATA_DIR = Path(os.environ.get("AI2_DATA_DIR", str(Path.home() / "TEAM_AI2_Pro_Data"))).expanduser()
DB = DATA_DIR / "slates.sqlite3"

def connect():
    return open_store(DATA_DIR)

def save_upload(sport, kind, file):
    save_file(DATA_DIR, sport, kind, file.name, file.getvalue())

def get_upload(sport, kind):
    return load_file(DATA_DIR, sport, kind)

st.set_page_config(page_title="TEAM AI² Pro", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")
# Unified preview: suppress legacy multipage links without deleting their files.
# Legacy pages remain available in the repository for migration and audit.
st.markdown("""<style>
[data-testid="stSidebarNav"], [data-testid="stSidebarNavItems"] {display:none !important;}
</style>""", unsafe_allow_html=True)

# Force a coherent dark palette at Streamlit's theme-token layer, rather
# than relying only on text selectors that miss dynamically rendered widgets.
st.markdown("""<style>
:root, [data-theme="dark"], [data-testid="stAppViewContainer"] {
  --primary-color: #4db8ff;
  --text-color: #f1f5ff;
  --background-color: #080b12;
  --secondary-background-color: #18263b;
}
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {
  color: #f1f5ff !important;
}
.stApp :is(h1,h2,h3,h4,h5,h6,p,small,label),
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stWidgetLabel"],
.stApp [data-baseweb="tab"] {
  color: #f1f5ff !important;
}
.stApp [data-testid="stCaptionContainer"] {opacity: 1 !important;}
.stApp [data-testid="stCaptionContainer"] p {color: #c3d1e8 !important;}
.stApp [data-baseweb="tab"][aria-selected="false"] {color: #d0ddf2 !important;}
.stApp [data-testid="stAlert"] p {color: inherit !important;}
</style>""", unsafe_allow_html=True)
st.markdown("""<style>
.stApp {background: #080b12; color: #f1f5ff;}
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3,
.stApp [data-testid="stMarkdownContainer"] h4,
.stApp [data-testid="stWidgetLabel"] *,
.stApp [data-testid="stCaptionContainer"] *,
.stApp [data-testid="stTabs"] button,
.stApp label {color: #f1f5ff !important;}
.stApp [data-testid="stCaptionContainer"] * {color: #b8c6df !important;}
.stApp [data-testid="stAlert"] * {color: inherit;}
.stApp [data-testid="stButton"] button {min-height: 3rem; background: #173459; color: #f8fbff; border: 1px solid #427bb0;}
.stApp [data-testid="stButton"] button:disabled {color: #b1baca; background: #293241;}
.stApp [data-testid="stTabs"] button[aria-selected="true"] {color: #76c8ff !important;}
</style>""", unsafe_allow_html=True)
# A clickable image panel changes URL query parameters. Synchronize on
# every rerun, not just the first load of a Streamlit session.
requested = str(st.query_params.get("sport", "")).upper()
if requested in SPORTS:
    st.session_state.sport = requested
elif "sport" not in st.session_state:
    st.session_state.sport = None

def go_home():
    st.session_state.sport = None
    if "sport" in st.query_params:
        del st.query_params["sport"]

if st.session_state.sport is None:
    st.title("TEAM AI² PRO")
    # New wide photo dashboard is preferred when installed; keep the
    # original four-sport emblem as a safe fallback.
    banner = Path(__file__).parent / "assets" / "team_ai2_sports_dashboard.jpg"
    logo = Path(__file__).parent / "assets" / "team_ai2_sports_emblem.png"
    if banner.exists():
        encoded = base64.b64encode(banner.read_bytes()).decode("ascii")
        # Image order is NFL, PGA, NASCAR, NBA, left to right.
        # The top brand masthead is not clickable; sport panels are.
        st.markdown(f"""<div style="position:relative;max-width:1100px;margin:auto">
          <img src="data:image/jpeg;base64,{encoded}" alt="TEAM AI² Pro photo dashboard: NFL, PGA, NASCAR and NBA"
               style="width:100%;height:auto;display:block">
          <a href="?sport=NFL" target="_self" aria-label="Open NFL"
             style="position:absolute;left:0%;top:25%;width:25%;height:75%"></a>
          <a href="?sport=PGA" target="_self" aria-label="Open PGA"
             style="position:absolute;left:25%;top:25%;width:25%;height:75%"></a>
          <a href="?sport=NASCAR" target="_self" aria-label="Open NASCAR"
             style="position:absolute;left:50%;top:25%;width:25%;height:75%"></a>
          <a href="?sport=NBA" target="_self" aria-label="Open NBA"
             style="position:absolute;left:75%;top:25%;width:25%;height:75%"></a>
        </div>""", unsafe_allow_html=True)
    elif logo.exists():
        encoded = base64.b64encode(logo.read_bytes()).decode("ascii")
        st.markdown(f"""<div style="position:relative;max-width:640px;margin:auto">
          <img src="data:image/png;base64,{encoded}" alt="TEAM AI² Pro four-sport emblem"
               style="width:100%;height:auto;display:block">
          <a href="?sport=PGA" target="_self" aria-label="Open PGA"
             style="position:absolute;left:0;top:0;width:50%;height:50%"></a>
          <a href="?sport=NFL" target="_self" aria-label="Open NFL"
             style="position:absolute;left:50%;top:0;width:50%;height:50%"></a>
          <a href="?sport=NASCAR" target="_self" aria-label="Open NASCAR"
             style="position:absolute;left:0;top:50%;width:50%;height:50%"></a>
          <a href="?sport=NBA" target="_self" aria-label="Open NBA"
             style="position:absolute;left:50%;top:50%;width:50%;height:50%"></a>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("The existing TEAM AI² logo asset is missing.")
    st.caption("Select a sport. Your saved slate remains until you replace it.")
    # Accessible navigation buttons; clickable image quadrants can be added later.
    cols = st.columns(4)
    for col,(sport,meta) in zip(cols,SPORTS.items()):
        with col:
            if st.button(f"{meta['icon']} {sport}", key=f"open_{sport}", use_container_width=True):
                st.session_state.sport = sport
                st.query_params["sport"] = sport
                st.rerun()
            st.caption(meta["description"])
else:
    sport = st.session_state.sport
    if st.button("← All sports"):
        go_home()
        st.rerun()
    st.title(f"{SPORTS[sport]['icon']} {sport}")
    st.caption("Unified platform development preview · Research-only NFL engine available; no sport engine certified.")
    tab_data,tab_build,tab_market,tab_results,tab_health = st.tabs(
        ["My Slate", "Build / Simulate", "Market Intelligence", "My Results", "Engine Health"]
    )
    with tab_data:
        st.subheader("Persistent slate files")
        st.info("Files are saved in SQLite. Configure AI2_DATA_DIR to a durable disk before deployment; ephemeral hosting storage will not survive a rebuild.")
        kind = st.selectbox("File type",["DraftKings salaries","Contest entries","Projections","Props / odds"],key=f"kind_{sport}")
        saved = get_upload(sport,kind)
        if saved:
            filename,payload,when = saved
            st.success(f"Saved: {filename} · {len(payload):,} bytes · {when}")
            st.download_button("Download saved file",payload,file_name=filename,key=f"download_{sport}_{kind}")
        else:
            st.warning("No saved file for this category.")
        upload = st.file_uploader("Choose a file to replace the saved version",type=["csv","xlsx","json"],key=f"upload_{sport}_{kind}")
        if st.button("Save / replace file",disabled=upload is None,key=f"save_{sport}_{kind}"):
            try:
                save_upload(sport,kind,upload)
                if sport == "NFL" and kind in ("DraftKings salaries", "Projections"):
                    st.session_state.pop("nfl_candidates", None)
                st.success("File saved. Previous NFL candidate results were cleared if the slate changed.")
                st.rerun()
            except (OSError,sqlite3.Error,ValueError) as exc:
                st.error(f"Could not save file: {exc}")
        st.caption(f"Local data folder: {DATA_DIR}. Files remain after app restarts as long as this folder is retained and writable.")
        if saved:
            with st.expander("Previous versions of this file"):
                history = list_history(DATA_DIR, sport, kind)
                if not history:
                    st.caption("No earlier versions saved.")
                for version_id, old_name, old_bytes, old_time in history:
                    st.download_button(f"Archived: {old_name} · {old_time}", old_bytes,
                                       file_name=f"archive_{version_id}_{old_name}", key=f"archive_{sport}_{kind}_{version_id}")
        st.divider()
        st.subheader("Saved build preferences")
        current_mode = load_setting(DB, sport, "portfolio_mode", "Balanced")
        mode = st.selectbox("Portfolio approach", ["Balanced", "Tournament upside", "Cash-oriented"],
                            index=["Balanced", "Tournament upside", "Cash-oriented"].index(current_mode)
                            if current_mode in ["Balanced", "Tournament upside", "Cash-oriented"] else 0,
                            key=f"mode_{sport}")
        if st.button("Save preferences", key=f"prefs_{sport}"):
            save_setting(DB, sport, "portfolio_mode", mode)
            st.success("Preferences saved on the configured data volume.")
    with tab_build:
        if sport != "NFL":
            st.info("Coming next: this sport's simulation engine is not connected.")
        else:
            salary_saved = get_upload("NFL", "DraftKings salaries")
            projection_saved = get_upload("NFL", "Projections")
            st.subheader("Saved slate readiness")
            status_cols = st.columns(2)
            with status_cols[0]:
                if salary_saved:
                    st.success(f"Salaries ready: {salary_saved[0]}")
                    st.caption(f"Saved {salary_saved[2]}")
                else:
                    st.warning("No saved DraftKings salaries")
            with status_cols[1]:
                if projection_saved:
                    st.success(f"Projections saved: {projection_saved[0]}")
                    st.caption(f"Saved {projection_saved[2]}")
                else:
                    st.warning("No saved independent projections")
            if salary_saved and projection_saved:
                try:
                    saved_rows, saved_issues = inspect_saved_nfl_inputs(salary_saved, projection_saved)
                    if saved_issues:
                        st.error("Slate data issues: " + "; ".join(saved_issues))
                    else:
                        verified_rows = eligible_nfl_rows(saved_rows)
                        st.caption(f"{len(saved_rows)} matched projection rows · {len(verified_rows)} pass player eligibility checks. Flags are supplied by the data and are not independent certification.")
                        if len(verified_rows) < 9:
                            st.warning("Current saved projections cannot produce a valid nine-player NFL Classic lineup.")
                except (ValueError, KeyError, TypeError) as exc:
                    st.error(f"Saved slate needs attention: {exc}")
            st.divider()
            st.caption("Research integration: 6d8dde0 · development branch")
            with st.expander("Research lab · observed-stat simulation (not lineup export)", expanded=False):
                st.caption("Experimental game-level football simulation. Requires a verified pool CSV and observed historical-rate CSV. No betting-market fields are used, and this does not certify a slate.")
                lab_pool = st.file_uploader("Verified simulation pool CSV", type=["csv"], key="nfl_lab_pool")
                if lab_pool is not None:
                    with st.expander("Current-week availability screen · advisory only", expanded=False):
                        try:
                            import pandas as pd
                            from ai2_nfl_availability import availability_report
                            preview_pool = pd.read_csv(lab_pool)
                            screening = availability_report(preview_pool)
                            counts = screening["availability_reason"].value_counts()
                            st.caption("DK statuses are not official active-roster verification. All players still require current-week participation and opportunity checks.")
                            st.dataframe(screening[["Name", "TeamAbbrev", "Position", "availability_status", "availability_reason"]], hide_index=True)
                            st.caption(f"Rows screened: {len(screening)} · Reported unavailable: {int(screening['availability_gate'].eq('BLOCK').sum())}")
                        except (ValueError, KeyError, pd.errors.ParserError) as exc:
                            st.warning(f"Availability screening unavailable: {exc}")
                st.caption("Optional public-data helper: generate historical player rates automatically from nflverse. This does not verify the current-week roster, injuries, or opportunity.")
                with st.expander("Fetch public NFL historical rates", expanded=False):
                    history_season = st.number_input("Target season", min_value=2020, max_value=2030, value=2026, step=1, key="nfl_history_season")
                    history_week = st.number_input("Target week", min_value=1, max_value=22, value=7, step=1, key="nfl_history_week")
                    # Never silently reuse an earlier week when the controls change.
                    requested_history_key = (int(history_season), int(history_week))
                    if st.session_state.get("nfl_public_rates_key") != requested_history_key:
                        st.session_state.pop("nfl_public_rates", None)
                        st.session_state.pop("nfl_public_history", None)
                        st.session_state.pop("nfl_baseline_audit", None)
                        st.session_state.pop("nfl_baseline_audit_rows", None)
                        st.session_state.pop("nfl_dk_audit_rows", None)
                        st.session_state.pop("nfl_public_rates_meta", None)
                    if st.button("Fetch and build historical rates", key="nfl_fetch_rates"):
                        # Import pandas before the guarded fetch: the exception
                        # tuple references pd.errors.ParserError even if the
                        # optional props_data import fails.
                        import pandas as pd
                        try:
                            from props_data import fetch_weekly, make_rates
                            from datetime import datetime, timezone
                            history = fetch_weekly([int(history_season)-1, int(history_season)])
                            rate_frame = make_rates(history, int(history_season), int(history_week))
                            if rate_frame.empty:
                                raise ValueError("No observed player histories returned")
                            st.session_state["nfl_public_history"] = history.copy()
                            st.session_state["nfl_public_rates"] = rate_frame
                            st.session_state["nfl_public_rates_key"] = requested_history_key
                            st.session_state["nfl_public_rates_meta"] = {
                                "target": f"{history_season} Week {history_week}",
                                "fetched_utc": datetime.now(timezone.utc).isoformat(),
                                "players": len(rate_frame),
                            }
                        except (ValueError, KeyError, OSError, ImportError, pd.errors.ParserError) as exc:
                            st.session_state.pop("nfl_public_rates", None)
                            st.session_state.pop("nfl_public_history", None)
                            st.session_state.pop("nfl_public_rates_meta", None)
                            st.error(f"Public data retrieval failed: {exc}")
                    if "nfl_public_rates" in st.session_state:
                        meta = st.session_state["nfl_public_rates_meta"]
                        st.success(f"Fetched {meta['players']} player baselines for {meta['target']} at {meta['fetched_utc']}")
                        st.caption("Source: nflverse stats_player public release; excludes target week and later. This is historical data, not certified projections.")
                        st.caption("Changing the target season or week invalidates previously fetched rates.")
                        st.download_button("Download observed rates CSV",
                                           st.session_state["nfl_public_rates"].to_csv(index=False).encode("utf-8"),
                                           file_name="nfl_observed_rates.csv", mime="text/csv")
                with st.expander("Historical walk-forward accuracy · observed statistics", expanded=False):
                    st.caption("Evaluate pregame weighted stat baselines against later observed games. This is NOT a DFS simulation backtest or proof of lineup profitability.")
                    if "nfl_public_history" not in st.session_state:
                        st.info("Fetch public historical rates above to enable this audit.")
                    elif int(history_week) <= 4:
                        st.info("Select target Week 5 or later for an observed Week 4+ backtest.")
                    elif st.button("Run historical baseline backtest", key="nfl_backtest_baselines"):
                        try:
                            from props_data import walkforward, walkforward_dk_offense
                            import pandas as pd
                            results = walkforward(
                                st.session_state["nfl_public_history"],
                                season=int(history_season),
                                first_week=4,
                                last_week=max(4, int(history_week) - 1),
                                window=8,
                            )
                            if results.empty:
                                st.warning("No eligible observed comparisons were available.")
                            else:
                                metrics = (results.groupby("stat", as_index=False)
                                           .agg(observations=("absolute_error", "size"),
                                                mean_absolute_error=("absolute_error", "mean"),
                                                mean_signed_error=("error", "mean")))
                                st.session_state["nfl_baseline_audit"] = metrics
                                st.session_state["nfl_baseline_audit_rows"] = results
                                dk_rows = walkforward_dk_offense(
                                    st.session_state["nfl_public_history"],
                                    season=int(history_season),
                                    first_week=4,
                                    last_week=int(history_week) - 1,
                                    window=8,
                                )
                                st.session_state["nfl_dk_audit_rows"] = dk_rows
                        except (ValueError, KeyError, TypeError) as exc:
                            st.error(f"Historical audit unavailable: {exc}")
                    if "nfl_baseline_audit" in st.session_state:
                        st.dataframe(st.session_state["nfl_baseline_audit"], use_container_width=True, hide_index=True)
                        dk_rows = st.session_state.get("nfl_dk_audit_rows")
                        if dk_rows is not None and not dk_rows.empty:
                            st.caption("Partial DraftKings offensive scoring only: no fumbles lost, return TDs or DST. Mean-stat bonus thresholds are a baseline, not simulation expectations.")
                            st.metric("Partial DK offensive mean absolute error",
                                      f"{dk_rows['absolute_error'].mean():.2f} points")
                            by_position = (dk_rows.groupby("position", as_index=False)
                                           .agg(player_games=("absolute_error", "size"),
                                                mean_absolute_error=("absolute_error", "mean"),
                                                signed_bias=("error", "mean")))
                            st.caption("Error by position; signed bias above zero means the baseline overpredicted.")
                            st.dataframe(by_position, use_container_width=True, hide_index=True)
                            st.caption(f"Evaluated {len(dk_rows):,} historical player-games")
                            st.download_button("Download partial DK scoring audit",
                                               dk_rows.to_csv(index=False).encode("utf-8"),
                                               file_name="nfl_partial_dk_walkforward.csv",
                                               mime="text/csv", key="nfl_download_dk_audit")
                        st.download_button("Download historical comparison rows",
                                           st.session_state["nfl_baseline_audit_rows"].to_csv(index=False).encode("utf-8"),
                                           file_name="nfl_baseline_walkforward.csv", mime="text/csv",
                                           key="nfl_download_baseline_audit")
                lab_rates = st.file_uploader("Observed player-rate CSV (optional when public rates fetched)", type=["csv"], key="nfl_lab_rates")
                lab_sims = st.selectbox("Simulation draws", [100, 1000, 10000], index=1, key="nfl_lab_sims")
                if st.button("Run research simulation", disabled=not (lab_pool and (lab_rates or "nfl_public_rates" in st.session_state)), key="nfl_lab_run"):
                    try:
                        import pandas as pd
                        from ai2_nfl_simulation_adapter import run_verified_stat_simulation
                        sim_pool = pd.read_csv(lab_pool)
                        sim_rates = pd.read_csv(lab_rates) if lab_rates else st.session_state["nfl_public_rates"].copy()
                        players, draws, metadata = run_verified_stat_simulation(
                            sim_pool, sim_rates, n_sims=int(lab_sims))
                        if draws.shape != (lab_sims, len(players)):
                            raise ValueError("Unexpected simulation matrix shape")
                        import numpy as np
                        if not np.isfinite(draws).all():
                            raise ValueError("Simulation contains nonfinite fantasy scores")
                        report = pd.DataFrame({
                            "Player": players["Name"].astype(str),
                            "Team": players["TeamAbbrev"].astype(str),
                            "Position": players["Position"].astype(str),
                            "Mean DK": draws.mean(axis=0).round(2),
                            "P90 DK": np.percentile(draws, 90, axis=0).round(2),
                        }).sort_values("Mean DK", ascending=False)
                        st.session_state.pop("nfl_distribution_diagnostics", None)
                        st.session_state.pop("nfl_distribution_summary", None)
                        st.session_state["nfl_research_draws"] = draws
                        st.session_state["nfl_research_players"] = players.copy()
                        st.session_state["nfl_research_report"] = report
                        st.success(f"Completed {lab_sims:,} experimental game-level draws. Not calibrated or authorized for contest entry.")
                    except (ValueError, KeyError, TypeError, ImportError, IndexError) as exc:
                        st.session_state.pop("nfl_research_report", None)
                        st.session_state.pop("nfl_research_draws", None)
                        st.session_state.pop("nfl_research_players", None)
                        st.error(f"Research simulation blocked: {exc}")
                if "nfl_research_report" in st.session_state:
                    st.dataframe(st.session_state["nfl_research_report"], use_container_width=True, hide_index=True)
                    with st.expander("Postgame simulation distribution audit", expanded=False):
                        st.caption("After games finish, upload observed DraftKings points for this exact slate. CSV columns: ID, actual_dk. This does not validate an independent historical holdout unless inputs were frozen before kickoff.")
                        actual_upload = st.file_uploader("Observed DK scores CSV", type=["csv"], key="nfl_research_actuals")
                        if st.button("Evaluate simulated distribution", disabled=actual_upload is None, key="nfl_evaluate_distribution"):
                            try:
                                import pandas as pd
                                from ai2_nfl_distribution_audit import distribution_diagnostics, distribution_summary
                                observations = pd.read_csv(actual_upload, dtype={"ID": str})
                                diagnostics = distribution_diagnostics(
                                    st.session_state["nfl_research_players"],
                                    st.session_state["nfl_research_draws"], observations)
                                summary = distribution_summary(diagnostics)
                                st.session_state["nfl_distribution_diagnostics"] = diagnostics
                                st.session_state["nfl_distribution_summary"] = summary
                            except (ValueError, KeyError, TypeError, ImportError) as exc:
                                st.session_state.pop("nfl_distribution_diagnostics", None)
                                st.session_state.pop("nfl_distribution_summary", None)
                                st.error(f"Distribution audit blocked: {exc}")
                        if "nfl_distribution_summary" in st.session_state:
                            summary = st.session_state["nfl_distribution_summary"]
                            st.metric("Mean absolute error", f"{summary['mean_absolute_error']:.2f} DK")
                            st.caption(f"Central 50% interval coverage: {summary['central_50_coverage']:.1%} (target ~50%); central 80% coverage: {summary['central_80_coverage']:.1%} (target ~80%); {summary['players']} players")
                            st.dataframe(st.session_state["nfl_distribution_diagnostics"], use_container_width=True, hide_index=True)
                            st.download_button("Download distribution audit",
                                               st.session_state["nfl_distribution_diagnostics"].to_csv(index=False).encode("utf-8"),
                                               file_name="nfl_distribution_audit.csv", mime="text/csv",
                                               key="nfl_distribution_download")
                st.warning("Research only. No lineup construction or DraftKings entry export from these draws.")
            st.divider()
            st.subheader("NFL Classic · Sunday candidate builder")
            st.caption("Projection-ranked legal candidates, NOT correlated simulations or validated contest-ROI predictions. No sportsbook inputs.")
            st.info("First save DraftKings salaries and an independent weekly projection CSV under My Slate. Projections must contain ID, proj, projection_source, weekly_projection_valid and opportunity_verified. The two verification flags must be true and must reflect independently checked data.")
            lineup_count = st.slider("Candidate lineups", 1, 20, 20, key="nfl_candidate_count")
            exposure_limit = st.slider("Maximum player exposure", 20, 100, 35, 5, key="nfl_candidate_exposure")
            if st.button("Build NFL Classic candidates", key="nfl_build_candidates"):
                try:
                    salaries = get_upload("NFL", "DraftKings salaries")
                    projections = get_upload("NFL", "Projections")
                    rows, issues = inspect_saved_nfl_inputs(salaries, projections)
                    if issues:
                        raise ValueError("; ".join(issues))
                    eligible = eligible_nfl_rows(rows)
                    if len(eligible) < 9:
                        raise ValueError("Insufficient verified eligible players. Check weekly projections, opportunity flags, and injuries.")
                    candidates = build_classic_candidates(eligible, count=lineup_count,
                                                          max_exposure=exposure_limit / 100)
                    st.session_state.pop("nfl_candidates", None)
                    if not candidates:
                        st.warning("No legal candidates found within the search budget. This does not prove the slate is infeasible.")
                    else:
                        st.session_state["nfl_candidates"] = candidates
                        fingerprint = slate_fingerprint(salaries, projections)
                        save_portfolio(DATA_DIR, fingerprint, candidates)
                        st.success(f"Built {len(candidates)} legal projection-ranked candidates.")
                        if len(candidates) < lineup_count:
                            st.warning(f"Only {len(candidates)} of {lineup_count} requested lineups met the portfolio exposure constraints.")
                except (ValueError, KeyError, TypeError) as exc:
                    st.session_state.pop("nfl_candidates", None)
                    st.error(f"Build blocked: {exc}")
            saved_salary_file = get_upload("NFL", "DraftKings salaries")
            saved_projection_file = get_upload("NFL", "Projections")
            candidates = []
            if saved_salary_file and saved_projection_file:
                current_fingerprint = slate_fingerprint(saved_salary_file, saved_projection_file)
                latest = load_latest_portfolio(DATA_DIR, current_fingerprint)
                if latest:
                    candidates = latest["candidates"]
                    st.caption(f"Saved candidate portfolio #{latest['id']} · {latest['created_at']} · matches current uploaded slate")
            if candidates:
                summary = [{"Lineup": i + 1, "Projected DK points": x["projected_points"],
                            "Salary": x["salary"],
                            "Players": ", ".join(str(p.get("Name", p["ID"])) for p in x["players"])}
                           for i, x in enumerate(candidates)]
                st.dataframe(summary, use_container_width=True, hide_index=True)
                st.download_button("Download candidate review JSON",
                                   json.dumps(summary, indent=2), file_name="nfl_candidate_review.json",
                                   mime="application/json")
                try:
                    candidate_csv = export_classic_candidates(candidates)
                    st.download_button("Download DraftKings roster-slot candidate CSV (review only)",
                                       candidate_csv, file_name="nfl_classic_candidates_review.csv",
                                       mime="text/csv", key="nfl_review_roster_csv")
                except (ValueError, KeyError, TypeError) as exc:
                    st.error(f"Candidate export blocked: {exc}")
                st.warning("Review only: roster-slot CSV is NOT a validated DraftKings contest-entry upload. Confirm slate, statuses, ID mapping, entry template, and model quality before any use.")
    with tab_market:
        st.subheader("Market Intelligence · post-simulation only")
        st.caption("Independent simulation first. This screen never changes projections, reruns the model, or authorizes betting.")
        if sport not in ("NFL", "NBA"):
            st.info("Player-prop comparisons are currently planned for NFL and NBA.")
        else:
            st.info("Research preview: load a frozen simulation JSON with its SHA-256 digest, then compare post-freeze props. Live provider integration is not yet certified.")
            frozen_upload = st.file_uploader("Frozen simulation JSON", type=["json"], key=f"market_frozen_{sport}")
            props_upload = st.file_uploader("Post-freeze market props JSON", type=["json"], key=f"market_props_{sport}")
            identity_upload = st.file_uploader("Verified athlete identity map (optional CSV)", type=["csv"], key=f"market_ids_{sport}")
            if identity_upload:
                try:
                    mapping = parse_identity_csv(identity_upload.getvalue())
                    st.success(f"Loaded {len(mapping)} explicit event-scoped athlete mappings.")
                    st.caption("Mappings are for provider matching only. This preview still requires normalized prop JSON.")
                except ValueError as exc:
                    st.error(f"Identity mapping rejected: {exc}")
            st.caption("Live retrieval requires PARLAY_API_KEY in Streamlit secrets or the server environment. Credentials are never entered in the app.")
            live_mode = st.checkbox("Retrieve provider props after freeze (experimental)", key=f"market_live_{sport}")
            if live_mode:
                st.warning("Experimental: the provider response schema has not been validated against a live authenticated response. Missing or invalid data will block comparison.")
            threshold = st.slider("Flag probability difference (percentage points)", 5, 50, 15, 5, key=f"market_gap_{sport}")
            if st.button("Compare frozen simulation to props", disabled=not (frozen_upload and (live_mode or props_upload) and (not live_mode or identity_upload)), key=f"market_compare_{sport}"):
                try:
                    frozen_data = json.loads(frozen_upload.getvalue().decode("utf-8"))
                    if not isinstance(frozen_data, dict):
                        raise ValueError("Expected a simulation object")
                    frozen = freeze_simulation(frozen_data["run_id"], frozen_data["observations"], frozen_data["frozen_at"])
                    supplied_digest = frozen_data.get("digest")
                    if not supplied_digest or supplied_digest != frozen.digest:
                        raise ValueError("A matching frozen simulation digest is required; arbitrary uploaded samples cannot be treated as a verified freeze")
                    if live_mode:
                        mapping = parse_identity_csv(identity_upload.getvalue())
                        api_key = os.environ.get("PARLAY_API_KEY", "")
                        if not api_key:
                            try:
                                api_key = st.secrets.get("PARLAY_API_KEY", "")
                            except FileNotFoundError:
                                api_key = ""
                        if not api_key:
                            raise ValueError("PARLAY_API_KEY is not configured on the server")
                        props_data = retrieve_market_props(frozen, sport=sport, api_key=api_key, identity_map=mapping)
                    else:
                        props_data = json.loads(props_upload.getvalue().decode("utf-8"))
                    if not isinstance(props_data, list):
                        raise ValueError("Expected a list of prop records")
                    findings = compare_frozen_to_props(frozen, props_data, min_gap=threshold / 100)
                    st.success(f"Compared frozen run {frozen.run_id} · {len(props_data)} matched prop records · {len(findings)} review flags")
                    st.caption(f"Simulation fingerprint: {frozen.digest}")
                    if findings:
                        st.dataframe(findings, use_container_width=True)
                        st.download_button("Download review flags (JSON)", json.dumps(findings, indent=2),
                                           file_name=f"{sport.lower()}_market_review.json", mime="application/json")
                    else:
                        st.info("No qualifying discrepancies. Confirm player IDs, periods, timestamps and quote freshness if matches were expected.")
                except (ValueError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
                    st.error(f"Comparison blocked: {exc}")
    with tab_results:
        st.info("Saved portfolios, bets and post-event audits will appear here.")
    with tab_health:
        st.error("Engine status: NOT CONNECTED")
        if sport == "NFL":
            st.subheader("NFL quality gate (diagnostic only)")
            st.caption("Fail-closed policy: legacy DK PPG fallbacks and unverified roles cannot pass. No simulation or export is enabled.")
            saved_salaries = get_upload("NFL", "DraftKings salaries")
            if saved_salaries:
                st.info("A salary slate is saved, but salaries alone do not establish current-week projections or opportunity.")
            else:
                st.warning("No NFL salary slate saved.")
            saved_projections = get_upload("NFL", "Projections")
            try:
                rows, issues = inspect_saved_nfl_inputs(saved_salaries, saved_projections)
            except ValueError as exc:
                rows, issues = [], [str(exc)]
            for issue in issues:
                st.warning(issue)
            eligible_rows = eligible_nfl_rows(rows)
            feasibility = find_classic_lineup(eligible_rows)
            st.write(f"Lineup feasibility: {feasibility.reason} ({feasibility.examined:,} checked)")
            check = validate_pool(rows, source_fresh=False,
                                  historical_validation_passed=False,
                                  lineup_feasibility_passed=feasibility.feasible is True)
            st.write(f"Verified eligible players: {check.eligible_count}")
            for reason in check.reasons:
                st.write("• " + reason)
        st.write("Release policy: no real-money export until data freshness, role/opportunity checks, lineup legality, and historical validation pass.")
