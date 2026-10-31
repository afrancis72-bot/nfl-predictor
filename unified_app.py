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
    st.caption("Click a sport panel above to enter. Your saved slate remains until you replace it.")
else:
    sport = st.session_state.sport
    if st.button("← All sports"):
        go_home()
        st.rerun()
    st.title(f"{SPORTS[sport]['icon']} {sport}")
    st.caption("TEAM AI² Pro · NFL portfolio builder and NBA player-prop research engine. Models remain experimental.")
    if sport == "NFL":
        from hashlib import sha256 as _ai2_sha256
        from ai2_nfl_oneclick import build_oneclick_portfolio
        from ai2_nfl_portfolio_diversity import portfolio_overlap_audit
        from ai2_nfl_review import (review_tables, named_csv, named_excel,
                                    roster_id_csv, swap_contest_tiers,
                                    snapshot_approved, load_approved)
        from ai2_local_storage import save_file as _ai2_save_file
        import pandas as _ai2_pd

        st.caption("NFL Classic · Upload DraftKings salaries, then select 15, 20, 50, or 150 lineups. No projections upload required.")
        upload_one = st.file_uploader("DraftKings salary CSV", type=["csv"], key="nfl_one_upload")
        saved_one = get_upload("NFL", "DraftKings salaries")
        if saved_one:
            st.caption(f"Saved salary file: {saved_one[0]}")
        seed_one = st.number_input("Simulation seed (optional)", min_value=0,
                                   max_value=2147483647, value=42, step=1,
                                   key="nfl_one_seed")
        requested_count = st.selectbox("Number of lineups", [15, 20, 50, 150], index=1, key="nfl_one_count")
        if st.button(f"BUILD MY {requested_count} LINEUPS", type="primary",
                     key="nfl_one_build", use_container_width=True):
            try:
                if upload_one is not None:
                    from ai2_nfl_inputs import parse_csv_payload
                    parse_csv_payload(upload_one.getvalue())
                    save_upload("NFL", "DraftKings salaries", upload_one)
                salaries_one = get_upload("NFL", "DraftKings salaries")
                if not salaries_one:
                    raise ValueError("Upload your DraftKings salary CSV first.")
                with st.spinner(f"Checking current NFL injuries and simulating {requested_count} legal lineups..."):
                    output_one = build_oneclick_portfolio(salaries_one, seed=int(seed_one), count=int(requested_count))
                st.session_state["ai2_oneclick_result"] = output_one
                st.session_state["ai2_oneclick_seed"] = int(seed_one)
                st.session_state["ai2_oneclick_count"] = int(requested_count)
                st.session_state["ai2_oneclick_digest"] = _ai2_sha256(salaries_one[1]).hexdigest()
                # A new build starts a new review, never overwrites approvals.
                st.session_state["ai2_oneclick_review_ranked"] = output_one["ranked"]
            except (ValueError, KeyError, TypeError, ImportError, OSError) as exc:
                st.session_state.pop("ai2_oneclick_result", None)
                st.session_state.pop("ai2_oneclick_review_ranked", None)
                st.error(f"Build not completed: {exc}")

        output_one = st.session_state.get("ai2_oneclick_result")
        visible_payload = (upload_one.getvalue() if upload_one is not None
                           else saved_one[1] if saved_one else None)
        current_digest = (_ai2_sha256(visible_payload).hexdigest()
                          if visible_payload is not None else "")
        if (not current_digest or
                st.session_state.get("ai2_oneclick_seed") != int(seed_one) or
                st.session_state.get("ai2_oneclick_count") != int(requested_count) or
                st.session_state.get("ai2_oneclick_digest") != current_digest):
            output_one = None

        if output_one:
            ranked = st.session_state.get("ai2_oneclick_review_ranked") or output_one["ranked"]
            overview, details, exposures = review_tables(ranked)
            st.subheader(f"Your {len(ranked)} lineups — player names")
            st.warning(output_one.get("research_warning",
                       "Experimental projections. Review current injuries before entry."))
            st.dataframe(overview, use_container_width=True, hide_index=True)
            st.download_button("Download readable Excel review",
                               data=named_excel(ranked),
                               file_name=f"team_ai2_nfl_{len(ranked)}_lineups_NAMED.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="nfl_named_xlsx")
            st.download_button("Download readable CSV review",
                               data=named_csv(ranked),
                               file_name=f"team_ai2_nfl_{len(ranked)}_lineups_NAMED.csv",
                               mime="text/csv", key="nfl_named_csv")
            st.download_button("Download DraftKings roster-ID CSV",
                               data=roster_id_csv(ranked),
                               file_name=f"team_ai2_nfl_{len(ranked)}_lineups_DK_ids.csv",
                               mime="text/csv", key="nfl_dk_ids")
            st.caption("Roster-ID CSV contains player IDs, not an official contest-entry template. Verify the DraftKings import format.")

            with st.expander("Player simulation audit · explain projected DK points"):
                sim_rows = output_one.get("player_simulation_audit") or []
                if not sim_rows:
                    st.warning("Detailed player component audit is not available for this build. "
                               "Run BUILD again to generate it from the same 10,000 football scenarios.")
                    if output_one.get("player_simulation_audit_error"):
                        st.caption("Diagnostic issue: " + str(output_one["player_simulation_audit_error"]))
                else:
                    sim_table = _ai2_pd.DataFrame(sim_rows)
                    available = sim_table[sim_table["Audit status"].astype(str).str.startswith("Verified")].copy()
                    if available.empty:
                        st.info("No verified player-level scoring breakdown is available.")
                    else:
                        available["Selector"] = available["Player"] + " (" + available["Team"] + ", " + available["Position"] + ")"
                        labels = available["Selector"].tolist()
                        puka = next((i for i, label in enumerate(labels)
                                     if label.startswith("Puka Nacua (")), 0)
                        choice = st.selectbox("Player to inspect", labels, index=puka, key="nfl_sim_player_inspect")
                        record = available.loc[available["Selector"] == choice].iloc[0]
                        cmean, crebuild = st.columns(2)
                        cmean.metric("Simulated mean DK points", f"{record['Sim mean DK']:.2f}")
                        crebuild.metric("Rebuilt from football stats", f"{record['Rebuilt mean DK']:.2f}")
                        stat_names = ["Mean targets", "Mean receptions", "Mean receiving yards",
                                      "Mean receiving TD", "Chance 1+ receiving TD %",
                                      "Chance 2+ receiving TD %", "Chance 100+ receiving yards %",
                                      "Mean carries", "Mean rushing yards", "Mean rushing TD",
                                      "Mean passing yards", "Mean passing TD", "Mean interceptions",
                                      "Target allocation prior / game", "Target allocation share %",
                                      "Team pass attempts prior", "Team offensive TD prior"]
                        metric_rows = [{"Simulation metric": name, "Average / percentage": record[name]}
                                       for name in stat_names if name in record and _ai2_pd.notna(record[name])]
                        if metric_rows:
                            st.dataframe(_ai2_pd.DataFrame(metric_rows), hide_index=True, use_container_width=True)
                        scoring_rows = [{"DraftKings scoring component": name.replace("Points: ", ""),
                                         "Average DK points": record[name]}
                                        for name in record.index if name.startswith("Points: ")
                                        and _ai2_pd.notna(record[name])]
                        if scoring_rows:
                            st.markdown("**Where the projected fantasy points come from**")
                            st.dataframe(_ai2_pd.DataFrame(scoring_rows), hide_index=True, use_container_width=True)
                        st.caption("This is the component-average explanation from the actual seeded simulation. "
                                   "It is NOT a single 33-point game or a sportsbook-line estimate. "
                                   "Each player's component sums are checked against every simulated DK score.")
                    st.download_button(
                        "Download full player simulation audit (CSV)",
                        data=sim_table.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"team_ai2_nfl_simulation_audit_seed_{int(seed_one)}.csv",
                        mime="text/csv", key="nfl_player_simulation_audit_export")


            with st.expander("NFL injuries and news status", expanded=True):
                report = output_one.get("pregame_report") or {}
                checked = report.get("checked_utc") or "not available"
                source = report.get("official_injury_url") or "unknown source"
                st.caption(f"Official injury source: {source} · Retrieved UTC: {checked}")
                st.caption("Game designations and practice participation are distinct. Limited or full practice does NOT guarantee a full game workload.")
                rows = report.get("injuries") or []
                if rows:
                    st.dataframe(_ai2_pd.DataFrame(rows),
                                 use_container_width=True, hide_index=True, height=270)
                else:
                    st.info("No player-level injury details were returned in this build.")
                st.warning(report.get("news_feed",
                           "Breaking injury and depth-chart news are not independently verified."))
                st.caption("Refresh by pressing BUILD again with the latest salary slate. Recheck official Sunday inactives before locking entries.")

            with st.expander("Opposing defensive injuries and matchup intelligence"):
                matchup_rows = output_one.get("opponent_matchup_report") or []
                matchup_summary = output_one.get("opponent_matchup_summary") or {}
                st.caption("Opposing defensive injuries, recent position-by-position allowance and potential game-script implications. Historical opponent observations exclude the current game.")
                st.info(matchup_summary.get("game_script_status",
                        "Hypotheses only — not a calibrated fantasy-point adjustment."))
                if matchup_rows:
                    st.dataframe(_ai2_pd.DataFrame(matchup_rows),
                                 use_container_width=True, hide_index=True, height=400)
                    st.download_button("Download opponent matchup research CSV",
                                       data=_ai2_pd.DataFrame(matchup_rows).to_csv(index=False).encode("utf-8-sig"),
                                       file_name="ai2_nfl_opponent_matchups.csv",
                                       mime="text/csv", key="nfl_opponent_research_download")
                else:
                    st.warning("Opponent defensive coverage evidence not available for this slate.")
                if matchup_summary.get("historical_data_issue"):
                    st.warning("Historical opponent data unavailable: " +
                               str(matchup_summary["historical_data_issue"]))
                st.caption(matchup_summary.get("coverage_matchup_status",
                    "Exact receiver-cornerback routes and shadow coverages not verified."))
                st.caption("An OUT defender is a clue, not proof of a particular WR matchup. No automatic positive player projections are credited without measured alignment and calibrated evidence.")

            with st.expander("Portfolio overlap and distinct winning paths", expanded=True):
                diversity = portfolio_overlap_audit(ranked)
                summary = diversity["summary"]
                st.caption(
                    f"{summary['distinct_qbs']} distinct quarterbacks across {summary['lineups']} lineups · "
                    f"{summary['pairs_at_least_alert']} of {summary['pairs']} lineup pairs share at least seven players."
                )
                st.info(summary["concentration_policy"])
                if diversity["near_duplicate_pairs"]:
                    st.subheader("Highly similar lineup pairs")
                    st.dataframe(_ai2_pd.DataFrame(diversity["near_duplicate_pairs"]),
                                 hide_index=True, use_container_width=True)
                else:
                    st.success("No lineup pairs share seven or more players.")
                st.subheader("Quarterback exposure and stack scripts")
                st.dataframe(_ai2_pd.DataFrame(diversity["lineup_scripts"]),
                             hide_index=True, use_container_width=True)
                st.caption("High overlap is a risk/return decision, not automatically a defect. Game totals, market ownership and receiver coverage matchups are not yet validated inputs to this optimizer.")

            with st.expander("Player exposure and lineup explanations"):
                st.dataframe(exposures, hide_index=True, use_container_width=True)
                for line_no, entry in enumerate(ranked, 1):
                    st.write(f"Lineup {line_no} · {entry['contest_tier']} · "
                             f"Mean {entry['simulated_mean']:.1f} · "
                             f"P90 {entry['simulated_p90']:.1f} · P99 {entry['simulated_p99']:.1f}")
                    explanation = entry.get("selection_explanation")
                    if explanation:
                        st.write(explanation)

            if len(ranked) == 20:
                with st.expander("Review contest assignments · no roster changes"):
                    st.caption("Swap a $3 and a $0.50 lineup without changing any of the nine players. All five $3 entries remain subject to premium-risk and current-slate exclusion rules.")
                    premium_numbers = [i for i, row in enumerate(ranked, 1)
                                       if row["contest_tier"] == "$3"]
                    regular_numbers = [i for i, row in enumerate(ranked, 1)
                                       if row["contest_tier"] != "$3"]
                    premium_one = st.selectbox("Existing $3 lineup", premium_numbers,
                                                key="nfl_premium_swap_from")
                    regular_one = st.selectbox("Existing $0.50 lineup", regular_numbers,
                                                key="nfl_premium_swap_to")
                    if st.button("Swap contest assignments", key="nfl_confirm_tier_swap"):
                        try:
                            st.session_state["ai2_oneclick_review_ranked"] = swap_contest_tiers(
                                ranked, premium_one, regular_one,
                                premium_excluded_names=output_one["audit"].get(
                                    "premium_only_exclusions", ()))
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))

                if st.button("APPROVE AND SAVE THESE 20 LINEUPS", key="nfl_approve_20"):
                    try:
                        payload = snapshot_approved(ranked, current_digest, int(seed_one))
                        _ai2_save_file(DATA_DIR, "NFL", "Approved 20-lineup portfolio",
                                       "ai2_nfl_approved_20.json", payload)
                        st.success("Approved lineup rosters and contest tiers saved separately. Future model builds cannot overwrite them.")
                    except (ValueError, OSError, sqlite3.Error) as exc:
                        st.error(f"Could not save approval: {exc}")

        with st.expander("Protected approved lineups and previous entries"):
            st.caption("The app never silently replaces human-approved lineups. Save a portable backup because Streamlit-hosted local storage may be temporary.")
            approved_file = get_upload("NFL", "Approved 20-lineup portfolio")
            if approved_file:
                try:
                    snapshot = load_approved(approved_file[1])
                    st.success(f"Saved approval: {snapshot['approved_utc']}")
                    approved_table, _, _ = review_tables(snapshot["lineups"])
                    st.dataframe(approved_table, hide_index=True, use_container_width=True)
                    st.download_button("Download protected approved lineup snapshot",
                                       data=approved_file[1],
                                       file_name="team_ai2_APPROVED_portfolio.json",
                                       mime="application/json", key="nfl_saved_approval")
                except (ValueError, KeyError, TypeError) as exc:
                    st.error(f"Existing approval needs review: {exc}")
            else:
                st.caption("No 20-lineup portfolio has been approved and saved in this app.")
            restore_approval = st.file_uploader(
                "Restore a previously downloaded approved portfolio JSON (optional)",
                type=["json"], key="nfl_restore_approved_json")
            if st.button("Restore approved portfolio", key="nfl_restore_approval",
                         disabled=restore_approval is None):
                try:
                    original = restore_approval.getvalue()
                    restored = load_approved(original)
                    if len(restored["lineups"]) != 20:
                        raise ValueError("Only 20-lineup approved portfolio snapshots can be restored")
                    _ai2_save_file(DATA_DIR, "NFL", "Approved 20-lineup portfolio",
                                   restore_approval.name, original)
                    st.success("Saved approved lineups restored separately from automatic builds.")
                    st.rerun()
                except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
                    st.error(f"Could not restore approval: {exc}")
            prior_entries = st.file_uploader(
                "Import your already-approved DraftKings entries (optional; stored separately)",
                type=["csv"], key="nfl_existing_approved_entries")
            if st.button("Protect imported entries", key="nfl_save_existing",
                         disabled=prior_entries is None):
                try:
                    _ai2_save_file(DATA_DIR, "NFL", "Previously approved DraftKings entries",
                                   prior_entries.name, prior_entries.getvalue())
                    st.success("Your earlier entries are saved separately and were not modified.")
                except (ValueError, OSError, sqlite3.Error) as exc:
                    st.error(str(exc))
            old = get_upload("NFL", "Previously approved DraftKings entries")
            if old:
                st.download_button("Download preserved previous entries",
                                   old[1], file_name=old[0], mime="text/csv",
                                   key="nfl_preserved_previous_entries")

        st.caption("One click automatically fetches available NFL research. No lineup is exported when core source or roster checks fail. News reporting and prediction accuracy remain experimental.")
        st.stop()

    if sport == "PGA":
        # Same app, one native V11 PGA workspace (DFS, round 3-balls, Showdown).
        import runpy as _ai2_pga_runpy
        _ai2_pga_runpy.run_path(str(Path(__file__).parent / "ai2_pga_workspace.py"), run_name="__main__")
        st.stop()

    if sport == "NASCAR":
        # NASCAR uses the V2.14 race engine on one consolidated surface.
        # NFL and NBA routing remains unchanged.
        import runpy as _ai2_nascar_runpy
        _ai2_nascar_runpy.run_path(str(Path(__file__).parent / "ai2_nascar_workspace.py"), run_name="__main__")
        st.stop()

    if sport == "NBA":
        # NBA is a native one-click props / parlay engine in the existing unified UI.
        # Preserve the working NFL and other sport routes verbatim.
        import runpy as _ai2_runpy
        _ai2_runpy.run_path(str(Path(__file__).parent / "ai2_nba_workspace.py"), run_name="__main__")
        st.stop()

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
                    st.session_state.pop("nfl_sim_ranked_candidates", None)
                    st.session_state.pop("nfl_research_draws", None)
                    st.session_state.pop("nfl_research_players", None)
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
            st.subheader("NFL Classic eligibility limits")
            st.caption("Applied before optimization; RB/WR filters include FLEX. Low-salary exposure caps apply during correlated portfolio selection.")
            wr_floor = st.number_input("Minimum WR salary ($)", min_value=3000, max_value=10000, value=int(load_setting(DB, "NFL", "min_wr_salary", 4000)), step=100, key="nfl_wr_salary_floor")
            rb_floor = st.number_input("Minimum RB salary ($)", min_value=3000, max_value=10000, value=int(load_setting(DB, "NFL", "min_rb_salary", 4500)), step=100, key="nfl_rb_salary_floor")
            verify_role = st.checkbox("Require verified weekly opportunity", value=bool(load_setting(DB, "NFL", "require_opportunity", True)), key="nfl_require_opportunity")
            value_ceiling = st.number_input("Low-salary WR/RB band: below ($)", min_value=4000, max_value=10000, value=int(load_setting(DB, "NFL", "value_salary_ceiling", 5000)), step=100, key="nfl_value_salary_ceiling")
            value_max = st.number_input("Max low-salary player appearances, all contests", min_value=1, max_value=20, value=int(load_setting(DB, "NFL", "value_max_appearances", 2)), key="nfl_value_max")
            premium_value_max = st.number_input("Max low-salary player appearances in $3 contest", min_value=1, max_value=int(value_max), value=min(int(value_max), int(load_setting(DB, "NFL", "premium_value_max_appearances", 1))), key="nfl_premium_value_max")
            if st.button("Save NFL eligibility limits", key="nfl_save_eligibility"):
                for key, val in dict(min_wr_salary=int(wr_floor), min_rb_salary=int(rb_floor), require_opportunity=bool(verify_role), value_salary_ceiling=int(value_ceiling), value_max_appearances=int(value_max), premium_value_max_appearances=int(premium_value_max), p90_efficiency_retention=0.95).items():
                    save_setting(DB, "NFL", key, val)
                st.success("NFL limits saved; rerun optimizer and simulations for updated P90/$1,000.")
            nfl_eligibility_kwargs = dict(min_wr_salary=int(wr_floor), min_rb_salary=int(rb_floor), require_opportunity=bool(verify_role))
            nfl_portfolio_kwargs = dict(**nfl_eligibility_kwargs, value_salary_ceiling=int(value_ceiling), value_max_appearances=int(value_max), premium_value_max_appearances=int(premium_value_max))
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
                        verified_rows = eligible_nfl_rows(saved_rows, **nfl_eligibility_kwargs)
                        st.caption(f"{len(saved_rows)} matched projection rows · {len(verified_rows)} pass player eligibility checks. Flags are supplied by the data and are not independent certification.")
                        if len(verified_rows) < 9:
                            st.warning("Current saved projections cannot produce a valid nine-player NFL Classic lineup.")
                except (ValueError, KeyError, TypeError) as exc:
                    st.error(f"Saved slate needs attention: {exc}")
            st.subheader("Auto-build historical player baselines")
            st.caption("Fetches real NFL weekly statistics, matches them to your saved DraftKings player IDs, and estimates pregame partial fantasy-point baselines. This is NOT yet an injury-adjusted, independently verified weekly projection.")
            if st.button("Fetch real stats and calculate baselines", key="nfl_auto_baselines"):
                try:
                    from ai2_nfl_inputs import parse_csv_payload
                    from props_data import fetch_weekly
                    from ai2_nfl_auto_baselines import build_historical_baselines
                    saved_salary = get_upload("NFL", "DraftKings salaries")
                    if not saved_salary:
                        raise ValueError("Save DraftKings salaries in My Slate first.")
                    history = fetch_weekly([2025, 2026])
                    result = build_historical_baselines(
                        parse_csv_payload(saved_salary[1]), history, 2026, 7)
                    try:
                        from ai2_nfl_matchup_diagnostics import defensive_allowance, apply_matchup_diagnostics
                        allowance = defensive_allowance(history, 2026, 7)
                        result = apply_matchup_diagnostics(result, allowance)
                        st.session_state["nfl_matchup_allowance_csv"] = allowance.to_csv(index=False)
                        st.session_state.pop("nfl_matchup_allowance_error", None)
                    except (ValueError, KeyError, TypeError) as matchup_exc:
                        st.session_state.pop("nfl_matchup_allowance_csv", None)
                        st.session_state["nfl_matchup_allowance_error"] = str(matchup_exc)
                    try:
                        from ai2_nfl_injury_diagnostics import opportunity_redistribution
                        result, injury_audit = opportunity_redistribution(result)
                        st.session_state["nfl_injury_opportunity_audit_csv"] = injury_audit.to_csv(index=False)
                        st.session_state["nfl_injury_opportunity_audit_count"] = len(injury_audit)
                        st.session_state.pop("nfl_injury_opportunity_error", None)
                    except (ValueError, KeyError, TypeError) as injury_exc:
                        st.session_state.pop("nfl_injury_opportunity_audit_csv", None)
                        st.session_state["nfl_injury_opportunity_error"] = str(injury_exc)
                    try:
                        from ai2_nfl_dst_diagnostics import historical_dst_indicators, attach_dst_indicators
                        dst_indicators = historical_dst_indicators(history, 2026, 7)
                        result = attach_dst_indicators(result, dst_indicators)
                        st.session_state["nfl_dst_indicators_csv"] = dst_indicators.to_csv(index=False)
                        st.session_state.pop("nfl_dst_indicators_error", None)
                    except (ValueError, KeyError, TypeError) as dst_exc:
                        st.session_state.pop("nfl_dst_indicators_csv", None)
                        st.session_state["nfl_dst_indicators_error"] = str(dst_exc)
                    matched = int(result["match_status"].eq("matched").sum())
                    st.session_state["nfl_auto_baselines_csv"] = result.to_csv(index=False)
                    st.session_state["nfl_auto_baselines_summary"] = (matched, len(result))
                    st.session_state["nfl_auto_baselines_reasons"] = result["readiness_reason"].value_counts().to_dict()
                    # Walk-forward uses only games before each evaluated week;
                    # never treat these diagnostics as future-week certification.
                    from props_data import walkforward_dk_offense
                    try:
                        backtest = walkforward_dk_offense(history, 2026, first_week=4, last_week=6)
                    except (ValueError, KeyError, TypeError) as audit_exc:
                        import pandas as pd
                        backtest = pd.DataFrame()
                        st.session_state["nfl_auto_baselines_backtest_error"] = str(audit_exc)
                    else:
                        st.session_state.pop("nfl_auto_baselines_backtest_error", None)
                    st.session_state["nfl_auto_baselines_backtest_csv"] = backtest.to_csv(index=False)
                    if backtest.empty:
                        st.session_state["nfl_auto_baselines_backtest_summary"] = None
                    else:
                        st.session_state["nfl_auto_baselines_backtest_summary"] = {
                            "observations": len(backtest),
                            "mae": float(backtest["absolute_error"].mean()),
                            "bias": float(backtest["error"].mean()),
                        }
                    st.session_state["nfl_public_history"] = history
                    from props_data import make_rates
                    st.session_state["nfl_public_rates"] = make_rates(history, 2026, 7)
                    st.session_state["nfl_public_rates_key"] = (2026, 7)
                    st.session_state["nfl_public_rates_meta"] = {
                        "target": "2026 Week 7",
                        "fetched_utc": datetime.now(timezone.utc).isoformat(),
                        "players": len(st.session_state["nfl_public_rates"]),
                    }
                except (ValueError, KeyError, TypeError, ImportError, OSError) as exc:
                    st.session_state.pop("nfl_auto_baselines_csv", None)
                    st.error(f"Automatic historical baseline failed: {exc}")
            if st.session_state.get("nfl_auto_baselines_csv"):
                matched, total = st.session_state["nfl_auto_baselines_summary"]
                st.success(f"Historical baselines prepared: {matched} of {total} salary rows matched with complete partial scoring data.")
                st.download_button("Download historical baseline review CSV",
                    st.session_state["nfl_auto_baselines_csv"],
                    file_name="nfl_week7_historical_baselines_review.csv",
                    mime="text/csv", key="nfl_auto_baselines_download")
                diagnostics = st.session_state.get("nfl_auto_baselines_backtest_summary")
                if diagnostics:
                    st.info(
                        f"Historical walk-forward check (2026 Weeks 4–6): "
                        f"{diagnostics['observations']} player-week comparisons · "
                        f"mean absolute error {diagnostics['mae']:.2f} partial DK points · "
                        f"average prediction minus actual {diagnostics['bias']:+.2f}. "
                        "This evaluates the naive historical baseline, not the full weekly simulator."
                    )
                    st.download_button(
                        "Download historical backtest audit CSV",
                        st.session_state["nfl_auto_baselines_backtest_csv"],
                        file_name="nfl_week7_historical_walkforward_audit.csv",
                        mime="text/csv", key="nfl_auto_baselines_backtest_download")
                else:
                    st.caption("Historical walk-forward validation has no complete observations for Weeks 4–6.")
                if st.session_state.get("nfl_matchup_allowance_csv"):
                    st.download_button(
                        "Download observed opponent allowance audit CSV",
                        st.session_state["nfl_matchup_allowance_csv"],
                        file_name="nfl_week7_opponent_allowance_audit.csv",
                        mime="text/csv", key="nfl_matchup_allowance_download")
                    st.caption("Matchup-adjusted partial scores appear in the baseline review CSV. Opponent factors are shrunk toward league averages and capped to limit sparse-sample effects; they are exploratory, not verified projections.")
                elif st.session_state.get("nfl_matchup_allowance_error"):
                    st.caption("Matchup adjustment unavailable: " + st.session_state["nfl_matchup_allowance_error"])
                if st.session_state.get("nfl_injury_opportunity_audit_count"):
                    st.download_button(
                        "Download provisional injury opportunity conservation audit",
                        st.session_state["nfl_injury_opportunity_audit_csv"],
                        file_name="nfl_week7_injury_opportunity_audit.csv",
                        mime="text/csv", key="nfl_injury_opportunity_audit_download")
                    st.caption("Adjusted carries and targets appear in the baseline review CSV. They reflect DraftKings OUT/IR designations, not independently confirmed official injuries; no player is certified.")
                elif st.session_state.get("nfl_injury_opportunity_error"):
                    st.caption("Injury opportunity diagnostic unavailable: " + st.session_state["nfl_injury_opportunity_error"])
                if st.session_state.get("nfl_dst_indicators_csv"):
                    st.download_button(
                        "Download historical DST opponent indicators",
                        st.session_state["nfl_dst_indicators_csv"],
                        file_name="nfl_week7_dst_indicators.csv",
                        mime="text/csv", key="nfl_dst_indicators_download")
                    st.caption("DST indicators include opponent passing interceptions and offensive TDs; sacks, fumble recoveries, return TDs, and official points allowed are not yet covered. No DST is marked verified.")
                elif st.session_state.get("nfl_dst_indicators_error"):
                    st.caption("DST historical indicators unavailable: " + st.session_state["nfl_dst_indicators_error"])
                st.caption("Unresolved reasons across the complete DraftKings salary pool:")
                for reason, count in st.session_state.get("nfl_auto_baselines_reasons", {}).items():
                    st.write(f"• {count} — {reason}")
                st.warning("Provisional projections are calculated from historical football statistics, not DK fantasy-point averages. They are NOT independently verified current-week projections; DST and player roles still require additional modeling and checks. No lineup export is authorized.")
            st.divider()
            st.subheader("Real-player historical simulation preview")
            st.caption("Uses your saved DraftKings player IDs, real salaries and nflverse historical statistics. Current-week opportunity is NOT verified. This is a provisional review, not a contest-ready portfolio.")
            if st.button("Run 10,000 football-stat scenarios with REAL players", key="nfl_real_preview"):
                try:
                    import pandas as pd
                    from ai2_nfl_inputs import parse_csv_payload
                    from ai2_nfl_real_slate_preview import prepare_real_slate_preview
                    from props_data import fetch_weekly
                    from dfs_stats_engine import simulate_stat_driven_dfs
                    from ai2_nfl_sim_portfolio import select_simulated_portfolio
                    salaries = get_upload("NFL", "DraftKings salaries")
                    if not salaries:
                        raise ValueError("Save your real DraftKings salary CSV in My Slate first.")
                    history = st.session_state.get("nfl_public_history")
                    if history is None or history.empty:
                        history = fetch_weekly([2025, 2026])
                    from ai2_nfl_official_injuries import (
                        fetch_official_injury_statuses, match_injury_statuses)
                    salary_rows = parse_csv_payload(salaries[1])
                    official_overrides = {}
                    try:
                        official_statuses, injury_meta = fetch_official_injury_statuses()
                        official_overrides = match_injury_statuses(salary_rows, official_statuses)
                        st.caption(
                            f"Official NFL injury report checked {injury_meta['checked_utc']} UTC; "
                            f"{len(official_overrides)} explicit game designations matched to DK IDs. "
                            "Missing designation does not establish availability.")
                    except (ValueError, OSError, TimeoutError) as injury_exc:
                        st.warning(
                            f"Official injury status unavailable ({injury_exc}). "
                            "Continuing historical PREVIEW using saved DK statuses only; "
                            "do not use as final contest entries.")
                    pool, rates = prepare_real_slate_preview(
                        salary_rows, history, 2026, 7, current_statuses=official_overrides)
                    st.info(f"Matched {len(pool)} real eligible historical-baseline players. Running actual football-statistics engine.")
                    players, draws, stat_draws = simulate_stat_driven_dfs(
                        pool, rates, n_sims=10000, allow_role_proxies=False)
                    # Scenario: exclude Hurst and questionable Nacua; retain Tucker with reduced exposure.
                    excluded_names = {"ted hurst iii", "puka nacua"}
                    build_pool = pool[~pool["Name"].astype(str).str.casefold().isin(excluded_names)].copy()
                    st.caption("Scenario exclusions: Ted Hurst III and Puka Nacua. Tre Tucker remains eligible.")
                    candidates = build_classic_candidates(
                        build_pool.to_dict("records"), count=1200, attempts=120000, max_exposure=1.0,
                        salary_floor=49600, **nfl_eligibility_kwargs)
                    if len(candidates) < 20:
                        raise ValueError(f"Only {len(candidates)} legal real-player candidates; need at least 20.")
                    ranked = select_simulated_portfolio(
                        candidates, players, draws, count=20, max_exposure=0.65,
                        upside_share=0.25, **nfl_portfolio_kwargs)
                    st.session_state["nfl_real_preview_players"] = players
                    st.session_state["nfl_real_preview_draws"] = draws
                    st.session_state["nfl_real_preview_stat_draws"] = stat_draws
                    st.session_state["nfl_real_preview_ranked"] = ranked
                    # Injury uncertainty remains visible for human review;
                    # questionable players are not automatically declared out.
                    flagged = {}
                    for item in ranked:
                        for player in item["candidate"]["players"]:
                            status = str(player.get("dk_status", "")).upper().strip()
                            if status in {"Q", "QUESTIONABLE", "LIMITED"}:
                                key = (str(player["ID"]), str(player["Name"]), status)
                                flagged[key] = flagged.get(key, 0) + 1
                    st.session_state["nfl_real_preview_injury_flags"] = pd.DataFrame(
                        [{"ID": key[0], "Player": key[1], "DK status": key[2],
                          "Lineups": n, "Exposure": f"{n * 5}%"}
                         for key, n in sorted(flagged.items(), key=lambda entry: -entry[1])])
                    st.session_state["nfl_real_preview_table"] = pd.DataFrame([
                        {"Lineup": i + 1, "Type": item["portfolio_role"],
                         "Players": ", ".join(p["Name"] for p in item["candidate"]["players"]),
                         "Salary": item["candidate"]["salary"],
                         "Simulated mean": round(item["simulated_mean"], 1)}
                        for i, item in enumerate(ranked)])
                    st.session_state["nfl_real_preview_csv"] = export_classic_candidates(
                        [item["candidate"] for item in ranked])
                    st.success("20 real-player lineups produced from historical-stat scenarios. NOT independently verified for current-week entry.")
                except (ValueError, KeyError, TypeError, IndexError, ImportError, OSError) as exc:
                    st.session_state.pop("nfl_real_preview_csv", None)
                    st.error(f"Real-player preview could not finish: {exc}")
            if st.session_state.get("nfl_real_preview_ranked") and st.session_state.get("nfl_real_preview_players") is not None:
                import numpy as np
                import pandas as pd
                sim_players = st.session_state["nfl_real_preview_players"]
                sim_draws = st.session_state["nfl_real_preview_draws"]
                player_report = sim_players[[x for x in ["ID", "Name", "Position", "TeamAbbrev", "Salary"] if x in sim_players.columns]].copy()
                player_report["sim_mean"] = np.mean(sim_draws, axis=0).round(2)
                player_report["sim_median"] = np.median(sim_draws, axis=0).round(2)
                player_report["sim_p10"] = np.percentile(sim_draws, 10, axis=0).round(2)
                player_report["sim_p90"] = np.percentile(sim_draws, 90, axis=0).round(2)
                if "Salary" in player_report:
                    player_report["points_per_1000"] = (player_report["sim_mean"] / (pd.to_numeric(player_report["Salary"], errors="coerce") / 1000)).round(2)
                lineup_report = pd.DataFrame([{
                    "lineup": i + 1,
                    "role": item.get("portfolio_role", ""),
                    "players": ", ".join(str(x.get("Name", x.get("ID"))) for x in item["candidate"]["players"]),
                    "salary": item["candidate"]["salary"],
                    "sim_mean": round(item["simulated_mean"], 2),
                    "sim_p90": round(item["simulated_p90"], 2),
                } for i, item in enumerate(st.session_state["nfl_real_preview_ranked"])])
                from ai2_nfl_historical_replay import audit_simulation_outputs
                integrity = audit_simulation_outputs(sim_players, sim_draws,
                    st.session_state["nfl_real_preview_ranked"], expected_sims=10000,
                    max_exposure=0.65)
                with st.expander("Model integrity audit (technical details)", expanded=False):
                    if integrity["status"] == "PASS":
                        st.success("Simulation matrix, lineup scoring, roster rules and exposure passed structural checks.")
                    else:
                        st.error("Simulation integrity FAILED. Do not enter these lineups.")
                        for issue in integrity["issues"]:
                            st.write("• " + issue)
                    st.caption("This does not validate player projections, historical calibration or expected ROI.")
                with st.expander("Evaluate our hand-built Stroud / CMC lineup", expanded=True):
                    st.caption("Scores this exact nine-player roster against the SAME 10,000 correlated scenarios already in memory. Does not add individual P90 scores.")
                    if st.button("Evaluate hand-built lineup", key="nfl_handbuilt_eval"):
                        try:
                            from ai2_nfl_lineup import check_classic_lineup
                            wanted = [
                                ("QB", "C.J. Stroud"), ("RB", "Christian McCaffrey"),
                                ("RB", "Chuba Hubbard"), ("WR", "Puka Nacua"),
                                ("WR", "Ja'Marr Chase"), ("WR", "Jaylin Noel"),
                                ("TE", "Tyler Higbee"), ("TE", "Michael Mayer"),
                                ("DST", "Colts")]
                            roster = []
                            indexes = []
                            for pos, name in wanted:
                                matched = sim_players.index[
                                    sim_players["Name"].astype(str).str.casefold().eq(name.casefold())
                                    & sim_players["Position"].astype(str).str.upper().eq(pos)
                                ].tolist()
                                if len(matched) != 1:
                                    raise ValueError(f"Expected one simulated {pos} named {name}; found {len(matched)}.")
                                idx = matched[0]
                                indexes.append(sim_players.index.get_loc(idx))
                                roster.append(sim_players.loc[idx].to_dict())
                            errors = check_classic_lineup(roster, salary_floor=49600, salary_cap=50000)
                            if errors:
                                raise ValueError("; ".join(errors))
                            totals = np.asarray(sim_draws)[:, indexes].sum(axis=1)
                            summary = {
                                "Salary": int(sum(float(p["Salary"]) for p in roster)),
                                "Mean": round(float(totals.mean()), 2),
                                "Median": round(float(np.median(totals)), 2),
                                "P90": round(float(np.percentile(totals, 90)), 2),
                                "P99": round(float(np.percentile(totals, 99)), 2),
                                "Chance 142+": round(float(np.mean(totals >= 142) * 100), 2),
                                "Chance 160+": round(float(np.mean(totals >= 160) * 100), 2),
                                "Chance 180+": round(float(np.mean(totals >= 180) * 100), 2),
                                "Chance 200+": round(float(np.mean(totals >= 200) * 100), 2),
                                "Chance 214+": round(float(np.mean(totals >= 214) * 100), 2),
                            }
                            st.session_state["nfl_handbuilt_result"] = summary
                            st.session_state["nfl_handbuilt_scenarios"] = pd.DataFrame({
                                "scenario": np.arange(1, len(totals) + 1), "dk_points": totals})
                        except (ValueError, KeyError, IndexError, TypeError) as exc:
                            st.error(f"Could not score hand-built lineup: {exc}")
                    if st.session_state.get("nfl_handbuilt_result"):
                        st.dataframe(pd.DataFrame([st.session_state["nfl_handbuilt_result"]]), hide_index=True)
                        st.download_button("Download hand-built lineup scenario scores (CSV)",
                            st.session_state["nfl_handbuilt_scenarios"].to_csv(index=False),
                            file_name="TEAM_AI2_HANDBUILT_SCENARIOS.csv", mime="text/csv",
                            key="nfl_handbuilt_scenarios_download")
                        st.warning("Provisional historical-baseline simulation only; current player availability and forecast calibration remain unverified.")
                st.subheader("Simulation statistics")
                st.caption("10,000 shared historical-stat scenarios; model estimates, not validated current-week forecasts.")
                st.dataframe(lineup_report, hide_index=True)
                from ai2_nfl_export import slot_classic_lineup
                import csv
                import io
                named_buffer = io.StringIO()
                named_writer = csv.writer(named_buffer)
                named_writer.writerow(["Lineup", "QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST", "Salary", "Sim mean", "Sim P90"])
                for i, item in enumerate(st.session_state["nfl_real_preview_ranked"], 1):
                    named_writer.writerow([i] + [
                        str(p.get("Name") or "UNKNOWN") for p in slot_classic_lineup(item["candidate"]["players"])
                    ] + [item["candidate"]["salary"], round(item["simulated_mean"], 2), round(item["simulated_p90"], 2)])
                st.download_button("Download named lineup review (CSV)", named_buffer.getvalue(),
                    file_name="TEAM_AI2_NAMED_LINEUP_REVIEW.csv", mime="text/csv", key="nfl_named_lineup_download")
                st.download_button("Download player simulation stats (CSV)", player_report.to_csv(index=False),
                    file_name="TEAM_AI2_PLAYER_SIM_STATS.csv", mime="text/csv", key="nfl_player_sim_stats_download")
                st.download_button("Download lineup simulation stats (CSV)", lineup_report.to_csv(index=False),
                    file_name="TEAM_AI2_LINEUP_SIM_STATS.csv", mime="text/csv", key="nfl_lineup_sim_stats_download")
            # Sportsbook prop integration paused; prioritize DFS simulation and lineup validation.
            if st.session_state.get("nfl_real_preview_csv"):
                st.dataframe(st.session_state["nfl_real_preview_table"], hide_index=True)
                st.warning("PROVISIONAL: Historical inputs, unverified roles/injuries, incomplete DST prior. Review every player; do not enter these as finalized contest lineups.")
                flags = st.session_state.get("nfl_real_preview_injury_flags")
                if flags is not None and not flags.empty:
                    st.warning(f"{len(flags)} questionable/limited players appear in this portfolio. Confirm official status before entry.")
                    st.dataframe(flags, hide_index=True)

                st.download_button("Download REAL-player provisional portfolio (review only)",
                                   st.session_state["nfl_real_preview_csv"],
                                   file_name="TEAM_AI2_REAL_PLAYERS_PROVISIONAL.csv",
                                   mime="text/csv", key="nfl_real_preview_download")
            st.divider()
            st.subheader("Build NFL Classic Portfolio")
            st.caption("One-button target: upload a DraftKings salary slate, then build. Automated verified current-week projections are not yet available, so the engine will stop rather than create misleading picks.")
            st.caption("Uses saved DraftKings salaries, independently verified weekly projections, and observed historical player rates. Simulates football statistics before DraftKings scoring; no sportsbook inputs.")
            st.caption("Uses eligible weekly projections when available; historical rates alone do not verify current roles or injuries.")
            if st.button("Build lineups · run 10,000 simulations", key="nfl_run_full_sim"):
                try:
                    import pandas as pd
                    import numpy as np
                    from ai2_nfl_simulation_adapter import run_verified_stat_simulation
                    from ai2_nfl_sim_portfolio import select_simulated_portfolio
                    salaries = get_upload("NFL", "DraftKings salaries")
                    projections = get_upload("NFL", "Projections")
                    if not salaries or not projections:
                        raise ValueError("Save both DraftKings salaries and independently verified weekly projections in My Slate first.")
                    rates = st.session_state.get("nfl_public_rates")
                    # Cached rates from another target week can leak later results.
                    if st.session_state.get("nfl_public_rates_key") != (2026, 7):
                        rates = None
                    if rates is None or rates.empty:
                        # A Streamlit reboot clears session state. Restore historical
                        # rates from the public source when a verified slate is run,
                        # rather than forcing a separate research-button click.
                        from props_data import fetch_weekly, make_rates
                        history = fetch_weekly([2025, 2026])
                        rates = make_rates(history, 2026, 7)
                        if rates.empty:
                            raise ValueError("No pregame historical player rates were available.")
                        st.session_state["nfl_public_rates"] = rates
                        st.session_state["nfl_public_rates_key"] = (2026, 7)
                        st.session_state["nfl_public_rates_meta"] = {
                            "source": "nflverse weekly stats",
                            "target_season": 2026,
                            "target_week": 7,
                            "latest_observed_season": int(rates["latest_observed_season"].max()),
                            "latest_observed_week": int(rates.loc[
                                rates["latest_observed_season"].eq(2026),
                                "latest_observed_week"].max()) if rates["latest_observed_season"].eq(2026).any() else None,
                        }

                    rows, issues = inspect_saved_nfl_inputs(salaries, projections)
                    if issues:
                        raise ValueError("; ".join(issues))
                    eligible = eligible_nfl_rows(rows, **nfl_eligibility_kwargs)
                    if len(eligible) < 9:
                        raise ValueError("Too few independently verified, eligible players for a legal lineup.")
                    pool = pd.DataFrame(eligible)
                    if "optimizer_eligible" not in pool.columns:
                        pool["optimizer_eligible"] = True
                    players, draws, stat_draws = run_verified_stat_simulation(
                        pool, rates, n_sims=10000)
                    if draws.shape != (10000, len(players)) or not np.isfinite(draws).all():
                        raise ValueError("Simulation results are incomplete or nonfinite.")
                    candidates = build_classic_candidates(eligible, count=600, attempts=120000, max_exposure=1.0,
                        salary_floor=49600, **nfl_eligibility_kwargs)
                    if len(candidates) < 20:
                        raise ValueError(f"Only {len(candidates)} legal candidate lineups were found; 20 required.")
                    ranked = select_simulated_portfolio(
                        candidates, players, draws, count=20,
                        max_exposure=0.35, upside_share=0.25, **nfl_portfolio_kwargs)
                    if len(ranked) != 20:
                        raise ValueError("Simulation did not produce a complete 20-lineup portfolio.")
                    csv_data = export_classic_candidates([entry["candidate"] for entry in ranked])
                    st.session_state["nfl_research_players"] = players.copy()
                    st.session_state["nfl_research_draws"] = draws
                    st.session_state["nfl_candidates"] = candidates
                    st.session_state["nfl_sim_ranked_candidates"] = ranked
                    st.session_state["nfl_full_sim_export"] = csv_data
                    st.success("Completed 10,000 correlated football-stat scenarios and selected 20 simulation-ranked lineups.")
                except (ValueError, KeyError, TypeError, ImportError, IndexError) as exc:
                    st.session_state.pop("nfl_full_sim_export", None)
                    st.error(f"Build blocked (no unverified fallback): {exc}")
            if st.session_state.get("nfl_full_sim_export"):
                st.download_button(
                    "Download 20-lineup DraftKings roster CSV (review required)",
                    st.session_state["nfl_full_sim_export"],
                    file_name="team_ai2_nfl_simulated_20_review.csv",
                    mime="text/csv", key="nfl_full_sim_download")
                st.warning("Review current official player statuses, projections, contest slate, and entry IDs before using this experimental portfolio. Simulated results do not establish profitability.")
            st.divider()
            st.caption("Advanced research tools · historical-rate fetch, diagnostics, and manual simulations")

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
                        meta = st.session_state.get("nfl_public_rates_meta") or {"target": f"{history_season} Week {history_week}", "fetched_utc": "not recorded", "players": len(st.session_state["nfl_public_rates"])}
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
                                from ai2_nfl_empirical_benchmark import walkforward_empirical_distribution
                                empirical = walkforward_empirical_distribution(
                                    st.session_state["nfl_public_history"],
                                    season=int(history_season),
                                    first_week=4,
                                    last_week=int(history_week) - 1,
                                    window=8, min_history=3)
                                st.session_state["nfl_empirical_audit_rows"] = empirical
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
                        empirical = st.session_state.get("nfl_empirical_audit_rows")
                        if empirical is not None and not empirical.empty:
                            st.caption("Historical player-score distribution benchmark (minimum three previous games). This is an empirical benchmark, not the correlated simulator.")
                            st.metric("Empirical benchmark mean absolute error",
                                      f"{empirical['absolute_error'].mean():.2f} DK")
                            st.caption(f"Central 50% coverage: {empirical['covered_50'].mean():.1%} (target ~50%); central 80% coverage: {empirical['covered_80'].mean():.1%} (target ~80%); {len(empirical):,} player-games")
                            st.dataframe(
                                empirical.groupby("position", as_index=False)
                                .agg(player_games=("absolute_error", "size"),
                                     mean_absolute_error=("absolute_error", "mean"),
                                     central_80_coverage=("covered_80", "mean")),
                                use_container_width=True, hide_index=True)
                            st.download_button("Download empirical distribution benchmark",
                                               empirical.to_csv(index=False).encode("utf-8"),
                                               file_name="nfl_empirical_distribution_walkforward.csv",
                                               mime="text/csv", key="nfl_download_empirical_audit")
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
                        st.session_state.pop("nfl_sim_ranked_candidates", None)
                        st.session_state["nfl_research_draws"] = draws
                        st.session_state["nfl_research_players"] = players.copy()
                        from datetime import datetime, timezone
                        st.session_state["nfl_research_run_utc"] = datetime.now(timezone.utc).isoformat()
                        st.session_state["nfl_research_report"] = report
                        st.success(f"Completed {lab_sims:,} experimental game-level draws. Not calibrated or authorized for contest entry.")
                    except (ValueError, KeyError, TypeError, ImportError, IndexError) as exc:
                        st.session_state.pop("nfl_research_report", None)
                        st.session_state.pop("nfl_research_draws", None)
                        st.session_state.pop("nfl_research_players", None)
                        st.error(f"Research simulation blocked: {exc}")
                if "nfl_research_draws" in st.session_state and st.session_state.get("nfl_candidates"):
                    with st.expander("Simulation-ranked NFL Classic portfolio · experimental", expanded=False):
                        st.caption("Ranks existing legal candidate lineups using shared correlated game draws. Requires simulated outcomes for every roster member including DST. This is not projected contest ROI.")
                        if st.button("Rank candidate portfolio by simulation", key="nfl_rank_sim_portfolio"):
                            try:
                                from ai2_nfl_sim_portfolio import select_simulated_portfolio
                                ranked = select_simulated_portfolio(
                                    st.session_state["nfl_candidates"],
                                    st.session_state["nfl_research_players"],
                                    st.session_state["nfl_research_draws"],
                                    count=st.session_state.get("nfl_candidate_count", 20),
                                    max_exposure=st.session_state.get("nfl_candidate_exposure", 35) / 100,
                                    upside_share=0.25, premium_count=min(5, st.session_state.get("nfl_candidate_count", 20)),
                                    **nfl_portfolio_kwargs)
                                st.session_state["nfl_sim_ranked_candidates"] = ranked
                            except (ValueError, KeyError, TypeError) as exc:
                                st.session_state.pop("nfl_sim_ranked_candidates", None)
                                st.error(f"Simulation ranking blocked: {exc}")
                        ranked = st.session_state.get("nfl_sim_ranked_candidates")
                        if ranked:
                            st.dataframe(pd.DataFrame([dict(contest=x["contest_tier"], p90_per_1000=x["p90_per_1000"], role=x["portfolio_role"],
                                mean=round(x["simulated_mean"], 2),
                                p90=round(x["simulated_p90"], 2),
                                p99=round(x["simulated_p99"], 2))
                                for x in ranked]), use_container_width=True, hide_index=True)
                            st.caption(f"Selected {len(ranked)} lineups under eligibility and exposure requirements.")
                            with st.expander("Why was each player selected?", expanded=False):
                                for j, item in enumerate(ranked, 1):
                                    st.write(f"Lineup {j} | {item['contest_tier']} | P90 {item['simulated_p90']:.2f} | P99 {item['simulated_p99']:.2f} | P90/$1K {item['p90_per_1000']:.2f}")
                                    st.caption(item["selection_explanation"]["contest_assignment"] + "; " + item["selection_explanation"]["qb_stack_note"])
                                    for p in item["candidate"]["players"]:
                                        evidence = item["selection_explanation"]["player_opportunity_evidence"][str(p["ID"])]
                                        st.write(f"{p.get('Position')} {p.get('Name', p['ID'])}: {evidence}")
                            st.download_button("Download simulation-ranked DraftKings CSV",
                                export_classic_candidates([x["candidate"] for x in ranked]),
                                file_name="nfl_simulation_ranked_candidates.csv",
                                mime="text/csv", key="nfl_sim_ranked_export")
                if "nfl_research_report" in st.session_state:
                    st.dataframe(st.session_state["nfl_research_report"], use_container_width=True, hide_index=True)
                    with st.expander("Postgame simulation distribution audit", expanded=False):
                        st.caption("After games finish, upload observed DraftKings points for this exact slate. CSV columns: ID, actual_dk; optional game labels are checked when provided on both sides. This does not validate an independent historical holdout unless inputs were frozen before kickoff.")
                        st.caption(f"Simulation generated (UTC): {st.session_state.get('nfl_research_run_utc', 'unknown')}. Record this timestamp with the slate and model version before kickoff.")
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
            st.subheader("Advanced · candidate diagnostics")
            st.caption("Projection-ranked legal candidates, NOT correlated simulations or validated contest-ROI predictions. No sportsbook inputs.")
            st.info("Save independent weekly projections and DK salaries. Low-cost WR/RB require verified opportunity plus numeric projected_targets/recent_targets_per_game/route_participation, or projected_touches/recent_touches_per_game/snap_share; a documented verified injury role is an alternative. Unknown evidence is rejected. All metrics must be independently verified.")
            lineup_count = st.slider("Candidate lineups", 1, 20, 15, key="nfl_candidate_count")
            exposure_limit = st.slider("Maximum player exposure", 20, 100, 35, 5, key="nfl_candidate_exposure")
            if st.button("Build NFL Classic candidates", key="nfl_build_candidates"):
                try:
                    salaries = get_upload("NFL", "DraftKings salaries")
                    projections = get_upload("NFL", "Projections")
                    rows, issues = inspect_saved_nfl_inputs(salaries, projections)
                    if issues:
                        raise ValueError("; ".join(issues))
                    eligible = eligible_nfl_rows(rows, **nfl_eligibility_kwargs)
                    if len(eligible) < 9:
                        raise ValueError("Insufficient verified eligible players. Check weekly projections, opportunity flags, and injuries.")
                    candidates = build_classic_candidates(eligible, count=120,
                                                          max_exposure=1.0)
                    st.session_state.pop("nfl_candidates", None)
                    if not candidates:
                        st.warning("No legal candidates found within the search budget. This does not prove the slate is infeasible.")
                    else:
                        st.session_state["nfl_candidates"] = candidates
                        st.session_state.pop("nfl_sim_ranked_candidates", None)
                        fingerprint = slate_fingerprint(salaries, projections)
                        save_portfolio(DATA_DIR, fingerprint, candidates)
                        st.success(f"Built {len(candidates)} legal projection-ranked candidates for portfolio selection.")
                        if len(candidates) < lineup_count:
                            st.warning(f"Only {len(candidates)} candidate lineups found; requested portfolio size is {lineup_count}.")
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
                    st.session_state["nfl_candidates"] = candidates
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
            eligible_rows = eligible_nfl_rows(rows, **nfl_eligibility_kwargs)
            feasibility = find_classic_lineup(eligible_rows)
            st.write(f"Lineup feasibility: {feasibility.reason} ({feasibility.examined:,} checked)")
            check = validate_pool(rows, source_fresh=False,
                                  historical_validation_passed=False,
                                  lineup_feasibility_passed=feasibility.feasible is True)
            st.write(f"Verified eligible players: {check.eligible_count}")
            for reason in check.reasons:
                st.write("• " + reason)
        st.write("Release policy: no real-money export until data freshness, role/opportunity checks, lineup legality, and historical validation pass.")
