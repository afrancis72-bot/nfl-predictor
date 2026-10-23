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
from ai2_nfl_preflight import validate_pool
from ai2_nfl_inputs import inspect_saved_nfl_inputs
from ai2_nfl_eligibility import eligible_nfl_rows
from ai2_nfl_feasibility import find_classic_lineup
from ai2_nfl_candidate_builder import build_classic_candidates
from ai2_market_challenge import freeze_simulation, compare_frozen_to_props
from ai2_market_identity import parse_identity_csv
from ai2_market_provider import retrieve_market_props

SPORTS = {
    "PGA": {"icon": "⛳", "description": "Classic · Showdown · 3-ball matchups"},
    "NFL": {"icon": "🏈", "description": "Classic · Showdown · Player props"},
    "NASCAR": {"icon": "🏁", "description": "Race simulations · DFS"},
    "NBA": {"icon": "🏀", "description": "DFS · Player props · Parlays"},
}
DATA_DIR = Path(os.environ.get("AI2_DATA_DIR", "./.ai2_data"))
DB = DATA_DIR / "slates.sqlite3"

def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS uploads (
        sport TEXT NOT NULL, kind TEXT NOT NULL, filename TEXT NOT NULL,
        content BLOB NOT NULL, uploaded_at TEXT NOT NULL,
        PRIMARY KEY (sport, kind))""")
    return conn

def save_upload(sport, kind, file):
    payload = file.getvalue()
    if not payload:
        raise ValueError("Cannot save an empty upload.")
    with connect() as conn:
        conn.execute(
            """INSERT INTO uploads(sport,kind,filename,content,uploaded_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(sport,kind) DO UPDATE SET
            filename=excluded.filename,content=excluded.content,
            uploaded_at=excluded.uploaded_at""",
            (sport,kind,file.name,payload,datetime.now(timezone.utc).isoformat()),
        )

def get_upload(sport, kind):
    with connect() as conn:
        return conn.execute(
            "SELECT filename,content,uploaded_at FROM uploads WHERE sport=? AND kind=?",
            (sport,kind),
        ).fetchone()

st.set_page_config(page_title="TEAM AI² Pro", page_icon="🎯", layout="wide")
st.markdown("""<style>
.stApp {background: #080b12;}
div[data-testid="stButton"] button {min-height: 3rem;}
</style>""", unsafe_allow_html=True)
if "sport" not in st.session_state:
    requested = str(st.query_params.get("sport", "")).upper()
    st.session_state.sport = requested if requested in SPORTS else None

def go_home():
    st.session_state.sport = None
    if "sport" in st.query_params:
        del st.query_params["sport"]

if st.session_state.sport is None:
    st.title("TEAM AI² PRO")
    logo = Path(__file__).parent / "assets" / "team_ai2_sports_emblem.png"
    if logo.exists():
        encoded = base64.b64encode(logo.read_bytes()).decode("ascii")
        # The links cover the actual four sport quadrants, while retaining
        # the original logo artwork untouched. The buttons below remain an
        # accessible, mobile-friendly alternative.
        st.markdown(f"""<div style="position:relative;max-width:640px;margin:auto">
          <img src="data:image/png;base64,{encoded}" alt="TEAM AI² Pro four-sport emblem"
               style="width:100%;height:auto;display:block">
          <a href="?sport=PGA" target="_self" aria-label="Open PGA"
             title="PGA" style="position:absolute;left:0;top:0;width:50%;height:50%"></a>
          <a href="?sport=NFL" target="_self" aria-label="Open NFL"
             title="NFL" style="position:absolute;left:50%;top:0;width:50%;height:50%"></a>
          <a href="?sport=NASCAR" target="_self" aria-label="Open NASCAR"
             title="NASCAR" style="position:absolute;left:0;top:50%;width:50%;height:50%"></a>
          <a href="?sport=NBA" target="_self" aria-label="Open NBA"
             title="NBA" style="position:absolute;left:50%;top:50%;width:50%;height:50%"></a>
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
    st.caption("Unified platform prototype · No sport engine connected or certified yet.")
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
                        st.success(f"Built {len(candidates)} legal projection-ranked candidates.")
                        if len(candidates) < lineup_count:
                            st.warning(f"Only {len(candidates)} of {lineup_count} requested lineups met the portfolio exposure constraints.")
                except (ValueError, KeyError, TypeError) as exc:
                    st.session_state.pop("nfl_candidates", None)
                    st.error(f"Build blocked: {exc}")
            candidates = st.session_state.get("nfl_candidates", [])
            if candidates:
                summary = [{"Lineup": i + 1, "Projected DK points": x["projected_points"],
                            "Salary": x["salary"],
                            "Players": ", ".join(str(p.get("Name", p["ID"])) for p in x["players"])}
                           for i, x in enumerate(candidates)]
                st.dataframe(summary, use_container_width=True, hide_index=True)
                st.download_button("Download candidate review JSON",
                                   json.dumps(summary, indent=2), file_name="nfl_candidate_review.json",
                                   mime="application/json")
                st.warning("Review only: DraftKings CSV export and real-money readiness remain disabled pending slate verification, source freshness and historical backtesting.")
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
