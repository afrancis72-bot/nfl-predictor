"""TEAM AI² Pro unified platform prototype.

Run: streamlit run unified_app.py
Persistent uploads use SQLite on a durable volume configured by AI2_DATA_DIR.
This prototype does not claim any of the sport engines are validated.
"""
import os
import base64
import html
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import streamlit as st
from ai2_settings import load_setting, save_setting

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
    tab_data,tab_build,tab_results,tab_health = st.tabs(
        ["My Slate", "Build / Simulate", "My Results", "Engine Health"]
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
                st.success("File saved.")
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
        st.warning("Not ready: lineup generation is disabled until the sport engine and its data checks are integrated.")
    with tab_results:
        st.info("Saved portfolios, bets and post-event audits will appear here.")
    with tab_health:
        st.error("Engine status: NOT CONNECTED")
        st.write("Release policy: no real-money export until data freshness, role/opportunity checks, lineup legality, and historical validation pass.")
