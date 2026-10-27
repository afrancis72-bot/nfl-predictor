from pathlib import Path
from datetime import date
import pandas as pd
import re
import streamlit as st

from international_ingestion import search_courses, geocode_course, location_detail, fetch_course_detail, fetch_course_holes, fetch_weather, IngestionError
from portfolio_optimizer import PortfolioSettings, optimize_portfolio
from pga_predictor_pro import Config, predict_from_dataframes
from golf_first_v11 import simulate_golf_first
from rrg_course_fit import normalize_rrg_course_fit, attach_source_comparison
from rrg_strokes_gained import normalize_rrg_strokes_gained
from mexico_course_evidence import mexico_course_evidence, mexico_course_thesis
from featured_entry import choose_featured_lineup
from round_parlay import fetch_groupings, normalize_groupings_upload, fetch_live_results, fetch_official_prior_rounds, normalize_live_results, build_round_ratings, simulate_groups, build_six_leg_tickets, golfchannel_round_url

ROOT = Path(__file__).resolve().parent

st.title("⛳ Team AI² Pro | PGA")
st.caption("One setup page. Clean model pages. Transparent Course DNA.")

@st.cache_data
def load_repo_csv(name):
    path = ROOT / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def load_weekly(upload, name):
    return pd.read_csv(upload) if upload is not None else load_repo_csv(name)

def validate_advanced_input(df, kind):
    """Validate optional advanced inputs without changing model logic.

    Repository fallbacks that do not satisfy the current contract are ignored
    instead of crashing a weekly build. Explicit uploads remain visible errors.
    """
    if df is None or df.empty:
        return pd.DataFrame(), "not supplied"

    work = df.copy()
    work.columns = [str(c).replace("\ufeff", "").strip() for c in work.columns]
    lower = {c.casefold(): c for c in work.columns}

    required = {
        "player_stats": {"player"},
        "results": {"player", "date"},
        "course_history": {"player"},
    }[kind]

    missing = [c for c in required if c not in lower]
    if missing:
        return pd.DataFrame(), "incompatible: missing " + ", ".join(missing)

    # Normalize required identity/date column casing when necessary.
    ren = {}
    for req in required:
        actual = lower[req]
        if actual != req:
            ren[actual] = req
    if ren:
        work = work.rename(columns=ren)

    if "player" in work.columns:
        work["player"] = work["player"].astype(str).str.strip()
        work = work[~work["player"].str.casefold().isin({"", "nan", "none", "player", "name"})].copy()

    if kind == "results":
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work[work["date"].notna()].copy()
        if work.empty:
            return pd.DataFrame(), "incompatible: no valid dated result rows"

    return work.reset_index(drop=True), "valid"


def _clean_cols(df):
    out = df.copy()
    out.columns = [str(c).replace("\ufeff", "").strip() for c in out.columns]
    return out


def normalize_otis_course_fit(df):
    """Normalize OTIS Advanced Course-Fit export for the current weekly field.

    Rank and Model are retained for audit only and are never used as predictive inputs.
    OTIS component scores are within-field percentiles, so they are used only within
    the current tournament field. Missing values remain missing.
    """
    if df is None or df.empty:
        return pd.DataFrame(), "empty"
    w = _clean_cols(df)
    lookup = {c.casefold(): c for c in w.columns}
    required = ["player", "true skill", "course fit", "form"]
    missing = [c for c in required if c not in lookup]
    if missing:
        return pd.DataFrame(), "unrecognized OTIS Course-Fit CSV (missing " + ", ".join(missing) + ")"
    ren = {
        lookup["player"]: "player",
        lookup["true skill"]: "otis_true_skill",
        lookup["course fit"]: "otis_course_fit",
        lookup["form"]: "otis_form",
    }
    optional = {
        "rank": "otis_rank", "model": "otis_model_audit",
        "fit: app": "otis_fit_app", "fit: ott": "otis_fit_ott",
        "fit: arg": "otis_fit_arg", "fit: putt": "otis_fit_putt",
        "fit: history": "otis_fit_history", "form rds": "otis_form_rds",
        "venue rds": "otis_venue_rds",
    }
    for src, dst in optional.items():
        if src in lookup: ren[lookup[src]] = dst
    w = w.rename(columns=ren)
    keep = [c for c in ["player", "otis_rank", "otis_model_audit", "otis_true_skill", "otis_course_fit", "otis_form", "otis_fit_app", "otis_fit_ott", "otis_fit_arg", "otis_fit_putt", "otis_fit_history", "otis_form_rds", "otis_venue_rds"] if c in w.columns]
    w = w[keep].copy()
    w["player"] = w["player"].astype(str).str.strip()
    for c in keep:
        if c != "player": w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w[~w["player"].str.casefold().isin({"", "nan", "none", "player", "name"})].copy()
    w = w.drop_duplicates(subset=["player"], keep="first").reset_index(drop=True)
    advanced = all(c in w.columns for c in ["otis_fit_app","otis_fit_ott","otis_fit_arg","otis_fit_putt","otis_fit_history","otis_form_rds","otis_venue_rds"])
    return w, ("OTIS Advanced Course-Fit" if advanced else "OTIS Course-Fit")

def normalize_otis_season_stats(df, target_season=2026):
    """Normalize an OTIS Season stats CSV into player_stats.csv-compatible rows."""
    if df is None or df.empty:
        return pd.DataFrame(), "empty"
    w = _clean_cols(df)
    lookup = {c.casefold(): c for c in w.columns}
    # OTIS documents player + season and the stat slugs below. Be tolerant of name/player_name.
    pcol = next((lookup[x] for x in ("player","player_name","name") if x in lookup), None)
    scol = next((lookup[x] for x in ("season","year") if x in lookup), None)
    if pcol is None:
        return pd.DataFrame(), "unrecognized OTIS season stats (missing player/name)"
    if scol is not None:
        season_text = w[scol].astype(str).str.strip()
        mask = season_text.eq(str(target_season))
        if mask.any():
            w = w[mask].copy()
    if pcol != "player": w = w.rename(columns={pcol:"player"})
    w["player"] = w["player"].astype(str).str.strip()
    # Keep OTIS native stat names; build_features already consumes published stat columns when present.
    numeric = [
        "sg_total","sg_off_the_tee","sg_approach","sg_around_green","sg_putting","sg_tee_to_green",
        "measured_rounds","driving_distance","driving_accuracy_pct","gir_pct","scrambling_pct",
        "putts_per_round","scoring_avg","birdie_avg","par3_scoring_avg","par4_scoring_avg",
        "par5_scoring_avg","events_played","cuts_made","top10s","wins","earnings_usd"
    ]
    for c in numeric:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w[~w["player"].str.casefold().isin({"","nan","none","player","name"})].copy()
    return w.drop_duplicates(subset=["player"], keep="first").reset_index(drop=True), "OTIS season stats"

def normalize_otis_event_results(df):
    """Normalize an OTIS Event results CSV into results.csv-compatible rows."""
    if df is None or df.empty:
        return pd.DataFrame(), "empty"
    w = _clean_cols(df)
    lookup = {c.casefold(): c for c in w.columns}
    pcol = next((lookup[x] for x in ("player","player_name","name") if x in lookup), None)
    dcol = next((lookup[x] for x in ("date","event_date","start_date","tournament_date") if x in lookup), None)
    if pcol is None:
        return pd.DataFrame(), "unrecognized OTIS event results (missing player/name)"
    if dcol is None:
        return pd.DataFrame(), "unrecognized OTIS event results (missing date)"
    ren={}
    if pcol!="player": ren[pcol]="player"
    if dcol!="date": ren[dcol]="date"
    if ren: w=w.rename(columns=ren)
    w["player"]=w["player"].astype(str).str.strip()
    w["date"]=pd.to_datetime(w["date"], errors="coerce")
    w=w[w["date"].notna() & ~w["player"].str.casefold().isin({"","nan","none","player","name"})].copy()
    return w.reset_index(drop=True), "OTIS event results"

def build_course_history_from_otis(results_df, course_name):
    """Derive current-venue history from OTIS event results; never invent missing history."""
    if results_df is None or results_df.empty or not str(course_name).strip():
        return pd.DataFrame(), "not available"
    w=results_df.copy()
    lookup={str(c).casefold():c for c in w.columns}
    ccol=next((lookup[x] for x in ("course","course_name","venue") if x in lookup), None)
    if ccol is None:
        return pd.DataFrame(), "OTIS results have no course column"
    def norm(x):
        return "".join(ch for ch in str(x).casefold() if ch.isalnum())
    target=norm(course_name)
    vals=w[ccol].astype(str)
    exact=vals.map(norm).eq(target)
    if not exact.any():
        # Conservative token containment fallback for branding variants.
        tokens=[t for t in re.split(r"\W+", str(course_name).casefold()) if len(t)>=4 and t not in {"golf","course","club"}]
        if tokens:
            exact=vals.str.casefold().map(lambda x: all(t in x for t in tokens))
    h=w[exact].copy()
    if h.empty:
        return pd.DataFrame(), "no OTIS starts matched current course"
    # build_features accepts a player-keyed history table; preserve published OTIS fields.
    return h.reset_index(drop=True), f"{len(h)} OTIS course-history starts"

def normalize_dk_players(df):
    """Accept native DraftKings PGA salary exports or the app's normalized players.csv."""
    if df is None or df.empty:
        return pd.DataFrame(), "empty"

    work = df.copy()
    # Strip BOM/whitespace that can appear in downloaded CSV headers.
    work.columns = [str(c).replace("\ufeff", "").strip() for c in work.columns]
    lookup = {str(c).strip().casefold(): c for c in work.columns}

    # Already-normalized weekly input.
    if "player" in lookup and "salary" in lookup:
        player_col, salary_col = lookup["player"], lookup["salary"]
        source = "normalized"
    else:
        # Native DraftKings PGA exports normally use Name + Salary.
        player_col = None
        for candidate in ("name", "player name", "golfer", "player"):
            if candidate in lookup:
                player_col = lookup[candidate]
                break
        salary_col = None
        for candidate in ("salary", "dk salary"):
            if candidate in lookup:
                salary_col = lookup[candidate]
                break
        if player_col is None or salary_col is None:
            return work, "unrecognized"
        source = "DraftKings native"

    work["player"] = work[player_col].astype(str).str.strip()
    work["salary"] = pd.to_numeric(
        work[salary_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )

    # Remove blank/header-like/bad rows while preserving all useful DK columns.
    bad_names = {"", "nan", "none", "name", "player"}
    work = work[
        work["player"].str.casefold().notna()
        & ~work["player"].str.casefold().isin(bad_names)
        & work["salary"].notna()
        & (work["salary"] > 0)
    ].copy()
    work["salary"] = work["salary"].astype(int)

    # DraftKings native salary exports can flag withdrawn/inactive golfers in a Status column.
    # Treat OUT as a hard exclusion before simulation so an inactive golfer can never reach
    # projections or the optimizer. Preserve every other status unchanged.
    status_col = next((lookup[k] for k in ("status", "player status") if k in lookup), None)
    if status_col is not None:
        status = work[status_col].fillna("").astype(str).str.strip().str.casefold()
        work = work[~status.isin({"out"})].copy()

    # A native salary file should contain one row per golfer; protect the model from duplicates.
    work = work.drop_duplicates(subset=["player"], keep="first").reset_index(drop=True)
    return work, source

@st.cache_data(ttl=3600)
def cached_course_search(query, api_key, country_hint):
    return search_courses(query, api_key, country_hint)

@st.cache_data(ttl=86400)
def cached_osm_search(query, country_hint, resolver_version="10.6.4a"):
    # resolver_version deliberately invalidates cached misses after resolver updates.
    return geocode_course(query, country_hint)

@st.cache_data(ttl=3600)
def cached_course_detail(course_id, api_key):
    return fetch_course_detail(course_id, api_key)

@st.cache_data(ttl=3600)
def cached_course_holes(course_id, tournament, api_key):
    return fetch_course_holes(course_id, tournament, api_key)

@st.cache_data(ttl=1800)
def cached_weather_gca(course_id, tournament, start_date, api_key):
    detail = cached_course_detail(course_id, api_key)
    return fetch_weather(detail, tournament, start_date=start_date, days=7)

@st.cache_data(ttl=1800)
def cached_weather_location(lat, lon, tournament, start_date):
    return fetch_weather({"latitude": lat, "longitude": lon}, tournament, start_date=start_date, days=7)


# ---------------- V10.8 unified navigation ----------------
if "pga_bundle" not in st.session_state:
    st.session_state.pga_bundle = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.prediction_key = None

page = st.sidebar.radio(
    "PGA MODEL",
    ["🏠 Setup / Inputs", "🏆 Tournament DFS", "🏌️ Round Showdown", "🎯 Round Parlays", "🧬 Course DNA", "📊 Results / Calibration"],
    index=0,
)
st.sidebar.caption("V10.9.6 • Showdown Distribution Upgrade")


def _course_dna(otis):
    """Reconstruct relative component emphasis from the weekly OTIS Course-Fit layer.
    This is an audit view, not a claim that the app knows hidden OTIS internals.
    """
    if otis is None or otis.empty or "otis_course_fit" not in otis.columns:
        return pd.DataFrame(), "Course-Fit data unavailable."
    labels = {
        "otis_fit_app":"Approach", "otis_fit_ott":"Off the Tee", "otis_fit_arg":"Around Green",
        "otis_fit_putt":"Putting", "otis_fit_history":"Venue / History"
    }
    cols=[c for c in labels if c in otis.columns and pd.to_numeric(otis[c],errors="coerce").notna().sum()>=8]
    if not cols:
        return pd.DataFrame(), "Advanced APP/OTT/ARG/PUTT/History component columns are not present."
    d=otis[["otis_course_fit"]+cols].apply(pd.to_numeric,errors="coerce").dropna()
    if len(d)<8:
        return pd.DataFrame(), "Not enough complete component rows to construct Course DNA."
    # Standardized least-squares reconstruction of overall Course Fit from its visible components.
    y=(d["otis_course_fit"]-d["otis_course_fit"].mean())/(d["otis_course_fit"].std(ddof=0) or 1)
    X=[]
    for c in cols:
        x=d[c]; X.append(((x-x.mean())/(x.std(ddof=0) or 1)).to_numpy())
    import numpy as np
    A=np.column_stack(X)
    coef=np.linalg.lstsq(A,y.to_numpy(),rcond=None)[0]
    pred=A@coef
    ssr=float(((y.to_numpy()-pred)**2).sum()); sst=float(((y.to_numpy()-y.mean())**2).sum())
    r2=max(0.0,1-ssr/sst) if sst>0 else 0.0
    pos=np.clip(coef,0,None)
    weights=(pos/pos.sum()*100) if pos.sum()>0 else (np.abs(coef)/np.abs(coef).sum()*100)
    rows=[]
    for c,b,w in zip(cols,coef,weights):
        level="HIGH" if w>=25 else ("MODERATE" if w>=15 else "LOW")
        rows.append({"Requirement":labels[c],"Constructed weight %":round(float(w),1),"Standardized influence":round(float(b),3),"Emphasis":level})
    out=pd.DataFrame(rows).sort_values("Constructed weight %",ascending=False).reset_index(drop=True)
    return out, f"Visible components explain {r2:.0%} of weekly Course Fit variation across {len(d)} golfers."


def _dna_thesis(dna, course):
    if dna.empty:
        return "The current weekly file does not expose enough component detail to construct a defensible course thesis."
    top=dna.iloc[0]
    second=dna.iloc[1] if len(dna)>1 else None
    low=dna.iloc[-1]
    txt=f"For **{course or 'this course'}**, the weekly Course-Fit layer is asking most strongly for **{top['Requirement']}** ({top['Constructed weight %']:.1f}% of reconstructed emphasis)."
    if second is not None:
        txt+=f" **{second['Requirement']}** is the secondary demand ({second['Constructed weight %']:.1f}%)."
    txt+=f" **{low['Requirement']}** carries the least reconstructed emphasis ({low['Constructed weight %']:.1f}%)."
    return txt


if page == "🏠 Setup / Inputs":
    st.header("🏠 Setup / Inputs")
    st.caption("Load the tournament once. The other pages use this shared setup and do not ask you to upload the same files again.")
    prev=st.session_state.pga_bundle or {}
    a,b,c=st.columns([1.3,1.3,1])
    tournament_name=a.text_input("Tournament",value=prev.get("tournament_name", ""),placeholder="Current tournament")
    course_query=b.text_input("Course",value=prev.get("course_query", ""),placeholder="Current course")
    tournament_start=c.date_input("Tournament-week start",value=prev.get("tournament_start",date.today()))

    st.subheader("Tournament data")
    st.caption("Observed golf stats drive our model. External FIT rankings remain separate for comparison.")
    u1,u2=st.columns(2)
    players_up=u1.file_uploader("1 · DraftKings PGA salary slate (required)",type="csv",key="setup_players")
    rrg_sg_up=u2.file_uploader("2 · RickRunGood L50 strokes gained (recommended)",type="csv",key="setup_rrg_sg")
    with st.expander("External ranking comparison (optional)",expanded=False):
        e1,e2=st.columns(2)
        rrg_up=e1.file_uploader("RickRunGood course-fit FIT",type="csv",key="setup_rrg")
        otis_fit_up=e2.file_uploader("OTIS Advanced Course-Fit",type="csv",key="setup_otis")
        use_otis_in_model=st.checkbox("Experimental: use OTIS in primary projections instead of an independent comparison",value=False)
        if use_otis_in_model:
            st.warning("OTIS-assisted runs are not independent. Leave unchecked for model-vs-provider audit.")

    st.subheader("Automatic course & weather")
    c1,c2,c3=st.columns([1,1,1])
    country_hint=c1.text_input("Country / region hint",value=prev.get("country_hint",""),placeholder="Japan, Florida, Mexico...")
    gca_key=c2.text_input("Golf Courses API key (optional)",type="password")
    find_course=c3.button("Find / refresh course",use_container_width=True)
    if "course_matches" not in st.session_state: st.session_state.course_matches=[]
    if "selected_course_id" not in st.session_state: st.session_state.selected_course_id=None
    if "course_lookup_note" not in st.session_state: st.session_state.course_lookup_note=""
    if find_course and course_query.strip():
        try:
            gca_matches=cached_course_search(course_query,gca_key,country_hint) if gca_key.strip() else []
            for r in gca_matches: r["source"]="Golf Courses API"
            if gca_matches:
                st.session_state.course_matches=gca_matches; st.session_state.course_lookup_note="Rich course database match found."
            else:
                osm_matches=cached_osm_search(course_query,country_hint)
                st.session_state.course_matches=osm_matches
                st.session_state.course_lookup_note="Global OpenStreetMap location lookup used." if osm_matches else "No automatic course/location match found."
        except Exception as exc:
            st.session_state.course_matches=[]; st.session_state.course_lookup_note=f"Lookup failed: {exc}"
    if st.session_state.course_lookup_note:
        st.info(st.session_state.course_lookup_note)
    selected_course=None
    if st.session_state.course_matches:
        labels=[f"[{r.get('source','source')}] {r['name']} — {r.get('city','')}, {r.get('state','')}, {r.get('country','')}" for r in st.session_state.course_matches]
        choice=st.selectbox("Matched course/location",range(len(labels)),format_func=lambda i:labels[i])
        selected_course=st.session_state.course_matches[choice]
        st.session_state.selected_course_id=selected_course["id"]

    with st.expander("⚙️ Advanced / Manual Data Overrides",expanded=False):
        st.caption("Normally leave these alone. They are fallbacks for missing or custom data.")
        stats_up=st.file_uploader("player_stats.csv override",type="csv",key="setup_stats")
        results_up=st.file_uploader("results.csv override",type="csv",key="setup_results")
        history_up=st.file_uploader("course_history.csv override",type="csv",key="setup_history")
        holes_up=st.file_uploader("course_holes.csv override",type="csv",key="setup_holes")
        weather_up=st.file_uploader("weather.csv override",type="csv",key="setup_weather")

    st.subheader("V11 • Golf-first fantasy engine")
    st.caption("V11 simulates holes, applies DK Classic scoring, and optimizes fantasy distributions. Provisional scoring probabilities require calibration; no actual tournament results are used.")
    sims=st.selectbox("Tournament simulations",[1000,3000,5000],index=1)
    tournament_format=st.radio("Tournament format",["No cut — all four rounds","Standard cut — after Round 2"],horizontal=True,index=1)
    cut_size=st.number_input("Approximate cut position (ties included)",min_value=1,max_value=100,value=65,disabled=tournament_format.startswith("No cut"))
    mexico_2026=("mexico" in tournament_name.casefold() or "vidanta" in course_query.casefold()) and tournament_start.year==2026
    course_par=st.selectbox("Course par",[70,71,72,73],index=1 if mexico_2026 else 2)
    if mexico_2026:
        st.caption("Official 2026 Vidanta Vallarta: par 71, 7,337 yards; course re-routed. Verify hole-by-hole pars separately.")
    st.warning("Confirm cut/no-cut, par, and course hole pars before building. The app cannot infer tournament rules reliably from the tournament name.")
    if st.button("Load inputs & build tournament model",type="primary",use_container_width=True):
        if not tournament_name.strip() or not course_query.strip():
            st.error("Enter the tournament and course.")
        elif players_up is None:
            st.error("Upload the DraftKings player salaries. OTIS is not required.")
        else:
            try:
                players_raw=pd.read_csv(players_up); players,dk_format=normalize_dk_players(players_raw)
                if players.empty: raise ValueError("DraftKings file did not contain usable PGA golfers.")
                if rrg_up is not None:
                    rrg_fit,rrg_info=normalize_rrg_course_fit(pd.read_csv(rrg_up),players)
                    if rrg_fit.empty: raise ValueError("RickRunGood file had no valid FIT data.")
                else:
                    rrg_fit=pd.DataFrame()
                    rrg_info={"source":"not supplied","matched":0,"total":len(players),"coverage":0.0}
                rrg_count=int(rrg_info["matched"])
                if rrg_sg_up is not None:
                    sg_stats,sg_info=normalize_rrg_strokes_gained(pd.read_csv(rrg_sg_up),players)
                    if sg_info["matched"] < max(6,int(0.50*len(players))):
                        raise ValueError(f"Only {sg_info['matched']}/{len(players)} golfers matched the SG sheet; verify the tournament.")
                else:
                    sg_stats=pd.DataFrame(columns=["player"])
                    sg_info={"source":"Not supplied","field_size":len(players),"matched":0,
                             "full_sg_components":0,"sg_total_available":0,"at_least_12_rounds":0,
                             "no_sample_count":len(players),"mean_rounds":0.0}
                if otis_fit_up is not None:
                    otis_raw=pd.read_csv(otis_fit_up); otis_fit,otis_status=normalize_otis_course_fit(otis_raw)
                    if otis_fit.empty: raise ValueError(f"OTIS file could not be used: {otis_status}")
                else:
                    otis_fit=pd.DataFrame(columns=["player"]); otis_status="Not supplied"
                dk_names=set(players.player.astype(str).str.strip().str.casefold())
                onames=set(otis_fit.player.astype(str).str.strip().str.casefold())
                match_count=len(dk_names & onames)
                if use_otis_in_model and not len(otis_fit):
                    raise ValueError("OTIS-assisted modeling was selected, but no OTIS file was uploaded.")
                if use_otis_in_model and match_count/len(players)<.70:
                    raise ValueError(f"OTIS model coverage {match_count}/{len(players)} is below the 70% safety floor.")
                if stats_up is not None:
                    player_stats,stats_status=validate_advanced_input(pd.read_csv(stats_up),"player_stats")
                    if player_stats.empty:
                        raise ValueError(f"Uploaded player stats could not be used: {stats_status}")
                    model_mode="Independent custom golf stats override"
                elif not sg_stats.empty and use_otis_in_model:
                    player_stats=sg_stats.merge(otis_fit,on="player",how="left",validate="one_to_one")
                    stats_status="RickRunGood observed L50 SG + OTIS experimental blend"
                    model_mode="OTIS-assisted (not independent)"
                elif not sg_stats.empty:
                    player_stats=sg_stats.copy()
                    stats_status="RickRunGood observed L50 SG"
                    model_mode="Independent Team AI² model using observed RickRunGood SG"
                elif use_otis_in_model:
                    player_stats,stats_status=otis_fit.copy(),otis_status
                    model_mode="OTIS-assisted (not independent)"
                else:
                    player_stats=pd.DataFrame(columns=["player"])
                    stats_status="Only DraftKings historical fantasy averages"
                    model_mode="UNVALIDATED DK PPG fallback"
                if rrg_sg_up is not None and sg_info["full_sg_components"]<len(players):
                    st.caption(f"Complete SG components: {sg_info['full_sg_components']}/{len(players)} golfers. "
                               "Incomplete data remains missing and receives reduced confidence.")
                if rrg_up is not None and rrg_count/len(players)<.70:
                    st.warning(f"Only {rrg_count}/{len(players)} golfers match RickRunGood. "
                               "Those rankings are for comparison only; missing fits are not imputed.")
                if results_up is not None: results,results_status=validate_advanced_input(pd.read_csv(results_up),"results")
                else: results,results_status=validate_advanced_input(load_repo_csv("results.csv"),"results")
                if history_up is not None: history,history_status=validate_advanced_input(pd.read_csv(history_up),"course_history")
                else: history,history_status=validate_advanced_input(load_repo_csv("course_history.csv"),"course_history")
                if holes_up is not None:
                    holes=pd.read_csv(holes_up); holes_source="uploaded"
                elif selected_course is not None and selected_course.get("source")=="Golf Courses API":
                    holes,_=cached_course_holes(st.session_state.selected_course_id,tournament_name,gca_key); holes_source="Golf Courses API"
                else:
                    holes=pd.DataFrame(); holes_source="not supplied"
                if weather_up is not None:
                    weather=pd.read_csv(weather_up); weather_source="uploaded"
                elif selected_course is not None:
                    if selected_course.get("source")=="Golf Courses API": weather=cached_weather_gca(st.session_state.selected_course_id,tournament_name,tournament_start,gca_key)
                    else: weather=cached_weather_location(float(selected_course["latitude"]),float(selected_course["longitude"]),tournament_name,tournament_start)
                    weather_source="Open-Meteo"
                else:
                    weather=pd.DataFrame(); weather_source="not supplied"
                with st.spinner(f"Running {int(sims):,} tournament simulations..."):
                    pred=predict_from_dataframes(Config(tournament=tournament_name,sims=int(sims),seed=42),players,player_stats,results,history,holes,weather)
                    hole_pars=None
                    if isinstance(holes,pd.DataFrame) and not holes.empty:
                        for par_col in ("par","Par","hole_par"):
                            if par_col in holes.columns:
                                vals=pd.to_numeric(holes[par_col],errors="coerce").dropna().astype(int).tolist()
                                if len(vals)==18 and sum(vals)==int(course_par): hole_pars=vals
                                break
                    pred,dk_samples,v11_diag=simulate_golf_first(pred,sims=int(sims),seed=42,course_par=int(course_par),
                        no_cut=tournament_format.startswith("No cut"),cut_size=int(cut_size),hole_pars=hole_pars)
                    pred["dk_points_proxy"]=pred["v11_mean_dk"]
                    pred["dk_value_per_1000"]=pred["v11_salary_value"]
                    pred["make_cut_pct"]=pred["v11_make_cut_pct"]/100.0
                    pred["top10_pct"]=pred["v11_top10_pct"]/100.0
                    pred["win_pct"]=pred["v11_win_pct"]/100.0
                    pred["v11_gpp_objective"]=0.45*pred["v11_mean_dk"]+0.55*pred["v11_p90_dk"]
                    pred["v11_balanced_objective"]=0.80*pred["v11_mean_dk"]+0.20*pred["v11_p75_dk"]
                    pred["v11_cut_objective"]=0.85*pred["v11_mean_dk"]+0.15*pred["v11_p75_dk"]
                    # Independent source ranks are added AFTER simulations and NEVER alter the prediction.
                    pred=attach_source_comparison(pred,rrg_fit,otis_fit)
                bundle={"tournament_name":tournament_name,"course_query":course_query,"tournament_start":tournament_start,"country_hint":country_hint,
                        "players":players,"otis_fit":otis_fit,"rrg_fit":rrg_fit,"rrg_info":rrg_info,
                        "rrg_sg_info":sg_info,"model_mode":model_mode,
                        "player_stats":player_stats,"results":results,"history":history,"holes":holes,"weather":weather,
                        "holes_source":holes_source,"weather_source":weather_source,"dk_format":dk_format,"otis_status":otis_status,"otis_match_count":match_count,
                        "sims":int(sims),"prediction":pred,"selected_course":selected_course,"v11_diagnostics":v11_diag,"tournament_format":tournament_format}
                st.session_state.pga_bundle=bundle; st.session_state.prediction=pred
                st.success(f"Setup ready — {len(players)} golfers; {int(sims):,} simulations. {model_mode}.")
                st.caption(f"Measured SG: {sg_info['full_sg_components']}/{len(players)}; "
                           f"RRG course fit: {rrg_count}/{len(players)}; OTIS: {match_count}/{len(players)}.")
                if sg_info["full_sg_components"]<0.5*len(players) and stats_up is None and not use_otis_in_model:
                    st.warning("No sufficiently complete current golf statistics. The fallback uses historical DK scoring "
                               "and is NOT a validated fantasy projection or cut-probability model.")
                elif sg_info["full_sg_components"]>=0.5*len(players) and stats_up is None:
                    st.info("Measured last-50 SG now drives the independent projection. Small samples are shrunk toward neutral. "
                            "Golf-first simulations remain uncalibrated; no invented hole stats or cut probabilities.")
            except Exception as exc:
                st.error(str(exc))

    b=st.session_state.pga_bundle
    if b:
        st.subheader("Current setup status")
        q1,q2,q3,q4,q5,q6=st.columns(6)
        q1.metric("Tournament",b["tournament_name"])
        q2.metric("Field",len(b["players"]))
        q3.metric("SG",f"{b.get('rrg_sg_info',{}).get('full_sg_components',0)}/{len(b['players'])}")
        q4.metric("RRG FIT",f"{b['rrg_info']['matched']}/{len(b['players'])}")
        q5.metric("OTIS",f"{b['otis_match_count']}/{len(b['players'])}")
        q6.metric("Weather",("Auto" if b["weather_source"]=="Open-Meteo" else b["weather_source"]))
        st.caption("Model source: " + b["model_mode"] + ". RRG FIT and OTIS ranks never change the model unless "
                   "you explicitly select experimental OTIS blend. RRG SG and RRG FIT overlap in underlying data.")
        st.success("Setup is stored for this session. Use the sidebar to go directly to DFS, Round Parlays, Course DNA, or Results.")
    st.stop()

bundle=st.session_state.pga_bundle
if not bundle:
    st.warning("No tournament setup loaded. Go to **🏠 Setup / Inputs** and upload the DraftKings field. RRG/OTIS are optional.")
    st.stop()

tournament_name=bundle["tournament_name"]; course_query=bundle["course_query"]; tournament_start=bundle["tournament_start"]
players=bundle["players"]; otis_fit=bundle["otis_fit"]; player_stats=bundle["player_stats"]; results=bundle["results"]; history=bundle["history"]
holes=bundle["holes"]; weather=bundle["weather"]; sims=bundle["sims"]; mc=bundle["prediction"]

st.caption(f"**{tournament_name}** • {course_query} • {len(players)} golfers • {sims:,} tournament sims")

if page == "🏆 Tournament DFS":
    st.header("🏆 Tournament DFS")
    c1,c2,c3,c4=st.columns(4); c1.metric("Field",len(mc)); c2.metric("Simulations",f"{sims:,}"); c3.metric("Salary cap","$50,000"); c4.metric("RRG coverage",f"{bundle['rrg_info']['matched']}/{len(players)}")
    st.caption("Independent model, RRG and OTIS ranks are kept separate unless you explicitly opted into OTIS-assisted projection.")
    st.info(f"V11 golf-first active • {bundle.get('tournament_format','format not specified')} • "
            f"{bundle.get('v11_diagnostics',{}).get('hole_pars_source','unknown hole pars')}. "
            "DK scoring from simulated golf; probabilities remain provisional until calibrated.")
    tab1,tab2,tab3=st.tabs(["Model Rankings","Model vs RRG vs OTIS","Build Lineups"])
    with tab1:
        display_cols=[c for c in ["player","salary","rrg_sg_rounds","model_data_confidence","sg_total","sg_approach","sg_ott","sg_putting","model_baseline_source","win_pct","top5_pct","top10_pct","top20_pct","make_cut_pct","expected_finish","v11_mean_dk","v11_median_dk","v11_p75_dk","v11_p90_dk","v11_p95_dk","v11_avg_birdies","v11_avg_bogeys","dk_points_proxy","course_fit_ceiling"] if c in mc.columns]
        sort_options=[c for c in ["v11_mean_dk","v11_p90_dk","win_pct","top10_pct","make_cut_pct","course_fit_ceiling","salary"] if c in mc.columns]
        sort_col=st.selectbox("Sort by",sort_options)
        st.dataframe(mc.sort_values(sort_col,ascending=(sort_col=="salary"))[display_cols],width="stretch",hide_index=True)
        st.download_button("Download projections",mc.to_csv(index=False),f"{tournament_name.replace(' ','_')}_projections.csv","text/csv")
    with tab2:
        st.caption("Separate PRE-TOURNAMENT rankings. Positive rank delta means our model ranks a golfer worse than the external source. Missing source values stay blank.")
        cols=[c for c in ("player","salary","model_rank","rrg_rank","otis_rank",
                          "model_vs_rrg_rank_delta","model_vs_otis_rank_delta",
                          "rrg_fit","otis_true_skill","otis_course_fit") if c in mc.columns]
        st.dataframe(mc[cols].sort_values("model_rank"),width="stretch",hide_index=True)
        st.download_button("Download independent source comparison",
                           mc[cols].to_csv(index=False),"pga_source_rank_comparison.csv",mime="text/csv")
        if bundle["otis_match_count"]==0:
            st.info("OTIS not yet released. The independent model and RRG comparison are available now.")
        st.warning("A shared ranking does not validate win, top-10 or cut probabilities. Source agreement is a research signal, not a guarantee.")
    with tab3:
        a,b,c,d=st.columns(4); lineup_count=a.number_input("Lineups",1,20,11); max_exposure=b.slider("Max exposure",.10,1.0,.37,.01); min_unique=c.number_input("Minimum unique golfers",1,5,3); salary_floor=d.number_input("Minimum lineup salary",40000,50000,46500,100)
        min_player_salary=st.number_input("Minimum golfer salary",6000,10000,6500,100)
        sg_enabled="rrg_sg_rounds" in mc.columns
        min_sg_rounds=st.selectbox("Minimum measured SG rounds",[0,8,12,20],index=2,
                                  disabled=not sg_enabled,
                                  help="Blocks unsupported very-small-sample golfers when RickRunGood measured data is supplied. Zero disables.")
        strategy=st.selectbox("Strategy",["GPP Ceiling","Balanced / Single Entry","Cut Equity"],index=1)
        if sg_enabled:
            st.caption("The default 12-round sample floor excludes golfers without credible SG evidence. "
                       "Missing SG never receives fabricated values.")
        names=sorted(mc.player.dropna().astype(str).unique())
        locks=st.multiselect("Lock golfers (ALL lineups)",names)
        excludes=st.multiselect("Exclude golfers from ALL lineups",[n for n in names if n not in locks])
        mexico_feature_week = (
            ("mexico" in tournament_name.casefold() or "vidanta" in course_query.casefold())
            and tournament_start.year == 2026
        )
        mexico_feature_defaults=["Tony Finau"] if mexico_feature_week and "Tony Finau" in names else []
        feature_exclusions=st.multiselect(
            "Exclude from $5 feature lineup ONLY (eligible for ten $0.50 entries)",
            names,default=mexico_feature_defaults,key="pga_feature_exclusions",
            help="Human-judgment veto on the single $5 entry. The model can still use these golfers in the $0.50 lineups.",
        )
        if st.button("Generate portfolio",type="primary"):
            settings=PortfolioSettings(lineup_count=int(lineup_count),salary_cap=50000,salary_floor=int(salary_floor),min_player_salary=int(min_player_salary),min_measured_sg_rounds=int(min_sg_rounds),roster_size=6,max_exposure=float(max_exposure),min_unique=int(min_unique),strategy=strategy,seed=42)
            try:
                progress=st.progress(0,text="Preparing DFS optimizer…")
                def _dfs_progress(frac,msg):
                    progress.progress(min(1.0,max(0.0,float(frac))),text=msg)
                portfolio,summary,exposure=optimize_portfolio(mc,settings,locks,excludes,progress_callback=_dfs_progress)
                summary,featured_id=choose_featured_lineup(
                    summary,portfolio,mc,feature_exclusions=feature_exclusions
                )
                st.session_state.dfs_portfolio={
                    "portfolio":portfolio,"summary":summary,"exposure":exposure,
                    "featured_lineup":featured_id,"feature_exclusions":list(feature_exclusions)
                }
                progress.empty()
                st.success(f"Generated {len(summary)} lineups. Highest-rated eligible $5 feature candidate: Lineup {featured_id}.")
                if feature_exclusions:
                    st.info("Feature-only veto applied: "+", ".join(feature_exclusions)+
                            ". These golfers remain eligible for the $0.50 entries.")
                st.caption("The feature index ranks estimated DK means, cut equity, top-10 strength, and summed player P90. "
                           "It is NOT a calibrated first-place or profitable-entry probability.")
            except Exception as exc:
                try: progress.empty()
                except Exception: pass
                st.error(str(exc))
        po=st.session_state.get("dfs_portfolio")
        if po:
            featured=po.get("featured_lineup")
            if featured:
                st.subheader(f"⭐ Proposed $5 single-entry: Lineup {featured}")
                featured_players=po["portfolio"].loc[po["portfolio"]["lineup"].eq(featured),["player","salary","make_cut_pct","dk_points_proxy"]]
                st.dataframe(featured_players,hide_index=True,width="stretch")
                st.caption("Review this candidate against the RickRunGood and OTIS comparisons before final lock. "
                           "The other ten portfolios are intended for the $0.50 contest.")
            st.dataframe(po["summary"],width="stretch",hide_index=True); st.dataframe(po["portfolio"],width="stretch",hide_index=True)
            st.subheader("Exposure"); st.dataframe(po["exposure"],width="stretch",hide_index=True)
            st.download_button("Download lineups CSV",po["portfolio"].to_csv(index=False),"pga_lineups.csv","text/csv")

elif page == "🏌️ Round Showdown":
    if otis_fit is None or otis_fit.empty:
        st.info("The tournament DFS model runs without OTIS. Round Showdown still needs an OTIS pre-round file plus verified completed-round results; no substitute is fabricated.")
        st.stop()
    from round_showdown import build_showdown_pool, optimize_showdown
    st.header("🏌️ Round Showdown — R2 / R3 / R4")
    st.caption("Dedicated DraftKings single-round lineups. Uses OTIS pre-round strength, independent Course DNA and verified completed-round scores. Showdown uses a simulated single-round fantasy proxy distribution from round strength, verified prior-round evidence and available OTIS skill components. It is not an official DraftKings projection.")
    x1,x2,x3,x4=st.columns(4)
    sd_round=int(x1.selectbox("Showdown round",[2,3,4],key="sd_round"))
    sd_count=int(x2.number_input("Lineups",1,20,4,key="sd_count"))
    sd_cap=float(x3.slider("Max exposure",0.10,1.0,0.50,0.05,key="sd_exposure"))
    sd_floor=int(x4.number_input("Salary floor",40000,50000,46500,100,key="sd_floor"))
    sd_up=st.file_uploader("DraftKings PGA Showdown salary CSV (round-specific)",type="csv",key="sd_dk")
    sd_prior_up=st.file_uploader("Completed prior-round results CSV (optional fallback)",type="csv",key=f"sd_live_{sd_round}")
    sd_tid=st.text_input("PGA TOUR tournament ID (optional; autodetected)",value="R2026527" if "baycurrent" in tournament_name.casefold() and tournament_start.year==2026 else "",key="sd_tid")
    if st.button("Build round Showdown",type="primary",use_container_width=True):
        if sd_up is None:
            st.error("Upload the round-specific DraftKings Showdown salary file.")
        else:
            try:
                dk,fmt=normalize_dk_players(pd.read_csv(sd_up))
                if dk.empty: raise ValueError("No valid golfer salaries found in the Showdown CSV.")
                if sd_prior_up is not None:
                    sd_live=normalize_live_results(pd.read_csv(sd_prior_up)); source="Uploaded completed-round results"
                else:
                    sd_live,source=fetch_official_prior_rounds(tournament_name,int(tournament_start.year),sd_round,sd_tid)
                if sd_live.empty:
                    raise ValueError("No verified completed-round scores available. Showdown blocked rather than using R1 ratings. " + source)
                ratings=build_round_ratings(otis_fit,sd_live,sd_round,tournament_name=tournament_name,use_course_dna=True)
                pool=build_showdown_pool(dk,ratings,sd_round,otis=otis_fit)
                covered=int((pool.live_rounds>0).sum())
                if covered<max(3,int(len(pool)*0.50)):
                    raise ValueError(f"Only {covered}/{len(pool)} golfers matched completed-round data. Check the CSV/field names.")
                portfolio,exposure=optimize_showdown(pool,sd_count,sd_cap,sd_floor,seed=42)
                st.session_state.pga_showdown={"round":sd_round,"pool":pool,"portfolio":portfolio,"exposure":exposure,"source":source,"coverage":covered}
            except Exception as exc:
                st.error(str(exc))
    sd=st.session_state.get("pga_showdown")
    if sd and sd.get("round")==sd_round:
        st.success(f"R{sd_round} Showdown • {sd['coverage']}/{len(sd['pool'])} golfers with prior-round evidence • {sd['source']}")
        p1,p2,p3=st.tabs(["Lineups","Golfer ratings","Exposure"])
        with p1:
            st.dataframe(sd["portfolio"],width="stretch",hide_index=True)
            st.download_button("Download Showdown lineups",sd["portfolio"].to_csv(index=False),f"PGA_R{sd_round}_Showdown_Lineups.csv","text/csv")
        with p2:
            st.dataframe(sd["pool"],width="stretch",hide_index=True)
        with p3:
            st.dataframe(sd["exposure"],width="stretch",hide_index=True)

elif page == "🎯 Round Parlays":
    if otis_fit is None or otis_fit.empty:
        st.info("Tournament modeling and independent RRG comparisons work without OTIS. Round-parlay ratings still require the OTIS source and prior-round evidence; this page will be enabled when supported inputs are available.")
        st.stop()
    st.header("🎯 Round Parlays")
    st.caption("Groupings are pulled automatically from the official PGA TOUR structured feed. Manual controls are hidden unless the automatic source fails.")
    a,b,c,d=st.columns(4)
    round_no=int(a.selectbox("Round",[1,2,3,4])); round_sims=int(b.selectbox("Simulations",[25000,50000,100000],index=2)); ticket_count=int(c.number_input("6-leg tickets",1,10,5)); max_overlap=int(d.slider("Max shared picks",0,5,4))
    year=int(tournament_start.year); default_tid="R2026527" if "baycurrent" in tournament_name.casefold() and year==2026 else ""
    with st.expander("⚙️ Manual grouping / leaderboard fallback",expanded=False):
        tournament_id=st.text_input("PGA TOUR tournament ID",value=default_tid)
        grouping_url=st.text_input("Grouping URL override",value="")
        grouping_up=st.file_uploader("Grouping CSV fallback",type="csv",key=f"groupings_r{round_no}")
        leaderboard_url=st.text_input("Leaderboard URL override",value="",disabled=(round_no==1))
        live_up=st.file_uploader("Prior-round results CSV fallback",type="csv",key=f"live_r{round_no}",disabled=(round_no==1))
    if st.button("Run round simulation",type="primary",use_container_width=True):
        if grouping_up is not None: groups=normalize_groupings_upload(pd.read_csv(grouping_up),round_no); gsrc="uploaded grouping CSV"; gnote="manual fallback"
        else:
            with st.spinner("Pulling official PGA TOUR groupings..."): groups,gsrc,gnote=fetch_groupings(tournament_name,year,round_no,grouping_url,tournament_id)
        live=pd.DataFrame(); lnote="R1: no in-tournament adjustment"
        if round_no>1:
            if live_up is not None: live=normalize_live_results(pd.read_csv(live_up)); lnote="uploaded prior-round results"
            else:
                with st.spinner("Pulling official completed-round scores..."):
                    live,lnote=fetch_official_prior_rounds(tournament_name,year,round_no,tournament_id)
                if live.empty and leaderboard_url.strip():
                    live,html_note=fetch_live_results(leaderboard_url,round_no)
                    lnote += " | HTML fallback: " + html_note
        if groups.empty: st.error(f"Could not load validated groupings ({gnote}). Open the manual fallback only if needed.")
        else:
            # Do not quietly present R2+ as live-informed when all prior scores are missing.
            if round_no > 1 and live.empty:
                st.error("No verified completed-round scores loaded. Round-specific simulation blocked to prevent an R1-like rerun. " + lnote)
                st.stop()
            # Main simulation: independent course DNA ON.
            ratings=build_round_ratings(otis_fit,live,round_no,tournament_name=tournament_name,use_course_dna=True)
            if round_no > 1:
                covered=int((ratings.live_rounds>0).sum())
                st.info(f"Prior-round evidence: {covered}/{len(ratings)} golfers matched • {len(live)} completed-round scores. Source: {lnote}")
                if covered < max(3,int(len(ratings)*0.50)):
                    st.error("Prior-round data matches fewer than half the golfers; check names or upload the prior-round CSV before running.")
                    st.stop()
            probs,_=simulate_groups(groups,ratings,n_sims=round_sims,seed=42)
            summary,legs=build_six_leg_tickets(probs,ticket_count=ticket_count,max_player_overlap=max_overlap)

            # Audit simulation: same groups, same live data, same seed, but course DNA OFF.
            # This isolates the effect of the independent Yokohama DNA from Monte Carlo noise.
            ratings_off=build_round_ratings(otis_fit,live,round_no,tournament_name=tournament_name,use_course_dna=False)
            probs_off,_=simulate_groups(groups,ratings_off,n_sims=round_sims,seed=42)
            audit=(probs[probs.status.eq("OK")][["group","tee_time","player","round_rating","win_pct","push_pct","non_loss_pct"]]
                   .merge(probs_off[probs_off.status.eq("OK")][["group","player","round_rating","win_pct","push_pct","non_loss_pct"]],
                          on=["group","player"],how="left",suffixes=("_dna_on","_dna_off")))
            audit=audit.merge(ratings[["player","pre_round_rating","course_fit_signal","course_fit_source"]],on="player",how="left")
            audit["rating_delta"]=audit["round_rating_dna_on"]-audit["round_rating_dna_off"]
            audit["win_pct_delta"]=audit["win_pct_dna_on"]-audit["win_pct_dna_off"]
            audit["non_loss_delta"]=audit["non_loss_pct_dna_on"]-audit["non_loss_pct_dna_off"]
            on_pick=(audit.sort_values(["group","win_pct_dna_on"],ascending=[True,False]).groupby("group",as_index=False).head(1)[["group","player"]].rename(columns={"player":"pick_dna_on"}))
            off_pick=(audit.sort_values(["group","win_pct_dna_off"],ascending=[True,False]).groupby("group",as_index=False).head(1)[["group","player"]].rename(columns={"player":"pick_dna_off"}))
            pick_compare=on_pick.merge(off_pick,on="group",how="outer")
            pick_compare["pick_changed"]=pick_compare["pick_dna_on"]!=pick_compare["pick_dna_off"]
            audit=audit.merge(pick_compare,on="group",how="left")
            audit["DNA changed group pick"]=audit["pick_changed"].map({True:"YES",False:"No"})
            st.session_state.round_parlay={"groups":groups,"ratings":ratings,"probs":probs,"summary":summary,"legs":legs,"gsrc":gsrc,"gnote":gnote,"lnote":lnote,"round":round_no,"audit":audit,"pick_compare":pick_compare,"live":live,"round_data_note":lnote}
    rp=st.session_state.get("round_parlay")
    if rp and rp.get("round")==round_no:
        st.success(f"Round {round_no} ready • {len(rp['groups'])} groups • {rp['gsrc']}")
        st.caption("Round evidence: " + rp.get("round_data_note","not recorded"))
        ok=rp["probs"][rp["probs"].status.eq("OK")].copy()
        # V10.9.5a calibration diagnostics: expose the exact prior-vs-live decomposition
        # used to create each final round rating so sensitivity tests can be run
        # without changing the production model.
        diag_cols=["player","pre_round_rating","live_form_adj","live_rounds"]
        diag=rp["ratings"][[c for c in diag_cols if c in rp["ratings"].columns]].copy()
        if "player" in diag.columns:
            ok=ok.merge(diag,on="player",how="left")
        t1,t2,t3=st.tabs(["3-Ball Probabilities","6-Leg Tickets","🧬 DNA Impact Audit"])
        with t1:
            st.dataframe(ok.sort_values(["group","win_pct"],ascending=[True,False]),width="stretch",hide_index=True)
            st.download_button("Download probabilities",ok.to_csv(index=False),f"round_{round_no}_3ball_probabilities.csv","text/csv")
        with t2:
            if rp["summary"].empty: st.warning("Fewer than six fully matched groups are available.")
            else:
                st.dataframe(rp["summary"],width="stretch",hide_index=True); st.dataframe(rp["legs"],width="stretch",hide_index=True)
                st.download_button("Download 6-leg tickets",rp["legs"].to_csv(index=False),f"round_{round_no}_six_leg_tickets.csv","text/csv")
        with t3:
            audit=rp.get("audit",pd.DataFrame()).copy()
            if audit.empty:
                st.info("Run the round simulation again to create the Course DNA ON vs OFF audit.")
            else:
                changed=int(rp.get("pick_compare",pd.DataFrame()).get("pick_changed",pd.Series(dtype=bool)).sum())
                max_move=float(audit["win_pct_delta"].abs().max()) if len(audit) else 0.0
                c1,c2,c3=st.columns(3)
                c1.metric("Groups where DNA changed the pick",changed)
                c2.metric("Largest win-probability move",f"{max_move:.2f} pts")
                c3.metric("Course-fit source",str(audit["course_fit_source"].iloc[0]))
                st.caption("DNA ON and OFF use the same groups, live-round inputs, simulation count and random seed. The delta therefore isolates the independent Course DNA signal rather than Monte Carlo noise.")
                show=audit[["group","tee_time","player","course_fit_signal","pre_round_rating","round_rating_dna_off","round_rating_dna_on","rating_delta","win_pct_dna_off","win_pct_dna_on","win_pct_delta","non_loss_pct_dna_off","non_loss_pct_dna_on","non_loss_delta","pick_dna_off","pick_dna_on","DNA changed group pick"]].copy()
                st.dataframe(show.sort_values(["group","win_pct_dna_on"],ascending=[True,False]),width="stretch",hide_index=True)
                st.download_button("Download DNA impact audit",show.to_csv(index=False),f"round_{round_no}_course_dna_audit.csv","text/csv")

elif page == "🧬 Course DNA":
    st.header("🧬 Course DNA")
    st.caption("What does our independent pre-event model think this course is asking golfers to do?")

    is_baycurrent = "baycurrent" in str(tournament_name).casefold() or "yokohama" in str(course_query).casefold()
    if is_baycurrent:
        st.subheader(f"{course_query} — Independent 2026 Model Thesis")
        st.info("Yokohama is primarily a **ball-striking test**: create greens/scoring chances with strong approach play, pair that with quality driving, and survive an unusually par-4-heavy routing. Short game and bentgrass putting matter, but they are secondary rather than the engine of the model.")

        pillars = pd.DataFrame([
            {"Model input":"SG: Approach","Weight %":22.0,"Demand":"PRIMARY","Why":"13 par 4s plus a 2025 profile where fairways were relatively easy to find but GIR/scoring opportunities were harder to create."},
            {"Model input":"SG: Off the Tee","Weight %":15.0,"Demand":"HIGH","Why":"2025 top finishers showed a meaningful driving signal; the layout rewards usable power and positioning around bunkers, doglegs and pinch points."},
            {"Model input":"Overall skill / SG Total","Weight %":14.0,"Demand":"HIGH","Why":"Keeps the course model anchored to complete golfer quality instead of overfitting one course trait."},
            {"Model input":"Par-4 scoring","Weight %":9.0,"Demand":"HIGH","Why":"Yokohama has 13 par 4s — an unusually large share of every round."},
            {"Model input":"Driving distance","Weight %":8.0,"Demand":"MODERATE-HIGH","Why":"Wide-ish fairways and manageable rough permit aggression, while 7,322 yards still rewards useful length."},
            {"Model input":"Bogey avoidance","Weight %":8.0,"Demand":"MODERATE-HIGH","Why":"Missed greens plus heavily bunkered complexes create recovery pressure."},
            {"Model input":"Driving accuracy","Weight %":5.0,"Demand":"MODERATE","Why":"Position still matters around landing-zone hazards, but pure accuracy is not treated as the dominant driving trait."},
            {"Model input":"Birdie rate","Weight %":5.0,"Demand":"MODERATE","Why":"When players create chances, the course is gettable; conversion and scoring remain relevant."},
            {"Model input":"SG: Around Green","Weight %":5.0,"Demand":"SECONDARY","Why":"Bunkering and green complexes punish misses, but recovery skill is not allowed to outrank ball striking."},
            {"Model input":"SG: Putting","Weight %":5.0,"Demand":"SECONDARY","Why":"Creeping bentgrass matters, but volatile putting is deliberately prevented from dominating the pre-event model."},
            {"Model input":"Par-3 scoring","Weight %":2.0,"Demand":"LOW","Why":"Only three par 3s per round."},
            {"Model input":"Par-5 scoring","Weight %":2.0,"Demand":"LOW","Why":"Only two par 5s per round, so there are fewer opportunities for this skill to separate the field."},
        ])
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Primary demand","Approach / GIR")
        c2.metric("Secondary demand","Off the Tee")
        c3.metric("Par 4s","13 of 18")
        c4.metric("Greens","Bentgrass")
        st.subheader("Constructed model weights")
        st.dataframe(pillars,width="stretch",hide_index=True)
        st.bar_chart(pillars.set_index("Model input")["Weight %"])
        st.caption("These weights are a pre-event 2026 Yokohama model thesis. They are not reverse-engineered from this week's results. The available APP/OTT/ARG/PUTT components are renormalized into an independent course-fit signal that feeds both Tournament DFS and Round Parlays; unavailable traits are never fabricated.")

        with st.expander("🔬 Evidence / Why",expanded=True):
            st.markdown("**Course structure:** 7,322-yard par 71; 13 par 4s, three par 3s and two par 5s. **Surfaces:** zoysia fairways/rough and creeping bentgrass greens. **2025 behavior:** fairways were comparatively easy to hit while GIR/scoring opportunities were harder to create. **Driving signal:** PGA TOUR's 2026 course-history review identified driving prowess as the clearest trait among many of last year's top-20 finishers. **Architecture:** fairway bunkers, doglegs and pinch points influence landing zones, while heavy bunkering protects many greens.")
            st.warning("Guardrail: we do not yet have defensible hole-by-hole approach-distance frequencies for this 2026 routing, so the model does **not** invent 100–125 / 125–150 / 150–175 / 175–200 / 200+ weights. Those remain a future granular layer.")

        with st.expander("🌬️ Round conditions are separate",expanded=False):
            st.write("Weather does not change the permanent Course DNA. Wind and other round conditions are applied in the round/tournament environment layer so a windy Friday can play differently from a calmer Thursday without rewriting what Yokohama fundamentally demands.")
    elif (
        ("mexico" in str(tournament_name).casefold() and "open" in str(tournament_name).casefold())
        or "vidanta" in str(course_query).casefold()
    ) and tournament_start.year == 2026:
        thesis=mexico_course_thesis()
        stats=mexico_course_evidence()
        st.subheader("Vidanta Vallarta — 2026 course-demand evidence")
        st.info("Primary: **overall ball striking and approach play**. The chart singles out "
                "**175–200 and 200–225 yard approaches**. GIR, scoring and tee-to-green skill "
                "matter more here than pure driving accuracy.")
        c1,c2,c3,c4=st.columns(4)
        for c,name in zip((c1,c2,c3,c4),
                          ("SG: Total","SG: Approach","Approach: 175–200 yards","Approach: 200–225 yards")):
            value=float(stats.loc[stats["Statistic"].eq(name),"Chart association"].iloc[0])
            c.metric(name,f"{value:.3f}")
        st.caption(thesis["source"]+". Figures are user-provided chart values, "
                   "NOT independently fitted predictor weights or forecast probabilities.")
        st.dataframe(stats,width="stretch",hide_index=True)
        st.download_button("Download course-demand evidence",
                           stats.to_csv(index=False),"pga_mexico_2026_course_evidence.csv",
                           mime="text/csv")
        st.subheader("Our measurable inputs vs unavailable inputs")
        st.write("**In the L50 strokes-gained CSV:** SG Total, Approach, Off-the-Tee, "
                 "Putting, Around-the-Green, driving distance and accuracy, and GIR.")
        st.warning("**Not in the supplied L50 golfer file:** individual 175–200 and 200–225 yard "
                   "approach performance, birdie-or-better rate, bogey avoidance and scoring average. "
                   "The engine does not infer those player statistics from generic SG: Approach.")
        st.caption("Chart strengths are not additive. SG Total, Tee-to-Green and Approach overlap; "
                   "the same performance must not be counted as three independent advantages. "
                   "RickRunGood course FIT also uses related inputs, so agreement isn't fully independent.")
        with st.expander("Model usage and research guardrails",expanded=True):
            st.write("These screenshot observations are **audited course evidence**, not an automated "
                     "numerical bonus in the currently committed simulations. "
                     "Our model consumes observed golf statistics independently, while this evidence "
                     "supports a future controlled Mexico Course-DNA ON/OFF comparison. "
                     "The 2026 routing changed, so historical course-fit figures are not assumed fully transferable.")
            st.caption(thesis["guardrail"])
    else:
        dna,note=_course_dna(otis_fit)
        rrg=bundle.get("rrg_fit",pd.DataFrame())
        if isinstance(rrg,pd.DataFrame) and not rrg.empty:
            st.subheader("RickRunGood course-fit decomposition (independent comparison)")
            rcols={"rrg_driving":"Driving","rrg_approach":"Approach","rrg_around_green":"Around green",
                   "rrg_putting":"Putting","rrg_scoring":"Scoring"}
            summaries=[{"Component":name,"Average contribution":round(float(pd.to_numeric(rrg[col],errors="coerce").mean()),4)}
                       for col,name in rcols.items() if col in rrg]
            st.dataframe(pd.DataFrame(summaries),hide_index=True,width="stretch")
            st.caption("RRG values are within-source FIT contributions, not strokes gained or Team AI²-derived weights. "
                       "The course layout and independent model thesis must be researched separately.")
        st.subheader(f"{course_query} — OTIS visible components (if available)")
        st.info(_dna_thesis(dna,course_query))
        if dna.empty:
            st.warning(note)
        else:
            st.dataframe(dna,width="stretch",hide_index=True)
            st.bar_chart(dna.set_index("Requirement")["Constructed weight %"])
            st.caption(note)
        with st.expander("🔬 Model Construction / Why",expanded=True):
            st.write("For courses without a researched independent profile, this page falls back to the visible weekly Course-Fit audit. It does not invent unavailable course characteristics.")

else:
    st.header("📊 Results / Calibration")
    st.caption("A clean home for post-tournament and round-by-round validation. No hindsight changes are made from this page.")
    c1,c2,c3=st.columns(3); c1.metric("Tournament model","Ready ✓"); c2.metric("Round model","Available ✓"); c3.metric("Calibration history","Awaiting accumulated results")
    st.subheader("Current tournament outputs")
    st.dataframe(mc[[c for c in ["player","win_pct","top5_pct","top10_pct","make_cut_pct","expected_finish","dk_points_proxy"] if c in mc.columns]].head(25),width="stretch",hide_index=True)
    rp=st.session_state.get("round_parlay")
    if rp:
        st.subheader(f"Latest Round {rp['round']} probabilities")
        ok=rp["probs"][rp["probs"].status.eq("OK")]
        st.dataframe(ok,width="stretch",hide_index=True)
    st.info("As rounds finish, this page will compare predicted win/push/loss probabilities with actual 3-ball outcomes and tournament percentiles. We will evaluate calibration across events rather than tune to one result.")

st.divider()
st.caption("Team AI² Pro unified PGA: DraftKings-only independent tournament modeling, optional RickRunGood and OTIS ranking comparisons, and explicit provenance. "
           "Mexico 2026 RickRunGood course evidence is recorded separately, with no fabricated approach-bucket projections.")
