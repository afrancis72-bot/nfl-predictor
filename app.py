from pathlib import Path
import io
import math
import random
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NFL Predictor Pro", page_icon="🏈", layout="wide")

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MC_FILE = DATA / "nfl_week4_2026_correlated_monte_carlo_50000_v8_airyards_redzone.csv"
COMP_FILE = DATA / "nfl_week4_2026_component_projections_v4_airyards_redzone.csv"
DST_FILE = DATA / "nfl_week4_2026_dst_model_v2.csv"
OWN_FILE = DATA / "nfl_week4_2026_ownership_leverage_v1.csv"
GAME_FILE = DATA / "nfl_week4_2026_game_environment_v3_verified.csv"

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def normalize_team(x):
    return str(x).strip().upper()

def load_base():
    mc = load_csv(MC_FILE)
    comp = load_csv(COMP_FILE)
    dst = load_csv(DST_FILE)
    own = load_csv(OWN_FILE)
    games = load_csv(GAME_FILE)
    return mc, comp, dst, own, games

def prepare_pool(mc, dst, own):
    skill = mc.copy()
    skill["Position"] = skill["Position"].astype(str).str.upper()
    skill["TeamAbbrev"] = skill["TeamAbbrev"].map(normalize_team)
    skill["proj"] = pd.to_numeric(skill["mean"], errors="coerce")
    skill["ceiling"] = pd.to_numeric(skill["p90"], errors="coerce")
    skill["p95_use"] = pd.to_numeric(skill["p95"], errors="coerce")
    skill["Salary"] = pd.to_numeric(skill["Salary"], errors="coerce")
    skill["game"] = skill["game"].astype(str)

    own2 = own[["Name","Position","TeamAbbrev","ownership_pct","leverage_score"]].copy()
    own2["Position"] = own2["Position"].astype(str).str.upper()
    own2["TeamAbbrev"] = own2["TeamAbbrev"].map(normalize_team)
    skill = skill.merge(
        own2, on=["Name","Position","TeamAbbrev"], how="left", suffixes=("","_own")
    )
    skill["ownership_pct"] = pd.to_numeric(skill["ownership_pct"], errors="coerce").fillna(8.0)
    skill["leverage_score"] = pd.to_numeric(skill["leverage_score"], errors="coerce").fillna(0.0)

    d = dst.copy()
    d["Position"] = "DST"
    d["TeamAbbrev"] = d["TeamAbbrev"].map(normalize_team)
    d["proj"] = pd.to_numeric(d["dst_mean"], errors="coerce")
    d["ceiling"] = pd.to_numeric(d["dst_p90"], errors="coerce")
    d["p95_use"] = pd.to_numeric(d["dst_p95"], errors="coerce")
    d["ownership_pct"] = pd.to_numeric(d["ownership_pct"], errors="coerce").fillna(8.0)
    d["leverage_score"] = pd.to_numeric(d["leverage_score"], errors="coerce").fillna(0.0)
    d["Salary"] = pd.to_numeric(d["Salary"], errors="coerce")
    d["opponent"] = d["Opponent"].map(normalize_team)
    d = d[["Name","Position","Salary","TeamAbbrev","game","opponent","proj","ceiling","p95_use","ownership_pct","leverage_score"]]

    keep = ["Name","Position","Salary","TeamAbbrev","game","opponent","proj","ceiling","p95_use","ownership_pct","leverage_score"]
    skill = skill[keep]

    # Add current-slate role/activity metadata from the component model.
    meta_cols = ["Name","Position","TeamAbbrev","AvgPointsPerGame","dk_status",
                 "usage_source_verified","games_played"]
    meta = comp[meta_cols].drop_duplicates(["Name","Position","TeamAbbrev"]).copy()
    meta["Position"] = meta["Position"].astype(str).str.upper()
    meta["TeamAbbrev"] = meta["TeamAbbrev"].map(normalize_team)
    skill = skill.merge(meta, on=["Name","Position","TeamAbbrev"], how="left")

    # Eligibility guardrail:
    # - QBs in the MC file are the modeled starter set.
    # - RB/WR/TE must have at least 1.0 DK PPG on the current salary slate.
    # - anyone explicitly marked OUT is excluded.
    skill["AvgPointsPerGame"] = pd.to_numeric(skill["AvgPointsPerGame"], errors="coerce")
    skill["dk_status"] = skill["dk_status"].fillna("").astype(str).str.upper()
    skill["optimizer_eligible"] = (
        ((skill["Position"] == "QB") |
         (skill["AvgPointsPerGame"].fillna(0) >= 1.0))
        & (skill["dk_status"] != "OUT")
    )

    d["AvgPointsPerGame"] = np.nan
    d["dk_status"] = ""
    d["usage_source_verified"] = True
    d["games_played"] = np.nan
    d["optimizer_eligible"] = True

    pool = pd.concat([skill, d], ignore_index=True)
    pool = pool.dropna(subset=["Name","Position","Salary","proj"]).copy()
    pool["Salary"] = pool["Salary"].astype(int)
    pool = pool[pool["Salary"] > 0]
    return pool

def opponent_from_game(game, team):
    try:
        a,b = str(game).split("@")
        return b if team == a else a
    except Exception:
        return ""

def lineup_valid(df, min_salary, max_salary, stack_required=True):
    if len(df) != 9: return False
    counts = df["Position"].value_counts().to_dict()
    if counts.get("QB",0)!=1 or counts.get("DST",0)!=1: return False
    if counts.get("RB",0) < 2 or counts.get("WR",0) < 3 or counts.get("TE",0) < 1: return False
    flex_count = counts.get("RB",0)+counts.get("WR",0)+counts.get("TE",0)
    if flex_count != 7: return False
    sal = int(df["Salary"].sum())
    if sal < min_salary or sal > max_salary: return False
    if stack_required:
        qb = df[df.Position=="QB"].iloc[0]
        mates = df[(df.TeamAbbrev==qb.TeamAbbrev) & (df.Position.isin(["WR","TE"]))]
        if len(mates) < 1: return False
    dst = df[df.Position=="DST"].iloc[0]
    opp = opponent_from_game(dst["game"], dst["TeamAbbrev"])
    if opp and any((df.TeamAbbrev==opp) & (df.Position.isin(["QB","RB","WR","TE"]))):
        return False
    return True

def lineup_score(df, strategy, own_weight, leverage_weight):
    if strategy == "Median":
        base = df["proj"].sum()
    elif strategy == "Balanced":
        base = (0.55*df["proj"] + 0.45*df["ceiling"]).sum()
    else:
        base = (0.30*df["proj"] + 0.70*df["ceiling"]).sum()
    return float(base - own_weight*df["ownership_pct"].sum()/10 + leverage_weight*df["leverage_score"].sum())

def random_candidate(pool, min_salary, max_salary, strategy, own_weight, leverage_weight, locks, excludes):
    p = pool[(pool["optimizer_eligible"] == True) & ~pool.Name.isin(excludes)].copy()
    locked = p[p.Name.isin(locks)].drop_duplicates("Name")
    if len(locked) != len(set(locks)): return None

    def weighted_sample(df, n):
        if n <= 0: return df.iloc[0:0]
        if len(df) < n: return None
        if strategy == "Median":
            w = df["proj"].clip(lower=0.1)
        elif strategy == "Balanced":
            w = (0.55*df["proj"]+0.45*df["ceiling"]).clip(lower=0.1)
        else:
            w = (0.30*df["proj"]+0.70*df["ceiling"]+0.15*df["leverage_score"]).clip(lower=0.1)
        # Randomized softmax-ish weights to generate diverse candidate portfolios.
        w = np.power(w.to_numpy(), 2.0)
        w = w / w.sum()
        idx = np.random.choice(df.index.to_numpy(), size=n, replace=False, p=w)
        return df.loc[idx]

    # Build around a QB, then satisfy base slots + FLEX.
    if any(locked.Position=="QB"):
        qb = locked[locked.Position=="QB"].iloc[[0]]
    else:
        qb = weighted_sample(p[p.Position=="QB"],1)
        if qb is None: return None
    qbrow = qb.iloc[0]
    selected = pd.concat([locked, qb]).drop_duplicates("Name")

    # Force at least one QB pass-catcher if not already locked.
    if not any((selected.TeamAbbrev==qbrow.TeamAbbrev)&selected.Position.isin(["WR","TE"])):
        mates = p[(p.TeamAbbrev==qbrow.TeamAbbrev)&p.Position.isin(["WR","TE"])&~p.Name.isin(selected.Name)]
        mate = weighted_sample(mates,1)
        if mate is None: return None
        selected = pd.concat([selected,mate])

    requirements = {"RB":2,"WR":3,"TE":1,"DST":1}
    for pos, minimum in requirements.items():
        have = int((selected.Position==pos).sum())
        need = max(0, minimum-have)
        cand = p[(p.Position==pos)&~p.Name.isin(selected.Name)]
        samp = weighted_sample(cand, need)
        if samp is None: return None
        selected = pd.concat([selected,samp])

    # Add FLEX from RB/WR/TE until 9 total.
    while len(selected) < 9:
        cand = p[p.Position.isin(["RB","WR","TE"])&~p.Name.isin(selected.Name)]
        samp = weighted_sample(cand,1)
        if samp is None: return None
        selected = pd.concat([selected,samp])

    if len(selected) > 9: return None
    selected = selected.drop_duplicates("Name")
    if not lineup_valid(selected, min_salary, max_salary, True): return None
    return selected

def build_portfolio(pool, n_lineups, max_exposure, min_unique, min_salary, max_salary,
                    strategy, own_weight, leverage_weight, locks, excludes, attempts=50000):
    lineups=[]
    exposure={}
    max_count=max(1, math.floor(n_lineups*max_exposure+1e-9))
    for _ in range(attempts):
        if len(lineups)>=n_lineups: break
        cand=random_candidate(pool,min_salary,max_salary,strategy,own_weight,leverage_weight,locks,excludes)
        if cand is None: continue
        names=set(cand.Name)
        if any(exposure.get(n,0)>=max_count for n in names): continue
        # 9-man lineup: at least min_unique different => overlap <= 9-min_unique
        if any(len(names & set(x.Name)) > 9-min_unique for x in lineups): continue
        score=lineup_score(cand,strategy,own_weight,leverage_weight)
        # Accept strong candidates more often; keep random diversity.
        if lineups and random.random() < 0.35 and score < np.median([lineup_score(x,strategy,own_weight,leverage_weight) for x in lineups]):
            continue
        lineups.append(cand.copy())
        for n in names: exposure[n]=exposure.get(n,0)+1
    return lineups, exposure, max_count

def lineups_to_df(lineups):
    rows=[]
    order={"QB":0,"RB":1,"WR":2,"TE":3,"DST":4}
    for i,l in enumerate(lineups,1):
        x=l.sort_values("Position", key=lambda s:s.map(order))
        names=x["Name"].tolist()
        rows.append({
            "Lineup":i,
            "QB":", ".join(x[x.Position=="QB"].Name),
            "RB":", ".join(x[x.Position=="RB"].Name),
            "WR":", ".join(x[x.Position=="WR"].Name),
            "TE":", ".join(x[x.Position=="TE"].Name),
            "DST":", ".join(x[x.Position=="DST"].Name),
            "Salary":int(x.Salary.sum()),
            "Projection":round(float(x.proj.sum()),2),
            "Ceiling":round(float(x.ceiling.sum()),2),
            "Ownership Sum":round(float(x.ownership_pct.sum()),1),
            "Players":" | ".join(names)
        })
    return pd.DataFrame(rows)

def exposure_df(lineups):
    allp=pd.concat(lineups,ignore_index=True)
    counts=allp.groupby(["Name","Position","TeamAbbrev"],as_index=False).size()
    counts["Exposure %"]=100*counts["size"]/len(lineups)
    return counts.rename(columns={"size":"Lineups"}).sort_values(["Exposure %","Name"],ascending=[False,True])

mc, comp, dst, own, games = load_base()
pool = prepare_pool(mc,dst,own)

st.title("🏈 NFL Predictor Pro")
st.caption("DraftKings NFL DFS • projections • correlated Monte Carlo • leverage • portfolio optimization")
st.warning("Week 4 model snapshot. Optimizer excludes explicit OUT players and low-role salary-list players; re-check final injury news, salaries and ownership before contest entry.")

view = st.sidebar.radio("View", ["Slate Setup","Player Projections","Simulation","Lineup Builder","Portfolio Analysis"])

if view == "Slate Setup":
    st.subheader("Slate Setup")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Skill players", f"{len(mc):,}")
    c2.metric("DSTs", f"{len(dst):,}")
    c3.metric("Simulation runs", "50,000")
    c4.metric("Games", f"{pool['game'].nunique()}")
    st.markdown("#### Verified game environment")
    st.dataframe(games, use_container_width=True, hide_index=True)
    st.markdown("#### Weekly DraftKings upload")
    up=st.file_uploader("Optional: upload a new DraftKings salary CSV for slate review", type=["csv"])
    if up is not None:
        dk=pd.read_csv(up)
        st.success(f"Loaded {len(dk):,} salary rows. This build keeps the current Week 4 model projections until the weekly model refresh is run.")
        st.dataframe(dk.head(50),use_container_width=True,hide_index=True)

elif view == "Player Projections":
    st.subheader("Player Projections")
    positions=st.multiselect("Position", ["QB","RB","WR","TE","DST"], default=["QB","RB","WR","TE","DST"])
    teams=st.multiselect("Team", sorted(pool.TeamAbbrev.unique()), default=[])
    q=st.text_input("Search player")
    eligible_only=st.toggle("Optimizer-eligible players only", value=True)
    x=pool[pool.Position.isin(positions)].copy()
    if eligible_only: x=x[x["optimizer_eligible"] == True]
    if teams: x=x[x.TeamAbbrev.isin(teams)]
    if q: x=x[x.Name.str.contains(q,case=False,na=False)]
    x["Value/1K"]=x["proj"]/(x["Salary"]/1000)
    show=x.sort_values("proj",ascending=False)[["Name","Position","TeamAbbrev","game","Salary","proj","ceiling","p95_use","ownership_pct","leverage_score","AvgPointsPerGame","optimizer_eligible","Value/1K"]]
    st.dataframe(show,use_container_width=True,hide_index=True,
                 column_config={"proj":st.column_config.NumberColumn("Mean",format="%.2f"),
                                "ceiling":st.column_config.NumberColumn("P90",format="%.2f"),
                                "p95_use":st.column_config.NumberColumn("P95",format="%.2f"),
                                "ownership_pct":st.column_config.NumberColumn("Own %",format="%.1f"),
                                "leverage_score":st.column_config.NumberColumn("Leverage",format="%.2f"),
                                "Value/1K":st.column_config.NumberColumn("Value/1K",format="%.2f")})

elif view == "Simulation":
    st.subheader("50,000-run Correlated Monte Carlo")
    st.caption("Current V8 model includes role-aware correlations, target share, aDOT and red-zone inputs.")
    metric=st.selectbox("Sort by",["mean","p90","p95","boom_25","boom_30"])
    pos=st.multiselect("Position",["QB","RB","WR","TE"],default=["QB","RB","WR","TE"])
    sx=mc[mc.Position.isin(pos)].sort_values(metric,ascending=False)
    st.dataframe(sx,use_container_width=True,hide_index=True)
    st.download_button("Download simulation CSV", mc.to_csv(index=False), "nfl_simulation_v8.csv","text/csv")

elif view == "Lineup Builder":
    st.subheader("DraftKings Portfolio Optimizer")
    a,b,c,d=st.columns(4)
    n_lineups=a.number_input("Lineups",1,150,15,1)
    max_exp=b.slider("Max exposure",0.05,1.0,0.30,0.05)
    min_unique=c.slider("Minimum unique players",1,8,4,1)
    min_salary=d.number_input("Minimum lineup salary",40000,50000,47500,100)
    e,f,g=st.columns(3)
    strategy=e.selectbox("Strategy",["GPP Ceiling","Balanced","Median"])
    own_weight=f.slider("Ownership fade weight",0.0,2.0,0.25,0.05)
    leverage_weight=g.slider("Leverage weight",0.0,2.0,0.35,0.05)

    eligible=pool[pool["optimizer_eligible"] == True].sort_values(["Position","proj"],ascending=[True,False])
    names=eligible["Name"].drop_duplicates().tolist()
    locks=st.multiselect("Lock players",names)
    excludes=st.multiselect("Exclude players",names)

    st.info("Default portfolio rules: 15 lineups • 30% max exposure • 4 minimum unique players • $47,500 salary floor • QB + pass-catcher stack • no offensive player against selected DST.")

    if st.button("Generate portfolio",type="primary"):
        with st.spinner("Generating diversified portfolio..."):
            lineups, expo, max_count=build_portfolio(
                pool,int(n_lineups),float(max_exp),int(min_unique),int(min_salary),50000,
                strategy,float(own_weight),float(leverage_weight),locks,excludes
            )
        if not lineups:
            st.error("No valid portfolio found. Relax locks/exclusions, uniqueness, exposure, or salary floor.")
        else:
            st.session_state["nfl_lineups"]=lineups
            st.success(f"Generated {len(lineups)} of {int(n_lineups)} requested lineups.")
            ldf=lineups_to_df(lineups)
            edf=exposure_df(lineups)
            st.markdown("#### Lineups")
            st.dataframe(ldf,use_container_width=True,hide_index=True)
            st.download_button("Download lineups",ldf.to_csv(index=False),"nfl_lineups.csv","text/csv")
            st.markdown("#### Exposure")
            st.dataframe(edf,use_container_width=True,hide_index=True)
            st.download_button("Download exposure",edf.to_csv(index=False),"nfl_exposure.csv","text/csv")

elif view == "Portfolio Analysis":
    st.subheader("Portfolio Analysis")
    lineups=st.session_state.get("nfl_lineups")
    if not lineups:
        st.info("Generate a portfolio in Lineup Builder first.")
    else:
        ldf=lineups_to_df(lineups)
        edf=exposure_df(lineups)
        sets=[set(x.Name) for x in lineups]
        max_overlap=0
        for i in range(len(sets)):
            for j in range(i+1,len(sets)):
                max_overlap=max(max_overlap,len(sets[i]&sets[j]))
        min_unique_actual=9-max_overlap if len(sets)>1 else 9
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Lineups",len(lineups))
        c2.metric("Max exposure",f"{edf['Exposure %'].max():.1f}%")
        c3.metric("Min pairwise unique",min_unique_actual)
        c4.metric("Salary range",f"${ldf.Salary.min():,}–${ldf.Salary.max():,}")
        st.markdown("#### Exposure")
        st.dataframe(edf,use_container_width=True,hide_index=True)
        st.markdown("#### Game exposure")
        rows=[]
        for i,l in enumerate(lineups,1):
            for game,n in l.groupby("game").size().items():
                rows.append({"Lineup":i,"Game":game,"Players":int(n)})
        gx=pd.DataFrame(rows)
        summary=gx.groupby("Game",as_index=False).agg(Lineups=("Lineup","nunique"),Max_players_in_one_lineup=("Players","max"))
        summary["Lineup exposure %"]=100*summary["Lineups"]/len(lineups)
        st.dataframe(summary.sort_values("Lineup exposure %",ascending=False),use_container_width=True,hide_index=True)
