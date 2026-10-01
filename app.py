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
MATCHUP_FILE = DATA / "nfl_week4_2026_individual_matchups_v1.csv"

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
    matchups = load_csv(MATCHUP_FILE)
    return mc, comp, dst, own, games, matchups

def prepare_pool(mc, dst, own, matchups):
    skill = mc.copy()
    skill["Position"] = skill["Position"].astype(str).str.upper()
    skill["TeamAbbrev"] = skill["TeamAbbrev"].map(normalize_team)
    skill["proj"] = pd.to_numeric(skill["mean"], errors="coerce")
    skill["ceiling"] = pd.to_numeric(skill["p90"], errors="coerce")
    skill["p95_use"] = pd.to_numeric(skill["p95"], errors="coerce")
    skill["p99_use"] = pd.to_numeric(skill["p99"], errors="coerce")
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

    # V3.1 PROJECTION-INTEGRITY GUARDRAIL
    # Some weekly MC rows can be corrupted when upstream usage fields are missing.
    # Detect only extreme failures, then use a conservative blend of season DK PPG
    # and same-position / nearby-salary peers. Reliable MC rows are untouched.
    ppg_map = comp[["Name","Position","TeamAbbrev","AvgPointsPerGame"]].drop_duplicates(["Name","Position","TeamAbbrev"]).copy()
    ppg_map["Position"] = ppg_map["Position"].astype(str).str.upper()
    ppg_map["TeamAbbrev"] = ppg_map["TeamAbbrev"].map(normalize_team)
    ppg_map["AvgPointsPerGame"] = pd.to_numeric(ppg_map["AvgPointsPerGame"], errors="coerce")
    skill = skill.merge(ppg_map, on=["Name","Position","TeamAbbrev"], how="left")

    skill["projection_repaired"] = False
    skill["projection_repair_note"] = ""
    reliable = (skill["proj"] >= 0.35 * skill["AvgPointsPerGame"].fillna(0)) | (skill["AvgPointsPerGame"].fillna(0) < 5)

    # Build salary-local positional baselines using only reliable rows.
    for pos in ["QB","RB","WR","TE"]:
        pos_idx = skill["Position"].eq(pos)
        bad_idx = pos_idx & ~reliable & (skill["AvgPointsPerGame"].fillna(0) >= 5)
        good = skill[pos_idx & reliable & skill["proj"].notna()].copy()
        if good.empty:
            continue
        for i in skill.index[bad_idx]:
            sal = skill.at[i,"Salary"]
            peers = good[(good["Salary"] >= sal - 1200) & (good["Salary"] <= sal + 1200)]
            if len(peers) < 5:
                peers = good
            salary_baseline = float(peers["proj"].median())
            ppg = float(skill.at[i,"AvgPointsPerGame"] or 0)
            repaired = 0.65 * ppg + 0.35 * salary_baseline
            # Keep fallback conservative and bounded around the observed scoring baseline.
            repaired = float(np.clip(repaired, 0.70 * ppg, 1.10 * ppg))
            p90_mult = float((good["ceiling"] / good["proj"].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).dropna().median())
            p95_mult = float((good["p95_use"] / good["proj"].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).dropna().median())
            p99_mult = float((good["p99_use"] / good["proj"].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).dropna().median())
            p90_mult = p90_mult if np.isfinite(p90_mult) else 1.75
            p95_mult = p95_mult if np.isfinite(p95_mult) else 2.15
            p99_mult = p99_mult if np.isfinite(p99_mult) else 2.65
            skill.at[i,"proj"] = repaired
            skill.at[i,"ceiling"] = repaired * p90_mult
            skill.at[i,"p95_use"] = repaired * p95_mult
            skill.at[i,"p99_use"] = repaired * p99_mult
            skill.at[i,"projection_repaired"] = True
            skill.at[i,"projection_repair_note"] = "MC integrity fallback: DK PPG + salary/position peers"

    # V3 INDIVIDUAL COVERAGE MATCHUP LAYER
    # Weekly research rows are player-specific. Unlisted players remain neutral (1.00).
    # Factors are deliberately bounded so matchup context cannot overpower volume/talent.
    m = matchups.copy()
    m["Position"] = m["Position"].astype(str).str.upper()
    m["TeamAbbrev"] = m["TeamAbbrev"].map(normalize_team)
    m["individual_matchup_factor"] = pd.to_numeric(m["individual_matchup_factor"], errors="coerce").clip(0.94, 1.06)
    matchup_cols = ["Name","Position","TeamAbbrev","expected_primary_coverage",
                    "coverage_matchup_grade","individual_matchup_factor","matchup_notes",
                    "matchup_source","matchup_source_url"]
    skill = skill.merge(m[matchup_cols], on=["Name","Position","TeamAbbrev"], how="left")
    skill["individual_matchup_factor"] = skill["individual_matchup_factor"].fillna(1.0)
    skill["coverage_matchup_grade"] = skill["coverage_matchup_grade"].fillna("Neutral / Not Researched")
    skill["expected_primary_coverage"] = skill["expected_primary_coverage"].fillna("")
    skill["matchup_notes"] = skill["matchup_notes"].fillna("")
    skill["pre_individual_matchup_proj"] = skill["proj"]
    skill["proj"] = skill["proj"] * skill["individual_matchup_factor"]
    skill["ceiling"] = skill["ceiling"] * skill["individual_matchup_factor"]
    skill["p95_use"] = skill["p95_use"] * skill["individual_matchup_factor"]
    skill["p99_use"] = skill["p99_use"] * skill["individual_matchup_factor"]
    skill["individual_matchup_delta"] = skill["proj"] - skill["pre_individual_matchup_proj"]

    d = dst.copy()
    d["Position"] = "DST"
    d["TeamAbbrev"] = d["TeamAbbrev"].map(normalize_team)
    d["proj"] = pd.to_numeric(d["dst_mean"], errors="coerce")
    d["ceiling"] = pd.to_numeric(d["dst_p90"], errors="coerce")
    d["p95_use"] = pd.to_numeric(d["dst_p95"], errors="coerce")
    # DST source has no P99 field; use P95 conservatively rather than inventing extra tail upside.
    d["p99_use"] = d["p95_use"]
    d["ownership_pct"] = pd.to_numeric(d["ownership_pct"], errors="coerce").fillna(8.0)
    d["leverage_score"] = pd.to_numeric(d["leverage_score"], errors="coerce").fillna(0.0)
    d["Salary"] = pd.to_numeric(d["Salary"], errors="coerce")
    d["opponent"] = d["Opponent"].map(normalize_team)
    d = d[["Name","Position","Salary","TeamAbbrev","game","opponent","proj","ceiling","p95_use","p99_use","ownership_pct","leverage_score"]]

    keep = ["Name","Position","Salary","TeamAbbrev","game","opponent","proj","ceiling","p95_use","p99_use","ownership_pct","leverage_score",
            "expected_primary_coverage","coverage_matchup_grade","individual_matchup_factor",
            "individual_matchup_delta","matchup_notes","matchup_source","matchup_source_url",
            "projection_repaired","projection_repair_note"]
    skill = skill[keep]

    # Add current-slate role/activity metadata from the component model.
    meta_cols = ["Name","Position","TeamAbbrev","AvgPointsPerGame","dk_status",
                 "usage_source_verified","games_played","dfs_research_pool",
                 "opportunity_score_v1","role_confidence","active_scenario",
                 "scenario_opportunity","role_change_flag","active_for_model"]
    meta = comp[meta_cols].drop_duplicates(["Name","Position","TeamAbbrev"]).copy()
    meta["Position"] = meta["Position"].astype(str).str.upper()
    meta["TeamAbbrev"] = meta["TeamAbbrev"].map(normalize_team)
    skill = skill.merge(meta, on=["Name","Position","TeamAbbrev"], how="left")

    # V2 ROLE-BASED ELIGIBILITY
    # Do not use a crude fantasy-points cutoff. A skill player must have a
    # credible usage/research signal in the Week 4 component model.
    for c in ["AvgPointsPerGame","opportunity_score_v1","role_confidence",
              "scenario_opportunity","games_played"]:
        skill[c] = pd.to_numeric(skill[c], errors="coerce")
    for c in ["usage_source_verified","dfs_research_pool","active_for_model"]:
        skill[c] = skill[c].fillna(False).astype(bool)
    skill["dk_status"] = skill["dk_status"].fillna("").astype(str).str.upper()

    role_signal = (
        skill["usage_source_verified"]
        | skill["dfs_research_pool"]
        | (
            skill["active_scenario"].notna()
            & (skill["role_confidence"].fillna(0) >= 0.45)
            & (skill["scenario_opportunity"].fillna(0) >= 3.0)
        )
    )

    skill["optimizer_eligible"] = (
        skill["active_for_model"]
        & (skill["dk_status"] != "OUT")
        & ((skill["Position"] == "QB") | role_signal)
    )

    # Transparent role labels shown in the UI.
    core = (
        (skill["Position"] == "QB")
        | (skill["opportunity_score_v1"].fillna(0) >= 45)
        | (skill["AvgPointsPerGame"].fillna(0) >= 15)
    )
    rotation = (
        (skill["opportunity_score_v1"].fillna(0) >= 30)
        | (skill["AvgPointsPerGame"].fillna(0) >= 8)
        | (
            skill["active_scenario"].notna()
            & (skill["role_confidence"].fillna(0) >= 0.55)
        )
    )
    skill["role_status"] = np.select(
        [
            ~skill["optimizer_eligible"],
            skill["optimizer_eligible"] & core,
            skill["optimizer_eligible"] & rotation,
        ],
        ["No Role / Blocked","Core / Starter","Rotation"],
        default="Thin GPP",
    )

    d["AvgPointsPerGame"] = np.nan
    d["dk_status"] = ""
    d["usage_source_verified"] = True
    d["games_played"] = np.nan
    d["dfs_research_pool"] = True
    d["opportunity_score_v1"] = np.nan
    d["role_confidence"] = np.nan
    d["active_scenario"] = np.nan
    d["scenario_opportunity"] = np.nan
    d["role_change_flag"] = np.nan
    d["active_for_model"] = True
    d["optimizer_eligible"] = True
    d["role_status"] = "DST"
    d["expected_primary_coverage"] = ""
    d["coverage_matchup_grade"] = "N/A"
    d["individual_matchup_factor"] = 1.0
    d["individual_matchup_delta"] = 0.0
    d["matchup_notes"] = ""
    d["matchup_source"] = ""
    d["matchup_source_url"] = ""
    d["projection_repaired"] = False
    d["projection_repair_note"] = ""

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
    # V3.1.3: tournament ranking uses more of the simulated right tail.
    # These are sums of PLAYER percentile outcomes (an upside index), not claims
    # that the resulting totals are true lineup-level P90/P95/P99 percentiles.
    if strategy == "Median":
        base = df["proj"].sum()
    elif strategy == "Balanced":
        base = (0.50*df["proj"] + 0.20*df["ceiling"] + 0.20*df["p95_use"] + 0.10*df["p99_use"]).sum()
    else:
        base = (0.20*df["proj"] + 0.25*df["ceiling"] + 0.30*df["p95_use"] + 0.25*df["p99_use"]).sum()
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
                    strategy, own_weight, leverage_weight, locks, excludes, projection_floor_pct=0.88, attempts=4000):
    """V3.1.5 fast array-based portfolio search.

    Candidate construction uses NumPy arrays and integer row IDs inside the hot
    loop. Pandas DataFrames are created only for valid completed candidates.
    This preserves the same roster, stack, DST-conflict, salary, exposure,
    uniqueness, quality-floor and ranking rules while avoiding thousands of
    DataFrame filters/concats during generation.
    """
    max_count=max(1, math.floor(n_lineups*max_exposure+1e-9))
    overlap_limit=9-int(min_unique)
    p=pool[(pool["optimizer_eligible"] == True) & ~pool.Name.isin(excludes)].copy().reset_index(drop=True)
    if not set(locks).issubset(set(p.Name)):
        return [],{},max_count,0.0,0.0

    names=p["Name"].astype(str).to_numpy()
    pos=p["Position"].astype(str).to_numpy()
    team=p["TeamAbbrev"].astype(str).to_numpy()
    game=p["game"].astype(str).to_numpy()
    salary=p["Salary"].astype(int).to_numpy()
    proj=p["proj"].astype(float).to_numpy()
    ceil=p["ceiling"].astype(float).to_numpy()
    p95=p["p95_use"].astype(float).to_numpy()
    p99=p["p99_use"].astype(float).to_numpy()
    own=p["ownership_pct"].astype(float).to_numpy()
    lev=p["leverage_score"].astype(float).to_numpy()

    if strategy == "Median": base_w=np.maximum(proj,0.1)
    elif strategy == "Balanced": base_w=np.maximum(0.55*proj+0.45*ceil,0.1)
    else: base_w=np.maximum(0.30*proj+0.70*ceil+0.15*lev,0.1)
    base_w=np.square(base_w)

    by_pos={q:np.flatnonzero(pos==q) for q in ["QB","RB","WR","TE","DST"]}
    skill_idx=np.flatnonzero(np.isin(pos,["RB","WR","TE"]))
    lock_idx=[int(np.flatnonzero(names==n)[0]) for n in locks]
    if len(set(lock_idx)) != len(lock_idx):
        return [],{},max_count,0.0,0.0
    rng=np.random.default_rng(315)

    def pick(candidates, selected):
        if candidates.size==0: return None
        if selected:
            candidates=candidates[~np.isin(candidates,np.fromiter(selected,dtype=int))]
        if candidates.size==0: return None
        w=base_w[candidates]; sw=float(w.sum())
        if not np.isfinite(sw) or sw<=0: return int(rng.choice(candidates))
        return int(rng.choice(candidates,p=w/sw))

    def make_candidate():
        selected=set(lock_idx)
        # Reject impossible lock structures early.
        if sum(pos[i]=="QB" for i in selected)>1 or sum(pos[i]=="DST" for i in selected)>1: return None
        if len(selected)>9: return None
        qlocks=[i for i in selected if pos[i]=="QB"]
        qi=qlocks[0] if qlocks else pick(by_pos["QB"],selected)
        if qi is None:return None
        selected.add(qi)
        # Force QB pass catcher.
        if not any(team[i]==team[qi] and pos[i] in ("WR","TE") for i in selected if i!=qi):
            mates=np.flatnonzero((team==team[qi]) & np.isin(pos,["WR","TE"]))
            mi=pick(mates,selected)
            if mi is None:return None
            selected.add(mi)
        for q,minimum in (("RB",2),("WR",3),("TE",1),("DST",1)):
            while sum(pos[i]==q for i in selected)<minimum:
                j=pick(by_pos[q],selected)
                if j is None:return None
                selected.add(j)
        while len(selected)<9:
            j=pick(skill_idx,selected)
            if j is None:return None
            selected.add(j)
        if len(selected)!=9:return None
        ids=np.fromiter(selected,dtype=int)
        counts={q:int(np.sum(pos[ids]==q)) for q in ["QB","RB","WR","TE","DST"]}
        if counts["QB"]!=1 or counts["DST"]!=1 or counts["RB"]<2 or counts["WR"]<3 or counts["TE"]<1:return None
        if counts["RB"]+counts["WR"]+counts["TE"]!=7:return None
        sal=int(salary[ids].sum())
        if sal<min_salary or sal>max_salary:return None
        di=ids[pos[ids]=="DST"][0]
        opp=opponent_from_game(game[di],team[di])
        if opp and any(team[i]==opp and pos[i] in ("QB","RB","WR","TE") for i in ids):return None
        # score directly from arrays
        if strategy=="Median": base=float(proj[ids].sum())
        elif strategy=="Balanced": base=float((0.50*proj[ids]+0.20*ceil[ids]+0.20*p95[ids]+0.10*p99[ids]).sum())
        else: base=float((0.20*proj[ids]+0.25*ceil[ids]+0.30*p95[ids]+0.25*p99[ids]).sum())
        score=base-own_weight*float(own[ids].sum())/10+leverage_weight*float(lev[ids].sum())
        return tuple(sorted(ids.tolist())),float(proj[ids].sum()),float(score)

    bank={}
    target_bank=max(1200,int(n_lineups)*50)
    max_attempts=max(8000,int(n_lineups)*400)
    for _ in range(max_attempts):
        z=make_candidate()
        if z is None:continue
        key=z[0]
        if key not in bank:bank[key]=z
        if len(bank)>=target_bank:break
    if not bank:return [],{},max_count,0.0,0.0

    vals=np.array([v[1] for v in bank.values()],dtype=float)
    reference_proj=float(np.percentile(vals,99))
    min_projection_required=reference_proj*float(projection_floor_pct)
    candidates=[v for v in bank.values() if v[1]>=min_projection_required]
    candidates.sort(key=lambda z:(z[2],z[1]),reverse=True)

    def greedy(order):
        chosen=[]; chosen_sets=[]; exposure={}; total=0.0
        for ids,pr,sc in order:
            nset={names[i] for i in ids}
            if any(exposure.get(n,0)>=max_count for n in nset):continue
            if any(len(nset & prev)>overlap_limit for prev in chosen_sets):continue
            chosen.append(ids);chosen_sets.append(nset);total+=sc
            for n in nset:exposure[n]=exposure.get(n,0)+1
            if len(chosen)>=n_lineups:break
        return chosen,exposure,total

    best=greedy(candidates)
    if len(best[0])<n_lineups and len(candidates)>n_lineups:
        scores=np.array([z[2] for z in candidates]); scale=max(float(np.std(scores)),1.0)
        for _ in range(180):
            order=[candidates[i] for i in np.argsort(-(scores+rng.normal(0,0.16*scale,len(scores))))]
            trial=greedy(order)
            if len(trial[0])>len(best[0]) or (len(trial[0])==len(best[0]) and trial[2]>best[2]):best=trial
            if len(best[0])>=n_lineups:break

    lineups=[p.iloc[list(ids)].copy() for ids in best[0]]
    return lineups,best[1],max_count,reference_proj,min_projection_required

def lineups_to_df(lineups, strategy=None, own_weight=0.25, leverage_weight=0.35, stack_rank=False):
    rows=[]
    order={"QB":0,"RB":1,"WR":2,"TE":3,"DST":4}
    for i,l in enumerate(lineups,1):
        x=l.sort_values("Position", key=lambda s:s.map(order))
        names=x["Name"].tolist()
        row={
            "Lineup":i,
            "QB":", ".join(x[x.Position=="QB"].Name),
            "RB":", ".join(x[x.Position=="RB"].Name),
            "WR":", ".join(x[x.Position=="WR"].Name),
            "TE":", ".join(x[x.Position=="TE"].Name),
            "DST":", ".join(x[x.Position=="DST"].Name),
            "Salary":int(x.Salary.sum()),
            "Projection":round(float(x.proj.sum()),2),
            "P90 Upside":round(float(x.ceiling.sum()),2),
            "P95 Upside":round(float(x.p95_use.sum()),2),
            "P99 Upside":round(float(x.p99_use.sum()),2),
            "Ownership Sum":round(float(x.ownership_pct.sum()),1),
            "Players":" | ".join(names)
        }
        if strategy is not None:
            row["GPP Score"] = round(lineup_score(x,strategy,own_weight,leverage_weight),2)
        rows.append(row)
    out=pd.DataFrame(rows)
    if stack_rank and "GPP Score" in out.columns:
        out=out.sort_values(["GPP Score","P99 Upside","Projection"],ascending=[False,False,False]).reset_index(drop=True)
        out.insert(0,"Rank",range(1,len(out)+1))
    return out

def exposure_df(lineups):
    allp=pd.concat(lineups,ignore_index=True)
    counts=allp.groupby(["Name","Position","TeamAbbrev"],as_index=False).size()
    counts["Exposure %"]=100*counts["size"]/len(lineups)
    return counts.rename(columns={"size":"Lineups"}).sort_values(["Exposure %","Name"],ascending=[False,True])

mc, comp, dst, own, games, matchups = load_base()
pool = prepare_pool(mc,dst,own,matchups)

st.title("🏈 NFL Predictor Pro")
st.caption("DraftKings NFL DFS • projections • correlated Monte Carlo • leverage • portfolio optimization")
st.warning("Week 4 model snapshot. V3 adds a bounded individual coverage-matchup layer on top of V2 role eligibility. Unresearched players remain neutral; re-check final injury news, salaries, coverage assignments and ownership before contest entry.")

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
    show=x.sort_values("proj",ascending=False)[["Name","Position","TeamAbbrev","game","Salary","proj","ceiling","p95_use","p99_use","ownership_pct","leverage_score","AvgPointsPerGame","role_status","coverage_matchup_grade","individual_matchup_factor","expected_primary_coverage","individual_matchup_delta","projection_repaired","optimizer_eligible","Value/1K"]]
    st.dataframe(show,use_container_width=True,hide_index=True,
                 column_config={"proj":st.column_config.NumberColumn("Mean",format="%.2f"),
                                "ceiling":st.column_config.NumberColumn("P90",format="%.2f"),
                                "p95_use":st.column_config.NumberColumn("P95",format="%.2f"),
                                "p99_use":st.column_config.NumberColumn("P99",format="%.2f"),
                                "ownership_pct":st.column_config.NumberColumn("Own %",format="%.1f"),
                                "leverage_score":st.column_config.NumberColumn("Leverage",format="%.2f"),
                                "individual_matchup_factor":st.column_config.NumberColumn("Coverage Factor",format="%.3f"),
                                "individual_matchup_delta":st.column_config.NumberColumn("Matchup Δ",format="%+.2f"),
                                "Value/1K":st.column_config.NumberColumn("Value/1K",format="%.2f")})
    st.markdown("#### V3 researched individual matchups")
    st.caption("Only researched player-specific matchups receive a non-neutral factor. The adjustment is capped and layered after the team-defense matchup model.")
    board = pool[(pool["Position"].isin(["WR","TE"])) & (pool["coverage_matchup_grade"] != "Neutral / Not Researched")][["Name","TeamAbbrev","opponent","expected_primary_coverage","coverage_matchup_grade","individual_matchup_factor","individual_matchup_delta","matchup_notes"]].drop_duplicates("Name")
    st.dataframe(board.sort_values("individual_matchup_factor",ascending=False),use_container_width=True,hide_index=True)

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
    own_weight=f.slider("Ownership fade weight",0.0,2.0,0.15,0.05, help="Lower values are more willing to roster strong chalk. Ownership still matters, but good high-owned plays are not heavily penalized.")
    leverage_weight=g.slider("Leverage weight",0.0,2.0,0.35,0.05)
    projection_floor_pct=st.slider("Projection quality floor (% of strong reference lineup)",0.75,1.00,0.88,0.01, help="Requires every generated lineup to retain this percentage of a strong reference projection from the current slate. 88% is a balanced GPP default.")

    eligible=pool[pool["optimizer_eligible"] == True].sort_values(["Position","proj"],ascending=[True,False])
    names=eligible["Name"].drop_duplicates().tolist()
    locks=st.multiselect("Lock players",names)
    excludes=st.multiselect("Exclude players",names)

    st.info("Default portfolio rules: 15 lineups • 30% max exposure • 4 minimum unique players • $47,500 salary floor • 88% projection-quality floor • QB + pass-catcher stack • no offensive player against selected DST. Ownership fade defaults to a chalk-friendly 0.15; leverage remains 0.35. V3.1.4 uses the same Mean + P90 + P95 + P99 GPP ranking with a faster reusable candidate-bank optimizer.")

    if st.button("Generate portfolio",type="primary"):
        with st.spinner("Generating diversified portfolio..."):
            lineups, expo, max_count, reference_proj, min_proj_required=build_portfolio(
                pool,int(n_lineups),float(max_exp),int(min_unique),int(min_salary),50000,
                strategy,float(own_weight),float(leverage_weight),locks,excludes,float(projection_floor_pct)
            )
        if not lineups:
            st.error("No valid portfolio found. Relax locks/exclusions, uniqueness, exposure, or salary floor.")
        else:
            st.session_state["nfl_lineups"]=lineups
            st.session_state["nfl_rank_settings"]={"strategy":strategy,"own_weight":float(own_weight),"leverage_weight":float(leverage_weight)}
            st.success(f"Generated {len(lineups)} of {int(n_lineups)} requested lineups.")
            st.caption(f"Projection quality guardrail: strong reference {reference_proj:.1f} DK points • minimum accepted {min_proj_required:.1f} ({projection_floor_pct:.0%}).")
            ldf=lineups_to_df(lineups,strategy,float(own_weight),float(leverage_weight),stack_rank=True)
            edf=exposure_df(lineups)
            st.markdown("#### Lineups — stack ranked best to worst")
            st.caption("V3.1.4 GPP Rank blends Mean + P90 + P95 + P99 player-level upside, then applies the selected ownership fade and leverage weights. P90/P95/P99 Upside are comparison indexes (sums of player percentiles), not literal lineup percentiles.")
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
        rank_settings=st.session_state.get("nfl_rank_settings",{"strategy":"GPP Ceiling","own_weight":0.25,"leverage_weight":0.35})
        ldf=lineups_to_df(lineups,rank_settings["strategy"],rank_settings["own_weight"],rank_settings["leverage_weight"],stack_rank=True)
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
        st.markdown("#### Stack-ranked lineups")
        st.caption("GPP Rank emphasizes simulated right-tail upside while preserving projection quality. P90/P95/P99 Upside are player-percentile sums used for comparison, not literal lineup percentiles. Rank is relative to this portfolio, not a guarantee of contest outcome.")
        st.dataframe(ldf,use_container_width=True,hide_index=True)
        st.download_button("Download ranked lineups",ldf.to_csv(index=False),"nfl_lineups_ranked.csv","text/csv")
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
