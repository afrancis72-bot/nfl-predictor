from pathlib import Path
import io
import math
import random
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, vstack, csr_matrix

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
                    strategy, own_weight, leverage_weight, locks, excludes, projection_floor_pct=0.88, attempts=50000):
    lineups=[]
    exposure={}
    max_count=max(1, math.floor(n_lineups*max_exposure+1e-9))

    # V3.1.2 QUALITY FLOOR: estimate a strong reference projection from valid
    # candidates, then require portfolio lineups to retain a chosen percentage
    # of that scoring strength. This prevents ownership/leverage from rescuing
    # lineups that give away too many expected DK points.
    reference_projections=[]
    calibration_attempts=min(6000, max(1500, attempts // 8))
    for _ in range(calibration_attempts):
        probe=random_candidate(pool,min_salary,max_salary,strategy,own_weight,leverage_weight,locks,excludes)
        if probe is not None:
            reference_projections.append(float(probe["proj"].sum()))
    if reference_projections:
        # 99th percentile is more stable than one lucky randomized maximum.
        best_reference_projection=float(np.percentile(reference_projections,99))
        min_projection_required=best_reference_projection*float(projection_floor_pct)
    else:
        best_reference_projection=0.0
        min_projection_required=0.0

    for _ in range(attempts):
        if len(lineups)>=n_lineups: break
        cand=random_candidate(pool,min_salary,max_salary,strategy,own_weight,leverage_weight,locks,excludes)
        if cand is None: continue
        if float(cand["proj"].sum()) < min_projection_required: continue
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
    return lineups, exposure, max_count, best_reference_projection, min_projection_required

def generate_candidate_bank(pool, bank_size, min_salary, max_salary, strategy,
                            own_weight, leverage_weight, locks, excludes,
                            projection_floor_pct=0.88, attempts=None):
    """V3.1.6a: fast deduplicated candidate-bank generation.

    The common no-lock/no-exclusion path constructs rosters with NumPy/index
    operations and materializes a DataFrame only after a roster is legal.
    Locks/exclusions retain the legacy generator for compatibility.
    """
    bank_size=int(bank_size)
    p=pool[(pool["optimizer_eligible"] == True) & ~pool.Name.isin(excludes)].copy().reset_index(drop=True)

    # Preserve full legacy behavior for special lock/exclusion builds.
    if locks or excludes:
        max_attempts = int(attempts or max(12000, bank_size*45))
        calibration_attempts=min(1200, max(500, bank_size))
        refs=[]
        for _ in range(calibration_attempts):
            probe=random_candidate(pool,min_salary,max_salary,strategy,own_weight,leverage_weight,locks,excludes)
            if probe is not None: refs.append(float(probe["proj"].sum()))
        ref=float(np.percentile(refs,99)) if refs else 0.0
        floor=ref*float(projection_floor_pct)
        bank={}
        for _ in range(max_attempts):
            if len(bank)>=bank_size: break
            cand=random_candidate(pool,min_salary,max_salary,strategy,own_weight,leverage_weight,locks,excludes)
            if cand is None or float(cand["proj"].sum()) < floor: continue
            key=tuple(sorted(cand["Name"].astype(str)))
            sc=lineup_score(cand,strategy,own_weight,leverage_weight)
            if key not in bank or sc>bank[key][1]: bank[key]=(cand.copy(),sc)
        return sorted(bank.values(),key=lambda z:z[1],reverse=True),ref,floor

    names=p["Name"].astype(str).to_numpy()
    pos=p["Position"].astype(str).to_numpy()
    teams=p["TeamAbbrev"].astype(str).to_numpy()
    games=p["game"].astype(str).to_numpy()
    salary=p["Salary"].to_numpy(dtype=float)
    proj=p["proj"].to_numpy(dtype=float)
    ceil=p["ceiling"].to_numpy(dtype=float)
    lev=p["leverage_score"].to_numpy(dtype=float)

    if strategy == "Median": raw=np.clip(proj,0.1,None)
    elif strategy == "Balanced": raw=np.clip(0.55*proj+0.45*ceil,0.1,None)
    else: raw=np.clip(0.30*proj+0.70*ceil+0.15*lev,0.1,None)
    weights=np.square(raw)

    by_pos={k:np.where(pos==k)[0] for k in ["QB","RB","WR","TE","DST"]}
    flex_idx=np.where(np.isin(pos,["RB","WR","TE"]))[0]
    mates={}
    for q in by_pos["QB"]:
        mates[q]=np.where((teams==teams[q]) & np.isin(pos,["WR","TE"]))[0]

    rng=np.random.default_rng()
    def pick(candidates, selected):
        if len(candidates)==0: return None
        if selected:
            candidates=candidates[~np.isin(candidates,np.fromiter(selected,dtype=int))]
        if len(candidates)==0: return None
        w=weights[candidates]; total=w.sum()
        if not np.isfinite(total) or total<=0: return int(rng.choice(candidates))
        return int(rng.choice(candidates,p=w/total))

    def fast_roster():
        sel=set()
        q=pick(by_pos["QB"],sel)
        if q is None: return None
        sel.add(q)
        m=pick(mates.get(q,np.array([],dtype=int)),sel)
        if m is None: return None
        sel.add(m)
        requirements={"RB":2,"WR":3,"TE":1,"DST":1}
        for k,need in requirements.items():
            while sum(pos[i]==k for i in sel)<need:
                z=pick(by_pos[k],sel)
                if z is None: return None
                sel.add(z)
        while len(sel)<9:
            z=pick(flex_idx,sel)
            if z is None: return None
            sel.add(z)
        idx=np.fromiter(sel,dtype=int)
        sal=float(salary[idx].sum())
        if sal<min_salary or sal>max_salary: return None
        d=idx[pos[idx]=="DST"]
        if len(d)!=1: return None
        di=int(d[0]); opp=opponent_from_game(games[di],teams[di])
        if opp and any((teams[i]==opp and pos[i] in ("QB","RB","WR","TE")) for i in idx): return None
        return idx

    # Fast calibration: enough samples for a stable 99th-percentile reference,
    # without thousands of expensive DataFrame builds.
    refs=[]
    calibration_attempts=max(500,min(1200,bank_size))
    for _ in range(calibration_attempts):
        idx=fast_roster()
        if idx is not None: refs.append(float(proj[idx].sum()))
    ref=float(np.percentile(refs,99)) if refs else 0.0
    floor=ref*float(projection_floor_pct)

    bank={}
    max_attempts=int(attempts or max(10000,bank_size*35))
    for _ in range(max_attempts):
        if len(bank)>=bank_size: break
        idx=fast_roster()
        if idx is None or float(proj[idx].sum())<floor: continue
        key=tuple(sorted(names[idx].tolist()))
        if key in bank: continue
        cand=p.iloc[idx].copy()
        bank[key]=(cand,lineup_score(cand,strategy,own_weight,leverage_weight))

    items=sorted(bank.values(),key=lambda z:z[1],reverse=True)
    return items,ref,floor


def select_portfolio_milp(candidate_items, n_lineups, max_exposure, min_unique, time_limit=20.0):
    """Choose the entire portfolio simultaneously from a candidate bank."""
    if len(candidate_items) < n_lineups:
        return [], {"success":False,"message":"Candidate bank smaller than requested portfolio."}
    lineups=[x[0] for x in candidate_items]
    scores=np.array([x[1] for x in candidate_items],dtype=float)
    n=len(lineups)
    max_count=max(1, math.floor(n_lineups*max_exposure+1e-9))
    sets=[set(x["Name"].astype(str)) for x in lineups]

    # Rows: exact lineup count; per-player exposure; incompatible overlap pairs.
    rows=[]; lbs=[]; ubs=[]
    rows.append(np.ones(n)); lbs.append(float(n_lineups)); ubs.append(float(n_lineups))

    players=sorted(set().union(*sets))
    for player in players:
        row=np.fromiter((1.0 if player in s else 0.0 for s in sets),dtype=float,count=n)
        rows.append(row); lbs.append(-np.inf); ubs.append(float(max_count))

    max_overlap=9-int(min_unique)
    # Candidate bank is capped in UI because pairwise incompatibility constraints are O(N^2).
    for i in range(n):
        si=sets[i]
        for j in range(i+1,n):
            if len(si & sets[j]) > max_overlap:
                row=np.zeros(n,dtype=float); row[i]=1.0; row[j]=1.0
                rows.append(row); lbs.append(-np.inf); ubs.append(1.0)

    A=csr_matrix(np.vstack(rows))
    constraints=LinearConstraint(A,np.asarray(lbs),np.asarray(ubs))
    res=milp(c=-scores, integrality=np.ones(n,dtype=int), bounds=Bounds(0,1),
             constraints=constraints,
             options={"time_limit":float(time_limit),"mip_rel_gap":0.001,"presolve":True})
    if res.x is None:
        return [], {"success":False,"message":str(res.message),"status":int(res.status)}
    chosen=np.where(res.x > 0.5)[0].tolist()
    selected=[lineups[i].copy() for i in chosen]
    selected.sort(key=lambda x: lineup_score(x,"GPP Ceiling",0.0,0.0), reverse=True)
    meta={"success":len(selected)==n_lineups,"message":str(res.message),"status":int(res.status),
          "candidate_count":n,"constraint_count":len(rows),"max_count":max_count,
          "objective_score":float(scores[chosen].sum()) if chosen else 0.0,
          "mip_gap":getattr(res,"mip_gap",None)}
    return selected, meta


def validate_portfolio(lineups, n_lineups, max_exposure, min_unique, min_salary, max_salary):
    issues=[]
    if len(lineups) != int(n_lineups):
        issues.append(f"Expected {int(n_lineups)} lineups; found {len(lineups)}.")
    max_count=max(1, math.floor(n_lineups*max_exposure+1e-9))
    counts={}
    keys=[]
    for i,l in enumerate(lineups,1):
        if not lineup_valid(l,min_salary,max_salary,True):
            issues.append(f"Lineup {i} fails lineup validity rules.")
        names=set(l["Name"].astype(str)); keys.append(tuple(sorted(names)))
        for name in names: counts[name]=counts.get(name,0)+1
    over={k:v for k,v in counts.items() if v>max_count}
    if over:
        issues.append("Exposure violations: "+", ".join(f"{k} {v}/{n_lineups}" for k,v in sorted(over.items())))
    if len(set(keys)) != len(keys): issues.append("Duplicate lineups detected.")
    max_overlap=0; worst=None
    sets=[set(k) for k in keys]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            ov=len(sets[i]&sets[j])
            if ov>max_overlap: max_overlap=ov; worst=(i+1,j+1)
    if max_overlap > 9-int(min_unique):
        issues.append(f"Uniqueness violation: lineups {worst[0]} and {worst[1]} overlap by {max_overlap} players.")
    return {"pass":not issues,"issues":issues,"max_overlap":max_overlap,"max_count":max_count,"counts":counts}

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

st.title("🏈 NFL Predictor Pro — V3.1.6")
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

    portfolio_mode=st.radio("Portfolio construction",["V3.1.6a Portfolio Optimize","V3.1.5 Sequential (control)"],horizontal=True)
    bank_size=st.slider("Candidate bank size",300,2000,1200,100,disabled=(portfolio_mode!="V3.1.6a Portfolio Optimize"),help="V3.1.6 generates this many strong legal candidates, then chooses the full portfolio simultaneously.")
    solver_seconds=st.slider("Portfolio solver time limit (seconds)",5,60,20,5,disabled=(portfolio_mode!="V3.1.6a Portfolio Optimize"))

    st.info("Default portfolio rules: 15 lineups • 30% max exposure • 4 minimum unique players • $47,500 salary floor • 88% projection-quality floor • QB + pass-catcher stack • no offensive player against selected DST. Ownership fade defaults to a chalk-friendly 0.15; leverage remains 0.35. V3.1.6 adds optional candidate-bank portfolio optimization. GPP ranking still uses Mean + P90 + P95 + P99 upside.")

    if st.button("Generate portfolio",type="primary"):
        with st.spinner("Generating diversified portfolio..."):
            solver_meta=None
            if portfolio_mode == "V3.1.6a Portfolio Optimize":
                candidate_items, reference_proj, min_proj_required = generate_candidate_bank(
                    pool,int(bank_size),int(min_salary),50000,strategy,float(own_weight),
                    float(leverage_weight),locks,excludes,float(projection_floor_pct),
                    attempts=max(60000,int(bank_size)*120)
                )
                lineups, solver_meta = select_portfolio_milp(
                    candidate_items,int(n_lineups),float(max_exp),int(min_unique),float(solver_seconds)
                )
                expo={}; max_count=max(1,math.floor(int(n_lineups)*float(max_exp)+1e-9))
                for l in lineups:
                    for nm in set(l.Name): expo[nm]=expo.get(nm,0)+1
            else:
                lineups, expo, max_count, reference_proj, min_proj_required=build_portfolio(
                    pool,int(n_lineups),float(max_exp),int(min_unique),int(min_salary),50000,
                    strategy,float(own_weight),float(leverage_weight),locks,excludes,float(projection_floor_pct)
                )
        if not lineups:
            st.error("No valid portfolio found. Relax locks/exclusions, uniqueness, exposure, or salary floor.")
        else:
            st.session_state["nfl_lineups"]=lineups
            st.session_state["nfl_rank_settings"]={"strategy":strategy,"own_weight":float(own_weight),"leverage_weight":float(leverage_weight)}
            qc=validate_portfolio(lineups,int(n_lineups),float(max_exp),int(min_unique),int(min_salary),50000)
            if qc["pass"]:
                st.success(f"Generated {len(lineups)} of {int(n_lineups)} requested lineups. FINAL QC: PASS")
            else:
                st.error("FINAL QC: FAIL — export is not considered tournament-ready.")
                for issue in qc["issues"]: st.write("• "+issue)
            if solver_meta is not None:
                st.caption(f"V3.1.6a candidate bank: {solver_meta.get('candidate_count',0):,} • portfolio constraints: {solver_meta.get('constraint_count',0):,} • solver: {solver_meta.get('message','')}")
            st.caption(f"Projection quality guardrail: strong reference {reference_proj:.1f} DK points • minimum accepted {min_proj_required:.1f} ({projection_floor_pct:.0%}).")
            ldf=lineups_to_df(lineups,strategy,float(own_weight),float(leverage_weight),stack_rank=True)
            edf=exposure_df(lineups)
            st.markdown("#### Lineups — stack ranked best to worst")
            st.caption("V3.1.3 GPP Rank blends Mean + P90 + P95 + P99 player-level upside, then applies the selected ownership fade and leverage weights. P90/P95/P99 Upside are comparison indexes (sums of player percentiles), not literal lineup percentiles.")
            st.dataframe(ldf,use_container_width=True,hide_index=True)
            if qc["pass"]:
                st.download_button("Download lineups",ldf.to_csv(index=False),"nfl_lineups_v316.csv","text/csv")
            else:
                st.warning("Download disabled until final portfolio QC passes.")
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
