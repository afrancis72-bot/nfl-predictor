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



def parse_classic_dk_csv(uploaded):
    """Parse a DraftKings NFL Classic salary CSV and normalize it for slate ingestion."""
    dk=pd.read_csv(uploaded)
    required={"Position","Name","ID","Salary","Game Info","TeamAbbrev","AvgPointsPerGame"}
    missing=required-set(dk.columns)
    if missing:
        raise ValueError("Missing DraftKings columns: "+", ".join(sorted(missing)))
    x=dk.copy()
    x["Position"]=x["Position"].astype(str).str.upper().replace({"D":"DST","DEF":"DST"})
    x["TeamAbbrev"]=x["TeamAbbrev"].map(normalize_team)
    x["Salary"]=pd.to_numeric(x["Salary"],errors="coerce")
    x["AvgPointsPerGame"]=pd.to_numeric(x["AvgPointsPerGame"],errors="coerce").fillna(0.0)
    # V3.1.6b: DraftKings availability status is part of Classic eligibility.
    # DK commonly uses Status, but tolerate alternate export labels.
    status_col=next((c for c in ["Status","Injury Status","InjuryStatus","Roster Status"] if c in x.columns),None)
    if status_col is None:
        x["dk_status"]=""
    else:
        x["dk_status"]=x[status_col].fillna("").astype(str).str.strip().str.upper()
    # Normalize common variants/abbreviations without treating Q/D as automatic outs.
    x["dk_status"]=x["dk_status"].replace({
        "O":"OUT","I.R.":"IR","INJURED RESERVE":"IR","RESERVE/INJURED":"IR",
        "INACT":"INACTIVE","INA":"INACTIVE","QUESTIONABLE":"Q","DOUBTFUL":"D"
    })
    x["game"]=x["Game Info"].astype(str).str.split().str[0].str.upper()
    x=x[x["Position"].isin(["QB","RB","WR","TE","DST"])].dropna(subset=["Name","Salary","TeamAbbrev"])
    x=x[x["Salary"]>0].drop_duplicates(["Name","Position","TeamAbbrev"],keep="first")
    x["opponent"]=[opponent_from_game(g,t) for g,t in zip(x["game"],x["TeamAbbrev"])]
    return x

def classic_pool_from_upload(dk, base_pool):
    """Use uploaded DK slate as the source of truth for roster, salary, IDs and games.
    Weekly researched projections are matched when available; otherwise a clearly
    labeled DK-PPG fallback is used so stale players can never leak into a new slate.
    """
    key=["Name","Position","TeamAbbrev"]
    b=base_pool.copy()
    # Prevent stale salary/game fields from surviving the merge.
    model_cols=[c for c in b.columns if c not in ["Salary","game","opponent","AvgPointsPerGame"]]
    merged=dk.merge(b[model_cols],on=key,how="left",suffixes=("","_model"))
    matched=merged["proj"].notna()
    merged["projection_source"]=np.where(matched,"Weekly researched model","DK PPG fallback")
    # V3.1.7 projection confidence: fallback data remains usable, but it should not
    # compete one-for-one with refreshed weekly research during candidate generation.
    merged["projection_confidence"]=np.where(matched,1.00,0.82)
    ppg=merged["AvgPointsPerGame"].fillna(0.0).astype(float)
    # Conservative fallback: DK season scoring baseline, with restrained tournament tails.
    fallback_mean=np.maximum(0.0,0.95*ppg)
    pos_mult=merged["Position"].map({"QB":1.00,"RB":1.00,"WR":1.00,"TE":1.00,"DST":0.90}).fillna(1.0)
    fallback_mean=fallback_mean*pos_mult
    merged["proj"]=merged["proj"].fillna(pd.Series(fallback_mean,index=merged.index))
    merged["ceiling"]=merged["ceiling"].fillna(merged["proj"]*1.55)
    merged["p95_use"]=merged["p95_use"].fillna(merged["proj"]*1.85)
    merged["p99_use"]=merged["p99_use"].fillna(merged["proj"]*2.25)
    merged["ownership_pct"]=pd.to_numeric(merged.get("ownership_pct",8.0),errors="coerce").fillna(8.0)
    merged["leverage_score"]=pd.to_numeric(merged.get("leverage_score",0.0),errors="coerce").fillna(0.0)
    # Fallback eligibility requires a non-trivial DK scoring history; QB/DST remain usable.
    fallback_ok=(merged["Position"].isin(["QB","DST"])) | (ppg>=2.0)
    existing_ok=merged.get("optimizer_eligible",pd.Series(False,index=merged.index)).fillna(False).astype(bool)
    merged["optimizer_eligible"]=np.where(matched,existing_ok,fallback_ok)
    # V3.1.6b: hard availability gate from the uploaded DK slate.
    # OUT/IR/Inactive players stay visible for audit but can never reach optimization.
    if "dk_status" not in merged.columns:
        merged["dk_status"]=""
    merged["dk_status"]=merged["dk_status"].fillna("").astype(str).str.strip().str.upper()
    blocked_statuses={"OUT","IR","INACTIVE","SUSPENDED","PUP","NFI"}
    merged["injury_blocked"]=merged["dk_status"].isin(blocked_statuses)
    merged["injury_flagged"]=merged["dk_status"].isin({"Q","D"})
    merged.loc[merged["injury_blocked"],"optimizer_eligible"]=False
    merged["role_status"]=merged.get("role_status",pd.Series("",index=merged.index)).fillna("")
    merged.loc[~matched,"role_status"]="DK Fallback"
    for c,default in {"coverage_matchup_grade":"Neutral / Uploaded Slate","individual_matchup_factor":1.0,"individual_matchup_delta":0.0,"expected_primary_coverage":"","projection_repaired":False}.items():
        if c not in merged: merged[c]=default
        merged[c]=merged[c].fillna(default)
    return merged



def current_slate_environment(pool, environment_rows):
    """Return only environment rows belonging to the active slate and verify full coverage.
    A new DK slate must never inherit unrelated prior-week game rows.
    """
    active_games=sorted({str(g).strip().upper() for g in pool.get("game", pd.Series(dtype=str)).dropna() if str(g).strip()})
    if environment_rows is None or len(environment_rows)==0 or "game" not in environment_rows.columns:
        return pd.DataFrame({"game":active_games}), False, active_games
    env=environment_rows.copy()
    env["game"]=env["game"].astype(str).str.strip().str.upper()
    env=env[env["game"].isin(active_games)].drop_duplicates("game",keep="last")
    total_col=next((c for c in ["game_total","total","vegas_total","over_under"] if c in env.columns),None)
    if total_col is None:
        return env, False, active_games
    env[total_col]=pd.to_numeric(env[total_col],errors="coerce")
    covered=set(env.loc[env[total_col].notna(),"game"])
    missing=[g for g in active_games if g not in covered]
    return env, len(missing)==0 and len(active_games)>0, missing

def environment_editor_seed(pool, researched_games):
    """Build an editor containing exactly the active DK games; prior-week rows are never displayed."""
    active=sorted(pool["game"].dropna().astype(str).str.strip().str.upper().unique().tolist())
    seed=pd.DataFrame({"game":active,"game_total":np.nan,"spread":np.nan})
    if researched_games is None or len(researched_games)==0 or "game" not in researched_games.columns:
        return seed
    rg=researched_games.copy(); rg["game"]=rg["game"].astype(str).str.strip().str.upper()
    total_col=next((c for c in ["game_total","total","vegas_total","over_under"] if c in rg.columns),None)
    spread_col=next((c for c in ["spread","home_spread","line"] if c in rg.columns),None)
    keep=["game"]+([total_col] if total_col else [])+([spread_col] if spread_col else [])
    rg=rg[keep].drop_duplicates("game",keep="last")
    rename={}
    if total_col: rename[total_col]="game_total"
    if spread_col: rename[spread_col]="spread"
    rg=rg.rename(columns=rename)
    seed=seed.drop(columns=["game_total","spread"]).merge(rg,on="game",how="left")
    if "game_total" not in seed: seed["game_total"]=np.nan
    if "spread" not in seed: seed["spread"]=np.nan
    return seed[["game","game_total","spread"]]

def calibrate_classic_tails(pool):
    """V3.1.6: bound pathological player tails while preserving ordering and upside."""
    x=pool.copy()
    mean=pd.to_numeric(x["proj"],errors="coerce").clip(lower=0.01)
    p90=pd.to_numeric(x["ceiling"],errors="coerce").fillna(mean)
    p95=pd.to_numeric(x["p95_use"],errors="coerce").fillna(p90)
    p99=pd.to_numeric(x["p99_use"],errors="coerce").fillna(p95)
    # Caps are intentionally broad; they stop corrupted/extreme tails, not legitimate ceilings.
    p90=np.minimum(np.maximum(p90,mean),mean*2.00)
    p95=np.minimum(np.maximum(p95,p90),mean*2.45)
    p99=np.minimum(np.maximum(p99,p95),mean*2.90)
    x["ceiling"],x["p95_use"],x["p99_use"]=p90,p95,p99
    x["tail_ratio_p99"]=x["p99_use"]/mean
    return x

def attach_game_environment(pool, games):
    """Attach a bounded 0-10 game environment score from weekly game research when available."""
    x=pool.copy(); x["game_env_score"]=5.0
    if games is None or len(games)==0 or "game" not in games.columns: return x
    g=games.copy(); g["game"]=g["game"].astype(str).str.upper()
    total_col=next((c for c in ["game_total","total","vegas_total","over_under"] if c in g.columns),None)
    spread_col=next((c for c in ["spread","home_spread","line"] if c in g.columns),None)
    if total_col is None: return x
    total=pd.to_numeric(g[total_col],errors="coerce")
    if total.notna().sum()==0: return x
    # 38 -> ~2, 44 -> ~5, 50 -> ~8, 54+ -> 10. Competitive games get a small boost.
    env=5.0+(total-44.0)*0.50
    if spread_col is not None:
        spread=pd.to_numeric(g[spread_col],errors="coerce").abs()
        env += np.clip((7.0-spread)/7.0,0,1)*0.75
    g["game_env_score"]=np.clip(env,0,10)
    return x.merge(g[["game","game_env_score"]].drop_duplicates("game"),on="game",how="left",suffixes=("","_weekly")).assign(
        game_env_score=lambda z:z.get("game_env_score_weekly",z["game_env_score"]).fillna(z["game_env_score"])
    ).drop(columns=[c for c in ["game_env_score_weekly"] if c in x.merge(g[["game","game_env_score"]].drop_duplicates("game"),on="game",how="left",suffixes=("","_weekly")).columns])

def lineup_context_scores(df):
    """Bounded game-environment and correlation diagnostics for a Classic lineup."""
    env=float(pd.to_numeric(df.get("game_env_score",5.0),errors="coerce").fillna(5.0).mean())
    qb=df[df.Position=="QB"]
    corr=0.0
    if len(qb):
        q=qb.iloc[0]
        mates=df[(df.TeamAbbrev==q.TeamAbbrev)&df.Position.isin(["WR","TE"])]
        corr += min(2,len(mates))*1.25
        opp=opponent_from_game(q.get("game",""),q.TeamAbbrev)
        bring=df[(df.TeamAbbrev==opp)&df.Position.isin(["RB","WR","TE"])] if opp else df.iloc[0:0]
        if len(bring): corr += 1.0
    return env,float(min(corr,3.5))

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
    return True

def lineup_score(df, strategy, own_weight, leverage_weight):
    # V3.1.6: P99 is a small tail signal, not a primary ranking engine.
    # Game environment and stack correlation are explicit, bounded bonuses.
    if strategy == "Median":
        base = df["proj"].sum()
    elif strategy == "Balanced":
        base = (0.60*df["proj"] + 0.22*df["ceiling"] + 0.15*df["p95_use"] + 0.03*df["p99_use"]).sum()
    else:
        base = (0.35*df["proj"] + 0.30*df["ceiling"] + 0.30*df["p95_use"] + 0.05*df["p99_use"]).sum()
    env,corr=lineup_context_scores(df)
    context_bonus=(env-5.0)*0.80 + corr
    # Confidence is deliberately a modest lineup-level adjustment. Weekly research
    # gets full credit; DK-PPG fallback rows carry uncertainty without being banned.
    conf=pd.to_numeric(df.get("projection_confidence",pd.Series(1.0,index=df.index)),errors="coerce").fillna(1.0)
    confidence_penalty=float((1.0-conf).sum())*1.50
    return float(base + context_bonus - confidence_penalty - own_weight*df["ownership_pct"].sum()/10 + leverage_weight*df["leverage_score"].sum())

def random_candidate(pool, min_salary, max_salary, strategy, own_weight, leverage_weight, locks, excludes):
    p = pool[(pool["optimizer_eligible"] == True) & ~pool.Name.isin(excludes)].copy()
    # V3.1.7 upstream tournament signals. These affect candidate opportunity, not
    # final exposure requirements, so stars are represented without being forced.
    p["projection_confidence"]=pd.to_numeric(p.get("projection_confidence",1.0),errors="coerce").fillna(1.0).clip(0.70,1.0)
    p["game_env_score"]=pd.to_numeric(p.get("game_env_score",5.0),errors="coerce").fillna(5.0).clip(0,10)
    eligible_skill=p[p.Position.isin(["RB","WR","TE"])]
    elite_cut=float(eligible_skill["p95_use"].quantile(0.85)) if len(eligible_skill) else float("inf")
    slate_cut=float(eligible_skill["p95_use"].quantile(0.95)) if len(eligible_skill) else float("inf")
    p["elite_ceiling_mult"]=1.0
    p.loc[p["p95_use"]>=elite_cut,"elite_ceiling_mult"]=1.18
    p.loc[p["p95_use"]>=slate_cut,"elite_ceiling_mult"]=1.35
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
        # V3.1.7: move bounded game environment, projection confidence and elite
        # ceiling representation upstream into candidate sampling.
        env_mult=(0.85+0.03*df["game_env_score"]).clip(0.85,1.15)
        w=w*env_mult*df["projection_confidence"]*df["elite_ceiling_mult"]
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
    # V3.1.7 candidate-opportunity signals. They alter sampling probability only;
    # the MILP still decides the final portfolio under the existing exposure/QC rules.
    p["projection_confidence"]=pd.to_numeric(p.get("projection_confidence",1.0),errors="coerce").fillna(1.0).clip(0.70,1.0)
    p["game_env_score"]=pd.to_numeric(p.get("game_env_score",5.0),errors="coerce").fillna(5.0).clip(0,10)
    skill_mask=p["Position"].isin(["RB","WR","TE"])
    elite_cut=float(p.loc[skill_mask,"p95_use"].quantile(0.85)) if skill_mask.any() else float("inf")
    slate_cut=float(p.loc[skill_mask,"p95_use"].quantile(0.95)) if skill_mask.any() else float("inf")
    p["elite_ceiling_mult"]=1.0
    p.loc[skill_mask & (p["p95_use"]>=elite_cut),"elite_ceiling_mult"]=1.18
    p.loc[skill_mask & (p["p95_use"]>=slate_cut),"elite_ceiling_mult"]=1.35

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
    env=p["game_env_score"].to_numpy(dtype=float)
    conf=p["projection_confidence"].to_numpy(dtype=float)
    elite=p["elite_ceiling_mult"].to_numpy(dtype=float)

    if strategy == "Median": raw=np.clip(proj,0.1,None)
    elif strategy == "Balanced": raw=np.clip(0.55*proj+0.45*ceil,0.1,None)
    else: raw=np.clip(0.30*proj+0.70*ceil+0.15*lev,0.1,None)
    # 5.0 environment is neutral; 10.0 earns only a 15% opportunity boost.
    env_mult=np.clip(0.85+0.03*env,0.85,1.15)
    weights=np.square(raw*env_mult*conf*elite)

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
        # V2: correlation is rewarded by simulations, not forced as a universal rule.
        # Most candidates still seed a pass-catcher, while a minority explore legal no-stack scripts.
        if rng.random() < 0.78:
            m=pick(mates.get(q,np.array([],dtype=int)),sel)
            if m is not None: sel.add(m)
        # V3.1.7 game-stack seeding: in strong environments, some candidates get
        # an opponent bring-back before generic slots are filled. This creates more
        # correlated shootout candidates without requiring them in every lineup.
        if env[q] >= 7.0 and rng.random() < 0.40:
            opp=opponent_from_game(games[q],teams[q])
            bring=np.where((teams==opp) & np.isin(pos,["RB","WR","TE"]))[0] if opp else np.array([],dtype=int)
            b=pick(bring,sel)
            if b is not None: sel.add(b)
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
        env,corr=lineup_context_scores(x)
        row["Game Env"] = round(env,2)
        row["Correlation"] = round(corr,2)
        row["P99/Mean"] = round(float(x.p99_use.sum()/max(x.proj.sum(),0.01)),2)
        if strategy is not None:
            row["GPP Score"] = round(lineup_score(x,strategy,own_weight,leverage_weight),2)
        rows.append(row)
    out=pd.DataFrame(rows)
    if stack_rank and "GPP Score" in out.columns:
        out=out.sort_values(["GPP Score","P95 Upside","Projection"],ascending=[False,False,False]).reset_index(drop=True)
        out.insert(0,"Rank",range(1,len(out)+1))
    return out

def exposure_df(lineups):
    allp=pd.concat(lineups,ignore_index=True)
    counts=allp.groupby(["Name","Position","TeamAbbrev"],as_index=False).size()
    counts["Exposure %"]=100*counts["size"]/len(lineups)
    return counts.rename(columns={"size":"Lineups"}).sort_values(["Exposure %","Name"],ascending=[False,True])




# =========================
# DFS ENGINE V2 — shared simulation / portfolio layer
# =========================
def _binned_mode(values, width=2.0):
    v=np.asarray(values,dtype=float); v=v[np.isfinite(v)]
    if not len(v): return np.nan
    lo=np.floor(v.min()/width)*width; hi=np.ceil(v.max()/width)*width+width
    edges=np.arange(lo,hi+1e-9,width)
    if len(edges)<2: return float(v[0])
    hist,edges=np.histogram(v,bins=edges); k=int(np.argmax(hist))
    return float((edges[k]+edges[k+1])/2)

def simulation_summary(names, sims, actual=None, bin_width=2.0):
    rows=[]; actual=actual or {}
    for j,name in enumerate(names):
        v=np.asarray(sims[:,j],dtype=float); a=actual.get(str(name),np.nan)
        rows.append({'Name':name,'Mean':v.mean(),'Median':np.median(v),'Mode (binned)':_binned_mode(v,bin_width),
                     'P10':np.percentile(v,10),'P25':np.percentile(v,25),'P75':np.percentile(v,75),
                     'P90':np.percentile(v,90),'P95':np.percentile(v,95),'Actual':a,
                     'Actual Percentile':(100*np.mean(v<=a) if pd.notna(a) else np.nan),
                     'Error vs Mean':(a-v.mean() if pd.notna(a) else np.nan)})
    return pd.DataFrame(rows)

def simulate_classic_v2(pool, n_sims=10000, seed=20261006):
    """Correlated full-slate simulation with game scripts, heavy tails and model uncertainty.
    One row is one coherent Sunday scenario; player outcomes are not independent.
    """
    x=pool[pool['optimizer_eligible']==True].copy().reset_index(drop=True)
    rng=np.random.default_rng(seed); ns=int(n_sims); n=len(x)
    games_u=x['game'].astype(str).unique().tolist(); teams_u=x['TeamAbbrev'].astype(str).unique().tolist()
    pace={g:rng.standard_t(6,size=ns)/np.sqrt(1.5) for g in games_u}
    # Explicit script regimes widen tails: slow/defensive, neutral, shootout/blowout-like environments.
    regime={g:rng.choice([-1,0,1,2],size=ns,p=[.18,.55,.20,.07]) for g in games_u}
    off={t:rng.normal(size=ns) for t in teams_u}; pas={t:rng.normal(size=ns) for t in teams_u}; rush={t:rng.normal(size=ns) for t in teams_u}
    out=np.zeros((ns,n),dtype=np.float32)
    for j,r in x.iterrows():
        pos=str(r['Position']); t=str(r['TeamAbbrev']); g=str(r['game']); opp=opponent_from_game(g,t)
        mean=max(.05,float(r['proj'])); p90=max(mean,float(r.get('ceiling',mean*1.5)))
        empirical=max(.75,(p90-mean)/1.2816)
        pos_floor={'QB':4.5,'RB':5.0,'WR':5.5,'TE':4.5,'DST':4.0}.get(pos,4.5)
        sd=max(empirical,pos_floor,mean*{'QB':.30,'RB':.48,'WR':.58,'TE':.58,'DST':.65}.get(pos,.5))
        # Model uncertainty shifts the center itself, separately from game-to-game variance.
        unc=rng.normal(0,{'QB':.07,'RB':.10,'WR':.12,'TE':.13,'DST':.16}.get(pos,.12),size=ns)*mean
        idio=rng.standard_t(5,size=ns)/np.sqrt(5/3)
        rg=regime[g].astype(float)
        if pos=='QB': z=.18*pace[g]+.36*off[t]+.35*pas[t]+.18*rg+.70*idio
        elif pos in ['WR','TE']: z=.15*pace[g]+.28*off[t]+.34*pas[t]+.16*rg+.73*idio
        elif pos=='RB': z=.10*pace[g]+.32*off[t]+.34*rush[t]-.12*pas[t]+.12*rg+.73*idio
        elif pos=='DST': z=-.38*off.get(opp,np.zeros(ns))-.15*pace[g]-.13*rg+.80*idio
        else: z=.15*pace[g]+.28*off[t]+.78*idio
        z=(z-z.mean())/(z.std()+1e-9)
        out[:,j]=np.clip(mean+unc+sd*z,0,None).astype(np.float32)
    return x,out

def v2_rescore_classic_candidates(candidate_items, sim_players, sims):
    if not candidate_items: return []
    idx={str(n):i for i,n in enumerate(sim_players['Name'])}; enriched=[]; best=np.full(sims.shape[0],-1e9,dtype=np.float32)
    for lineup,_old in candidate_items:
        cols=[idx[n] for n in lineup['Name'].astype(str) if n in idx]
        if len(cols)!=9: continue
        pts=sims[:,cols].sum(axis=1).astype(np.float32)
        rec={'lineup':lineup,'sim':pts,'mean':float(pts.mean()),'median':float(np.median(pts)),
             'p75':float(np.percentile(pts,75)),'p90':float(np.percentile(pts,90)),'p95':float(np.percentile(pts,95))}
        enriched.append(rec); best=np.maximum(best,pts)
    for r in enriched:
        r['optimal_rate']=float(np.mean(r['sim']>=best-1e-5))
        own=float(pd.to_numeric(r['lineup']['ownership_pct'],errors='coerce').fillna(8).sum())
        lev=float(pd.to_numeric(r['lineup']['leverage_score'],errors='coerce').fillna(0).sum())
        r['score']=.30*r['mean']+.15*r['median']+.20*r['p90']+.25*r['p95']+12*r['optimal_rate']-.025*own+.12*lev
    return enriched

def select_v2_portfolio(records, n_lineups, max_exposure=.65, min_unique=3):
    """Greedy marginal-scenario portfolio: rewards strong lineups AND new right-tail coverage."""
    if not records: return [],{}
    records=sorted(records,key=lambda r:r['score'],reverse=True); selected=[]; counts={}; ns=len(records[0]['sim']); covered=np.full(ns,-1e9,dtype=np.float32)
    max_count=max(1,math.ceil(int(n_lineups)*float(max_exposure)-1e-12)); max_overlap=9-int(min_unique)
    pool_scores=np.array([r['score'] for r in records]); mu=pool_scores.mean(); sig=pool_scores.std()+1e-9
    while len(selected)<int(n_lineups):
        best_i=None; best_val=-1e18
        for i,r in enumerate(records):
            if r.get('_used'): continue
            names=set(r['lineup']['Name'].astype(str))
            if any(counts.get(n,0)>=max_count for n in names): continue
            if any(len(names & set(q['lineup']['Name'].astype(str)))>max_overlap for q in selected): continue
            marginal=float(np.mean(np.maximum(r['sim']-covered,0))) if selected else float(np.mean(r['sim']))
            overlap=(np.mean([len(names & set(q['lineup']['Name'].astype(str)))/9 for q in selected]) if selected else 0)
            val=(r['score']-mu)/sig + .18*marginal - 1.75*overlap
            if val>best_val: best_val=val; best_i=i
        if best_i is None: break
        r=records[best_i]; r['_used']=True; selected.append(r); covered=np.maximum(covered,r['sim'])
        for n in set(r['lineup']['Name'].astype(str)): counts[n]=counts.get(n,0)+1
    return [r['lineup'] for r in selected],{'selected_records':selected,'max_count':max_count,'scenario_coverage_mean':float(np.mean(covered)) if selected else np.nan}

def select_showdown_portfolio_v2(cands, players, n_lineups, max_player_exp, max_cpt_exp, min_unique, locks, excludes, cpt_excludes):
    """Portfolio selection by marginal scenario coverage, not a stack-ranked list of near-duplicates."""
    lockset=set(locks); exset=set(excludes); cex=set(cpt_excludes); n_lineups=int(n_lineups)
    maxp,maxc=_showdown_exposure_counts(n_lineups,max_player_exp,max_cpt_exp); selected=[]; counts={}; ccounts={}; covered=None
    vals=np.array([c.get('rank_score',0.) for c in cands],dtype=float); mu=vals.mean() if len(vals) else 0; sig=vals.std()+1e-9
    while len(selected)<n_lineups:
        bi=None; bv=-1e18
        for i,c in enumerate(cands):
            if c.get('_v2used'): continue
            ids=[c['cpt']]+list(c['flex']); names=set(players.iloc[ids]['Name'].astype(str)); cp=str(players.iloc[c['cpt']]['Name'])
            if lockset and not lockset.issubset(names): continue
            if names&exset or cp in cex or ccounts.get(cp,0)>=maxc or any(counts.get(n,0)>=maxp for n in names): continue
            if any(len(names & set(q['names']))>6-int(min_unique) for q in selected): continue
            marginal=float(np.mean(c['sim'])) if covered is None else float(np.mean(np.maximum(c['sim']-covered,0)))
            overlap=np.mean([len(names & set(q['names']))/6 for q in selected]) if selected else 0
            val=(c.get('rank_score',0)-mu)/sig + .22*marginal - 2.0*overlap
            if val>bv: bv=val; bi=i
        if bi is None: break
        c=cands[bi]; c['_v2used']=True; ids=[c['cpt']]+list(c['flex']); names=set(players.iloc[ids]['Name'].astype(str)); cp=str(players.iloc[c['cpt']]['Name'])
        z=dict(c); z['names']=list(names); selected.append(z); covered=c['sim'].copy() if covered is None else np.maximum(covered,c['sim'])
        for n in names: counts[n]=counts.get(n,0)+1
        ccounts[cp]=ccounts.get(cp,0)+1
    return selected,{'requested':n_lineups,'max_player_count':maxp,'max_cpt_count':maxc,'relaxed':False,'scenario_coverage_mean':float(np.mean(covered)) if covered is not None else np.nan}

# -------------------------
# Single-Game Showdown V1.8a
# -------------------------
CLASSIC_DEFAULTS = {
    "classic_n_lineups": 15, "classic_max_exp": 0.65, "classic_min_unique": 3,
    "classic_min_salary": 47500, "classic_strategy": "GPP Ceiling",
    "classic_own_weight": 0.15, "classic_leverage_weight": 0.35,
    "classic_projection_floor": 0.88, "classic_portfolio_mode": "DFS Engine V2 Scenario Portfolio",
    "classic_bank_size": 1200, "classic_solver_seconds": 20,
    "classic_locks": [], "classic_excludes": [],
}
SHOWDOWN_DEFAULTS = {
    "sd_n_lineups": 20, "sd_max_player_exp": 0.75, "sd_max_cpt_exp": 0.50,
    "sd_min_unique": 2, "sd_min_salary": 38000, "sd_sims": 20000,
    "sd_candidate_bank": 6000, "sd_strategy": "Tournament Ceiling",
    "sd_min_standard": "Off", "sd_allow_deep_punt": True,
    "sd_locks": [], "sd_excludes": [], "sd_cpt_excludes": [],
}

def _init_defaults(defaults):
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

def _reset_defaults(defaults):
    for k,v in defaults.items(): st.session_state[k]=v

def _clean_name(x):
    return " ".join(str(x).lower().replace(".","").replace("'","").replace("-"," ").split())

def parse_showdown_csv(uploaded):
    df=pd.read_csv(uploaded)
    req={"Position","Name","ID","Roster Position","Salary","Game Info","TeamAbbrev","AvgPointsPerGame"}
    missing=req-set(df.columns)
    if missing: raise ValueError("Missing DraftKings columns: "+", ".join(sorted(missing)))
    df=df.copy(); df["Roster Position"]=df["Roster Position"].astype(str).str.upper()
    df["TeamAbbrev"]=df["TeamAbbrev"].map(normalize_team)
    df["Position"]=df["Position"].astype(str).str.upper()
    df["Salary"]=pd.to_numeric(df["Salary"],errors="coerce")
    df["AvgPointsPerGame"]=pd.to_numeric(df["AvgPointsPerGame"],errors="coerce")
    df["Status"]=df.get("Status","").fillna("").astype(str).str.upper()
    game=str(df["Game Info"].dropna().iloc[0]) if df["Game Info"].notna().any() else "Uploaded Showdown"
    teams=sorted(df["TeamAbbrev"].dropna().unique().tolist())
    if len(teams)!=2: raise ValueError(f"Expected exactly two teams; found {teams}")
    flex=df[df["Roster Position"].str.contains("FLEX",na=False)].copy()
    cpt=df[df["Roster Position"].eq("CPT")][["Name","TeamAbbrev","Salary","ID"]].rename(columns={"Salary":"CPT Salary","ID":"CPT ID"})
    flex=flex.merge(cpt,on=["Name","TeamAbbrev"],how="left")
    flex=flex.rename(columns={"Salary":"FLEX Salary","ID":"FLEX ID"})
    flex=flex[~flex["Status"].isin(["OUT","IR"])].copy()
    # Remove zero-production backup QBs when that team has an active QB with a scoring baseline.
    for tm in flex["TeamAbbrev"].dropna().unique():
        active_qb=(flex["TeamAbbrev"].eq(tm) & flex["Position"].eq("QB") & (flex["AvgPointsPerGame"].fillna(0)>0))
        if active_qb.any():
            flex=flex[~(flex["TeamAbbrev"].eq(tm) & flex["Position"].eq("QB") & (flex["AvgPointsPerGame"].fillna(0)<=0))].copy()
    flex=flex.dropna(subset=["CPT Salary","CPT ID","FLEX Salary","FLEX ID"])
    flex["key"]=flex["Name"].map(_clean_name)+"|"+flex["TeamAbbrev"]
    return flex.reset_index(drop=True),game,teams


def apply_showdown_current_role_layer(x):
    """V1.8 current-opportunity layer.

    Role status is independent of salary and is applied before simulation.
    The mapping is intentionally isolated so it can be refreshed for each
    single-game slate without changing model weights or optimizer rules.
    """
    y=x.copy()
    # Verified current depth-chart roles for ATL @ NO, 2026-10-05.
    # Multipliers represent expected NORMAL-GAME opportunity, not talent.
    role_map={
        "Michael Penix Jr.":("Starter",1.00,True),
        "Cooper Rush":("Reserve QB",0.03,False),
        "Bijan Robinson":("Starter",1.00,True),
        "Brian Robinson Jr.":("RB2 / rotation",0.72,True),
        "Drake London":("Starter",1.00,True),
        "Kyle Pitts":("TE1",1.00,True),
        "Austin Hooper":("Reserve TE",0.42,False),
        "Tyler Shough":("Starter",1.00,True),
        "Chris Olave":("Starter",1.00,True),
        "Devaughn Vele":("Starter",1.00,True),
        "Juwan Johnson":("TE1",1.00,True),
        "Noah Fant":("TE2 / questionable",0.58,False),
        "Alvin Kamara":("RB1",1.00,True),
        "Kendre Miller":("RB2 / expanded role",0.76,True),
    }
    y["Current Role"]="Unverified / model role"
    y["Opportunity Multiplier"]=1.0
    y["Role CPT Eligible"]=True
    for i,r in y.iterrows():
        name=str(r["Name"])
        if name in role_map:
            label,mult,cpt_ok=role_map[name]
            y.at[i,"Current Role"]=label
            y.at[i,"Opportunity Multiplier"]=float(mult)
            y.at[i,"Role CPT Eligible"]=bool(cpt_ok)

    # Apply only to skill/QB projections. K/DST keep their defined team roles.
    affected=y["Position"].isin(["QB","RB","WR","TE"])
    y.loc[affected,"Base Mean"]=y.loc[affected,"Base Mean"]*y.loc[affected,"Opportunity Multiplier"]

    # Compress reserve distributions as well as means. This prevents stale P90/P95
    # tails from making a reserve look like a normal starter after mean adjustment.
    reserve=affected & (y["Opportunity Multiplier"]<0.90)
    y.loc[reserve,"Sim SD"]=y.loc[reserve,"Sim SD"]*np.sqrt(y.loc[reserve,"Opportunity Multiplier"].clip(lower=.05))

    # Keep a small contingent ceiling for reserves, but never a normal starter floor.
    y.loc[reserve,"Base Mean"]=y.loc[reserve,"Base Mean"].clip(lower=0.10)
    return y

def showdown_projection_table(flex, comp, mc, dst):
    x=flex.copy()
    # Prefer the weekly model when the uploaded game's teams are represented.
    cm=comp.copy(); cm["TeamAbbrev"]=cm["TeamAbbrev"].map(normalize_team); cm["key"]=cm["Name"].map(_clean_name)+"|"+cm["TeamAbbrev"]
    cm["model_mean"]=pd.to_numeric(cm.get("matchup_adjusted_projection"),errors="coerce")
    cm["model_mean"]=cm["model_mean"].fillna(pd.to_numeric(cm.get("projected_dk_points"),errors="coerce"))
    cm=cm.sort_values("model_mean",ascending=False).drop_duplicates("key")[["key","model_mean"]]
    x=x.merge(cm,on="key",how="left")
    mm=mc.copy(); mm["TeamAbbrev"]=mm["TeamAbbrev"].map(normalize_team); mm["key"]=mm["Name"].map(_clean_name)+"|"+mm["TeamAbbrev"]
    mm=mm.drop_duplicates("key")[["key","mean","p90","p95"]]
    x=x.merge(mm,on="key",how="left")
    # DST lives in a separate model table.
    if dst is not None and not dst.empty:
        dd=dst.copy(); dd["TeamAbbrev"]=dd["TeamAbbrev"].map(normalize_team); dd["key"]=dd["Name"].map(_clean_name)+"|"+dd["TeamAbbrev"]
        if "dst_mean" in dd: dd=dd.drop_duplicates("key")[["key","dst_mean","dst_p90","dst_p95"]]
        else: dd=pd.DataFrame(columns=["key","dst_mean","dst_p90","dst_p95"])
        x=x.merge(dd,on="key",how="left")
    else:
        x["dst_mean"]=np.nan; x["dst_p90"]=np.nan; x["dst_p95"]=np.nan
    x["Base Mean"]=pd.to_numeric(x["model_mean"],errors="coerce")
    x["Base Mean"]=x["Base Mean"].fillna(pd.to_numeric(x["mean"],errors="coerce"))
    dst_mask=x["Position"].isin(["DST","D"])
    x.loc[dst_mask,"Base Mean"]=x.loc[dst_mask,"Base Mean"].fillna(pd.to_numeric(x.loc[dst_mask,"dst_mean"],errors="coerce"))
    x["Projection Source"]=np.where(x["Base Mean"].notna(),"NFL weekly model","DK slate fallback")
    # V1.1 role-aware fallback. DK PPG is useful, but it can overstate tiny-sample
    # backups. Discount the median for low-salary skill players while preserving
    # enough variance for legitimate TD-dependent punt outcomes. Weekly-model rows
    # are never altered by this fallback calibration.
    fallback_mask=x["Base Mean"].isna()
    dk_ppg=pd.to_numeric(x["AvgPointsPerGame"],errors="coerce").fillna(0.0)
    sal=pd.to_numeric(x["FLEX Salary"],errors="coerce").fillna(0.0)
    skill_mask=x["Position"].isin(["RB","WR","TE"])
    role_mult=pd.Series(1.0,index=x.index)
    role_mult.loc[skill_mask & (sal < 1000)] = 0.28
    role_mult.loc[skill_mask & (sal >= 1000) & (sal < 2500)] = 0.38
    role_mult.loc[skill_mask & (sal >= 2500) & (sal < 4000)] = 0.58
    role_mult.loc[skill_mask & (sal >= 4000) & (sal < 6500)] = 0.82
    fallback_mean=dk_ppg*role_mult
    # Kickers/DST/QB use their slate scoring baseline directly; their roles are
    # less ambiguous than reserve RB/WR/TE roles in a salary-only fallback.
    fallback_mean.loc[~skill_mask]=dk_ppg.loc[~skill_mask]
    x.loc[fallback_mask,"Base Mean"]=fallback_mean.loc[fallback_mask]
    x["Fallback Role Multiplier"]=np.where(fallback_mask,role_mult,1.0)

    # V1.7 weekly-model role calibration.  Showdown is especially sensitive to a
    # stale or over-aggressive one-game projection because the optimizer can turn
    # that error into 60-70% exposure.  DK's current slate PPG is NOT a projection
    # and never replaces our weekly model; it is used only as a bounded role sanity
    # anchor.  Large model-vs-role disagreements are partially shrunk rather than
    # accepted at full strength.  Salary is deliberately absent from this step.
    x["Raw Weekly Mean"]=x["Base Mean"]
    x["DK Role Baseline"]=dk_ppg
    weekly_skill=(~fallback_mask) & skill_mask & (dk_ppg>0)
    role_anchor=0.80*x["Base Mean"] + 0.20*dk_ppg
    upper_guard=np.maximum(dk_ppg*1.45, dk_ppg+3.5)
    lower_guard=np.minimum(dk_ppg*0.60, np.maximum(dk_ppg-3.5,0.10))
    calibrated=np.minimum(role_anchor,upper_guard)
    calibrated=np.maximum(calibrated,lower_guard)
    x.loc[weekly_skill,"Base Mean"]=calibrated.loc[weekly_skill]
    x["Role Calibration"]="None"
    x.loc[weekly_skill,"Role Calibration"]="Weekly model + bounded DK role anchor"

    x["Role Tier"]="Weekly model"
    x.loc[fallback_mask & skill_mask & (sal>=6500),"Role Tier"]="Core skill"
    x.loc[fallback_mask & skill_mask & (sal>=4000) & (sal<6500),"Role Tier"]="Secondary skill"
    x.loc[fallback_mask & skill_mask & (sal>=2500) & (sal<4000),"Role Tier"]="Low-volume skill"
    x.loc[fallback_mask & skill_mask & (sal<2500),"Role Tier"]="Punt / TD-dependent"
    x.loc[fallback_mask & ~skill_mask,"Role Tier"]="Defined role"
    # V1.3 opportunity/participation assumptions for salary-only fallback skill players.
    # These are not exposure caps: they shape the simulated scoring distribution so
    # low-volume players have realistic dud frequency and conditional upside.
    x["Low Score Prob"]=0.0
    x.loc[fallback_mask & skill_mask & (sal>=6500),"Low Score Prob"]=0.05
    x.loc[fallback_mask & skill_mask & (sal>=4000) & (sal<6500),"Low Score Prob"]=0.14
    x.loc[fallback_mask & skill_mask & (sal>=2500) & (sal<4000),"Low Score Prob"]=0.32
    x.loc[fallback_mask & skill_mask & (sal<2500),"Low Score Prob"]=0.50
    x["Spike Prob"]=0.0
    x.loc[fallback_mask & skill_mask & (sal>=6500),"Spike Prob"]=0.03
    x.loc[fallback_mask & skill_mask & (sal>=4000) & (sal<6500),"Spike Prob"]=0.04
    x.loc[fallback_mask & skill_mask & (sal>=2500) & (sal<4000),"Spike Prob"]=0.06
    x.loc[fallback_mask & skill_mask & (sal<2500),"Spike Prob"]=0.08
    pos_floor={"QB":2.0,"RB":0.10,"WR":0.10,"TE":0.10,"K":2.0,"DST":1.0,"D":1.0}
    x["Base Mean"]=x.apply(lambda r: max(float(r["Base Mean"]) if pd.notna(r["Base Mean"]) else 0.0,pos_floor.get(r["Position"],0.10)),axis=1)
    p90=pd.to_numeric(x["p90"],errors="coerce")
    p90=np.where(dst_mask,pd.to_numeric(x["dst_p90"],errors="coerce"),p90)
    # Preserve a long right tail for cheap skill players, but don't let the cheap
    # salary itself create a high median projection.
    base_sd=x["Base Mean"]*x["Position"].map({"QB":0.42,"RB":0.65,"WR":0.75,"TE":0.75,"K":0.42,"DST":0.70,"D":0.70}).fillna(0.65)
    punt_boost=pd.Series(1.0,index=x.index)
    punt_boost.loc[fallback_mask & skill_mask & (sal<4000)] = 1.35
    fallback_sd=(base_sd*punt_boost).clip(lower=np.where(fallback_mask & skill_mask & (sal<4000),1.15,1.5))
    tail_sd=(pd.Series(p90,index=x.index)-x["Base Mean"])/1.2816
    x["Sim SD"]=tail_sd.where((~fallback_mask) & (tail_sd>1.0),fallback_sd).fillna(fallback_sd)
    x=apply_showdown_current_role_layer(x)
    return x.reset_index(drop=True)

def simulate_showdown_players(players, teams, n_sims=20000, seed=315):
    rng=np.random.default_rng(seed); n=len(players); ns=int(n_sims)
    pace=rng.standard_t(6,size=ns)/np.sqrt(1.5)
    team_off={t:rng.standard_t(7,size=ns)/np.sqrt(7/5) for t in teams}
    pass_script={t:rng.normal(size=ns) for t in teams}
    rush_script={t:rng.normal(size=ns) for t in teams}
    game_regime=rng.choice([-1,0,1,2],size=ns,p=[.18,.55,.20,.07])
    # Negative relationship between opponent offense and DST outcomes.
    out=np.zeros((ns,n),dtype=np.float32)
    for j,r in players.iterrows():
        t=r["TeamAbbrev"]; opp=teams[1] if t==teams[0] else teams[0]; pos=r["Position"]
        idio=rng.standard_t(5,size=ns)/np.sqrt(5/3)
        if pos=="QB": z=.18*pace+.38*team_off[t]+.34*pass_script[t]+.14*game_regime+.70*idio
        elif pos in ["WR","TE"]: z=.16*pace+.30*team_off[t]+.32*pass_script[t]+.12*game_regime+.72*idio
        elif pos=="RB": z=.12*pace+.34*team_off[t]+.30*rush_script[t]-.10*pass_script[t]+.10*game_regime+.73*idio
        elif pos in ["DST","D"]: z=-.42*team_off[opp]-.16*pace-.12*game_regime+.18*rush_script[t]+.72*idio
        elif pos=="K": z=.18*pace+.32*team_off[t]+.78*idio
        else: z=.15*pace+.30*team_off[t]+.78*idio
        # Normalize factor variance so Sim SD remains interpretable.
        z=(z-z.mean())/(z.std()+1e-9)
        unc=rng.normal(0,{"QB":.07,"RB":.10,"WR":.12,"TE":.13,"K":.08,"DST":.16,"D":.16}.get(pos,.12),size=ns)*float(r["Base Mean"])
        score=np.clip(float(r["Base Mean"])+unc+float(r["Sim SD"])*z,0,None)
        # V1.3: low-volume fallback skill players are mixture distributions, not
        # smooth bell curves. Most retain normal opportunity, but a role-tier-based
        # share of simulations are true low-opportunity/dud outcomes. Rare spike
        # branches preserve the legitimate TD/broken-play ceiling.
        if r.get("Projection Source","")=="DK slate fallback" and pos in ["RB","WR","TE"]:
            q=float(r.get("Low Score Prob",0.0) or 0.0)
            sp=float(r.get("Spike Prob",0.0) or 0.0)
            if q>0:
                dud=rng.random(ns)<q
                score[dud]*=rng.uniform(0.05,0.35,size=int(dud.sum()))
                live=~dud
                # Modest live-branch lift keeps upside conditional on actually
                # earning opportunity without making salary itself create ceiling.
                score[live]*=(1.0+0.20*q)
            if sp>0:
                spike=(rng.random(ns)<sp) & (~dud if q>0 else np.ones(ns,dtype=bool))
                if spike.any():
                    score[spike]+=rng.gamma(shape=2.0,scale=2.0,size=int(spike.sum()))
        out[:,j]=score.astype(np.float32)
    return out

def showdown_value_diagnostics(players, sims):
    rows=[]
    for j,r in players.iterrows():
        v=sims[:,j]
        rows.append({
            "Name":r["Name"],"Pos":r["Position"],"Team":r["TeamAbbrev"],
            "Salary":int(r["FLEX Salary"]),"Role Tier":r.get("Role Tier",""),
            "Raw Weekly Mean":float(r.get("Raw Weekly Mean",r.get("Base Mean",0.0))),
            "DK Role Baseline":float(r.get("DK Role Baseline",0.0)),
            "Calibrated Mean":float(r.get("Base Mean",0.0)),"Current Role":r.get("Current Role",""),"Opportunity Multiplier":float(r.get("Opportunity Multiplier",1.0)),
            "Sim Mean":float(np.mean(v)),"P75":float(np.percentile(v,75)),
            "P90":float(np.percentile(v,90)),"P95":float(np.percentile(v,95)),
            "≤3 pts %":100*float(np.mean(v<=3.0)),"10+ pts %":100*float(np.mean(v>=10.0)),
            "Source":r.get("Projection Source","")
        })
    return pd.DataFrame(rows)

def showdown_viability_flags(players, sims, standard="Balanced GPP"):
    # V1.4 minimum-viability screen. Salary never determines eligibility.
    # A player qualifies from the simulated scoring distribution; a marginal sixth
    # player can still be used when Allow one deep punt is enabled.
    presets={
        "Loose GPP":   {"mean":1.25,"p75":1.75,"p90":3.5,"p3":0.18,"p5":0.08,"need":2},
        "Balanced GPP":{"mean":2.00,"p75":3.00,"p90":5.0,"p3":0.25,"p5":0.12,"need":3},
        "Strong Floor":{"mean":3.00,"p75":4.00,"p90":6.5,"p3":0.40,"p5":0.20,"need":3},
    }
    rows=[]
    if standard=="Off":
        return pd.DataFrame({"Viable":[True]*len(players),"Deep Punt OK":[True]*len(players),"Viability Tests":[5]*len(players)},index=players.index)
    cfg=presets.get(standard,presets["Balanced GPP"])
    for j,r in players.iterrows():
        v=sims[:,j]; mean=float(np.mean(v)); p75=float(np.percentile(v,75)); p90=float(np.percentile(v,90))
        prob3=float(np.mean(v>=3.0)); prob5=float(np.mean(v>=5.0)); prob10=float(np.mean(v>=10.0))
        tests=sum([mean>=cfg["mean"],p75>=cfg["p75"],p90>=cfg["p90"],prob3>=cfg["p3"],prob5>=cfg["p5"]])
        viable=tests>=cfg["need"]
        # A deep punt must still own a plausible scoring path. This prevents the
        # optimizer from using essentially dead players solely to unlock five studs.
        deep_ok=(p90>=max(3.0,cfg["p90"]*.70)) or (prob5>=max(.06,cfg["p5"]*.60)) or (prob10>=.025)
        rows.append({"Viable":bool(viable),"Deep Punt OK":bool(deep_ok),"Viability Tests":int(tests)})
    return pd.DataFrame(rows,index=players.index)

def generate_showdown_candidates(players, sims, teams, bank_size=6000, min_salary=44000, max_salary=50000, seed=316, min_standard="Balanced GPP", allow_deep_punt=True):
    """V1.7a performance fix: same V1.7 construction logic, precomputed arrays.

    The expensive player metrics and roster metadata are calculated once. Candidate
    attempts use NumPy arrays instead of repeated pandas slicing. Simulation quality,
    bank-size target, role thresholds, salary rules and lineup rules are unchanged.
    """
    rng=np.random.default_rng(seed); n=len(players); seen=set(); rows=[]; tries=0; target=int(bank_size)
    flags=showdown_viability_flags(players,sims,min_standard)

    # Player distribution metrics: calculate ONCE.
    sim_mean=np.mean(sims,axis=0).astype(float)
    sim_p75=np.percentile(sims,75,axis=0).astype(float)
    sim_p90=np.percentile(sims,90,axis=0).astype(float)
    sim_p95=np.percentile(sims,95,axis=0).astype(float)
    prob5=np.mean(sims>=5.0,axis=0).astype(float)
    prob10=np.mean(sims>=10.0,axis=0).astype(float)

    # Convert all roster metadata to NumPy once; do not use DataFrame.iloc in the hot loop.
    pos=players["Position"].astype(str).to_numpy()
    team=players["TeamAbbrev"].astype(str).to_numpy()
    flex_salary=pd.to_numeric(players["FLEX Salary"],errors="coerce").fillna(0).to_numpy(dtype=np.int32)
    cpt_salary=pd.to_numeric(players["CPT Salary"],errors="coerce").fillna(0).to_numpy(dtype=np.int32)
    viable=flags["Viable"].to_numpy(dtype=bool)
    deep_ok=flags["Deep Punt OK"].to_numpy(dtype=bool)

    core=np.zeros(n,dtype=bool); relief=np.zeros(n,dtype=bool); captain_ok=np.zeros(n,dtype=bool)
    for i in range(n):
        if pos[i] in ["QB","K","DST","D"]:
            core[i]=(sim_mean[i]>=3.0 and sim_p75[i]>=4.0 and sim_p90[i]>=6.0)
        else:
            core[i]=(sim_mean[i]>=3.25 and sim_p75[i]>=4.0 and sim_p90[i]>=7.0 and prob5[i]>=0.20)
        relief[i]=(sim_p90[i]>=6.0 and prob5[i]>=0.15 and sim_mean[i]>=1.75)
        role_cpt_ok=bool(players.iloc[i].get("Role CPT Eligible",True))
        captain_ok[i]=role_cpt_ok

    mean_scale=np.maximum(sim_mean,0.10); p90_scale=np.maximum(sim_p90,0.10); ceiling_scale=np.maximum(sim_p95,0.10)
    weights=(0.40*mean_scale/mean_scale.max()+0.40*p90_scale/p90_scale.max()+0.20*ceiling_scale/ceiling_scale.max())
    weights=np.maximum(weights,0.005)
    weights=np.where(core,weights,weights*0.55)
    weights=np.where((~core)&(~relief),weights*0.35,weights)
    weights=weights/weights.sum()
    cpt_weights=weights*captain_ok.astype(float)
    if cpt_weights.sum()>0: cpt_weights=cpt_weights/cpt_weights.sum()

    all_idx=np.arange(n,dtype=np.int16)
    # Same generous search ceiling as V1.7, but the hot path is now array-only.
    max_tries=target*140
    while len(rows)<target and tries<max_tries:
        tries+=1
        if cpt_weights.sum()<=0: break
        c=int(rng.choice(n,p=cpt_weights))
        if not captain_ok[c]: continue
        avail=all_idx[all_idx!=c]
        pw=weights[avail]; pw=pw/pw.sum()
        flex_idx=np.sort(rng.choice(avail,size=5,replace=False,p=pw)).astype(int)
        key=(c,tuple(flex_idx.tolist()))
        if key in seen: continue
        ids=np.concatenate(([c],flex_idx))

        salary=int(cpt_salary[c]+flex_salary[flex_idx].sum())
        if salary<min_salary or salary>max_salary: continue
        if len(np.unique(team[ids]))<2: continue

        viable_count=int(viable[ids].sum())
        nonviable=ids[~viable[ids]]
        if min_standard!="Off":
            if allow_deep_punt:
                if viable_count<5 or len(nonviable)>1: continue
                if len(nonviable) and not deep_ok[nonviable[0]]: continue
            elif viable_count<6: continue

        p=pos[ids]

        seen.add(key)
        pts=(1.5*sims[:,c]+sims[:,flex_idx].sum(axis=1)).astype(np.float32,copy=False)
        # One percentile call instead of three full passes.
        q75,q90,q95=np.percentile(pts,[75,90,95])
        rows.append({"cpt":c,"flex":tuple(flex_idx.tolist()),"salary":salary,
                     "mean":float(pts.mean()),"p75":float(q75),"p90":float(q90),"p95":float(q95),
                     "sim":pts,"viable_count":viable_count,"core_count":int(core[ids].sum())})
    return rows,flags

def rank_showdown_candidates(cands, strategy):
    if not cands: return cands
    # Optimal rate is computed against the generated candidate bank, simulation by simulation.
    ns=len(cands[0]["sim"]); best=np.full(ns,-1e9,dtype=np.float32)
    for c in cands: best=np.maximum(best,c["sim"])
    for c in cands:
        c["optimal_rate"]=float(np.mean(c["sim"]>=best-1e-5))
        if strategy=="Median / Mean": c["rank_score"]=c["mean"]
        elif strategy=="Balanced": c["rank_score"]=.55*c["mean"]+.20*c["p90"]+.25*c["p95"]
        else: c["rank_score"]=.25*c["mean"]+.30*c["p90"]+.35*c["p95"]+10*c["optimal_rate"]
        # V1.1 salary-efficiency sanity check. This is deliberately a soft hurdle,
        # not a hard spend rule: unusual low-salary builds can still win if their
        # simulated ceiling is strong enough to overcome the penalty.
        unused=max(0,50000-c["salary"])
        c["unused_salary"]=unused
        if unused>3000:
            c["rank_score"]-=0.70*((unused-3000)/1000.0)
    return sorted(cands,key=lambda z:z["rank_score"],reverse=True)

def _showdown_exposure_counts(n_lineups, max_player_exp, max_cpt_exp):
    # Small portfolios are discrete: with 4 entries, 65% cannot literally be represented.
    # Round UP to the nearest attainable lineup count so the UI percentage is a target,
    # not an accidental hard floor that makes a requested portfolio impossible.
    maxp=max(1, min(n_lineups, math.ceil(n_lineups*max_player_exp-1e-12)))
    maxc=max(1, min(n_lineups, math.ceil(n_lineups*max_cpt_exp-1e-12)))
    return maxp,maxc

def select_showdown_portfolio(cands, players, n_lineups, max_player_exp, max_cpt_exp, min_unique, locks, excludes, cpt_excludes):
    lockset=set(locks); exset=set(excludes); cex=set(cpt_excludes)
    base_maxp,base_maxc=_showdown_exposure_counts(n_lineups,max_player_exp,max_cpt_exp)

    # Controlled relaxation order. Preserve uniqueness first; relax exposure only enough
    # to fill the requested portfolio. Never relax locks/exclusions or lineup legality.
    attempts=[]
    for addp,addc in [(0,0),(1,0),(0,1),(1,1),(2,1),(2,2)]:
        attempts.append((min(n_lineups,base_maxp+addp),min(n_lineups,base_maxc+addc)))
    seen=set(); attempts=[x for x in attempts if not (x in seen or seen.add(x))]

    best=[]; used_limits=(base_maxp,base_maxc); relaxed=False
    for maxp,maxc in attempts:
        selected=[]; counts={}; cpt_counts={}
        for c in cands:
            ids=[c["cpt"]]+list(c["flex"]); names=set(players.iloc[ids]["Name"]); cpt_name=players.iloc[c["cpt"]]["Name"]
            if lockset and not lockset.issubset(names): continue
            if names & exset or cpt_name in cex: continue
            if cpt_counts.get(cpt_name,0)>=maxc: continue
            if any(counts.get(nm,0)>=maxp for nm in names): continue
            if any(len(names & set(x["names"]))>6-int(min_unique) for x in selected): continue
            c2=dict(c); c2["names"]=list(names); selected.append(c2)
            for nm in names: counts[nm]=counts.get(nm,0)+1
            cpt_counts[cpt_name]=cpt_counts.get(cpt_name,0)+1
            if len(selected)>=n_lineups: break
        if len(selected)>len(best): best=selected; used_limits=(maxp,maxc)
        if len(selected)>=n_lineups:
            relaxed=(maxp,maxc)!=(base_maxp,base_maxc)
            return selected,{"requested":n_lineups,"max_player_count":maxp,"max_cpt_count":maxc,"base_player_count":base_maxp,"base_cpt_count":base_maxc,"relaxed":relaxed}
    return best,{"requested":n_lineups,"max_player_count":used_limits[0],"max_cpt_count":used_limits[1],"base_player_count":base_maxp,"base_cpt_count":base_maxc,"relaxed":used_limits!=(base_maxp,base_maxc)}

def showdown_results_df(selected, players):
    rows=[]
    for i,c in enumerate(selected,1):
        cp=players.iloc[c["cpt"]]; fps=players.iloc[list(c["flex"])]
        rows.append({"Rank":i,"CPT":cp["Name"],"FLEX 1":fps.iloc[0]["Name"],"FLEX 2":fps.iloc[1]["Name"],"FLEX 3":fps.iloc[2]["Name"],"FLEX 4":fps.iloc[3]["Name"],"FLEX 5":fps.iloc[4]["Name"],"Salary":c["salary"],"Mean":c["mean"],"P75":c["p75"],"P90":c["p90"],"P95":c["p95"],"Optimal %":100*c["optimal_rate"]})
    return pd.DataFrame(rows)

def showdown_dk_export(selected, players):
    rows=[]
    for c in selected:
        cp=players.iloc[c["cpt"]]; fps=players.iloc[list(c["flex"])]
        row={"CPT":f'{cp["Name"]} ({int(cp["CPT ID"])})'}
        for k,(_,r) in enumerate(fps.iterrows(),1): row[f"FLEX{k}"]=f'{r["Name"]} ({int(r["FLEX ID"])})'
        rows.append(row)
    return pd.DataFrame(rows)

mc, comp, dst, own, games, matchups = load_base()
base_pool = attach_game_environment(calibrate_classic_tails(prepare_pool(mc,dst,own,matchups)),games)
if "classic_uploaded_pool" not in st.session_state:
    st.session_state["classic_uploaded_pool"]=None
if "classic_active_environment" not in st.session_state:
    st.session_state["classic_active_environment"]=games.copy()
if "classic_environment_verified" not in st.session_state:
    st.session_state["classic_environment_verified"]=True
pool = st.session_state["classic_uploaded_pool"] if st.session_state["classic_uploaded_pool"] is not None else base_pool

st.title("🏈 NFL Predictor Pro — DFS Engine V2.0.1")
st.caption("DraftKings NFL DFS • correlated game scripts • variance + uncertainty • scenario portfolios • calibration")
st.warning("Classic V3.1.6b adds DraftKings injury-status eligibility gating. Classic V3.1.6 recalibrates tournament tails and adds bounded game-environment/correlation scoring. Uploaded DK slates are the roster/salary source of truth; unmatched players are explicitly labeled DK PPG fallback. Re-check final injury news and ownership before contest entry.")

view = st.sidebar.radio("View", ["Slate Setup","Player Projections","Simulation","Lineup Builder","Single Game Showdown","Simulation Validation","Portfolio Analysis"])

if view == "Slate Setup":
    st.subheader("Slate Setup")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Skill players", f"{len(mc):,}")
    c2.metric("DSTs", f"{len(dst):,}")
    c3.metric("Simulation runs", "50,000")
    c4.metric("Games", f"{pool['game'].nunique()}")
    st.markdown("#### Active game environment")
    active_env, env_verified, env_missing = current_slate_environment(pool, st.session_state.get("classic_active_environment"))
    if st.session_state.get("classic_uploaded_pool") is not None and not env_verified:
        st.error("Current slate environment is NOT verified. Prior-week game data has been blocked. Enter/activate totals for every active game below before generating a V2 portfolio.")
    elif env_verified:
        st.success("Game environment verified for every game on the active slate.")
    st.dataframe(active_env, use_container_width=True, hide_index=True)
    st.markdown("#### Weekly DraftKings Main Slate upload")
    st.caption("Upload the DraftKings Classic salary CSV here each week. The uploaded file becomes the active roster, salary, player-ID and game source for Classic projections and lineup construction.")
    up=st.file_uploader("Upload DraftKings NFL Classic salary CSV", type=["csv"], key="classic_salary_upload")
    u1,u2=st.columns([1,1])
    if up is not None:
        try:
            dk=parse_classic_dk_csv(up)
            uploaded_pool=classic_pool_from_upload(dk,base_pool)
            uploaded_pool=calibrate_classic_tails(uploaded_pool)
            # V2.0.1: a newly uploaded slate starts UNVERIFIED. Never attach stale prior-week environments.
            slate_games=sorted(uploaded_pool["game"].dropna().astype(str).str.strip().str.upper().unique().tolist())
            previous_games=st.session_state.get("classic_uploaded_games",[])
            if slate_games != previous_games:
                st.session_state["classic_active_environment"]=environment_editor_seed(uploaded_pool,games)
                st.session_state["classic_environment_verified"]=False
                st.session_state["classic_uploaded_games"]=slate_games
            env_now, env_ok, _ = current_slate_environment(uploaded_pool, st.session_state.get("classic_active_environment"))
            if env_ok and st.session_state.get("classic_environment_verified",False):
                uploaded_pool=attach_game_environment(uploaded_pool,env_now)
            else:
                uploaded_pool["game_env_score"]=np.nan
            st.session_state["classic_uploaded_pool"]=uploaded_pool
            matched=int((uploaded_pool["projection_source"]=="Weekly researched model").sum())
            fallback=len(uploaded_pool)-matched
            blocked=uploaded_pool[uploaded_pool["injury_blocked"]].copy()
            flagged=uploaded_pool[uploaded_pool["injury_flagged"]].copy()
            eligible_n=int(uploaded_pool["optimizer_eligible"].sum())
            st.success(f"Activated {len(uploaded_pool):,} DK players • {eligible_n} optimizer-eligible • {matched} weekly-model matches • {fallback} DK fallback rows.")
            if len(blocked):
                st.warning(f"Injury-status gate excluded {len(blocked)} player(s) from optimization: "+", ".join(blocked["Name"].astype(str)+" ("+blocked["dk_status"].astype(str)+")"))
            else:
                st.success("Injury-status gate: no OUT / IR / inactive players detected in this upload.")
            if len(flagged):
                st.info("Availability watch (not automatically excluded): "+", ".join(flagged["Name"].astype(str)+" ("+flagged["dk_status"].astype(str)+")"))
            if fallback:
                st.warning("Fallback rows do not have refreshed usage/air-yards/red-zone research. They remain clearly labeled and should not be treated as equivalent to weekly researched projections.")
            st.dataframe(uploaded_pool[["Name","Position","TeamAbbrev","game","Salary","ID","dk_status","injury_blocked","proj","projection_source","projection_confidence","optimizer_eligible"]].sort_values(["Position","Salary"],ascending=[True,False]),use_container_width=True,hide_index=True)
        except Exception as e:
            st.error(f"Could not activate DraftKings slate: {e}")
    active_upload=st.session_state.get("classic_uploaded_pool")
    if active_upload is not None:
        st.markdown("#### Verify this slate's game environment")
        st.caption("The table contains ONLY games from the uploaded DK slate. Enter the current market game total for every game; spread is optional but improves the environment score. This prevents any prior-week environment from leaking into a new slate.")
        seed=environment_editor_seed(active_upload, st.session_state.get("classic_active_environment"))
        edited_env=st.data_editor(seed,hide_index=True,use_container_width=True,disabled=["game"],key="classic_env_editor")
        if st.button("Activate current game environment",type="primary",key="activate_current_env"):
            checked, ok, missing=current_slate_environment(active_upload,edited_env)
            if not ok:
                st.session_state["classic_environment_verified"]=False
                st.error("Environment not activated. Add a game total for: "+", ".join(missing))
            else:
                st.session_state["classic_active_environment"]=checked.copy()
                st.session_state["classic_environment_verified"]=True
                refreshed=active_upload.drop(columns=["game_env_score"],errors="ignore")
                refreshed=attach_game_environment(refreshed,checked)
                st.session_state["classic_uploaded_pool"]=refreshed
                st.success("Current-slate environment activated. No prior-week game rows are being used.")
                st.rerun()
    if u1.button("Use built-in weekly model slate"):
        st.session_state["classic_uploaded_pool"]=None
        st.session_state["classic_active_environment"]=games.copy()
        st.session_state["classic_environment_verified"]=True
        st.session_state["classic_uploaded_games"]=[]
        st.rerun()
    active=st.session_state.get("classic_uploaded_pool")
    if active is not None:
        st.info(f"ACTIVE CLASSIC SLATE: uploaded DraftKings CSV • {len(active)} players • {active['game'].nunique()} games")
    else:
        st.info("ACTIVE CLASSIC SLATE: built-in researched weekly model")

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
    st.subheader("Simulation")
    st.caption("Legacy weekly Monte Carlo inputs are shown below. DFS Engine V2 additionally runs coherent full-slate game-script simulations during portfolio construction, with heavy tails and model uncertainty.")
    metric=st.selectbox("Sort by",["mean","p90","p95","boom_25","boom_30"])
    pos=st.multiselect("Position",["QB","RB","WR","TE"],default=["QB","RB","WR","TE"])
    sx=mc[mc.Position.isin(pos)].sort_values(metric,ascending=False)
    st.dataframe(sx,use_container_width=True,hide_index=True)
    st.download_button("Download simulation CSV", mc.to_csv(index=False), "nfl_simulation_v8.csv","text/csv")

elif view == "Lineup Builder":
    st.subheader("DraftKings Portfolio Optimizer")
    if st.session_state.get("classic_uploaded_pool") is not None and not st.session_state.get("classic_environment_verified",False):
        st.error("V2 SAFETY GATE: the uploaded slate's game environment has not been verified. Go to Slate Setup and activate current game totals first. Prior-week environment data will not be used as a fallback.")
        st.stop()
    a,b,c,d=st.columns(4)
    _init_defaults(CLASSIC_DEFAULTS)
    st.button("Reset to Recommended Defaults",key="classic_reset_btn",on_click=_reset_defaults,args=(CLASSIC_DEFAULTS,))
    n_lineups=a.number_input("Lineups",1,150,key="classic_n_lineups",step=1)
    max_exp=b.slider("Max exposure",0.05,1.0,key="classic_max_exp",step=0.05)
    min_unique=c.slider("Minimum unique players",1,8,key="classic_min_unique",step=1)
    min_salary=d.number_input("Minimum lineup salary",40000,50000,key="classic_min_salary",step=100)
    e,f,g=st.columns(3)
    strategy=e.selectbox("Strategy",["GPP Ceiling","Balanced","Median"],key="classic_strategy")
    own_weight=f.slider("Ownership fade weight",0.0,2.0,key="classic_own_weight",step=0.05, help="Lower values are more willing to roster strong chalk. Ownership still matters, but good high-owned plays are not heavily penalized.")
    leverage_weight=g.slider("Leverage weight",0.0,2.0,key="classic_leverage_weight",step=0.05)
    projection_floor_pct=st.slider("Projection quality floor (% of strong reference lineup)",0.75,1.00,key="classic_projection_floor",step=0.01, help="Requires every generated lineup to retain this percentage of a strong reference projection from the current slate. 88% is a balanced GPP default.")

    eligible=pool[pool["optimizer_eligible"] == True].sort_values(["Position","proj"],ascending=[True,False])
    names=eligible["Name"].drop_duplicates().tolist()
    locks=st.multiselect("Lock players",names,key="classic_locks")
    excludes=st.multiselect("Exclude players",names,key="classic_excludes")

    portfolio_mode=st.radio("Portfolio construction",["DFS Engine V2 Scenario Portfolio","V3.1.7 Portfolio Optimize (control)"],horizontal=True,key="classic_portfolio_mode")
    bank_size=st.slider("Candidate bank size",300,2000,key="classic_bank_size",step=100,disabled=False,help="V3.1.7 generates this many strong legal candidates, then chooses the full portfolio simultaneously.")
    solver_seconds=st.slider("Portfolio solver time limit (seconds)",5,60,key="classic_solver_seconds",step=5,disabled=False)

    st.info("V2 defaults: 15 lineups • 65% safety exposure ceiling • 3 minimum unique players • $47,500 salary floor • 88% projection-quality floor • QB + pass-catcher stack • no offensive player against selected DST. Ownership fade defaults to a chalk-friendly 0.15; leverage remains 0.35. V2 rescoring uses coherent full-slate scenarios with heavy-tailed player variance and model uncertainty, then selects lineups for marginal scenario coverage instead of simply taking the highest-ranked near-duplicates.")

    if st.button("Generate portfolio",type="primary"):
        with st.spinner("Generating diversified portfolio..."):
            solver_meta=None
            candidate_items, reference_proj, min_proj_required = generate_candidate_bank(
                pool,int(bank_size),int(min_salary),50000,strategy,float(own_weight),
                float(leverage_weight),locks,excludes,float(projection_floor_pct),
                attempts=max(60000,int(bank_size)*120)
            )
            if portfolio_mode == "DFS Engine V2 Scenario Portfolio":
                sim_players,classic_sims=simulate_classic_v2(pool,10000)
                records=v2_rescore_classic_candidates(candidate_items,sim_players,classic_sims)
                lineups,solver_meta=select_v2_portfolio(records,int(n_lineups),float(max_exp),int(min_unique))
                st.session_state['classic_v2_sim_players']=sim_players; st.session_state['classic_v2_sims']=classic_sims
            else:
                lineups, solver_meta = select_portfolio_milp(candidate_items,int(n_lineups),float(max_exp),int(min_unique),float(solver_seconds))
            expo={}; max_count=max(1,math.ceil(int(n_lineups)*float(max_exp)-1e-12))
            for l in lineups:
                for nm in set(l.Name): expo[nm]=expo.get(nm,0)+1
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
                st.caption(f"V3.1.6 candidate bank: {solver_meta.get('candidate_count',0):,} • portfolio constraints: {solver_meta.get('constraint_count',0):,} • solver: {solver_meta.get('message','')}")
            st.caption(f"Projection quality guardrail: strong reference {reference_proj:.1f} DK points • minimum accepted {min_proj_required:.1f} ({projection_floor_pct:.0%}).")
            ldf=lineups_to_df(lineups,strategy,float(own_weight),float(leverage_weight),stack_rank=True)
            edf=exposure_df(lineups)
            st.markdown("#### Lineups — stack ranked best to worst")
            st.caption("V3.1.6 GPP Rank emphasizes Mean + P90 + P95, uses P99 only as a small tail signal, then adds bounded game-environment and stack-correlation bonuses before ownership/leverage adjustments. Percentile columns are player-upside indexes, not literal lineup percentiles.")
            st.dataframe(ldf,use_container_width=True,hide_index=True)
            if qc["pass"]:
                st.download_button("Download lineups",ldf.to_csv(index=False),"nfl_lineups_v316.csv","text/csv")
            else:
                st.warning("Download disabled until final portfolio QC passes.")
            st.markdown("#### Exposure")
            st.dataframe(edf,use_container_width=True,hide_index=True)
            st.download_button("Download exposure",edf.to_csv(index=False),"nfl_exposure.csv","text/csv")

elif view == "Single Game Showdown":
    st.subheader("DraftKings Single-Game Showdown")
    st.caption("Upload the DraftKings Showdown salary CSV. The module detects the two teams, removes OUT players, matches the weekly NFL model when available, simulates correlated game outcomes, and optimizes 1 CPT + 5 FLEX lineups.")
    _init_defaults(SHOWDOWN_DEFAULTS)
    uploaded=st.file_uploader("Upload DraftKings Showdown salary CSV",type=["csv"],key="showdown_salary_upload")
    if uploaded is None:
        st.info("Upload a DraftKings NFL Showdown salary CSV to begin. Your Classic settings and data are unaffected.")
    else:
        try:
            flex,game_label,sd_teams=parse_showdown_csv(uploaded)
            players=showdown_projection_table(flex,comp,mc,dst)
            st.success(f"Detected {game_label} • {' vs '.join(sd_teams)} • {len(players)} active FLEX-eligible players")
            model_matches=int((players["Projection Source"]=="NFL weekly model").sum())
            c1,c2,c3=st.columns(3); c1.metric("Active players",len(players)); c2.metric("Weekly-model matches",model_matches); c3.metric("Fallback players",len(players)-model_matches)
            coverage=model_matches/max(1,len(players))
            if coverage<0.60:
                st.error(f"PROJECTION INTEGRITY GATE: only {coverage:.0%} of active players matched the weekly model. V2 will show diagnostics, but this slate should not be trusted for automated entry until the weekly projection feed is refreshed.")
            elif coverage<0.85:
                st.warning(f"Projection coverage is {coverage:.0%}. Review fallback rows before trusting the portfolio.")
            else:
                st.success(f"Projection integrity gate: {coverage:.0%} weekly-model coverage.")
            with st.expander("Player matching / projection audit"):
                st.dataframe(players[["Name","Position","TeamAbbrev","FLEX Salary","CPT Salary","Status","Current Role","Opportunity Multiplier","Raw Weekly Mean","DK Role Baseline","Base Mean","Sim SD","Role Tier","Role Calibration","Projection Source"]].sort_values("FLEX Salary",ascending=False),use_container_width=True,hide_index=True)
            st.button("Reset Showdown to Recommended Defaults",key="sd_reset_btn",on_click=_reset_defaults,args=(SHOWDOWN_DEFAULTS,))
            a,b,c,d=st.columns(4)
            n_lineups=a.number_input("Showdown lineups",1,150,key="sd_n_lineups",step=1)
            max_player=b.slider("Max player exposure",0.10,1.0,key="sd_max_player_exp",step=0.05)
            max_cpt=c.slider("Max Captain exposure",0.05,1.0,key="sd_max_cpt_exp",step=0.05)
            min_unique=d.slider("Minimum unique players",1,5,key="sd_min_unique",step=1)
            e,f,g,h=st.columns(4)
            min_salary=e.number_input("Minimum salary",30000,50000,key="sd_min_salary",step=500)
            n_sims=f.select_slider("Game simulations",options=[5000,10000,20000,30000,50000],key="sd_sims")
            bank=g.select_slider("Candidate bank",options=[1000,2000,4000,6000,8000,10000],key="sd_candidate_bank")
            strategy=h.selectbox("Ranking",["Tournament Ceiling","Balanced","Median / Mean"],key="sd_strategy")
            q1,q2=st.columns(2)
            min_standard=q1.selectbox("Minimum player standard",["Off","Loose GPP","Balanced GPP","Strong Floor"],key="sd_min_standard",help="Screens players by their simulated scoring distribution, not salary. Captain must pass the selected standard.")
            allow_deep_punt=q2.checkbox("Allow one deep punt",key="sd_allow_deep_punt",help="Allows at most one player who misses the full standard, but only if the simulation still shows a legitimate scoring path.")
            names=players["Name"].tolist()
            locks=st.multiselect("Lock player(s)",names,key="sd_locks")
            excludes=st.multiselect("Exclude player(s)",names,key="sd_excludes")
            cpt_excludes=st.multiselect("Exclude from Captain only",names,key="sd_cpt_excludes")
            eff_p,eff_c=_showdown_exposure_counts(int(n_lineups),float(max_player),float(max_cpt))
            st.caption(f"Effective small-portfolio limits: any player ≤ {eff_p}/{int(n_lineups)} lineups ({100*eff_p/int(n_lineups):.0f}%) • any Captain ≤ {eff_c}/{int(n_lineups)} ({100*eff_c/int(n_lineups):.0f}%). Percentages are rounded up to the nearest attainable lineup count.")
            st.info("V2 defaults: 20 lineups • 75% safety player ceiling • 50% Captain ceiling • 2 minimum uniques • $38,000 salary floor • 20,000 correlated game simulations • 6,000 candidates • viability screen Off. Rare legal constructions are probability-weighted rather than prohibited. V1.7 keeps opportunity-first candidate generation and adds a bounded role sanity calibration for weekly-model skill players. DK PPG is used only as a 20% role anchor when available; it never replaces the weekly projection. V2 no longer bans unusual legal constructions (including DST Captain, double-DST, no-QB, or unusual salary usage) merely because they are rare; simulated outcomes and portfolio value determine representation.")
            if st.button("Simulate game + build Showdown portfolio",type="primary",key="sd_generate"):
                with st.spinner("Simulating correlated game outcomes and optimizing Showdown lineups..."):
                    sim=simulate_showdown_players(players,sd_teams,int(n_sims))
                    cands,viability=generate_showdown_candidates(players,sim,sd_teams,int(bank),int(min_salary),50000,min_standard=min_standard,allow_deep_punt=bool(allow_deep_punt))
                    ranked=rank_showdown_candidates(cands,strategy)
                    diagnostics=showdown_value_diagnostics(players,sim)
                    diagnostics=diagnostics.join(viability[["Viable","Deep Punt OK","Viability Tests"]])
                    selected,sd_meta=select_showdown_portfolio_v2(ranked,players,int(n_lineups),float(max_player),float(max_cpt),int(min_unique),locks,excludes,cpt_excludes)
                if not selected:
                    st.error("No valid Showdown portfolio found. Relax salary, uniqueness, locks, or exclusions.")
                else:
                    rdf=showdown_results_df(selected,players); dkdf=showdown_dk_export(selected,players)
                    st.session_state["showdown_selected"]=selected; st.session_state["showdown_players"]=players; st.session_state['showdown_sims']=sim
                    if len(selected)<int(n_lineups):
                        st.error(f"Requested {int(n_lineups)} lineups but only {len(selected)} could be built after controlled exposure relaxation. Increase candidate bank or relax minimum uniques / salary floor / locks before exporting.")
                    else:
                        st.success(f"Built all {len(selected)} requested Showdown lineups from {len(cands):,} legal simulated candidates.")
                    if sd_meta.get("relaxed"):
                        st.warning(f"To fill the portfolio, V1.2 relaxed exposure counts to player ≤ {sd_meta['max_player_count']}/{int(n_lineups)} and Captain ≤ {sd_meta['max_cpt_count']}/{int(n_lineups)}. Locks, exclusions, uniqueness and lineup legality were not relaxed.")
                    st.dataframe(rdf,use_container_width=True,hide_index=True,column_config={"Mean":st.column_config.NumberColumn(format="%.2f"),"P75":st.column_config.NumberColumn(format="%.2f"),"P90":st.column_config.NumberColumn(format="%.2f"),"P95":st.column_config.NumberColumn(format="%.2f"),"Optimal %":st.column_config.NumberColumn(format="%.3f")})
                    st.download_button("Download DraftKings Showdown CSV",dkdf.to_csv(index=False),"DK_Showdown_Lineups.csv","text/csv")
                    st.markdown("#### Showdown value diagnostics")
                    st.caption("V1.7 generates candidates from simulated opportunity and ceiling rather than points-per-dollar. Weekly-model skill projections receive a bounded 20% DK-role sanity anchor before simulation; salary is not part of that calibration. V2 evaluates legal constructions from simulated outcomes rather than hard-banning rare roster structures. Role-ineligible/inactive players remain blocked.")
                    diag_show=diagnostics.sort_values(["Salary","Sim Mean"],ascending=[True,False])
                    st.dataframe(diag_show,use_container_width=True,hide_index=True,column_config={"Sim Mean":st.column_config.NumberColumn(format="%.2f"),"P75":st.column_config.NumberColumn(format="%.2f"),"P90":st.column_config.NumberColumn(format="%.2f"),"P95":st.column_config.NumberColumn(format="%.2f"),"≤3 pts %":st.column_config.NumberColumn(format="%.1f%%"),"10+ pts %":st.column_config.NumberColumn(format="%.1f%%")})
                    all_names=[]; all_cpt=[]
                    for z in selected:
                        ids=[z["cpt"]]+list(z["flex"]); all_names.extend(players.iloc[ids]["Name"].tolist()); all_cpt.append(players.iloc[z["cpt"]]["Name"])
                    ex=pd.Series(all_names).value_counts().rename("Lineups").to_frame(); ex["Exposure %"]=100*ex["Lineups"]/len(selected)
                    cx=pd.Series(all_cpt).value_counts().rename("Captain Lineups").to_frame(); cx["Captain %"]=100*cx["Captain Lineups"]/len(selected)
                    st.markdown("#### Player exposure"); st.dataframe(ex.reset_index(names="Name"),use_container_width=True,hide_index=True)
                    st.markdown("#### Captain exposure"); st.dataframe(cx.reset_index(names="Name"),use_container_width=True,hide_index=True)
        except Exception as exc:
            st.error(f"Could not process this Showdown CSV: {exc}")

elif view == "Simulation Validation":
    st.subheader("Simulation Validation")
    st.caption("V2 compares the full simulated distribution with actual fantasy results. Mean, median and binned mode are kept separate; Actual Percentile is the calibration diagnostic.")
    source=st.radio("Simulation source",["NFL Main Slate","Showdown"],horizontal=True)
    if source=="NFL Main Slate":
        sp=st.session_state.get('classic_v2_sim_players'); ss=st.session_state.get('classic_v2_sims')
        if sp is None or ss is None:
            st.info("Generate a DFS Engine V2 Main Slate portfolio first.")
        else:
            names=sp['Name'].astype(str).tolist(); ss_use=ss
    else:
        sp=st.session_state.get('showdown_players'); ss_use=st.session_state.get('showdown_sims')
        if sp is None or ss_use is None:
            st.info("Generate a Showdown portfolio first.")
        else: names=sp['Name'].astype(str).tolist()
    if sp is not None and ss_use is not None:
        base=simulation_summary(names,ss_use)
        st.dataframe(base,use_container_width=True,hide_index=True)
        st.download_button("Download pre-event simulation validation baseline",base.to_csv(index=False),"simulation_validation_baseline.csv","text/csv")
        actual_up=st.file_uploader("After the games: upload actual DK points CSV (columns: Name, Actual)",type=['csv'],key='actual_validation_upload')
        if actual_up is not None:
            a=pd.read_csv(actual_up)
            if {'Name','Actual'}.issubset(a.columns):
                amap=dict(zip(a['Name'].astype(str),pd.to_numeric(a['Actual'],errors='coerce')))
                graded=simulation_summary(names,ss_use,amap)
                st.markdown("#### Graded simulation vs actual")
                st.dataframe(graded,use_container_width=True,hide_index=True)
                valid=graded.dropna(subset=['Actual Percentile'])
                if len(valid):
                    c1,c2,c3=st.columns(3); c1.metric("Actual above simulated median",f"{100*np.mean(valid['Actual Percentile']>50):.1f}%")
                    c2.metric("Actual above simulated P90",f"{100*np.mean(valid['Actual Percentile']>90):.1f}%")
                    c3.metric("Mean absolute error",f"{np.mean(np.abs(valid['Error vs Mean'])):.2f}")
                st.download_button("Download graded validation",graded.to_csv(index=False),"simulation_validation_graded.csv","text/csv")
            else: st.error("Actual-results CSV must contain Name and Actual columns.")

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
