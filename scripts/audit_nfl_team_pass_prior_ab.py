"""Research A/B of pregame team pass forecast blend: 50/50 vs 100% observed team.

No production promotion, no target-game outcomes used by the simulation.
The candidate value (1.0) is selected after examining historical periods:
all windows are exploratory, including the supplemental 2025 weeks 4-8.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai2_nfl_snap_roles import fetch_snap_counts
from props_data import fetch_weekly
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE,evaluate

PLAYER_KEYS=["season","week","game","team","player","position"]
WR_KEYS=["season","week","game","team","player"]
TEAM_KEYS=["season","week","game","team"]
WINDOWS={
    "2025_early_diagnostic":(2025,[4,5,6,7,8]),
    "2025_reference":(2025,[9,10,11,12]),
    "2026_retrospective":(2026,[4,5,6,7,8,9,10]),
}


def exact_paired(original,candidate,keys,identical_cols):
    a=pd.DataFrame(original)
    b=pd.DataFrame(candidate)
    if a.empty or b.empty:
        raise ValueError("Empty baseline or candidate historical observations")
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError("Duplicate player/team game in paired audit")
    joined=a.merge(b,on=keys,how="outer",validate="one_to_one",
                   suffixes=("_base","_candidate"),indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Baseline and candidate use different player/team games")
    for col in identical_cols:
        x,y=joined[col+"_base"],joined[col+"_candidate"]
        if pd.api.types.is_numeric_dtype(x):
            if not np.allclose(x.to_numpy(float),y.to_numpy(float),
                               rtol=0,atol=1e-8,equal_nan=True):
                raise ValueError("Retrospective ground truth/baseline mismatch: "+col)
        elif not x.fillna("<MISSING>").eq(y.fillna("<MISSING>")).all():
            raise ValueError("Retrospective label mismatch: "+col)
    return joined.drop(columns="_merge")


def errors(a,actual_col,baseline_col,candidate_col):
    truth=a[actual_col].to_numpy(float)
    v0=a[baseline_col].to_numpy(float)
    v1=a[candidate_col].to_numpy(float)
    if not all(np.isfinite(z).all() for z in (truth,v0,v1)):
        raise ValueError("Nonfinite forecast/truth in comparison")
    e0=v0-truth
    e1=v1-truth
    return {
        "n":len(a),
        "baseline_mae":round(float(np.mean(np.abs(e0))),4),
        "candidate_mae":round(float(np.mean(np.abs(e1))),4),
        "candidate_mae_lift_pct":round(float(100*(np.mean(np.abs(e0))-
            np.mean(np.abs(e1)))/max(np.mean(np.abs(e0)),1e-7)),3),
        "baseline_bias":round(float(e0.mean()),4),
        "candidate_bias":round(float(e1.mean()),4),
        "candidate_better_obs":int(np.sum(np.abs(e1)<np.abs(e0))),
        "candidate_worse_obs":int(np.sum(np.abs(e1)>np.abs(e0))),
    }


def by_groups(df,group_by,actual,baseline,candidate):
    groups={}
    for key,group in df.groupby(group_by):
        groups[str(key)]=errors(group,actual,baseline,candidate)
    return groups


def bootstrap_game_ci(df,actual,base,candidate,*,draws=2500,seed=20261117):
    """Paired game-cluster bootstrap; positive = candidate improves absolute error."""
    if len(df)<30:
        raise ValueError("Not enough matchable player-games")
    x=df.copy()
    x["__cluster"]=x["season"].astype(str)+":"+x["week"].astype(str)+":"+x["game"].astype(str)
    g=x.groupby("__cluster",sort=True)
    per_game=np.stack([
        g.apply(lambda f: pd.Series({
            "base":(f[base]-f[actual]).abs().sum(),
            "candidate":(f[candidate]-f[actual]).abs().sum(),
            "count":len(f),
        }),include_groups=False)[k].to_numpy(float)
        for k in ("base","candidate","count")],axis=1)
    rng=np.random.default_rng(seed)
    n=len(per_game)
    selected=rng.integers(0,n,size=(draws,n))
    summed=per_game[selected].sum(axis=1)
    lift=100*(summed[:,0]-summed[:,1])/np.maximum(summed[:,0],1e-9)
    q=np.percentile(lift,[2.5,97.5])
    return {"n_game_clusters":int(n),"delta_mae_pct_95ci":
            [round(float(q[0]),3),round(float(q[1]),3)]}


def compare_window(scores,wr,team):
    pc=exact_paired(scores["base"],scores["candidate"],PLAYER_KEYS,
                    ["actual_partial_dk","history_mean_partial_dk","opponent"])
    wc=exact_paired(wr["base"],wr["candidate"],WR_KEYS,
                    ["actual_targets","role","rank","prior_targets_pg","opponent"])
    tc=exact_paired(team["base"],team["candidate"],TEAM_KEYS,
                    ["actual_pass_attempts","historical_team_pass_baseline","passing_qb"])
    output={
        "player_partial_dk":errors(pc,"actual_partial_dk_base",
                 "simulated_mean_dk_base","simulated_mean_dk_candidate"),
        "player_by_position":by_groups(pc,"position","actual_partial_dk_base",
                 "simulated_mean_dk_base","simulated_mean_dk_candidate"),
        "player_cluster_bootstrap":bootstrap_game_ci(pc,"actual_partial_dk_base",
                 "simulated_mean_dk_base","simulated_mean_dk_candidate"),
        "wr_targets":errors(wc,"actual_targets_base",
                 "predicted_targets_base","predicted_targets_candidate"),
        "wr_by_role":by_groups(wc,"role_base","actual_targets_base",
                 "predicted_targets_base","predicted_targets_candidate"),
        "team_pass_attempts":errors(tc,"actual_pass_attempts_base",
                 "model_pass_attempts_base","model_pass_attempts_candidate"),
        "team_attempts_vs_pregame_history_baseline":errors(tc,"actual_pass_attempts_base",
                 "historical_team_pass_baseline_base","model_pass_attempts_candidate"),
        "mean_p90_coverage":{
            "baseline_pct":round(float(100*(pc["actual_partial_dk_base"]<=
                       pc["simulated_p90_dk_base"]).mean()),2),
            "candidate_pct":round(float(100*(pc["actual_partial_dk_base"]<=
                       pc["simulated_p90_dk_candidate"]).mean()),2),
        },
        "wr_p90_coverage":{
            "baseline_pct":round(float(100*(wc["actual_targets_base"]<=
                    wc["predicted_p90_base"]).mean()),2),
            "candidate_pct":round(float(100*(wc["actual_targets_base"]<=
                    wc["predicted_p90_candidate"]).mean()),2),
        },
        "matched_player_games":len(pc),"matched_wr_games":len(wc),
        "matched_team_games":len(tc),
    }
    return output,pc,wc,tc


def replay(weekly,schedules,snaps,*,windows=WINDOWS,n_sims=220):
    result={
        "title":"Pregame team passing-attempt forecast weighting sensitivity",
        "default_weight":0.5,
        "candidate_weight":1.0,
        "candidate_selection_was_informed_by_inspected_history":True,
        "independent_holdout":False,"production_approved":False,
        "one_click_modified":False,"no_injury_status_snapshot":True,
        "windows":{},
        "limitations":[
            "2025 and 2026 windows are exploratory: candidate chosen after inspecting them",
            "Future frozen pre-kickoff projections required for independent validation",
            "Historical synthetic DK player pool lacks dated salaries or official inactives",
            "Partial fantasy points exclude some DK scoring events",
            "Target shares and OTHER budgets automatically change with team pass forecast",
            "Game bootstrap reflects sampled games, not future predictive certainty",
            "220 scenarios/game versus eventual 10000-scenario production requirement",
        ],
    }
    files={}
    for window,(season,weeks) in windows.items():
        runs={}
        for label,weight in (("base",.5),("candidate",1.0)):
            roles=[];team=[]
            scores,failures=evaluate(
                weekly,schedules,snaps,season,weeks,
                games_per_week=None,n_sims=n_sims,
                team_pass_env_weight=weight,
                wr_role_rows=roles,team_pass_rows=team)
            if failures:
                raise ValueError(f"{window}/{label}: {len(failures)} game failures: {failures[:5]}")
            runs[label]={"scores":scores,"roles":pd.DataFrame(roles),
                         "team":pd.DataFrame(team)}
        comparison,pc,wc,tc=compare_window(
            {key:run["scores"] for key,run in runs.items()},
            {key:run["roles"] for key,run in runs.items()},
            {key:run["team"] for key,run in runs.items()})
        if comparison["matched_team_games"]<30:
            raise ValueError(f"Incomplete full-slate retrospective for {window}")
        result["windows"][window]=comparison
        files[window+"_player_pair"]=pc
        files[window+"_wr_pair"]=wc
        files[window+"_team_pair"]=tc
        print(window,json.dumps(comparison,sort_keys=True),flush=True)
    return result,files


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--n-sims",type=int,default=220)
    ap.add_argument("--out-dir",default="nfl_team_pass_prior_ab")
    a=ap.parse_args(argv)
    if not 100<=a.n_sims<=10000:
        ap.error("Simulation count must be 100..10000")
    data=fetch_weekly([2024,2025,2026])
    snaps=fetch_snap_counts([2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    result,files=replay(data,schedules,snaps,n_sims=a.n_sims)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    for key,frame in files.items():
        frame.to_csv(out/(key+".csv"),index=False)
    (out/"ab_validation.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print("RESEARCH A/B COMPLETE; NO PRODUCTION PROMOTION",flush=True)

if __name__=="__main__":
    main()
