"""Research A/B: route passing-yard reconciliation to OTHER before uniform WR boosts.

The simulated pregame game and all RNG draws are the same. 0.70 strength is
already present as a bounded optional implementation; this validates it,
rather than tuning new constants to previously inspected target-game outcomes.

ALL 2025/2026 comparisons here are exploratory, NOT independent holdouts.
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
from scripts.audit_nfl_team_pass_prior_ab import (
    WINDOWS, PLAYER_KEYS, exact_paired, errors, by_groups, bootstrap_game_ci,
)

STAT_KEYS=PLAYER_KEYS+["stat"]

def paired_results(score_a,score_b,stat_a,stat_b):
    """A paired test refusing any changed observed box scores or eligibility."""
    p=exact_paired(score_a,score_b,PLAYER_KEYS,
                   ["actual_partial_dk","history_mean_partial_dk","opponent"])
    s=exact_paired(stat_a,stat_b,STAT_KEYS,
                   ["actual","prior_baseline","category","direct_dk_weight"])
    out={
       "player":errors(p,"actual_partial_dk_base",
                       "simulated_mean_dk_base","simulated_mean_dk_candidate"),
       "by_position":by_groups(p,"position","actual_partial_dk_base",
                       "simulated_mean_dk_base","simulated_mean_dk_candidate"),
       "player_game_cluster_ci":bootstrap_game_ci(
                       p,"actual_partial_dk_base",
                       "simulated_mean_dk_base","simulated_mean_dk_candidate"),
       "wr":errors(p.loc[p.position.eq("WR")],"actual_partial_dk_base",
                       "simulated_mean_dk_base","simulated_mean_dk_candidate"),
       "stat_by_position":{},
       "player_p90_coverage_pct":{
          "baseline":round(float(100*(p.actual_partial_dk_base<=p.simulated_p90_dk_base).mean()),2),
          "candidate":round(float(100*(p.actual_partial_dk_base<=p.simulated_p90_dk_candidate).mean()),2),
       },
       "matched_player_games":int(len(p)),
       "matched_stat_rows":int(len(s)),
       "invariant_stats_all_exact":False,
       "predicted_yards_changed":False,
    }
    for (position,stat),rows in s.groupby(["position","stat"]):
        out["stat_by_position"][position+"."+stat]={
          **errors(rows,"actual_base","predicted_mean_base","predicted_mean_candidate"),
          "historic_baseline_mae":round(float(np.mean(
              np.abs(rows.prior_baseline_base-rows.actual_base))),4),
          "p90_base":round(float(100*(rows.actual_base<=rows.predicted_p90_base).mean()),2),
          "p90_candidate":round(float(100*(rows.actual_base<=rows.predicted_p90_candidate).mean()),2),
        }
    conserved=("targets","receptions","receiving_tds","carries","rushing_tds","rushing_yards")
    inv=s[s.stat.isin(conserved)]
    # Reconciliation must NOT create or destroy opportunity, TDs or carries.
    for col in ("predicted_mean","predicted_p10","predicted_p90"):
        if not np.allclose(inv[col+"_base"],inv[col+"_candidate"],
                           rtol=0,atol=1e-8):
            raise ValueError(f"Reconciliation modified non-yard stat {col}")
    # At least one modeled WR or QB passing-yard result must actually change.
    yr=s[s.stat.isin(("receiving_yards","passing_yards"))]
    changed=not np.allclose(yr.predicted_mean_base,yr.predicted_mean_candidate,
                            rtol=0,atol=1e-8)
    if not changed:
        raise ValueError("OTHER-yards candidate had no simulated effect")
    out["invariant_stats_all_exact"]=True
    out["predicted_yards_changed"]=True
    return out,p,s

def replay(weekly,schedules,snaps,*,n_sims=220,windows=WINDOWS):
    report={
        "title":"Experimental OTHER-first receiving-yard reconciliation A/B",
        "baseline_strength":0.0,
        "candidate_strength":0.7,
        "team_pass_env_weight":0.5,
        "parameter_preexisting_before_study":True,
        "prior_outcomes_inspected":True,
        "independent_holdout":False,
        "production_approved":False,
        "one_click_modified":False,
        "windows":{},
        "limitations":[
            "All 2025/2026 slices are exploratory; need future untouched pregame predictions",
            "Simulated yard corrections bounded by 3-22 yards/catch OTHER and original global scale",
            "Historical eligible reconstructed player pool, no dated salaries/inactives",
            "Expected yard totals are a partial box-score based fantasy projection",
            "220 simulations per game, not final 10k production acceptance",
            "OTHER aggregate is a bookkeeping bucket, not a verified active receiver",
            "P90 scoring coverage also reflects TD/volume calibration not just yard distribution",
        ]
    }
    files={}
    for window,(year,weeks) in windows.items():
        outcomes={}
        for label,strength in (("base",0.),("candidate",.7)):
            audit=[]
            score,failures=evaluate(weekly,schedules,snaps,year,weeks,
                games_per_week=None,n_sims=n_sims,
                other_yard_strength=strength,
                team_pass_env_weight=.5,
                stat_audit_rows=audit,
                audit_model_name=label)
            if failures or score.empty or not audit:
                raise ValueError(f"{window}/{label} failed: {failures[:5]}")
            outcomes[label]={"score":score,"stat":pd.DataFrame(audit)}
        comp,players,stats=paired_results(
            outcomes["base"]["score"],outcomes["candidate"]["score"],
            outcomes["base"]["stat"],outcomes["candidate"]["stat"])
        if comp["matched_player_games"]<150 or comp["matched_stat_rows"]<500:
            raise ValueError("Incomplete comparable historical slate")
        report["windows"][window]=comp
        files[window+"_players"]=players
        files[window+"_stats"]=stats
        print(window,json.dumps(comp,sort_keys=True),flush=True)
    return report,files

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--n-sims",type=int,default=220)
    ap.add_argument("--out-dir",default="nfl_receiver_yard_ab")
    args=ap.parse_args(argv)
    if not 100<=args.n_sims<=10000:
        ap.error("Invalid scenario count")
    weekly=fetch_weekly([2024,2025,2026])
    snaps=fetch_snap_counts([2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    report,files=replay(weekly,schedules,snaps,n_sims=args.n_sims)
    out=Path(args.out_dir);out.mkdir(exist_ok=True,parents=True)
    for key,value in files.items():
        value.to_csv(out/(key+".csv"),index=False)
    (out/"validation.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print("Research A/B finished; NO production release",flush=True)

if __name__=="__main__":
    main()
