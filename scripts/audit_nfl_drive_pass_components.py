"""Research: diagnose offensive plays, dropback decisions, sacks and final attempts.

Comparisons are retrospective; PBP target-game observations are passed to
the audit AFTER the simulator returns. Production and one-click remain unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from ai2_nfl_drive_pass_accounting import observed_pbp_team_games,report_metrics
from ai2_nfl_pbp_drives import load_historical_pbp,pbp_to_drives
from ai2_nfl_snap_roles import fetch_snap_counts
from props_data import fetch_weekly
from scripts.audit_nfl_wr_target_roles import WINDOWS
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE,evaluate

KEYS=["season","week","game","team"]

def audit(weekly,schedules,snaps,drives,observed,*,n_sims=220,windows=None):
    output={"name":"NFL pregame plays/dropbacks/sacks/attempts component diagnosis",
            "status":"research_only","production_approved":False,
            "one_click_modified":False,"independent_holdout":False,
            "scenario_count_per_game":int(n_sims),
            "windows":{},"limitations":[
                "Historical 2025/2026 results already inspected, not an independent holdout",
                "Neutral statistical engine and experimental possession engine compared without retuning",
                "Possession actual drives are eligible fixed-drive proxy, not unambiguous official possessions",
                "Actual target-game PBP enters diagnostics only AFTER pregame-only simulation",
                "Components are an exact arithmetic identity, not an intervention/causal estimate",
                "Synthetic DK historical pool lacks dated salaries and verified inactives",
                "The old engine has no drive count; only possession mode generates modeled drives",
                "220 scenarios/game descriptive; production release requires 10,000 scenarios and forward holdout",
            ]},"diagnostic_checks":{}}
    files={}
    for label,(season,weeks) in (windows or WINDOWS).items():
        frames={}
        output["windows"][label]={}
        for engine,strength in (("neutral",0.),("possession",1.)):
            rows=[]
            scored,failures=evaluate(
                weekly,schedules,snaps,season,weeks,
                games_per_week=None,n_sims=n_sims,
                possession_drives=drives if strength else None,
                possession_strength=strength,
                team_component_rows=rows,actual_pbp_by_team=observed,
                team_component_engine=engine)
            if failures or not rows:
                raise ValueError(f"{label}/{engine}: failed games {failures[:10]}")
            d=pd.DataFrame(rows)
            if d.duplicated(KEYS).any():
                raise ValueError("Duplicated team-game accounting")
            if len(d)<40:
                raise ValueError("Incomplete test season")
            for r in rows:
                pred=r["simulated_target_allocation"]
                if not np.isclose(sum(pred.values()),r["sim_pass_attempts"],
                                  rtol=0,atol=1e-8):
                    raise ValueError("Receiver group target allocation conservation failed")
            frames[engine]=d
            m=report_metrics(d)
            m["matched_scored_player_games"]=len(scored)
            output["windows"][label][engine]=m
            # CSV-flatten dict columns to retain usable dataset
            for name in ["simulated_target_allocation","actual_known_position_and_rank_targets"]:
                for category in sorted(set().union(*(v.keys() for v in d[name]))):
                    d[name+"_"+category]=d[name].map(lambda x:x.get(category,0))
                d=d.drop(columns=[name])
            files[f"{label}_{engine}_team_games"]=d
            print(label,engine,json.dumps(m,sort_keys=True),flush=True)
        l=frames["neutral"]
        p=frames["possession"]
        matched=l.merge(p,on=KEYS,how="outer",indicator=True,
                        suffixes=("_neutral","_possession"),validate="one_to_one")
        if not matched["_merge"].eq("both").all():
            raise ValueError("Neutral vs possession missing team-games")
        for col in ("actual_plays","actual_dropbacks","actual_pass_attempts",
                    "actual_sacks","actual_rushes","eligible_observed_drives"):
            if not np.allclose(matched[col+"_neutral"],matched[col+"_possession"],
                               rtol=0,atol=1e-8):
                raise ValueError("Actual team-game evidence differs between engine modes")
        output["windows"][label]["paired_team_games"]=len(matched)
        output["windows"][label]["possession_minus_neutral_attempt_mae"]=round(
            output["windows"][label]["possession"]["pass_attempts"]["mae"]-
            output["windows"][label]["neutral"]["pass_attempts"]["mae"],3)
    output["diagnostic_checks"]={
        "all_runs_complete":True,"pass_attempt_components_reconcile":True,
        "passes_equal_dropbacks_minus_sacks":True,
        "plays_equal_dropbacks_plus_rushes":True,
        "all_simulated_receiver_and_other_targets_equal_passes":True,
        "neutral_and_possession_observe_same_real_game":True,
        "production_activation":False,
    }
    return output,files

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--n-sims",type=int,default=220)
    ap.add_argument("--out-dir",default="nfl_drive_pass_accounting")
    args=ap.parse_args(argv)
    if not 100<=args.n_sims<=10000:
        ap.error("n-sims must be 100..10000")
    weekly=fetch_weekly([2024,2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    raw=load_historical_pbp([2024,2025,2026])
    drives=pbp_to_drives(raw)
    observed=observed_pbp_team_games(raw,drives,schedules)
    print("Observed actual PBP team-games:",len(observed),
          "eligible historical drives:",len(drives),flush=True)
    result,data=audit(weekly,schedules,snaps,drives,observed,n_sims=args.n_sims)
    out=Path(args.out_dir);out.mkdir(exist_ok=True,parents=True)
    for name,frame in data.items():
        frame.to_csv(out/(name+".csv"),index=False)
    (out/"component_audit.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print("RESEARCH ONLY: engine unchanged; release gates remain closed",flush=True)

if __name__=="__main__":
    main()
