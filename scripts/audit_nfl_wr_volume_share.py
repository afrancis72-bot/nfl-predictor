"""Full Sunday retrospective pass-volume vs WR target-share decomposition.

NO pregame forecasts use the game outcomes. Oracle pass-volume correction is
postgame diagnostic only; not deployable model output. No real historic DK
salaries/inactives, not independent untouched holdout.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from ai2_nfl_pass_volume_share_audit import decompose_receiver_errors,summary
from ai2_nfl_wr_role_audit import receiver_role_report
from ai2_nfl_snap_roles import fetch_snap_counts
from props_data import fetch_weekly
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE,evaluate
from scripts.audit_nfl_wr_target_roles import WINDOWS

def run_replay(weekly,schedules,snaps,*,windows=None,n_sims=300,thresholds=(3.5,2.0)):
    report={"name":"WR target-volume and allocation accounting audit",
            "production_approved":False,"independent_holdout":False,
            "uses_postgame_attempts_only_for_diagnosis":True,
            "windows":{}}
    outputs={}
    for label,(season,weeks) in (windows or WINDOWS).items():
        report["windows"][label]={}
        for threshold in thresholds:
            role_rows=[];team_rows=[];passes=[]
            scores,failures=evaluate(
                weekly,schedules,snaps,season,weeks,
                games_per_week=None,n_sims=n_sims,
                wr_min_targets_pg=threshold,
                wr_role_rows=role_rows,
                team_target_rows=team_rows,team_pass_rows=passes)
            if failures or not role_rows or not passes:
                raise ValueError(f"Replay failed {label} threshold {threshold}: {failures[:5]}")
            wr=decompose_receiver_errors(role_rows,passes)
            team=pd.DataFrame(passes)
            targets=pd.DataFrame(team_rows)
            if team.shape[0]!=targets.shape[0] or team.shape[0]<40:
                raise ValueError("Mismatch in complete team-game sample")
            role=receiver_role_report(role_rows,team_rows)
            group=summary(wr,team)
            group["coverage"]={
                "ranked_matched_wr":len(wr),
                "scored_player_games":len(scores),
                "historical_top3_not_modeled":role["historical_top3_wr_missing_model_spots"],
                "simulated_other_target_share_pct":role["simulated_other_target_bucket_share_pct"],
                "actual_nonmodeled_skill_targets_pct":role["actual_nonmodeled_rb_wr_te_target_share_pct"],
                "sim_other_not_equal_to_actual_nonmodeled":True,
            }
            key="min_"+str(threshold).replace(".","p")
            report["windows"][label][key]=group
            outputs[label+"_"+key+"_receivers"]=wr
            outputs[label+"_"+key+"_team_passes"]=team
            print(label,key,json.dumps(group,sort_keys=True),flush=True)
    report["limitations"]=[
        "Historical 2025/2026 game windows previously inspected, not independent holdout",
        "Synthetic historical DK eligible pool, not actual slate and no frozen game-day inactives",
        "No opponent coverage/route diagnostics or live lineup implications",
        "Oracle volume counterfactual is a postgame decomposition, NOT a pregame projection",
        "Predicted share uses simulated target mean divided by simulated pass-attempt mean",
        "Predicted and actual total team pass attempts include throwaway passes; actual WR target is attributed",
        "Mean component decomposition is an accounting identity, not causal attribution",
        "Simulator uses 300 scenarios per game here, not production 10000-scenario validation",
    ]
    return report,outputs

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--n-sims",type=int,default=300)
    parser.add_argument("--out-dir",default="wr_volume_share_audit")
    args=parser.parse_args()
    if not 100<=args.n_sims<=10000:parser.error("n-sims must be 100..10000")
    weekly=fetch_weekly([2024,2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    report,files=run_replay(weekly,schedules,snaps,n_sims=args.n_sims)
    dest=Path(args.out_dir);dest.mkdir(parents=True,exist_ok=True)
    for name,df in files.items():
        df.to_csv(dest/(name+".csv"),index=False)
    (dest/"wr_volume_share_audit.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print("RESEARCH ONLY: do not promote diagnostic or postgame oracle",flush=True)

if __name__=="__main__":
    main()
