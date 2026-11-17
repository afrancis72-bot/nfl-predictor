"""Full Sunday historical WR1/WR2/WR3 targets and unmodeled OTHER audit.

Rank roles using ONLY prior-week within-team receiving usage across all
recently seen NFL WRs, not target-week catches, future player activity or
postgame depth chart. Keep modeled eligibility separate from rank: missing
WR3 is a coverage problem, not evidence of zero targets.

Run:
 PYTHONPATH=. python -u scripts/audit_nfl_wr_target_roles.py
Outputs role_player_targets.csv, team_target_budget.csv,
wr_role_target_audit.json, and an optional paired experiment.

Data are retrospective with previously inspected 2025/2026 outcomes;
synthetic historical eligible player pools and no frozen pregame availability.
This audit cannot approve or modify paid-production projections.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai2_nfl_snap_roles import fetch_snap_counts
from ai2_nfl_wr_role_audit import receiver_role_report,_metrics
from props_data import fetch_weekly
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE,evaluate

WINDOWS={
    "2025_reference":(2025,[9,10,11,12]),
    "2026_retrospective":(2026,[4,5,6,7,8,9,10]),
}
KEYS=["season","week","game","team","player"]


def _paired_neutral_possession(neutral,candidate):
    if neutral.empty or candidate.empty:
        raise ValueError("Cannot compare role errors with empty modeled WR lists")
    for df in (neutral,candidate):
        if df.duplicated(KEYS).any():
            raise ValueError("Duplicate modeled WR on target week")
    merged=neutral.merge(candidate,on=KEYS,how="outer",
                         suffixes=("_neutral","_possession"),
                         indicator=True,validate="one_to_one")
    if not merged["_merge"].eq("both").all():
        raise ValueError("Different WR cohorts in neutral and possession simulations")
    for col in ("role","opponent","actual_targets","prior_targets_pg"):
        a=merged[f"{col}_neutral"]
        b=merged[f"{col}_possession"]
        if col in ("actual_targets","prior_targets_pg"):
            if not np.allclose(a.to_numpy(float),b.to_numpy(float),atol=1e-8,rtol=0):
                raise ValueError("Different actual/prior target values")
        elif not a.eq(b).all():
            raise ValueError("Different pregame WR role labels or opponent")
    result={}
    for role in ("WR1","WR2","WR3","WR4+"):
        sub=merged[merged["role_neutral"].eq(role)]
        if sub.empty:
            result[role]={"n":0}
            continue
        truth=sub["actual_targets_neutral"].to_numpy(float)
        one=float(np.mean(np.abs(sub["predicted_targets_neutral"]-truth)))
        two=float(np.mean(np.abs(sub["predicted_targets_possession"]-truth)))
        result[role]={
            "n":len(sub),"neutral_mae":round(one,3),
            "possession_mae":round(two,3),
            "possession_mae_lift_pct":round(100*(one-two)/one,3) if one else None,
            "neutral_bias":round(float(np.mean(sub["predicted_targets_neutral"]-truth)),3),
            "possession_bias":round(float(np.mean(sub["predicted_targets_possession"]-truth)),3),
        }
    return result,merged


def run_replay(weekly,schedules,snaps,*,windows=None,
               n_sims=300,compare_possession=False,drives=None):
    windows=windows or WINDOWS
    report={"name":"NFL pregame WR1 WR2 WR3 target and missing player bucket audit",
            "windows":{},"role_definition":"prior average targets ranked among all recent same-team WRs",
            "rank_uses_target_game_results":False,
            "independent_holdout":False,"production_approved":False,
            "historical_player_pool_only":True}
    outputs={}
    for label,(year,weeks) in windows.items():
        role_rows=[]
        team_rows=[]
        scores,failures=evaluate(
            weekly,schedules,snaps,year,weeks,games_per_week=None,
            n_sims=n_sims,wr_role_rows=role_rows,
            team_target_rows=team_rows,
            audit_model_name="neutral")
        if not role_rows or not team_rows:
            raise ValueError(f"No WR role or team budget reports: {label}: {failures[:10]}")
        wr=pd.DataFrame(role_rows)
        team=pd.DataFrame(team_rows)
        measured_wr=scores[scores["position"].eq("WR")][KEYS]
        matched=measured_wr.merge(wr[KEYS],on=KEYS,how="outer",
                                 indicator=True,validate="one_to_one")
        if not matched["_merge"].eq("both").all():
            raise ValueError(f"WR role observations do not cover matched WRs {label}")
        data=receiver_role_report(role_rows,team_rows)
        data.update({
            "year":year,"weeks":weeks,
            "skipped_simulator_games":failures[:30],
            "skipped_simulator_game_count":len(failures),
            "matched_wr_points":len(measured_wr),
            "ranked_wr_target_rows":len(wr),
            "teams_with_full_target_data":int(team["actual_skill_player_missing_targets"].eq(0).sum()),
        })
        outputs[label+"_WR_player_targets"]=wr
        outputs[label+"_team_other_budgets"]=team
        if compare_possession:
            if drives is None:
                raise ValueError("Possession comparison requires historical drives")
            rows2=[]
            games2=[]
            test,problems=evaluate(
                weekly,schedules,snaps,year,weeks,games_per_week=None,
                n_sims=n_sims,wr_role_rows=rows2,team_target_rows=games2,
                possession_drives=drives,possession_strength=1.,
                audit_model_name="possession")
            contrast,paired=_paired_neutral_possession(wr,pd.DataFrame(rows2))
            if len(problems)!=len(failures):
                raise ValueError("Different skipped game count in possession A/B")
            data["paired_neutral_vs_possession_role_targets"]=contrast
            outputs[label+"_paired_wr_targets"]=paired
            outputs[label+"_possession_team_other_budgets"]=pd.DataFrame(games2)
        report["windows"][label]=data
        print(label,"WR rows",len(wr),"team-game rows",len(team),
              "skipped games",len(failures),flush=True)
        for role,m in data["roles"].items():
            print(role,m,flush=True)
        print("TEAM_OTHER",{
            key:data[key] for key in (
                "sample_team_games",
                "rank3_omitted_team_games",
                "historical_top3_wr_missing_model_spots",
                "eligible_wr_per_team_game_mean",
                "teams_with_3_or_more_modeled_wr",
                "actual_nonmodeled_rb_wr_te_target_share_pct",
                "simulated_other_target_bucket_share_pct",
            )},flush=True)
        if compare_possession:
            print("POSSESSION_ROLE_AB",contrast,flush=True)
    report["limitations"]=[
        "2025 and 2026 were analyzed previously; results are not out-of-sample",
        "WR role is ranked from earlier games, not official WR depth chart, slot/outside alignment or coverage",
        "Historical receiver ranking can include a player later known OUT; no frozen historical pregame injury/status source",
        "Only matched modeled WRs receive prediction grades; missing-role roster coverage separately reported",
        "Simulated OTHER includes incompletions/throwaways/unmatched receivers while real nonmodeled player targets exclude unattributed throws; percentages are NOT directly comparable",
        "No actual historical DK salaries, 10,000-sim production replay or contest ROI here",
        "No claim that role errors are caused by opponent matchup without an independent conditioned test",
    ]
    return report,outputs


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims",type=int,default=300)
    ap.add_argument("--out-dir",default="nfl_wr_role_audit")
    ap.add_argument("--compare-possession",action="store_true")
    args=ap.parse_args(argv)
    if not 100<=args.n_sims<=10000:
        ap.error("--n-sims must be 100..10000")
    weekly=fetch_weekly([2024,2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    drives=None
    if args.compare_possession:
        from ai2_nfl_pbp_drives import load_historical_pbp,pbp_to_drives
        drives=pbp_to_drives(load_historical_pbp([2024,2025,2026]))
    report,files=run_replay(
        weekly,schedules,snaps,drives=drives,
        compare_possession=args.compare_possession,n_sims=args.n_sims)
    path=Path(args.out_dir)
    path.mkdir(parents=True,exist_ok=True)
    for label,frame in files.items():
        frame.to_csv(path/(label+".csv"),index=False)
    (path/"wr_role_target_audit.json").write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print("Research diagnostic complete; production approved:",
          report["production_approved"],flush=True)


if __name__=="__main__":
    main()
