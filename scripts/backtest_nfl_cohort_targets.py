"""Full-Sunday paired NFL WR same-team co-observation forecasting A/B.

Historical NFL weeks previously studied by Team AI² Pro; diagnostic ONLY.
Synthetic historical DK eligible pools are NOT official day-of-game rosters.
Both branches use equal seeds, same games, identical player identities, and
past-only input; every result is graded only AFTER its forecast was generated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from props_data import fetch_weekly
from ai2_nfl_snap_roles import fetch_snap_counts
from ai2_nfl_wr_role_audit import _metrics
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE,evaluate
from scripts.backtest_nfl_matchup_simulator import match_results,paired_metrics
from scripts.audit_nfl_wr_target_roles import _paired_neutral_possession


def run_study(weekly,schedules,snaps,*,n_sims=180,games_per_week=None,
              strength=.5,windows=None):
    if not 0<strength<=1:
        raise ValueError("Positive precommitted candidate strength required")
    windows=windows or {
        "2025_reference":(2025,(9,10,11,12)),
        "2026_retrospective":(2026,(4,5,6,7,8,9,10)),
    }
    result={
        "study":"Research-only WR opportunity conditioned on past teammate co-observation",
        "model_strength_precommitted":strength,
        "source_note":"Historical missing stat row != officially OUT; pool != confirmed active",
        "production_approved":False,"independent_holdout":False,
        "same_player_game_cohort":True,"same_seed_per_team":True,
        "n_simulations_per_game":n_sims,"games_per_week":games_per_week,
        "windows":{},
    }
    files={}
    for label,(year,weeks) in windows.items():
        baseline_wr=[];experimental_wr=[];ledger=[]
        baseline,skipped0=evaluate(
            weekly,schedules,snaps,year,weeks,
            games_per_week=games_per_week,n_sims=n_sims,
            isolate_team_rng=True,wr_role_rows=baseline_wr,
            audit_model_name="baseline")
        experiment,skipped1=evaluate(
            weekly,schedules,snaps,year,weeks,
            games_per_week=games_per_week,n_sims=n_sims,
            isolate_team_rng=True,wr_role_rows=experimental_wr,
            cohort_target_strength=strength,cohort_target_rows=ledger,
            audit_model_name="cohort")
        if skipped0!=skipped1:
            raise ValueError("Different skipped games in research A/B: "
                             +str(skipped0[:10])+"; "+str(skipped1[:10]))
        paired=match_results(baseline,experiment)
        matched=_paired_neutral_possession(
            pd.DataFrame(baseline_wr),pd.DataFrame(experimental_wr))[1]
        if paired.empty or matched.empty:
            raise ValueError("No complete matched NFL player/WR games")
        roles={}
        for role in ("WR1","WR2","WR3","WR4+"):
            group=matched[matched["role_neutral"].eq(role)]
            if group.empty:
                roles[role]={"n":0}
                continue
            old=_metrics(group.rename(columns={
                "actual_targets_neutral":"actual_targets",
                "prior_targets_pg_neutral":"prior_targets_pg",
                "predicted_targets_neutral":"predicted_targets",
                "predicted_p10_neutral":"predicted_p10",
                "predicted_p90_neutral":"predicted_p90"}))
            new=_metrics(group.rename(columns={
                "actual_targets_possession":"actual_targets",
                "prior_targets_pg_possession":"prior_targets_pg",
                "predicted_targets_possession":"predicted_targets",
                "predicted_p10_possession":"predicted_p10",
                "predicted_p90_possession":"predicted_p90"}))
            roles[role]={
                "n":len(group),"baseline":old,"candidate":new,
                "target_mae_lift_pct":round(
                    100*(old["sim_mae_targets"]-new["sim_mae_targets"])
                    /max(old["sim_mae_targets"],1e-8),3),
            }
        metrics=paired_metrics(paired)
        by_pos={str(pos):paired_metrics(group)
                for pos,group in paired.groupby("position")}
        report={
            "paired_dk":metrics,"by_position":by_pos,
            "role_target_metrics":roles,
            "matched_cohort_events":len(ledger),
            "changed_player_forecasts":int(sum(
                abs(float(x["delta_targets_pg"]))>1e-9 for x in ledger)),
            "mean_target_change_per_cohort_forecast":(
                float(pd.DataFrame(ledger)["delta_targets_pg"].mean())
                if ledger else None),
            "skipped_games":skipped0[:50],
            "skipped_game_count":len(skipped0),
            "source_game_day_availability_verified":False,
            "release_approved":False,
        }
        result["windows"][label]=report
        files[label+"_paired_dk"]=paired
        files[label+"_paired_wr_targets"]=matched
        files[label+"_cohort_forecast_changes"]=pd.DataFrame(ledger)
        print(label,"player-games",metrics["n"],
              "DK MAE",metrics["baseline_simulator_mae"],"to",
              metrics["matchup_simulator_mae"],"lift%",
              metrics["matchup_mae_improvement_pct"],
              "cohort changes",report["changed_player_forecasts"],
              "skipped games",len(skipped0),flush=True)
        for role in ("WR1","WR2","WR3"):
            rr=roles[role]
            print(role,"n",rr["n"],
                  "target MAE lift%",rr.get("target_mae_lift_pct"),
                  "baseline bias",rr.get("baseline",{}).get("sim_bias_targets"),
                  "candidate bias",rr.get("candidate",{}).get("sim_bias_targets"),
                  flush=True)
    return result,files


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-sims",type=int,default=180)
    p.add_argument("--games-per-week",type=int,default=0,
                   help="0: all Sunday games, else sample N games/week")
    p.add_argument("--strength",type=float,default=.5)
    p.add_argument("--out-dir",default="nfl_cohort_target_ab")
    a=p.parse_args(argv)
    if not 100<=a.n_sims<=10000 or a.games_per_week<0:
        p.error("Invalid research simulation size / games-per-week")
    print("Loading pregame NFL historical source; no target-week training data",flush=True)
    weekly=fetch_weekly([2024,2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    report,frames=run_study(
        weekly,schedules,snaps,n_sims=a.n_sims,
        games_per_week=a.games_per_week or None,strength=a.strength)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    for key,df in frames.items():
        df.to_csv(out/(key+".csv"),index=False)
    (out/"cohort_target_results.json").write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print("RESEARCH ONLY. No release, no lineup generation.",flush=True)


if __name__=="__main__":
    main()
