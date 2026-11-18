"""Single controlled STOP/GO evidence-weighted baseline study for Bentley.

This research makes only a partial-DK MEAN forecast correction. It neither
repairs targets/possessions nor calibrates P90 tails. Candidate modes/weights
are frozen in ai2_nfl_evidence_baseline.py; select from 2025 W9-12 only.
2026 W4-10 remains retrospective, NOT unseen. No live app integration.
STOP the entire blend approach if the predefined acceptance tests fail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai2_nfl_evidence_baseline import MODES,KEYS,evidence_blend
from ai2_nfl_snap_roles import fetch_snap_counts
from props_data import fetch_weekly
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE,evaluate
from scripts.audit_nfl_full_slates import game_cluster_lift_ci

MIN_MEAN_DK_GAIN_PCT=2.
MAX_POSITION_REGRESSION_PCT=2.
EXPECTED_ROLES=("QB","RB","WR","TE")
WINDOWS={
    "2025_training":(2025,(9,10,11,12)),
    "2026_retrospective":(2026,(4,5,6,7,8,9,10)),
}


def _mae(y,x):
    return float(np.mean(np.abs(np.asarray(y,float)-np.asarray(x,float))))


def grade(frame,pred_col,comparator_col):
    """Paired actual same-player/game MAE with game cluster CI."""
    if frame.empty:
        raise ValueError("Cannot grade an empty NFL cohort")
    x=frame.copy()
    if x.duplicated(list(KEYS)).any():
        raise ValueError("Duplicate player-game in forecast grade")
    y=x["actual_partial_dk"].to_numpy(float)
    predicted=x[pred_col].to_numpy(float)
    baseline=x[comparator_col].to_numpy(float)
    if not all(np.isfinite(v).all() for v in (y,predicted,baseline)):
        raise ValueError("Missing or non-finite actual and paired forecasts")
    mae=_mae(y,predicted); b=_mae(y,baseline)
    by_role={}
    for role,g in x.groupby("position"):
        aa=_mae(g["actual_partial_dk"],g[pred_col])
        bb=_mae(g["actual_partial_dk"],g[comparator_col])
        by_role[str(role)]={
            "n":len(g),"candidate_mae":round(aa,5),
            "comparator_mae":round(bb,5),
            "lift_pct":round(100*(bb-aa)/bb,3) if bb>0 else None,
        }
    bootstrap=x[[*KEYS,"actual_partial_dk",
                 pred_col,comparator_col]].copy()
    bootstrap=bootstrap.rename(columns={
        pred_col:"simulated_mean_dk",
        comparator_col:"history_mean_partial_dk",
    })
    ci=game_cluster_lift_ci(bootstrap)
    return {
        "n":len(x),
        "game_count":ci["games"],
        "candidate_mae":round(mae,5),
        "comparator_mae":round(b,5),
        "lift_pct":round(100*(b-mae)/b,3) if b>0 else None,
        "game_cluster_95_ci_pct":ci["lift_pct_95_ci"],
        "by_position":by_role,
        "candidate_p90_calibrated":False,
    }


def stop_rule(grade_result):
    return (
        grade_result["n"]>=400
        and grade_result["game_count"]>=20
        and grade_result["lift_pct"] is not None
        and grade_result["lift_pct"]>=MIN_MEAN_DK_GAIN_PCT
        and all(
            grade_result["by_position"].get(pos,{}).get("n",0)>=25
            and grade_result["by_position"][pos]["lift_pct"] is not None
            and grade_result["by_position"][pos]["lift_pct"]>=-MAX_POSITION_REGRESSION_PCT
            for pos in EXPECTED_ROLES
        )
    )


def one_window(history,schedules,snaps,season,weeks,n_sims):
    provenance=[]
    pred,failures=evaluate(
        history,schedules,snaps,season,weeks,n_sims=n_sims,
        games_per_week=None,defense_weight=0.,
        pregame_evidence_rows=provenance)
    if pred.empty:
        raise ValueError("Empty historical NFL mean forecast")
    if failures:
        raise ValueError(f"Historical NFL game coverage incomplete {season}: {failures[:8]}")
    if not provenance:
        raise ValueError("No pregame player-history evidence returned")
    evidence=pd.DataFrame(provenance)
    frames={mode:evidence_blend(pred,evidence,mode) for mode in MODES}
    return pred,frames


def run_study(history,schedules,snaps,*,n_sims=250):
    data={}
    for label,(season,weeks) in WINDOWS.items():
        df,variants=one_window(history,schedules,snaps,season,weeks,n_sims)
        data[label]=(df,variants)
    training=data["2025_training"][0]
    hist_train=_mae(training["actual_partial_dk"],
                    training["history_mean_partial_dk"])
    sim_train=_mae(training["actual_partial_dk"],
                   training["simulated_mean_dk"])
    champion=("history_mean_partial_dk" if hist_train<=sim_train
              else "simulated_mean_dk")
    report={
        "study":"One bounded study: pregame evidence-weighted partial DK mean",
        "weights_precommitted_in_code":list(MODES),
        "training_champion_standalone":champion,
        "training_hist_mae":round(hist_train,5),
        "training_sim_mae":round(sim_train,5),
        "min_mean_gain_pct":MIN_MEAN_DK_GAIN_PCT,
        "max_position_regression_pct":MAX_POSITION_REGRESSION_PCT,
        "stop_if_no_2025_qualification":True,
        "p90_calibration_not_improved_or_tested":True,
        "production_enabled":False,
        "independent_unseen_holdout":False,
        "complete_historical_sunday_games":True,
        "windows":{},
    }
    artifacts={}
    for window,(base,variants) in data.items():
        period={}
        oldp=_mae(base["actual_partial_dk"],base["simulated_mean_dk"])
        histp=_mae(base["actual_partial_dk"],base["history_mean_partial_dk"])
        for mode,f in variants.items():
            metric=grade(f,"blended_mean_partial_dk",champion)
            period[mode]={
                **metric,
                "vs_simulator":round(100*(oldp-metric["candidate_mae"])/oldp,3),
                "vs_historical":round(100*(histp-metric["candidate_mae"])/histp,3),
                "mean_history_weight":round(float(f["historical_weight"].mean()),3),
                "passes_mean_gate":stop_rule(metric),
            }
            if window=="2026_retrospective":
                artifacts[f"{window}_{mode}"]=f
        period["history_only_mae"]=round(histp,5)
        period["simulator_only_mae"]=round(oldp,5)
        report["windows"][window]=period
        print(window,"history",round(histp,4),"sim",round(oldp,4),flush=True)
        for mode in MODES:
            item=period[mode]
            print(mode,"MAE",item["candidate_mae"],
                  "gain vs better 2025 standalone%",item["lift_pct"],
                  "role guard",item["passes_mean_gate"],flush=True)
    train_results=report["windows"]["2025_training"]
    qualifiers=[m for m in MODES if train_results[m]["passes_mean_gate"]]
    chosen=min(qualifiers,key=lambda m:(
        train_results[m]["candidate_mae"],MODES.index(m))) if qualifiers else None
    report["selected_2025_model"]=chosen
    hold=report["windows"]["2026_retrospective"]
    selected_result=hold.get(chosen) if chosen is not None else None
    if selected_result is not None:
        ci=selected_result["game_cluster_95_ci_pct"]
        holdout_confirmation=bool(stop_rule(selected_result) and
                                  ci and ci[0]>0)
    else:
        holdout_confirmation=False
    report["2026_historical_confirmation"]=holdout_confirmation
    # No selection means stop the blending line of work, even if a 2026
    # candidate looks good in hindsight. No forward blind proof exists.
    report["development_decision"]=(
        "STOP_BASELINE_BLEND_RABBIT_HOLE"
        if not holdout_confirmation else
        "RESEARCH_CONTINUE_ONLY_NEEDS_FROZEN_PROSPECTIVE_TEST")
    report["production_approved"]=False
    report["classics_release_approved"]=False
    return report,artifacts


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-sims",type=int,default=250)
    p.add_argument("--out-dir",default="nfl_evidence_baseline_study")
    args=p.parse_args(argv)
    if args.n_sims<100 or args.n_sims>10000:
        p.error("n-sims must be 100..10000")
    print("Freeze study candidates; load historical NFL player/snap data",flush=True)
    weekly=fetch_weekly([2024,2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    report,artifacts=run_study(weekly,schedules,snaps,n_sims=args.n_sims)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    for label,df in artifacts.items():
        df.to_csv(out/f"{label}.csv",index=False)
    (out/"results.json").write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({
        "training_champion":report["training_champion_standalone"],
        "2025_selected":report["selected_2025_model"],
        "retrospective_2026_confirmation":report["2026_historical_confirmation"],
        "decision":report["development_decision"],
        "production_approved":False,
    },indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
