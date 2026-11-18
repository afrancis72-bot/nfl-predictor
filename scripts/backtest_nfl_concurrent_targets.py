"""Historical NFL paired A/B: contemporaneous team-week attempt target allocation.

Only pregame week data determine player weights and modeled/unmodeled/
unidentified attempt fractions. The comparison is RESEARCH ONLY: previous
2025/2026 games have been used during model development, and the historical
pool is not verified game-day ACTIVE. No production scorer changes by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ai2_nfl_snap_roles import fetch_snap_counts
from props_data import fetch_weekly
from ai2_nfl_wr_role_audit import _metrics
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE,evaluate
from scripts.backtest_nfl_matchup_simulator import match_results,paired_metrics
from scripts.audit_nfl_wr_target_roles import _paired_neutral_possession

KEYS=["season","week","game","team","player"]


def run_study(weekly,schedules,snaps,*,n_sims=160,games_per_week=2,
              windows=None):
    windows=windows or {
        "2025_reference":(2025,(9,10,11,12)),
        "2026_retrospective":(2026,(4,5,6,7,8,9,10)),
    }
    report={
        "study":"Concurrent team-week receiver target allocation vs original",
        "production_approved":False,"true_prospective":False,
        "real_game_day_status_verified":False,
        "source":"nflverse prior team weeks; target week used for grading ONLY",
        "n_simulation_worlds":n_sims,"games_per_week":games_per_week,
        "windows":{},
    }
    files={}
    for label,(season,weeks) in windows.items():
        base_wr=[];experiment_wr=[];base_diag=[];candidate_diag=[]
        base,base_skips=evaluate(
            weekly,schedules,snaps,season,weeks,
            games_per_week=games_per_week,n_sims=n_sims,
            isolate_team_rng=True,wr_role_rows=base_wr,
            concurrent_target_rows=base_diag,
            audit_model_name="original")
        candidate,candidate_skips=evaluate(
            weekly,schedules,snaps,season,weeks,
            games_per_week=games_per_week,n_sims=n_sims,
            isolate_team_rng=True,wr_role_rows=experiment_wr,
            concurrent_target_research=True,
            concurrent_target_rows=candidate_diag,
            audit_model_name="concurrent")
        if base_skips!=candidate_skips:
            raise ValueError(f"Different skipped games in paired research {label}: "
                             f"{base_skips[:8]} vs {candidate_skips[:8]}")
        paired=match_results(base,candidate)
        wr_pair=_paired_neutral_possession(
            pd.DataFrame(base_wr),pd.DataFrame(experiment_wr))[1]
        if paired.empty or wr_pair.empty:
            raise ValueError(f"Empty matched game/player sample for {label}")
        by_role={}
        for role in ("WR1","WR2","WR3","WR4+"):
            r=wr_pair[wr_pair["role_neutral"].eq(role)]
            if r.empty:
                by_role[role]={"n":0}
                continue
            baseline=_metrics(r.rename(columns={
                "actual_targets_neutral":"actual_targets",
                "predicted_targets_neutral":"predicted_targets",
                "prior_targets_pg_neutral":"prior_targets_pg",
                "predicted_p10_neutral":"predicted_p10",
                "predicted_p90_neutral":"predicted_p90",
            }))
            proposed=_metrics(r.rename(columns={
                "actual_targets_possession":"actual_targets",
                "predicted_targets_possession":"predicted_targets",
                "prior_targets_pg_possession":"prior_targets_pg",
                "predicted_p10_possession":"predicted_p10",
                "predicted_p90_possession":"predicted_p90",
            }))
            by_role[role]={
                "n":len(r),"original":baseline,"concurrent":proposed,
                "target_mae_gain_pct":round(
                    100*(baseline["sim_mae_targets"]-proposed["sim_mae_targets"])/
                    max(baseline["sim_mae_targets"],.0001),3),
                "absolute_bias_improved":(
                    abs(proposed["sim_bias_targets"])<
                    abs(baseline["sim_bias_targets"])),
            }
        diag=pd.DataFrame(candidate_diag)
        m=paired_metrics(paired)
        result={
            "paired_dk":m,
            "by_position_dk":{
                str(pos):paired_metrics(r)
                for pos,r in paired.groupby("position")},
            "by_wr_role_targets":by_role,
            "team_games":len(diag),
            "teams_with_concurrent_pregame_profile":int(diag["model_active"].sum()),
            "no_profile_team_games":int((~diag["model_active"]).sum()),
            "mean_modeled_receiver_share":float(diag["modeled_share"].mean()),
            "mean_unmodeled_receiver_share":float(diag["unmodeled_share"].mean()),
            "mean_unidentified_attempt_share":float(diag["no_identified_target_share"].mean()),
            "skipped_game_count":len(base_skips),
            "skipped_game_examples":base_skips[:15],
            "result_is_prospective_release_evidence":False,
        }
        report["windows"][label]=result
        files[label+"_paired_player_dk"]=paired
        files[label+"_paired_wr_targets"]=wr_pair
        files[label+"_concurrent_team_profiles"]=diag
        print(label,"player-games",m["n"],
              "DK MAE gain %",m["matchup_mae_improvement_pct"],
              "team-game profiles",result["teams_with_concurrent_pregame_profile"],
              "/",result["team_games"],flush=True)
        for name in ("WR1","WR2","WR3"):
            x=by_role[name]
            print(name,"count",x["n"],"original bias",
                  x.get("original",{}).get("sim_bias_targets"),
                  "candidate bias",x.get("concurrent",{}).get("sim_bias_targets"),
                  "target MAE lift %",x.get("target_mae_gain_pct"),flush=True)
    return report,files


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims",type=int,default=160)
    ap.add_argument("--games-per-week",type=int,default=2)
    ap.add_argument("--out-dir",default="research_concurrent_targets")
    args=ap.parse_args(argv)
    if not 100<=args.n_sims<=10000:
        ap.error("n-sims must be 100..10000")
    if args.games_per_week<0:
        ap.error("games-per-week must be nonnegative")
    history=fetch_weekly([2024,2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    report,files=run_study(
        history,schedules,snaps,n_sims=args.n_sims,
        games_per_week=args.games_per_week or None)
    dst=Path(args.out_dir);dst.mkdir(parents=True,exist_ok=True)
    for name,frame in files.items():
        frame.to_csv(dst/(name+".csv"),index=False)
    (dst/"validation.json").write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({
        "2025":report["windows"]["2025_reference"]["paired_dk"],
        "2026":report["windows"]["2026_retrospective"]["paired_dk"],
        "production_approved":False,
        "blind_forward_holdout":False,
    },indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
