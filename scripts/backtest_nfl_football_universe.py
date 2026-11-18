"""Retrospective A/B of pregame historical WR teammates in the football world.

Not current-week active status verification: never use this study to unlock
lineups. Every non-DK teammate remains INTERNAL and unselectable. Grades are
made only after predictions using target-week nflverse outcomes, with paired
player/game identity and the same per-team RNG seed in both model variants.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from props_data import fetch_weekly
from ai2_nfl_snap_roles import fetch_snap_counts
from ai2_nfl_wr_role_audit import _metrics
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE, evaluate
from scripts.backtest_nfl_matchup_simulator import match_results, paired_metrics
from scripts.audit_nfl_wr_target_roles import _paired_neutral_possession

MODES=("wr_only","all")


def run_study(history, schedules, snaps, *, n_sims=180,
              games_per_week=2, windows=None):
    windows=windows or {
        "2025_training":(2025,(9,10,11,12)),
        "2026_retro":(2026,(4,5,6,7,8,9,10)),
    }
    report={
        "study":"Historical non-DK teammate inclusion: internal football roles",
        "model_changes_production":False,
        "prospective":False,
        "status_proven_active":False,
        "source":"nflverse prior-week individual stats; target-game observations grade only",
        "same_team_seed_isolated":True,
        "n_sims_per_game":n_sims,
        "games_per_week":games_per_week,
        "windows":{},
        "acceptance":"RESEARCH_ONLY_NOT_RELEASE_PROOF",
    }
    output={}
    for label,(year,weeks) in windows.items():
        base_roles=[];base_universe=[]
        old,errors=evaluate(
            history,schedules,snaps,year,weeks,
            games_per_week=games_per_week,n_sims=n_sims,
            wr_role_rows=base_roles,football_universe_rows=base_universe,
            isolate_team_rng=True,audit_model_name="original")
        if old.empty:
            raise ValueError(f"Empty baseline football sample {label}: {errors[:6]}")
        experiments={}
        output[f"{label}_base_wr_targets"]=pd.DataFrame(base_roles)
        output[f"{label}_baseline_universe"]=pd.DataFrame(base_universe)
        for mode in MODES:
            candidate_roles=[];universe=[]
            new,failures=evaluate(
                history,schedules,snaps,year,weeks,
                games_per_week=games_per_week,n_sims=n_sims,
                wr_role_rows=candidate_roles,football_universe_rows=universe,
                isolate_team_rng=True,football_universe_research=mode,
                audit_model_name=mode)
            paired=match_results(old,new)
            by_pos={str(pos):paired_metrics(g)
                    for pos,g in paired.groupby("position")}
            wr_paired=_paired_neutral_possession(
                pd.DataFrame(base_roles),pd.DataFrame(candidate_roles))[1]
            wr_role_comparison={}
            for role in ("WR1","WR2","WR3","WR4+"):
                rows=wr_paired[wr_paired["role_neutral"].eq(role)]
                if rows.empty:
                    wr_role_comparison[role]={"n":0}
                    continue
                wr_role_comparison[role]={
                    "n":len(rows),
                    "original":_metrics(rows.rename(columns={
                        "actual_targets_neutral":"actual_targets",
                        "predicted_targets_neutral":"predicted_targets",
                        "prior_targets_pg_neutral":"prior_targets_pg",
                        "predicted_p10_neutral":"predicted_p10",
                        "predicted_p90_neutral":"predicted_p90",
                    })),
                    "candidate":_metrics(rows.rename(columns={
                        "actual_targets_possession":"actual_targets",
                        "predicted_targets_possession":"predicted_targets",
                        "prior_targets_pg_possession":"prior_targets_pg",
                        "predicted_p10_possession":"predicted_p10",
                        "predicted_p90_possession":"predicted_p90",
                    })),
                }
            info=pd.DataFrame(universe)
            if not info.empty and (info["internal_wr_count"]<0).any():
                raise ValueError("Impossible counted internal receivers")
            metrics=paired_metrics(paired)
            if failures or errors:
                # Complete-game coverage is essential. Report missing games
                # rather than treating the remaining players as validation.
                coverage_complete=False
            else:
                coverage_complete=True
            experiments[mode]={
                "paired_dk":metrics,
                "dk_by_position":by_pos,
                "wr_role_target_metrics":wr_role_comparison,
                "world_games":int(info.shape[0]),
                "mean_included_nondk_receivers":(
                    round(float(info["internal_wr_count"].mean()),3)
                    if not info.empty else None),
                "games_with_added_nondk_receivers":(
                    int(info["internal_wr_count"].gt(0).sum())
                    if not info.empty else 0),
                "skipped_baseline":errors[:12],
                "skipped_candidate":failures[:12],
                "identical_complete_game_coverage":coverage_complete,
                "research_release_approved":False,
            }
            output[f"{label}_{mode}_paired_dk"]=paired
            output[f"{label}_{mode}_paired_wr_targets"]=wr_paired
            output[f"{label}_{mode}_universe"]=info
            print(label,mode,"added WR/game",
                  experiments[mode]["mean_included_nondk_receivers"],
                  "DK lift%",metrics["matchup_mae_improvement_pct"],
                  "WR1 target bias:",wr_role_comparison["WR1"].get("candidate",{}).get("sim_bias_targets"),
                  flush=True)
        report["windows"][label]=experiments
    return report,output


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims",type=int,default=180)
    ap.add_argument("--games-per-week",type=int,default=2,
                    help="2 for quick smoke, 0 for full Sunday slate")
    ap.add_argument("--out-dir",default="nfl_full_universe_ab")
    args=ap.parse_args(argv)
    if not 100<=args.n_sims<=10000:
        ap.error("n-sims must be 100..10000")
    if args.games_per_week<0:
        ap.error("games-per-week must be >=0")
    weekly=fetch_weekly([2024,2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    report,frames=run_study(
        weekly,schedules,snaps,n_sims=args.n_sims,
        games_per_week=args.games_per_week or None)
    dst=Path(args.out_dir);dst.mkdir(parents=True,exist_ok=True)
    for key,frame in frames.items():
        frame.to_csv(dst/f"{key}.csv",index=False)
    (dst/"report.json").write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({
        "source":"historical nflverse; no current-week status validation",
        "research_only":True,
        "2026_summary":{
            mode:{
                "matched":report["windows"]["2026_retro"][mode]["paired_dk"]["n"],
                "lift_pct":report["windows"]["2026_retro"][mode]["paired_dk"]["matchup_mae_improvement_pct"],
                "WR1":report["windows"]["2026_retro"][mode]["wr_role_target_metrics"]["WR1"],
                "WR2":report["windows"]["2026_retro"][mode]["wr_role_target_metrics"]["WR2"],
            }
            for mode in MODES
        },
        "production_release_gate":"LOCKED",
    },indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
