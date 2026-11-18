"""Historical A/B for source-backed opponent WR1/WR2/WR3/RB1/TE1 STAT effects.

**Retrospective research only**. Picks a role-effect strength from 2025 weeks
9-12 WITHOUT looking at test results, then scores 2026 weeks 4-8. Full football
simulations share the same seeds, schedule, pregame rates and eligible players;
the candidate differs ONLY by the stat rates altered for the corresponding
pregame same-role defensive profiles.

2026 has informed earlier AI2 investigations, so this is NOT untouched forward
validation. No outcome can unlock the Bentley production release gate directly.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from props_data import fetch_weekly
from ai2_nfl_snap_roles import fetch_snap_counts
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE,evaluate
from scripts.backtest_nfl_matchup_simulator import match_results,paired_metrics

TRAIN=(9,10,11,12)
TEST=(4,5,6,7,8)
STRENGTHS=(0.,.25,.5,1.)


def run_comparison(history,schedules,snaps,*,n_sims=320,
                   games_per_week=2,train_weeks=TRAIN,test_weeks=TEST):
    def run(season,weeks,s):
        rows,errors=evaluate(
            history,schedules,snaps,season,weeks,
            games_per_week=games_per_week,n_sims=n_sims,
            role_defense_strength=s,component_strength=0.,
            defense_weight=0.,game_matchup_strength=0.,
            possession_strength=0.)
        if rows.empty:
            raise ValueError(f"Empty retrospective {season} strength={s}; "
                             f"first errors: {errors[:4]}")
        return rows,errors
    baseline,base_errors=run(2025,train_weeks,0.)
    train={}
    for s in STRENGTHS:
        candidate,errors=(baseline,base_errors) if s==0. else run(
            2025,train_weeks,s)
        matched=match_results(baseline,candidate)
        train[str(s)]={
            "metrics":paired_metrics(matched),
            "skipped_games":len(errors),"errors":errors[:8]}
    base_mae=train["0.0"]["metrics"]["baseline_simulator_mae"]
    qualified=[]
    for s in STRENGTHS:
        if not s:
            continue
        m=train[str(s)]["metrics"]
        improvement=(base_mae-m["matchup_simulator_mae"])/max(base_mae,1.e-9)
        if improvement>=.01:
            qualified.append(s)
    chosen=min(qualified,key=lambda s:(
        train[str(s)]["metrics"]["matchup_simulator_mae"],s)
        ) if qualified else 0.

    original,issues_original=run(2026,test_weeks,0.)
    changed,issues_changed=(original,issues_original) if not chosen else run(
        2026,test_weeks,chosen)
    paired=match_results(original,changed)
    overall=paired_metrics(paired)
    by_position={
        position:paired_metrics(df)
        for position,df in paired.groupby("position")}
    lift_ci=overall["paired_game_cluster_lift_ci"]["lift_pct_95_ci"]
    requirements={
        "training_strength_selected_without_test_results":True,
        "2025_training_1pct_improvement_over_neutral":chosen>0.,
        "2026_retrospective_mae_gain":bool(
            overall["matchup_mae_improvement_pct"] is not None
            and overall["matchup_mae_improvement_pct"]>0.),
        "2026_game_cluster_ci_lower_bound_positive":bool(
            lift_ci is not None and lift_ci[0]>0.),
        "2026_p90_coverage_85_to_95_percent":bool(
            85<=overall["actual_below_p90_pct"]<=95),
        "no_position_mean_mae_regression_worse_than_3pct":all(
            m["matchup_mae_improvement_pct"] is not None and
            m["matchup_mae_improvement_pct"]>=-3.
            for m in by_position.values()),
    }
    return {
        "name":"Leakage-free defense-versus-pregame-role per-stat multiplier A/B",
        "historical_training":"2025 weeks "+",".join(map(str,train_weeks)),
        "retrospective_test":"2026 weeks "+",".join(map(str,test_weeks)),
        "completed_sunday_games_per_week":games_per_week,
        "worlds_per_game":n_sims,
        "2025_training":train,
        "chosen_strength_trained_only_on_2025":chosen,
        "2026_test":overall,
        "2026_position_results":by_position,
        "2026_week_results":{
            str(w):paired_metrics(d) for w,d in paired.groupby("week")},
        "2026_baseline_skipped_games":issues_original[:30],
        "2026_candidate_skipped_games":issues_changed[:30],
        "qualification_flags":requirements,
        "retrospective_stat_engine_pass":all(requirements.values()),
        "production_release_gate": "LOCKED",
        "further_requirements":[
            "Prospective prekickoff-frozen real-slate holdouts not used for tuning",
            "Verified actual weekly transactions, full status and complete receivers",
            "Stat-channel calibration, game-wide conservation and 90pct P90 coverage",
            "All 9 Bentley release checks, full regression and real deployment acceptance"
        ],
    },paired


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims",type=int,default=320)
    parser.add_argument("--games-per-week",type=int,default=2)
    parser.add_argument("--out-dir",default="nfl_role_defense_stat_ab")
    args=parser.parse_args(argv)
    if not 100<=args.n_sims<=10000:
        parser.error("n-sims must be 100..10000")
    if args.games_per_week<1:
        parser.error("games-per-week must be >=1")
    weekly=fetch_weekly([2024,2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    result,paired=run_comparison(
        weekly,schedules,snaps,n_sims=args.n_sims,
        games_per_week=args.games_per_week)
    dest=Path(args.out_dir);dest.mkdir(parents=True,exist_ok=True)
    (dest/"results.json").write_text(json.dumps(result,indent=2,
                                   allow_nan=False),encoding="utf-8")
    paired.to_csv(dest/"paired_player_game_scores.csv",index=False)
    print(json.dumps({
        "2025_training_selected_strength":
            result["chosen_strength_trained_only_on_2025"],
        "2026_retrospective":result["2026_test"],
        "qualifications":result["qualification_flags"],
        "release_gate":"LOCKED","artifact_dir":str(dest)
    },indent=2,allow_nan=False))


if __name__=="__main__":
    main()
