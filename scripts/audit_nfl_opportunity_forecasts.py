"""Full-Sunday NFL player-stat audit: opportunities, efficiency outputs, TDs.

Builds *the same prior-week player pool* and player identity matches as the
existing neutral football simulator. Captures actual pass attempts, carries,
targets, catches, yardage and TDs independently of fantasy-point scoring.

The 2025 reference and 2026 retrospective weeks are both ALREADY used in
other AI2 research; this is diagnosis rather than untouched validation.
Football-stat MAE is compared with the SAME players' own historical rates
and, optionally, the experimental possession model.

Outputs:
 - neutral_player_stats.csv (every observed player/stat)
 - possession_player_stats.csv (only with --compare-possession)
 - paired_player_stat_errors.csv (only with --compare-possession)
 - opportunity_audit.json (position/stat MAE, P90 and game-cluster intervals)

No production one-click activation. No lineup ROI implied. Reconstructed
player pools lack frozen historical DK salaries and verified injury status.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai2_nfl_opportunity_audit import (
    REQUIRED, opportunity_report, stat_summary,
)
from ai2_nfl_pbp_drives import load_historical_pbp, pbp_to_drives
from ai2_nfl_snap_roles import fetch_snap_counts
from props_data import fetch_weekly
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE, evaluate


KEYS=("season","week","game","player","team","position","stat")
PERIODS={
    "2025_reference": (2025,[9,10,11,12]),
    "2026_retrospective": (2026,[4,5,6,7,8,9,10]),
}


def _distinct_player_games(frame):
    return int(frame[["season","week","game","player","team","position"]].
               drop_duplicates().shape[0])


def _strict_matched_player_games(scores,stat_rows):
    if scores.empty or stat_rows.empty:
        raise ValueError("Cannot audit zero matched player outcomes")
    keys=("season","week","game","player","team","position")
    expected=scores[list(keys)].drop_duplicates()
    observed=stat_rows[list(keys)].drop_duplicates()
    check=expected.merge(observed,how="outer",on=list(keys),
                         indicator=True,validate="one_to_one")
    if not check["_merge"].eq("both").all():
        missing=check.loc[~check["_merge"].eq("both"),list(keys)+["_merge"]]
        raise ValueError("Missing player-level opportunity forecast: "+
                         str(missing.head(6).to_dict("records")))


def compare_stat_rows(neutral,candidate):
    """Fail closed on any missing player/stat, changed actual or prior rates."""
    for frame in (neutral,candidate):
        missing=REQUIRED-set(frame.columns)
        if missing:
            raise ValueError(f"Incomplete stat audit: {sorted(missing)}")
        if frame.duplicated(list(KEYS)).any():
            raise ValueError("Duplicate player-stat identity in comparison")
    merged=neutral.merge(candidate,how="outer",on=list(KEYS),
                         suffixes=("_neutral","_possession"),
                         indicator=True,validate="one_to_one")
    if not merged["_merge"].eq("both").all():
        mismatch=merged.loc[~merged["_merge"].eq("both"),list(KEYS)+["_merge"]]
        raise ValueError("Neutral and possession model stat pools differ: "+
                         str(mismatch.head(5).to_dict("records")))
    for field in ("actual","prior_baseline"):
        a=merged[field+"_neutral"].to_numpy(float)
        b=merged[field+"_possession"].to_numpy(float)
        if not np.allclose(a,b,atol=1e-8,rtol=0):
            raise ValueError("Compared models have different football truth or priors")
    return merged.drop(columns=["_merge"])


def paired_stat_summary(frame, *, resamples=1200, seed=53):
    if frame.empty:
        raise ValueError("No matched football stats for opponent evaluation")
    y=frame["actual_neutral"].to_numpy(float)
    a=frame["predicted_mean_neutral"].to_numpy(float)
    b=frame["predicted_mean_possession"].to_numpy(float)
    err0=np.abs(a-y)
    err1=np.abs(b-y)
    neutral_mae=float(np.mean(err0))
    match_mae=float(np.mean(err1))
    frame=frame.copy()
    frame["_game"]=(frame["season"].astype(str)+"-"+
                    frame["week"].astype(str)+"-"+
                    frame["game"].astype(str))
    frame["_neutral_error"]=err0
    frame["_possession_error"]=err1
    grouped=frame.groupby("_game")[
        ["_neutral_error","_possession_error"]].sum().to_numpy(float)
    ci=None
    if len(grouped)>=5:
        rng=np.random.default_rng(seed)
        index=rng.integers(0,len(grouped),size=(resamples,len(grouped)))
        sums=grouped[index].sum(axis=1)
        delta=100*(sums[:,0]-sums[:,1])/np.maximum(sums[:,0],1e-8)
        ci=[round(float(z),3) for z in np.percentile(delta,[2.5,97.5])]
    return {
        "n":int(len(frame)),
        "games":int(len(grouped)),
        "neutral_mae":round(neutral_mae,4),
        "possession_mae":round(match_mae,4),
        "possession_improvement_over_neutral_pct":(
            round(100*(neutral_mae-match_mae)/neutral_mae,3)
            if neutral_mae else None),
        "paired_game_bootstrap_improvement_95_ci":ci,
        "neutral_p90_below_pct":round(
            100*float(np.mean(y<=frame["predicted_p90_neutral"].to_numpy(float))),2),
        "possession_p90_below_pct":round(
            100*float(np.mean(y<=frame["predicted_p90_possession"].to_numpy(float))),2),
    }


def run_audit(weekly,schedules,snaps,*,drives=None,compare_possession=True,
              n_sims=220,periods=None):
    periods=periods or PERIODS
    results={}
    csv_frames={}
    for label,(season,weeks) in periods.items():
        diagnostics=[]
        score_rows,issues=evaluate(
            weekly,schedules,snaps,season,weeks,games_per_week=None,
            n_sims=n_sims,stat_audit_rows=diagnostics,
            audit_model_name="neutral")
        base=pd.DataFrame(diagnostics)
        _strict_matched_player_games(score_rows,base)
        neutral=opportunity_report(base,skipped_games=len(issues))
        print(label,"neutral matched player-games:",
              _distinct_player_games(base),"stat observations:",len(base),
              "skipped games:",len(issues),flush=True)
        section={
            "season":season,"weeks":weeks,
            "n_simulations_per_game":n_sims,
            "neutral":neutral,
            "fantasy_point_player_game_count":len(score_rows),
            "neutral_fatal_game_errors":issues[:50],
        }
        csv_frames[label+"_neutral"]=base
        if compare_possession:
            if drives is None:
                raise ValueError("Possession comparison requested but no PBP drives supplied")
            contrast=[]
            points,problems=evaluate(
                weekly,schedules,snaps,season,weeks,games_per_week=None,
                n_sims=n_sims,possession_drives=drives,
                possession_strength=1.,stat_audit_rows=contrast,
                audit_model_name="possession")
            possession=pd.DataFrame(contrast)
            _strict_matched_player_games(points,possession)
            paired=compare_stat_rows(base,possession)
            summary={
                f"{role}.{stat}":paired_stat_summary(sub)
                for (role,stat),sub in paired.groupby(["position","stat"])
            }
            section["possession"]=opportunity_report(
                possession,skipped_games=len(problems))
            section["neutral_vs_possession_by_position_stat"]=summary
            section["neutral_vs_possession_all_stats_not_comparable_units"]=(
                paired_stat_summary(paired))
            section["possession_fatal_game_errors"]=problems[:50]
            section["match_consistent_player_stat_keys"]=True
            section["match_consistent_games"]=len(issues)==len(problems)
            csv_frames[label+"_possession"]=possession
            csv_frames[label+"_paired"]=paired
            print(label,"possession matched player-games:",
                  _distinct_player_games(possession),"stats:",len(possession),
                  "skipped games:",len(problems),flush=True)
        results[label]=section
    report={
        "name":"Team AI² NFL player-stat error diagnosis (opportunity vs efficiency vs TD)",
        "windows":results,
        "unit_warning":"Do not average raw per-stat errors across yards/counts/TDs; report positions/statistics separately",
        "research_only":True,
        "production_approved":False,
        "independent_untouched_test":False,
        "limitations":[
            "2025 and 2026 weeks previously used for research and design",
            "Matched active players only; reconstructed prior-week pool may omit late-emerging players",
            "No dated DK salaries or historical inactive snapshots",
            "Week-level historical rates use only earlier weeks but are not frozen exact kickoff-state predictions",
            "Historical P10/P90 coverage for zero-inflated TDs requires care",
            "Full contest ROI and 10,000-simulation app are not reproduced here",
        ],
    }
    return report,csv_frames


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims",type=int,default=220)
    parser.add_argument("--output-dir",default="nfl_opportunity_audit")
    parser.add_argument("--compare-possession",action="store_true")
    args=parser.parse_args(argv)
    if not 100<=args.n_sims<=10000:
        parser.error("Simulations per game must be between 100 and 10000")
    print("Loading historical nflverse player weeks and schedules",flush=True)
    weekly=fetch_weekly([2024,2025,2026])
    snaps=fetch_snap_counts([2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    drives=None
    if args.compare_possession:
        print("Loading prior historical drive data for direct comparison",flush=True)
        drives=pbp_to_drives(load_historical_pbp([2024,2025,2026]))
    report,frames=run_audit(weekly,schedules,snaps,drives=drives,
                            compare_possession=args.compare_possession,
                            n_sims=args.n_sims)
    directory=Path(args.output_dir)
    directory.mkdir(parents=True,exist_ok=True)
    for name,frame in frames.items():
        frame.to_csv(directory/(name+".csv"),index=False)
    (directory/"opportunity_audit.json").write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    last=report["windows"]["2026_retrospective"]
    watch=["QB.pass_attempts","QB.carries","RB.carries","RB.targets",
           "WR.targets","TE.targets","WR.receptions","RB.rushing_tds",
           "WR.receiving_tds"]
    print("2026 diagnostic specific stats compared with historical means:",flush=True)
    for stat in watch:
        m=last["neutral"]["summary_by_position_and_stat"].get(stat)
        print(stat, m,flush=True)
    if args.compare_possession:
        print("2026 neutral versus possession opportunity differences:",flush=True)
        for stat in watch:
            m=last["neutral_vs_possession_by_position_stat"].get(stat)
            print(stat,m,flush=True)
    print("Output:",str(directory),flush=True)
    print("Approved for paid lineup builds:",report["production_approved"],
          flush=True)


if __name__=="__main__":
    main()
