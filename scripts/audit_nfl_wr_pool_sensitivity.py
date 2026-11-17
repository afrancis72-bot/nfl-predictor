"""Retrospective WR roster cutoff sensitivity. No frozen inactive statuses: research only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from props_data import fetch_weekly
from ai2_nfl_snap_roles import fetch_snap_counts
from scripts.backtest_nfl_player_dk_means import evaluate,GAMES_SOURCE
from scripts.audit_nfl_wr_target_roles import WINDOWS,KEYS
from ai2_nfl_wr_role_audit import _metrics,receiver_role_report

def pair_roles(baseline,relaxed):
    a=pd.DataFrame(baseline); b=pd.DataFrame(relaxed)
    if a.empty or b.empty or a.duplicated(KEYS).any() or b.duplicated(KEYS).any():
        raise ValueError("Empty/ambiguous WR comparison")
    join=a.merge(b,on=KEYS,how="outer",indicator=True,suffixes=("_base","_relaxed"),validate="one_to_one")
    if join["_merge"].eq("left_only").any():
        raise ValueError("Relaxed threshold dropped a baseline player")
    common=join[join["_merge"].eq("both")]
    if common.empty: raise ValueError("No common WR cohort")
    for key in ("role","opponent","actual_targets","prior_targets_pg"):
        x=common[key+"_base"]; y=common[key+"_relaxed"]
        if key in ("actual_targets","prior_targets_pg"):
            if not np.allclose(x.to_numpy(float),y.to_numpy(float),atol=1e-8,rtol=0):
                raise ValueError("Inconsistent baseline/relaxed actual or prior targets")
        elif not x.eq(y).all():
            raise ValueError("Inconsistent role or opponent between A/B cohorts")
    roles={}
    for role in ("WR1","WR2","WR3","WR4+"):
        sub=common[common.role_base.eq(role)]
        if sub.empty:
            roles[role]={"n":0};continue
        actual=sub.actual_targets_base.to_numpy(float)
        x=sub.predicted_targets_base.to_numpy(float)
        y=sub.predicted_targets_relaxed.to_numpy(float)
        roles[role]={"n":len(sub),"base_mae":round(float(np.abs(x-actual).mean()),3),
          "relaxed_mae":round(float(np.abs(y-actual).mean()),3),
          "base_bias":round(float((x-actual).mean()),3),
          "relaxed_bias":round(float((y-actual).mean()),3)}
    added=b.merge(a[KEYS],on=KEYS,how="left",indicator=True,validate="one_to_one")
    added=added[added["_merge"].eq("left_only")].drop(columns="_merge")
    return {"common_role_errors":roles,"baseline_matched_wr":len(a),
      "relaxed_matched_wr":len(b),"new_matched_wr":len(added),
      "new_wr_metrics":_metrics(added)},join,added

def run_replay(weekly,schedules,snaps,*,windows=None,n_sims=300):
    report={"research_only":True,"production_approved":False,
        "independent_holdout":False,"windows":{}}
    frames={}
    for label,(season,weeks) in (windows or WINDOWS).items():
        by_threshold={}
        for threshold in (3.5,2.0):
            rows=[];team=[]
            results,failed=evaluate(weekly,schedules,snaps,season,weeks,
                games_per_week=None,n_sims=n_sims,
                wr_min_targets_pg=threshold,wr_role_rows=rows,team_target_rows=team)
            if failed or not rows or not team:
                raise ValueError(f"Invalid replay {label} {threshold}: {failed[:3]}")
            by_threshold[threshold]=(rows,team)
        base,bt=by_threshold[3.5];relaxed,rt=by_threshold[2.0]
        comparison,joined,added=pair_roles(base,relaxed)
        b=receiver_role_report(base,bt); r=receiver_role_report(relaxed,rt)
        if b["sample_team_games"]!=r["sample_team_games"] or b["sample_team_games"]<40:
            raise ValueError("Team-game cohorts differ or sample insufficient")
        comparison.update({"team_games":b["sample_team_games"],
          "base_top3_omitted":b["historical_top3_wr_missing_model_spots"],
          "relaxed_top3_omitted":r["historical_top3_wr_missing_model_spots"],
          "base_other_pct":b["simulated_other_target_bucket_share_pct"],
          "relaxed_other_pct":r["simulated_other_target_bucket_share_pct"],
          "base_wr_per_team":b["eligible_wr_per_team_game_mean"],
          "relaxed_wr_per_team":r["eligible_wr_per_team_game_mean"]})
        if comparison["relaxed_top3_omitted"]>comparison["base_top3_omitted"]:
            raise ValueError("Relaxation cannot omit additional historical top3 WRs")
        report["windows"][label]=comparison
        frames[label+"_paired.csv"]=joined
        frames[label+"_new.csv"]=added
        print(label,json.dumps(comparison,sort_keys=True),flush=True)
    report["limitations"]=[
      "2025 and 2026 previously inspected: no independent holdout",
      "Synthetic historical pool; no frozen pregame injury/active statuses or real DK salaries",
      "Common cohort compares same receiver games, but roster change alters multinomial allocation",
      "No revenue, ownership, lineup ceiling or production gate approval",
      "DO NOT activate production: thresholds tested only on historical observed player games"]
    return report,frames

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n-sims",type=int,default=300)
    ap.add_argument("--out-dir",default="wr_pool_sensitivity")
    args=ap.parse_args()
    if not 100<=args.n_sims<=10000: ap.error("n-sims out of bounds")
    weekly=fetch_weekly([2024,2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    report,frames=run_replay(weekly,schedules,snaps,n_sims=args.n_sims)
    p=Path(args.out_dir);p.mkdir(parents=True,exist_ok=True)
    for name,f in frames.items():f.to_csv(p/name,index=False)
    (p/"wr_pool_sensitivity.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print("Research-only WR pool comparison complete; production unchanged",flush=True)

if __name__=="__main__":
    main()
