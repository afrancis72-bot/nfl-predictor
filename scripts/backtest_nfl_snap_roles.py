"""No-lookahead NFL offensive snap / opportunity ablation.

Uses 2025 weeks 8-18 to choose the role blend weight, then tests independent
2026 weeks 4-8. Metrics: observed RB carries and QB/RB/WR/TE targets per game.
These are opportunity checks, NOT proof of DraftKings mean or contest cash ROI.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from ai2_nfl_snap_roles import fetch_snap_counts,predict_snap_roles
from ai2_nfl_team_environment import predict_team_environment
from ai2_nfl_auto_baselines import _norm_name
from ai2_nfl_matchups import canonical_team
from props_data import fetch_weekly,clean_weekly,make_rates


def collect(history,snaps,season,weeks):
    cleaned=clean_weekly(history)
    examples=[]
    for week in weeks:
        rates=make_rates(history,season,week)
        env=predict_team_environment(history,season,week)
        snap=predict_snap_roles(history,snaps,season,week,team_environment=env)
        actual=cleaned[(cleaned["season"].eq(season)) & (cleaned["week"].eq(week))].copy()
        actual["_name"]=actual["player_name"].map(_norm_name)
        actual["_team"]=actual["recent_team"].map(canonical_team)
        actual["_pos"]=actual["position"].astype(str).str.upper()
        actual=actual.drop_duplicates(subset=["_name","_team","_pos"],keep=False)
        actual_idx={(r["_name"],r["_team"],r["_pos"]):r
                    for r in actual.to_dict("records")}
        for r in rates.to_dict("records"):
            k=(_norm_name(r["player"]),canonical_team(r["team"]),str(r["position"]).upper())
            p=snap.get(k)
            truth=actual_idx.get(k)
            if not p or truth is None:
                continue
            for role,field,baseline in (
                ("targets","targets","targets_pg"),
                ("carries","carries","carries_pg")):
                if role=="carries" and k[2] not in ("RB",):
                    continue
                before=r.get(baseline)
                estimate=p.get("expected_"+role)
                outcome=truth.get(field)
                try:
                    numbers=np.array([before,estimate,outcome],dtype=float)
                except (ValueError,TypeError):
                    continue
                if not np.isfinite(numbers).all() or numbers[0]<.01 or numbers[2]<0:
                    continue
                examples.append(dict(season=season,week=week,role=role,
                                     baseline=float(numbers[0]),snap=float(numbers[1]),
                                     actual=float(numbers[2])))
    return pd.DataFrame(examples)


def score(examples,weight):
    if examples.empty:
        return {"observations":0,"mae":None}
    a=examples["baseline"].to_numpy(float)
    b=examples["snap"].to_numpy(float)
    fact=examples["actual"].to_numpy(float)
    predictions=np.clip(a*(1-weight)+b*weight,a*.75,a*1.25)
    return {"observations":len(examples),
            "mae":round(float(np.abs(predictions-fact).mean()),5),
            "rmse":round(float(np.sqrt(np.mean((predictions-fact)**2))),5)}


def audit(history,snaps):
    train=collect(history,snaps,2025,range(8,19))
    holdout=collect(history,snaps,2026,range(4,9))
    if train.empty or holdout.empty:
        raise ValueError("No full pregame snap-opportunity holdout sample")
    weights=[0.,.15,.25,.35,.50,.75,1.]
    train_results=[{"weight":w,**score(train,w)} for w in weights]
    winner=min(train_results,key=lambda x:(x["mae"],x["weight"]))["weight"]
    return {"training_period":"2025 regular season weeks 8-18",
            "heldout_period":"2026 regular season weeks 4-8",
            "training":train_results,"chosen_weight_pre_2026":winner,
            "holdout":[{"weight":w,**score(holdout,w)} for w in (0.,winner,.35,1.)],
            "holdout_by_role":[{"role":role,
                "history_only":score(g,0.),
                "snap_enhanced":score(g,winner)}
                for role,g in holdout.groupby("role")],
            "production_changes_tested":False,
            "caution":"Opportunities only, not simulated DK points, GPP ownership, or contest cash rates."}


if __name__=="__main__":
    history=fetch_weekly([2024,2025,2026])
    snaps=fetch_snap_counts([2025,2026])
    print(json.dumps(audit(history,snaps),indent=2))
