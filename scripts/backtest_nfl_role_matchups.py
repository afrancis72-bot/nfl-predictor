"""Chronological defense-versus-WR1/WR2/RB1/TE1 pregame evidence benchmark.

Training picks one global score-adjustment weight from completed 2025 weeks
9-12; 2026 weeks 4-8 never choose the weight. Player's *own* prior DK score
is the baseline, and opposing-defense role residual uses ONLY prior contests.
This is a descriptive player-game test, not a full NFL stat-model validation.
"""
import json
import numpy as np
from props_data import fetch_weekly
from ai2_nfl_role_matchups import role_observations, defense_role_profiles

WEIGHTS = (0.0, .25, .5, 1.0)

def weeks_result(history, season, weeks):
    rows=[]
    for wk in weeks:
        profiles,_=defense_role_profiles(history, season, wk)
        lookup={(r["defense"],r["role"]):r for r in profiles.to_dict("records")}
        completed=role_observations(history,season,wk,include_week=True)
        if completed.empty:
            continue
        for row in completed.to_dict("records"):
            if row["season"]!=season or row["week"]!=wk:
                continue
            p=lookup.get((row["defense"],row["role"]))
            rows.append({
                "season":season,"week":wk,"position":row["position"],
                "role":row["role"],"defense":row["defense"],
                "baseline":float(row["baseline_dk"]),
                "actual":float(row["actual_dk"]),
                "role_residual":float(p["shrunk_excess_dk"]) if p else 0.,
                "prior_defense_role_games":int(p["n"]) if p else 0,
            })
    if not rows:
        raise ValueError("No matched historical role-versus-defense player games")
    return rows

def score(rows,weight):
    baseline=np.array([x["baseline"] for x in rows],float)
    effect=np.array([x["role_residual"] for x in rows],float)
    actual=np.array([x["actual"] for x in rows],float)
    error=baseline+float(weight)*effect-actual
    return {
        "n":len(rows),
        "mae":round(float(np.abs(error).mean()),4),
        "rmse":round(float(np.sqrt(np.mean(error**2))),4),
        "bias":round(float(error.mean()),4),
        "evidence_coverage_pct":round(100*sum(
            x["prior_defense_role_games"]>0 for x in rows)/len(rows),1),
    }

def run(history):
    train=weeks_result(history,2025,(9,10,11,12))
    test=weeks_result(history,2026,(4,5,6,7,8))
    t={str(w):score(train,w) for w in WEIGHTS}
    chosen=min(WEIGHTS,key=lambda w:(t[str(w)]["mae"],w))
    h={str(w):score(test,w) for w in WEIGHTS}
    roles={}
    for role in ("QB1","RB1","RB2","WR1","WR2","WR3","TE1","TE2"):
        subset=[x for x in test if x["role"]==role]
        if subset:
            roles[role]={
                "neutral":score(subset,0.),
                "training_choice":score(subset,chosen),
            }
    return {
        "source":"nflverse completed player-week partial DraftKings points",
        "baseline":"Player's own prior eight games with 0.87 recency weighting",
        "role":"Predicted from teammate earlier QB attempts, RB carries+targets or WR/TE targets; no game-week targets",
        "training":"2025 weeks 9-12",
        "holdout":"2026 weeks 4-8, weight selected on 2025 only",
        "training_scores":t,
        "trained_weight":chosen,
        "holdout_scores":h,
        "holdout_mae_improved":h[str(chosen)]["mae"]<h["0.0"]["mae"],
        "role_breakdown":roles,
        "status":"Research only: do not use raw or shrunk role residual in current DFS score without stronger validation.",
    }

if __name__=="__main__":
    print(json.dumps(run(fetch_weekly([2024,2025,2026])),indent=2))
