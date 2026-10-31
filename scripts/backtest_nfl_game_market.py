"""Test whether published game totals improve historical NFL expected team scoring.

2025 season weeks 5-18 tune one market-weight; 2026 weeks 4-8 hold out.
A completed game's closing line was known prior to kickoff, but the file lacks
historical retrieval timestamps. Scores are read ONLY in the held-out outcome
column, never used in inputs. Team score contains DST/ST points (limitation).
"""
import json
import numpy as np
import pandas as pd
from props_data import fetch_weekly,make_rates
from ai2_nfl_matchups import canonical_team
from ai2_nfl_game_market import fetch_schedule_lines, market_totals_for_slate, market_tds_prior
from dfs_stats_engine import _team_scoring_budget

TRAIN=(2025,range(5,19))
HOLDOUT=(2026,range(4,9))
WEIGHTS=(0.,.25,.5,.75,1.)
FG_PRIOR=1.65


def prior_expected_points(rates,team):
    qb=rates[(rates["team"].map(canonical_team)==team)&
             (rates["position"].astype(str).eq("QB"))].copy()
    if qb.empty:
        return None
    qb["attempts"]=pd.to_numeric(qb["pass_attempts_pg"],errors="coerce")
    qb=qb.sort_values("attempts",ascending=False)
    passing=float(qb.iloc[0]["passing_tds_pg"])
    if not np.isfinite(passing):
        return None
    rush=rates[(rates["team"].map(canonical_team)==team)&
               (rates["position"].astype(str).isin(["RB","WR","TE"]))].copy()
    sum_carries=pd.to_numeric(rush["carries_pg"],errors="coerce").clip(lower=0).sum()
    team_rush=float(np.clip(sum_carries+float(qb.iloc[0].get("carries_pg",0)),12.,42.))
    rush_td_prior=max(.3,team_rush*.035)
    tds,fgs=_team_scoring_budget(passing,rush_td_prior)
    return float(tds),float(fgs)


def assemble(history,game_schedule,season,weeks):
    records=[]
    for week in weeks:
        rates=make_rates(history,season,week)
        matched=game_schedule[(game_schedule["season"].eq(season))&
                              (game_schedule["week"].eq(week))&
                              (game_schedule["game_type"].astype(str).eq("REG"))]
        market=market_totals_for_slate(matched,season,week)
        for _,game in matched.iterrows():
            away,home=canonical_team(game["away_team"]),canonical_team(game["home_team"])
            pre=market.get(tuple(sorted([away,home])))
            if not pre:
                continue
            for team,score_col in ((home,"home_score"),(away,"away_score")):
                try:
                    actual=float(game[score_col])
                except (ValueError,TypeError):
                    continue
                if not np.isfinite(actual):
                    continue
                estimate=prior_expected_points(rates,team)
                if estimate is None:
                    continue
                tds,fg=estimate
                records.append({
                    "season":season,"week":week,"team":team,
                    "historical_tds":tds,"fg_prior":fg,
                    "market_total":float(pre["total_points"]),
                    "actual_team_points":actual})
    return pd.DataFrame(records)


def metric(frame,weight):
    estimated=np.array([
        7.*market_tds_prior(r["historical_tds"],r["fg_prior"],r["market_total"],weight=weight)
        +3.*r["fg_prior"] for r in frame.to_dict("records")])
    actual=frame["actual_team_points"].to_numpy(float)
    return {"n":len(frame),"mae":round(float(np.abs(estimated-actual).mean()),4),
            "bias":round(float((estimated-actual).mean()),4),
            "rmse":round(float(np.sqrt(((estimated-actual)**2).mean())),4)}


def audit(history,schedules):
    train=assemble(history,schedules,*TRAIN)
    heldout=assemble(history,schedules,*HOLDOUT)
    if len(train)<30 or len(heldout)<15:
        raise ValueError(f"Not enough sourced team-score records for market holdout: train={len(train)}, test={len(heldout)}")
    train_scores=[{"weight":w,**metric(train,w)} for w in WEIGHTS]
    chosen=min(train_scores,key=lambda x:(x["mae"],x["weight"]))["weight"]
    test_scores=[{"weight":w,**metric(heldout,w)} for w in (0.,chosen,.5,1.)]
    return {
        "train":"2025 weeks 5-18, all published REG games with observed prior player stats",
        "holdout":"2026 weeks 4-8; no setting selected based on 2026",
        "chosen_weight_pre_2026":chosen,
        "training":train_scores,"holdout_scores":test_scores,
        "caution":"Observed team points include special teams and defensive touchdowns; market lines are closing lines, not timestamped quotes. Offense TD priors are approximations. Shadow benchmark only."
    }


def run():
    history=fetch_weekly([2024,2025,2026])
    schedule=fetch_schedule_lines()
    return audit(history,schedule)


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
