"""2026-11-01 historical replay of current unified NFL production engine.

Runs CURRENT ai2_nfl_oneclick (not archived predictions or a new model) with
user's authentic historical 562-row DraftKings salary slate.
All nflverse historic features use a strict pre-Week-8 cutoff.
Injury designations are frozen from USER'S salary file; contemporary NFL.com
injuries and postgame reserve status are intentionally disabled to avoid time
leakage. This is a retrospective approximation, NOT a verified Nov 1 12:59
snapshot of every public news item. Missing news must be disclosed.
Only AFTER the fresh model finishes do we load Week 8 realized stats for scoring.
The full DST actual result is not reconstructed from nflverse offensive stats.
"""
import csv
import json
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SLATE=ROOT/"tests"/"fixtures"/"nov1_2026_user_sunday_main_dk.csv"
OUT=ROOT/"nov1_current_model_results"
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
from props_data import clean_weekly, dk_classic_points, fetch_weekly
from ai2_nfl_auto_intelligence import collect_automatic_baselines
from ai2_nfl_auto_baselines import _norm_name
from ai2_nfl_matchups import canonical_team
import ai2_nfl_oneclick as model

csv_bytes=SLATE.read_bytes()
salary=pd.read_csv(SLATE)
assert len(salary)==562 and salary["ID"].is_unique
assert set(salary["Game Info"].str.extract(r"(\d\d/\d\d/\d{4})")[0])=={"11/01/2026"}
frozen_status={
    "".join(ch for ch in str(row["Name"]).casefold() if ch.isalnum()):
    str(row["Status"]).strip().upper()
    for row in salary.to_dict("records")
    if pd.notna(row["Status"]) and str(row["Status"]).strip()
}
assert len(frozen_status)>50, "Historic DK injury snapshot appears absent"
snapshot_meta={
    "source":"User-supplied Sunday final 2026-11-01 DraftKings salaries",
    "checked_utc":"NOT_ARCHIVED; historical source file saved after games",
    "practice_rows":[],
    "snapshot_limitation":"Status fields are original DK file, not verified pre-kickoff NFL.com news",
}
def historical_research(salaries, season, week):
    assert (season,week)==(2026,8), (season,week)
    # No LIVE statuses, no Week-8+ news, no postgame IR roster roll forward.
    history=fetch_weekly([2025,2026])
    # Strictly keep ONLY completed pregame weekly stats for all role,
    # rates, opportunity, efficiency and snap projections.
    history=history[(pd.to_numeric(history["season"],errors="coerce")<season)
                  |((pd.to_numeric(history["season"],errors="coerce")==season)
                    &(pd.to_numeric(history["week"],errors="coerce")<week))]
    def injury_fetcher():
        return frozen_status,snapshot_meta
    def no_postgame_roster(year):
        return pd.DataFrame({"season":pd.Series(dtype=int),
                             "week":pd.Series(dtype=int)})
    result=collect_automatic_baselines(
        salaries, season, week,
        history=history,injury_fetcher=injury_fetcher,
        roster_fetcher=no_postgame_roster)
    assert result["target_week"]==8
    return result

model.collect_automatic_baselines=historical_research

report={"kind":"CURRENT_UNIFIED_MODEL_HISTORICAL_REPLAY",
        "scenario":"November 1, 2026 Sunday 11-game DK Classic slate",
        "input_player_rows":len(salary),
        "slate_date":"2026-11-01",
        "historical_feature_weeks":"Strictly before 2026 regular season W8",
        "injury_context":"Historical user-supplied DK OUT/Q/IR statuses; live NFL.com disabled",
        "known_limits":[
            "Not verified as a complete, timestamped pre-kickoff injury/news snapshot",
            "No current-week coach interviews or late news feed in the model",
            "Offensive nflverse score omits fumbles, special teams and some 2-point plays",
            "DST scores must be validated separately; lineup OFFENSIVE subtotal only",
            "New model parameters were developed after event, so this is a retrospective fit, not a true out-of-time backtest",
        ]}
def save_json(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,default=str),encoding="utf-8")
def save_csv(name,rows):
    if not rows:
        return
    columns=list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(OUT/name,"w",newline="",encoding="utf-8-sig") as h:
        writer=csv.DictWriter(h,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)
# Capture the completed stat-first projections even when the portfolio
# allocator correctly refuses an infeasible 20-lineup export.
original_prop_decisions = model.player_prop_decisions
def capture_prop_forecasts(*args, **kwargs):
    data = original_prop_decisions(*args, **kwargs)
    save_csv("fresh_model_player_stat_forecasts.csv", data)
    print("CAPTURED new production stat-first forecasts: " + str(len(data)) +
          " player projections before portfolio optimization", flush=True)
    return data
model.player_prop_decisions = capture_prop_forecasts

# Save the exact production simulator's filtered candidate set once. This
# permits offline MILP feasibility probes without postgame score leakage or
# changing any role/overlap/salary constraints.
import ai2_nfl_sim_portfolio as portfolio_module
original_portfolio_solver = portfolio_module._solve_joint_portfolio
def capture_portfolio_candidates(scored, **kwargs):
    path=OUT/"new_model_portfolio_candidate_input.json"
    if not path.exists():
        payload=[]
        for item in scored:
            payload.append({
                "players":[{
                    k:p.get(k) for k in
                    ("ID","Name","TeamAbbrev","Position","Salary",
                     "observed_games","observed_carries_pg",
                     "observed_targets_pg","observed_rushing_yards_pg",
                     "observed_recent_four_games",
                     "observed_recent_four_rush_yd_pg",
                     "observed_recent_four_carries_pg",
                     "observed_recent_four_targets_pg",
                     "return_competition_reason")
                } for p in item["candidate"]["players"]],
                "salary":item["candidate"]["salary"],
                "mean":item["simulated_mean"],
                "p90":item["simulated_p90"],
                "p99":item["simulated_p99"],
            })
        path.write_text(json.dumps({
            "candidates":payload,
            "parameters":{k:v for k,v in kwargs.items()
                          if k not in ("premium_excluded_ids",)},
            "premium_excluded_ids":list(kwargs.get("premium_excluded_ids",())),
        },default=str),encoding="utf-8")
        print("CAPTURED exact model portfolio MILP input: "+
              str(len(payload))+" retained candidate lineups",flush=True)
    return original_portfolio_solver(scored, **kwargs)
portfolio_module._solve_joint_portfolio = capture_portfolio_candidates

try:
    print("START current production build from 562-row Nov1 DK slate",flush=True)
    o=model.build_oneclick_portfolio(
        ("nov1_2026_user_sunday_main_dk.csv",csv_bytes),
        seed=101010,count=20)
    fresh=o["player_prop_forecasts"]
    assert len(o["ranked"])==20 and len(fresh)>0
    save_csv("fresh_model_player_stat_forecasts.csv",fresh)
    selected=o["ranked"]
    player_by_id={str(r["DraftKings ID"]):r for r in fresh}
    offensive_ids=[]
    for x in selected:
        offensive_ids += [str(p["ID"]) for p in x["candidate"]["players"]
                          if str(p.get("Position","")).upper()!="DST"]
    assert all(key in player_by_id for key in set(offensive_ids))
    print("Fresh NEW version complete: "+str(len(fresh))+
          " forecast players and 20 lineups",flush=True)

    # Outcome retrieval happens ONLY AFTER running the model.
    games=clean_weekly(fetch_weekly([2026]))
    games=games[(games["season"]==2026)&(games["week"]==8)].copy()
    actual_map=defaultdict(list)
    actual_columns = {
      "Carries projection":"carries",
      "Rushing yards projection":"rushing_yards",
      "Targets projection":"targets",
      "Receptions projection":"receptions",
      "Receiving yards projection":"receiving_yards",
      "Passing yards projection":"passing_yards",
      "Passing TDs projection":"passing_tds",
    }
    for r in games.to_dict("records"):
        key=(_norm_name(r.get("player_name","")),
             canonical_team(r.get("recent_team","")),
             str(r.get("position","")).upper())
        actual_map[key].append(r)

    scored=[]
    actual_by_id={}
    for r in fresh:
        key=(_norm_name(r["Player"]),canonical_team(r["Team"]),
             str(r["Position"]).upper())
        recs=actual_map.get(key,[])
        matching = len(recs)==1 and str(r["Position"])!="DST"
        actual=recs[0] if matching else {}
        score=dk_classic_points(actual,r["Position"]) if matching else float("nan")
        if not np.isfinite(score): score=None
        item={
            "Player":r["Player"],"Team":r["Team"],"Position":r["Position"],
            "DraftKings ID":r["DraftKings ID"],"Salary":r["Salary"],
            "Model mean DK":r["Model mean DK"],
            "Model median DK":r["Model median DK"],
            "Model P10 DK":r["Model P10 DK"],"Model P90 DK":r["Model P90 DK"],
            "Pregame role":r["Historical role validation"],
            "Actual offensive partial DK":round(float(score),2) if score is not None else None,
            "Actual match":"matched and partial scored" if score is not None else
                           "No unique NFLverse Week-8 player/game stat",
            "Model error actual minus mean":
              round(float(score)-float(r["Model mean DK"]),2)
              if score is not None else None,
        }
        for forecast_name,actual_name in actual_columns.items():
            if r.get(forecast_name) is not None and matching:
                v=actual.get(actual_name)
                try: v=float(v)
                except (TypeError,ValueError): v=float("nan")
                if np.isfinite(v):
                    item["Model "+forecast_name]=r[forecast_name]
                    item["Actual "+actual_name]=round(v,2)
                    item["Error "+actual_name]=round(v-float(r[forecast_name]),2)
        scored.append(item)
        actual_by_id[str(r["DraftKings ID"])]=score
    save_csv("new_model_vs_actual_players.csv",scored)

    lineup_rows=[]
    for index,entry in enumerate(selected,1):
        roster=entry["candidate"]["players"]
        offensive=[p for p in roster if str(p["Position"])!="DST"]
        observed=[actual_by_id.get(str(p["ID"])) for p in offensive]
        known=sum(x is not None for x in observed)
        model_offense=sum(float(player_by_id[str(p["ID"])]["Model mean DK"])
                          for p in offensive)
        lineup_rows.append({
            "Lineup":index,"Seed":101010,"Contest":entry["contest_tier"],
            "Salary":entry["candidate"]["salary"],
            "NEW model 9-player mean":round(float(entry["simulated_mean"]),2),
            "NEW model 9-player P90":round(float(entry["simulated_p90"]),2),
            "NEW model 8-offense mean":round(model_offense,2),
            "Actual offense matched of 8":known,
            "Actual 8-offense partial DK":
              round(float(sum(observed)),2) if known==8 else None,
            "Actual offense minus NEW offense projection":
              round(float(sum(observed))-model_offense,2) if known==8 else None,
            "Actual DST":"UNAVAILABLE; NO FULL LINEUP SCORE CLAIMED",
            "Roster":" | ".join(str(p["Name"])+" ("+str(p["Position"])+")"
                                for p in roster),
        })
    save_csv("new_model_lineups_offensive_actual.csv",lineup_rows)
    errors=[float(r["Model error actual minus mean"]) for r in scored
            if r["Model error actual minus mean"] is not None]
    eligible=[r for r in scored if r["Position"] in ("QB","RB","WR","TE")]
    by_position={}
    for position in ("QB","RB","WR","TE"):
        rows=[r for r in eligible if r["Position"]==position and
              r["Model error actual minus mean"] is not None]
        by_position[position]={
            "n":len(rows),
            "mean_abs_error":round(float(np.mean(
                [abs(r["Model error actual minus mean"]) for r in rows])),2)
                if rows else None,
            "signed_error":round(float(np.mean(
                [r["Model error actual minus mean"] for r in rows])),2)
                if rows else None,
        }
    report.update({"status":"MODEL_BUILD_COMPLETED",
                   "fresh_forecast_players":len(fresh),
                   "fresh_lineups":len(selected),
                   "actual_scored_players":len(errors),
                   "partial_DK_MAE":round(float(np.mean(np.abs(errors))),2)
                        if errors else None,
                   "actual_vs_projection_bias":round(float(np.mean(errors)),2)
                        if errors else None,
                   "position_errors":by_position,
                   "complete_offense_lineups":sum(
                        r["Actual offense matched of 8"]==8 for r in lineup_rows),
                   "new_mean_lineup_mean":round(float(np.mean(
                        [r["NEW model 9-player mean"] for r in lineup_rows])),2)})
    print("MODEL BACKTEST COMPLETED",json.dumps(report,indent=2),flush=True)
except Exception as e:
    report.update({"status":"CURRENT_ENGINE_EXECUTION_FAILED",
                   "failure_type":type(e).__name__,
                   "failure":str(e),"traceback":traceback.format_exc()[-9000:]})
    print("MODEL BACKTEST FAILED",json.dumps(report,indent=2),flush=True)
    save_json("backtest_result.json",report)
    raise
save_json("backtest_result.json",report)
if "GITHUB_STEP_SUMMARY" in os.environ:
    with open(os.environ["GITHUB_STEP_SUMMARY"],"a") as f:
        f.write("## 2026-11-01 CURRENT NFL model historical replay\n")
        f.write("Status: "+report["status"]+"\n\n")
        f.write("Current model forecast players: "+str(report["fresh_forecast_players"])+"\n\n")
        f.write("Offensive actuals matched: "+str(report["actual_scored_players"])+"\n\n")
        f.write("MAE: "+str(report["partial_DK_MAE"])+"\n\n")
        f.write("**Not a complete full-lineup/DST score or verified pregame-news backtest.**\n")
