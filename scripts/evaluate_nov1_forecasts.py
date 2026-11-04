"""Evaluate *NEW current model* saved player predictions vs actual Nov1 2026.

Called only AFTER production one-click attempt. A failure of full lineup
generation must NOT erase the valid prior model stat forecasts.
Scores are nflverse DK-style offensive scores excluding fumbles lost and some
special-teams/two-point events; DST comparisons unavailable.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from props_data import fetch_weekly,clean_weekly,dk_classic_points
from ai2_nfl_auto_baselines import _norm_name
from ai2_nfl_matchups import canonical_team
OUT=ROOT/"nov1_current_model_results"
SOURCE=OUT/"fresh_model_player_stat_forecasts.csv"
if not SOURCE.exists():
    print("NO NEW MODEL PLAYER FORECASTS FOUND - cannot fake actual comparison",flush=True)
    sys.exit(2)
with SOURCE.open(encoding="utf-8-sig",newline="") as f:
    projected=list(csv.DictReader(f))
print("Read "+str(len(projected))+" actual fresh unified model projections",flush=True)

# Postgame data are loaded here, after current model forecast has completed.
games=clean_weekly(fetch_weekly([2026]))
games=games[(games["season"]==2026)&(games["week"]==8)]
lookup=defaultdict(list)
for r in games.to_dict("records"):
    key=(_norm_name(r.get("player_name","")),
         canonical_team(r.get("recent_team","")),
         str(r.get("position","")).upper())
    lookup[key].append(r)

fields={
    "Carries projection":"carries",
    "Rushing yards projection":"rushing_yards",
    "Targets projection":"targets",
    "Receptions projection":"receptions",
    "Receiving yards projection":"receiving_yards",
    "Passing yards projection":"passing_yards",
    "Passing TDs projection":"passing_tds",
    "Rushing TDs projection":"rushing_tds",
    "Receiving TDs projection":"receiving_tds",
    "Total scrimmage yards projection":None,
}
results=[]
for p in projected:
    ident=str(p.get("DraftKings ID",""))
    pos=str(p.get("Position","")).upper()
    key=(_norm_name(p.get("Player","")),
         canonical_team(p.get("Team","")),pos)
    matched=lookup.get(key,[])
    game=matched[0] if len(matched)==1 and pos!="DST" else None
    score=dk_classic_points(game,pos) if game is not None else float("nan")
    if not np.isfinite(score): score=None
    est=float(p["Model mean DK"])
    row={
        "Player":p["Player"],"Team":p["Team"],"Position":pos,
        "DraftKings ID":ident, "Salary":p["Salary"],
        "New model projected DK":round(est,2),
        "New model P10":p.get("Model P10 DK",""),
        "New model median":p.get("Model median DK",""),
        "New model P90":p.get("Model P90 DK",""),
        "Modeled non-TD DK":p.get("Model non-TD DK",""),
        "Modeled expected TD DK":p.get("Model expected TD DK",""),
        "Pregame role validation":p.get("Historical role validation",""),
        "Actual partial DK":round(float(score),2) if score is not None else "",
        "Difference actual - new projection":round(float(score)-est,2) if score is not None else "",
        "Actual resolved":"YES" if score is not None else
                          "NO - no unique W8 offensive box-score match",
    }
    if game is not None:
        for pred_key,stat in fields.items():
            v=p.get(pred_key,"")
            if v in ("",None):
                continue
            if stat is None:
                base=[game.get("rushing_yards",float("nan")),
                      game.get("receiving_yards",float("nan"))]
                if any(pd.isna(y) for y in base):
                    continue
                actual=float(base[0])+float(base[1])
            else:
                actual=game.get(stat,float("nan"))
                if pd.isna(actual):
                    continue
                actual=float(actual)
            row["Model "+pred_key]=v
            row["Actual "+stat if stat else "Actual total scrimmage yards"]=round(actual,2)
            row["Error "+stat if stat else "Error total scrimmage yards"]=round(actual-float(v),2)
    results.append(row)
columns=list(dict.fromkeys(col for row in results for col in row))
with (OUT/"new_model_vs_actual_players.csv").open("w",encoding="utf-8-sig",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=columns)
    writer.writeheader();writer.writerows(results)

key="Difference actual - new projection"
resolved=[r for r in results if r[key]!=""]
position={}
for pos in ("QB","RB","WR","TE"):
    rows=[r for r in resolved if r["Position"]==pos]
    position[pos]={
        "matched":len(rows),
        "mean_absolute_DK_error":round(float(np.mean([abs(r[key]) for r in rows])),2) if rows else None,
        "mean_signed_DK_error":round(float(np.mean([r[key] for r in rows])),2) if rows else None,
    }
rating=[]
for p in resolved:
    rating.append({k:p[k] for k in ["Player","Team","Position","Salary",
                                   "New model projected DK","Actual partial DK",
                                   "Difference actual - new projection",
                                   "Pregame role validation"]})
rating.sort(key=lambda r:abs(float(r[key])),reverse=True)
report={
    "status":"NEW CURRENT MODEL PLAYER STAT FORECAST COMPARED TO NOV1 RESULTS",
    "selected_current_model":"feature/unified-ai2-platform",
    "date":"2026-11-01",
    "complete_20_lineup_optimizer_status":"FAILED: hard exposure and overlap constraints, no full lineups exported",
    "new_model_stat_forecast_players":len(projected),
    "matched_nov1_realized_offensive_players":len(resolved),
    "average_absolute_DK_error":round(float(np.mean([abs(r[key]) for r in resolved])),2)
           if resolved else None,
    "signed_DK_error_actual_minus_projection":round(float(np.mean([r[key] for r in resolved])),2)
           if resolved else None,
    "by_position":position,
    "largest_12_player_forecast_misses":rating[:12],
    "notes":[
      "This is fresh model player-level forecast accuracy, not old lineups",
      "The model was trained/tuned after Week8, so this is a retrospective replay not a true out-of-time calibration study",
      "Partial DK scoring excludes fumbles lost and some special teams/2point plays",
      "No DFS contest payouts or real ownership claims",
      "Official NFL news archive unavailable; original DK salary injury designations frozen",
      "Players without unique NFLverse W8 offensive record or DST are not treated as known zero"
    ],
}
(OUT/"new_model_stat_backtest_summary.json").write_text(
    json.dumps(report,indent=2,default=str),encoding="utf-8")
print("MODEL PLAYER STAT FORECAST BACKTEST COMPLETED",json.dumps(report,indent=2),flush=True)
