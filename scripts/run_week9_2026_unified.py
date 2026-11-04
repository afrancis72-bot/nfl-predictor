"""Run current unified sports NFL one-click on the user's 2026-11-08 DK Classic slate.

Read the exact authenticated salary CSV saved in tests/fixtures, pull the
current live public inputs using the production pipeline (no frozen week-8
snapshot), and export review files. No paid feed or user sportsbook upload.
These are provisional Tuesday predictions, not Sunday-approved entries.
The official injury page may still show prior-week reports; do not assert
target-week clearance absent a dated report. Every exception is saved.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ai2_nfl_oneclick import build_oneclick_portfolio
import ai2_nfl_oneclick as production
from ai2_nfl_review import named_excel, named_csv, review_tables
from ai2_nfl_portfolio_audit import portfolio_audit_csv
from ai2_nfl_qb_audit import qb_and_partners
from ai2_nfl_correlation_guard import opposing_pass_catcher_dst_pairs
from ai2_nfl_portfolio_risk import maximum_player_appearances, MAX_SHARED_PLAYERS

SLATE=ROOT/"tests"/"fixtures"/"nov8_2026_user_sunday_main_dk.csv"
OUT=ROOT/"week9_2026_current_unified_results"
OUT.mkdir(exist_ok=True)
COUNT=50
SEED=101010
def json_file(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,default=str),encoding="utf-8")
def csv_file(name,rows):
    if not rows:
        return
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/name,index=False,encoding="utf-8-sig")
def collect_stat_predictions(*args,**kwargs):
    pred=_ORIGINAL_PROP(*args,**kwargs)
    csv_file("player_stat_by_stat_forecasts.csv",pred)
    print(f"FORECASTS CAPTURED: {len(pred)} players before optimizer selection",flush=True)
    return pred

file_bytes=SLATE.read_bytes()
salary=pd.read_csv(SLATE)
assert len(salary)==570, "Expected exactly 570 players and DST on uploaded Week9 slate"
assert salary["ID"].astype(str).is_unique
assert all(salary["Game Info"].str.contains("11/08/2026",regex=False)), "Must be 2026-11-08 slate only"
assert salary["Game Info"].nunique()==11, "Expected exactly 11 contests"
assert set(salary["Position"].unique())=={"QB","RB","WR","TE","DST"}
assert salary["Salary"].between(0,10000).all()
sha=hashlib.sha256(file_bytes).hexdigest()
summary={
    "kind":"NFL_CURRENT_UNIFIED_WEEK9_PREGAME",
    "slate":"2026-11-08 Sunday NFL Classic main",
    "generated_utc":datetime.now(timezone.utc).isoformat(),
    "repo_branch":"afrancis72-bot/nfl-predictor:feature/unified-ai2-platform",
    "source_commit":os.environ.get("GITHUB_SHA","unknown"),
    "slate_players_and_defenses":len(salary),
    "game_count":salary["Game Info"].nunique(),
    "salary_sha256":sha,
    "requested_lineups":COUNT,
    "seed":SEED,
    "provisional":True,
    "no_human_approval":True,
    "source_limits":[
        "No news/interview aggregator currently connected; official injury status and DK file only",
        "Current NFL.com report may describe a previous week's game and lacks guaranteed Week9 date binding",
        "The file includes 27 DK-questionable players; some may return before lock",
        "This is a Tuesday forecast. Friday reports and Sunday inactive announcements must be checked again",
        "Projected points and model probabilities are uncalibrated; no guarantee of positive DFS return",
        "2026 Week8 backtest had major selected-lineup overprediction even after player-level improvements"
    ]
}
print("WEEK9 RUN START",json.dumps({k:summary[k] for k in ["slate","slate_players_and_defenses","game_count","seed","source_commit"]}),flush=True)
_ORIGINAL_PROP=production.player_prop_decisions
production.player_prop_decisions=collect_stat_predictions
try:
    results=build_oneclick_portfolio(("DKSalaries (15)(1).csv",file_bytes),seed=SEED,count=COUNT)
    ranked=results["ranked"]
    assert len(ranked)==COUNT
    ov,detail,expo=review_tables(ranked)
    salary_map={str(row["ID"]):row for row in salary.to_dict("records")}
    appearances=Counter(str(p["ID"]) for r in ranked for p in r["candidate"]["players"])
    final_records={str(p["ID"]):p for r in ranked for p in r["candidate"]["players"]}
    invalid={
        key:{"count":num,"cap":maximum_player_appearances(final_records[key],COUNT,.28)}
        for key,num in appearances.items()
        if num>maximum_player_appearances(final_records[key],COUNT,.28)}
    if invalid:
        raise ValueError("Invalid player exposures: "+str(invalid))
    stacks=0
    opposing_dst=0
    questionable_selected=[]
    names=Counter()
    rosters=[]
    for idx,r in enumerate(ranked,1):
        ps=r["candidate"]["players"]
        rosters.append(set(str(p["ID"]) for p in ps))
        qb,partners=qb_and_partners(ps)
        stacks+=bool(partners)
        opposing_dst+=bool(opposing_pass_catcher_dst_pairs(ps))
        for p in ps:
            names[str(p["Name"])]+=1
            s=salary_map.get(str(p["ID"]),{})
            if str(s.get("Status","")).upper() in {"Q","QUESTIONABLE","OUT","O","IR","D","DOUBTFUL"}:
                questionable_selected.append({
                    "Lineup":idx,"Player":p["Name"],"Position":p["Position"],
                    "Status in uploaded DraftKings file":str(s.get("Status",""))})
    pair_overlap=max(
        len(rosters[i]&rosters[j])
        for i in range(COUNT) for j in range(i+1,COUNT))
    if pair_overlap>MAX_SHARED_PLAYERS:
        raise ValueError(f"Near-duplicate lineup pair shares {pair_overlap} players")
    if questionable_selected:
        raise ValueError("Known questionable/out player selected: "+str(questionable_selected[:8]))
    if not ov["Salary"].between(49600,50000).all():
        raise ValueError("Salary floor/cap violated")
    csv_file("named_lineups.csv",ov.to_dict("records"))
    csv_file("player_details.csv",detail.to_dict("records"))
    csv_file("player_exposures.csv",expo.to_dict("records"))
    (OUT/"named_lineups.xlsx").write_bytes(named_excel(ranked))
    (OUT/"draftkings_ids_NOT_APPROVED.csv").write_bytes(results["csv"])
    (OUT/"simulation_portfolio_audit.csv").write_bytes(portfolio_audit_csv(ranked,seed=SEED))
    for filename,tag in [
        ("pregame_role_evidence.csv","pregame_role_evidence"),
        ("qb_stack_selection_audit.csv","qb_selection_audit"),
        ("injury_report.csv","injuries"),
        ("returning_rb_watch.csv","returning_rb_watch"),
        ("role_projection_anomalies.csv","role_projection_anomalies"),
        ("projection_review_flags.csv","projection_review_flags"),
        ("role_risk_exposure.csv","role_risk_exposure"),
        ("opponent_matchup_report.csv","opponent_matchup_report"),
        ("snap_role_report.csv","snap_role_report"),
    ]:
        rows=results["pregame_report"].get(tag,[]) if tag=="injuries" else results.get(tag,[])
        if isinstance(rows,pd.DataFrame):
            rows=rows.to_dict("records")
        if isinstance(rows,list):
            csv_file(filename,rows)
    info=results.get("pregame_report",{})
    json_file("pregame_research_context.json",info)
    json_file("qb_selection_diagnostics.json",results.get("qb_selection_diagnostics",{}))
    json_file("data_input_lineage.json",results.get("input_lineage",[]))
    json_file("opportunity_risk_summary.json",{
        "injury_context_report":results.get("injury_context_report",[]),
        "reserve_status_issue":results.get("reserve_status_issue"),
        "snap_feed_issue":results.get("snap_feed_issue"),
        "blocked_low_salary_players":results.get("pregame_role_blocked_count"),
        "noted_research_warning":results.get("research_warning"),
        "unverified_late_news":info.get("news_feed")
    })
    qb_counts=Counter(
        str(next(p for p in r["candidate"]["players"] if p["Position"]=="QB")["Name"])
        for r in ranked)
    summary.update({
        "status":"FULL_50_LINEUP_BUILD_COMPLETE",
        "lineups":len(ranked),
        "forecast_count":len(results.get("player_prop_forecasts",[])),
        "max_player_exposure":max(appearances.values()),
        "exposure_cap":14,
        "maximum_pairwise_shared_players":pair_overlap,
        "qb_wr_te_stacks":stacks,
        "nonstacked_lineups":COUNT-stacks,
        "opposing_wr_te_dst_conflicted_lineups":opposing_dst,
        "quarterback_exposures":dict(qb_counts),
        "average_mean":round(float(ov["Mean"].mean()),2),
        "average_p90":round(float(ov["P90"].mean()),2),
        "average_p99":round(float(ov["P99"].mean()),2),
        "minimum_salary":int(ov["Salary"].min()),
        "maximum_salary":int(ov["Salary"].max()),
        "pregame_role_blocked_count":results.get("pregame_role_blocked_count"),
        "questionable_or_out_selected":questionable_selected,
        "injury_source_url":info.get("official_injury_url"),
        "injury_checked_utc":info.get("checked_utc"),
        "current_week_injury_verified":False,
        "source_feed_warning":"NFL report date does not independently certify Week9 designations"
    })
    print("WEEK9 50-LINEUP BUILD SUCCESS",json.dumps(summary,indent=2,default=str),flush=True)
except Exception as e:
    summary.update({
        "status":"FULL_50_LINEUP_BUILD_FAILED",
        "error_type":type(e).__name__,
        "error":str(e),
        "traceback":traceback.format_exc()[-7500:],
        "player_stat_forecasts_saved":(OUT/"player_stat_by_stat_forecasts.csv").exists()
    })
    print("WEEK9 BUILD FAILED",json.dumps(summary,indent=2,default=str),flush=True)
    json_file("build_summary.json",summary)
    raise
json_file("build_summary.json",summary)
