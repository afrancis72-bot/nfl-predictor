"""Reproducible Week9 4/16/20 core-diversified review from three frozen current-model runs.

Does NOT read postgame outcomes or fabricate ownership. Never alters saved
reviewed/approved lineups. For live use the same core selector is available
to the unified Streamlit app with full source-specific player role metadata.
"""
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ai2_nfl_core_diversifier import diversify_40_from_stat_simulation
from ai2_nfl_review import named_csv, named_excel, review_tables, roster_id_csv
from ai2_nfl_qb_audit import qb_and_partners
from ai2_nfl_lineup import check_classic_lineup
from ai2_nfl_portfolio_risk import maximum_player_appearances
from ai2_nfl_matchups import canonical_team, opponent_from_game_info

SLATE=ROOT/"tests"/"fixtures"/"nov8_2026_user_sunday_main_dk.csv"
BUILDS=[
 ("101010",ROOT/"week9_2026_current_unified_results"),
 ("47",ROOT/"week9_2026_alternate_seed_47"),
 ("89",ROOT/"week9_2026_alternate_seed_89"),
]
OUT=ROOT/"week9_2026_40_core_diversified_REVIEW"
OUT.mkdir(exist_ok=True)
salary=pd.read_csv(SLATE)
name_salary={
  (str(r["Name"]),canonical_team(r["TeamAbbrev"]),str(r["Position"])):r
  for r in salary.to_dict("records")
}
for _,p in BUILDS:
    if not (p/"named_lineups.csv").exists():
        raise ValueError("Missing independent model output "+str(p))
role_flags=set()
role_limited=set()
emerging=set()
portfolios=[]
roster_cols=("QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DST")
for seed,folder in BUILDS:
    summary=json.loads((folder/"build_summary.json").read_text())
    if summary.get("status")!="FULL_50_LINEUP_BUILD_COMPLETE":
        raise ValueError("An input build failed: "+seed)
    if summary.get("slate")!="2026-11-08 Sunday NFL Classic main":
        raise ValueError("Wrong slate or stale input seed: "+seed)
    a=pd.read_csv(folder/"role_projection_anomalies.csv")
    role_flags.update(str(x).strip() for x in a.get("Player",[]) if pd.notna(x))
    roles=pd.read_csv(folder/"pregame_role_evidence.csv")
    emerging.update(
        (str(r["Player"]),canonical_team(r["Team"]),str(r["Position"]))
        for r in roles.to_dict("records")
        if r.get("Role gate")=="EMERGING_LIMITED")
    roles_audit=pd.read_csv(folder/"simulation_portfolio_audit.csv")
    # These risk labels come from the original actual stat-simulation build
    # and are a review-only fallback when exporting the original ranked
    # objects isn't possible. They do NOT certify a returning starter.
    for row in roles_audit.to_dict("records"):
        flagged=str(row.get("Role-limited RBs") or "")
        for seg in flagged.split(";"):
            nm=seg.split(":",1)[0].strip()
            if nm and nm.lower()!="nan":
                role_limited.add(nm)
    d=pd.read_csv(folder/"named_lineups.csv")
    if len(d)!=50:
        raise ValueError("Expected 50 entries for seed "+seed)
    results=[]
    for _,r in d.iterrows():
        players=[]
        for col in roster_cols:
            s=str(r[col])
            if " (" not in s or not s.endswith(")"):
                raise ValueError("Malformed named roster entry "+s)
            nm,tm=s.rsplit(" (",1)
            tm=canonical_team(tm[:-1])
            position=(("RB" if col.startswith("RB") else "WR")
                      if col.startswith(("RB","WR")) else
                      col if col!="FLEX" else "")
            matches=[v for (n,t,p),v in name_salary.items()
                     if n==nm and t==tm and (not position or p==position)]
            if len(matches)!=1:
                raise ValueError("Ambiguous salary identity for "+s)
            v=matches[0]
            pos=str(v["Position"])
            k=(nm,tm,pos)
            players.append({
                "ID":str(v["ID"]),"Name":nm,"Position":pos,
                "TeamAbbrev":tm,"Salary":int(v["Salary"]),
                "opponent":(opponent_from_game_info(tm,v["Game Info"]) if pos=="QB" else ""),
                "emerging_role_limited":k in emerging,
                "emerging_role_evidence":"Two real current-year weekly games and peer decline"
                         if k in emerging else "",
                "return_competition_reason":"Original simulator RB role-risk audit"
                         if pos=="RB" and nm in role_limited else "",
                "proj":0.,
            })
        if check_classic_lineup(players,salary_floor=49600,salary_cap=50000):
            raise ValueError("An original source lineup is no longer legal")
        results.append({
            "candidate":{"players":players,"salary":int(r["Salary"])},
            "simulated_mean":float(r["Mean"]),
            "simulated_p90":float(r["P90"]),
            "simulated_p99":float(r["P99"]),
            "origin_seed":seed,"origin_lineup":int(r["Lineup"]),
        })
    portfolios.append(results)

selected,audit=diversify_40_from_stat_simulation(
    portfolios,role_anomalies=role_flags,milp_time_limit=75.)
overview,details,exposure=review_tables(selected)
# Include the exact source run and original lineup number for traceability.
bykey={
    tuple(sorted(str(p["ID"]) for p in x["candidate"]["players"])):x
    for batch in portfolios for x in batch
}
overview.insert(1,"Original seed",[
    r["origin_seed"] for r in selected])
overview.insert(2,"Original lineup",[
    r["origin_lineup"] for r in selected])
overview.to_csv(OUT/"40_named_lineups_REVIEW.csv",index=False,encoding="utf-8-sig")
details.to_csv(OUT/"40_player_details.csv",index=False,encoding="utf-8-sig")
exposure.to_csv(OUT/"40_exposure.csv",index=False,encoding="utf-8-sig")
(OUT/"40_named_lineups_REVIEW.xlsx").write_bytes(named_excel(selected))
(OUT/"40_roster_IDs_NOT_ENTRY_SHEET.csv").write_text(roster_id_csv(selected),encoding="utf-8-sig")
prem=selected[:4]
roster_slots=[set(str(p["ID"]) for p in x["candidate"]["players"])
              for x in prem]
names=Counter(str(p["Name"]) for x in selected for p in x["candidate"]["players"])
combination=sum(
    any(p["Name"]=="Puka Nacua" for p in x["candidate"]["players"]) and
    any(p["Name"]=="Chuba Hubbard" for p in x["candidate"]["players"])
    for x in selected)
both_or_either=sum(
    any(p["Name"] in {"Puka Nacua","Chuba Hubbard"}
        for p in x["candidate"]["players"]) for x in selected)
audit.update({
    "generated_utc":datetime.now(timezone.utc).isoformat(),
    "new_model_runs":[s for s,_ in BUILDS],
    "original_40_remains_untouched":True,
    "status":"REVIEW_NOT_APPROVED",
    "premium_risk":"No premium role-limited RB; model flags need separate Sunday verification",
    "Hubbard_40_count":names["Chuba Hubbard"],
    "Nacua_40_count":names["Puka Nacua"],
    "Hubbard_AND_Nacua_40_count":combination,
    "Hubbard_OR_Nacua_40_count":both_or_either,
    "Coleman_40_count":names["Jonah Coleman"],
    "Cardinals_DST_40_count":names["Cardinals"],
    "premium_by_original":[
        {"seed":x["origin_seed"],"lineup":x["origin_lineup"],
         "qb":next(p["Name"] for p in x["candidate"]["players"] if p["Position"]=="QB"),
         "mean":round(x["simulated_mean"],2),"p90":round(x["simulated_p90"],2)}
        for x in prem
    ],
    "premium_average_mean":round(sum(x["simulated_mean"] for x in prem)/4,2),
    "premium_average_p90":round(sum(x["simulated_p90"] for x in prem)/4,2),
    "official_week9_inactives_verified":False,
    "ownership_model":"NOT CONNECTED; no low-owned claims",
    "caveats":["Some role-risk metadata reconstructed from pregame per-lineup audit",
               "Ownership unknown; these are upside/role-supported pivots, not verified ownership pivots",
               "Portfolio changes not a forecast of better win rate or ROI",
               "User approval required for all 40 entries"]
})
(OUT/"40_portfolio_diagnostics.json").write_text(json.dumps(audit,indent=2))
print("NEW CORE-DIVERSIFIED 40",json.dumps(audit,indent=2),flush=True)
