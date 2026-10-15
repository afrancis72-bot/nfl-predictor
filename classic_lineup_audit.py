"""Strict DraftKings classic lineup validation for research exports."""
from __future__ import annotations
import pandas as pd

SLOTS = ("QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST")
FLEX_POSITIONS = {"RB", "WR", "TE"}

def audit_classic_lineups(lineups: pd.DataFrame, salaries: pd.DataFrame,
                          excluded_qbs=(), min_salary=47500) -> pd.DataFrame:
    """Fail closed: returns one row per lineup with explicit rule violations.

    Validates roster legality, salaries, player IDs, RB team overlap and QB-DST
    opposition. Does not verify current injuries or projection calibration.
    """
    missing=set(SLOTS)-set(lineups)
    if missing: raise ValueError("Missing lineup slots: "+", ".join(sorted(missing)))
    needed={"Name","ID","Position","Salary","TeamAbbrev","Game Info"}
    if needed-set(salaries): raise ValueError("Salary file missing required columns")
    s=salaries.copy()
    s["Position"]=s["Position"].astype(str).str.upper()
    results=[]
    for idx, lineup in lineups.iterrows():
        issues=[]; players=[]
        for slot in SLOTS:
            name=str(lineup[slot])
            matches=s[s["Name"]==name]
            if slot=="DST": matches=matches[matches.Position.isin(["DST","D"])]
            if len(matches)!=1:
                issues.append(f"{slot}: ambiguous or missing name ({name})")
                continue
            player=matches.iloc[0]
            players.append((slot,player))
            expected=slot.rstrip("123") if slot not in ("FLEX","DST") else slot
            allowed=FLEX_POSITIONS if slot=="FLEX" else ({"DST","D"} if slot=="DST" else {expected})
            if player.Position not in allowed: issues.append(f"{slot}: illegal position")
            if slot=="QB" and name in set(excluded_qbs): issues.append("Excluded quarterback: "+name)
        total=sum(int(p.Salary) for _,p in players)
        if len(players)!=9: issues.append("Unresolved roster slots")
        else:
            if len({str(p.ID) for _,p in players})!=9: issues.append("Duplicate player ID")
            rb_teams=[p.TeamAbbrev for slot,p in players if slot in ("RB1","RB2") or (slot=="FLEX" and p.Position=="RB")]
            if len(rb_teams)!=len(set(rb_teams)): issues.append("Multiple RBs from same team")
            qb=next((p for slot,p in players if slot=="QB"),None)
            dst=next((p for slot,p in players if slot=="DST"),None)
            if qb is not None and dst is not None:
                matchup=str(qb["Game Info"]).split(" ")[0].split("@")
                if dst.TeamAbbrev in matchup and dst.TeamAbbrev!=qb.TeamAbbrev:
                    issues.append("QB faces opposing DST")
        if total>50000 or total<min_salary: issues.append("Salary outside allowed range")
        if "Salary" in lineups and int(lineup["Salary"])!=total: issues.append("Listed salary mismatch")
        results.append({"lineup":idx+1,"verified_salary":total,"resolved_players":len(players),
                        "passed":not issues,"issues":"; ".join(issues)})
    return pd.DataFrame(results)
