"""NFL Classic roster feasibility checks; not an optimizer or release approval."""
from collections import Counter

ROSTER = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}
FLEX = {"RB", "WR", "TE"}

def check_classic_lineup(players, *, salary_cap=50000, salary_floor=47500):
    issues = []
    if len(players) != 9:
        issues.append("NFL Classic requires exactly nine roster entries")
    ids = [str(p.get("ID", "")).strip() for p in players]
    if not all(ids) or len(ids) != len(set(ids)):
        issues.append("Player IDs must be present and unique")
    counts = Counter(str(p.get("Position", "")).upper() for p in players)
    if any(counts[p] < count for p, count in ROSTER.items()):
        issues.append("Missing required QB/RB/WR/TE/DST roster positions")
    extra = counts.copy()
    for p, count in ROSTER.items():
        extra[p] -= count
    if sum(extra.values()) != 1 or any(n < 0 for n in extra.values()) or not any(extra[p] == 1 for p in FLEX):
        issues.append("Exactly one RB/WR/TE FLEX is required")
    try:
        salaries = [float(p["Salary"]) for p in players]
        total = sum(salaries)
        if any(s <= 0 for s in salaries) or not salary_floor <= total <= salary_cap:
            issues.append("Salary outside configured floor/cap or nonpositive salary")
    except (KeyError, ValueError, TypeError):
        issues.append("Salary missing or invalid")
    rb_teams = [str(p.get("TeamAbbrev", "")).upper() for p in players if str(p.get("Position", "")).upper() == "RB"]
    if len(rb_teams) != len(set(rb_teams)):
        issues.append("Multiple running backs from same team")
    qbs = [p for p in players if str(p.get("Position", "")).upper() == "QB"]
    dsts = [p for p in players if str(p.get("Position", "")).upper() == "DST"]
    if qbs and not str(qbs[0].get("opponent") or "").strip():
        issues.append("Quarterback opponent is missing")
    if qbs and dsts and str(qbs[0].get("opponent", "")).upper() == str(dsts[0].get("TeamAbbrev", "")).upper() and qbs[0].get("opponent"):
        issues.append("Quarterback against opposing defense")
    return issues
