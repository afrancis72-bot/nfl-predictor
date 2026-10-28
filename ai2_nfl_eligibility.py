"""Return only NFL rows meeting conservative player-level eligibility.

This is not independent certification of user-supplied flags.
"""
from math import isfinite
from ai2_nfl_preflight import BLOCKED, REQUIRED

def eligible_nfl_rows(rows, *, min_wr_salary=4000, min_rb_salary=4500, require_opportunity=True):
    result = []
    seen = set()
    for row in rows:
        if any(k not in row or row[k] is None for k in REQUIRED):
            continue
        identifier = str(row.get("ID", "")).strip()
        if not identifier or identifier in seen:
            continue
        if str(row.get("dk_status", "")).upper().strip() in BLOCKED:
            continue
        if row.get("weekly_projection_valid") is not True or (require_opportunity and row.get("opportunity_verified") is not True):
            continue
        if str(row["projection_source"]).strip().lower() in {"", "unknown", "dk ppg fallback"}:
            continue
        try:
            salary, projection = float(row["Salary"]), float(row["proj"])
        except (ValueError, TypeError):
            continue
        if not isfinite(salary) or salary <= 0 or not isfinite(projection) or projection <= 0:
            continue
        position = str(row.get("Position", "")).upper()
        if (position == "WR" and salary < min_wr_salary) or (position == "RB" and salary < min_rb_salary):
            continue
        if position not in {"QB", "RB", "WR", "TE", "DST"}:
            continue
        if not str(row.get("TeamAbbrev", "")).strip():
            continue
        if str(row.get("Position", "")).upper() == "QB" and not row.get("opponent"):
            continue
        seen.add(identifier)
        result.append(row)
    return result
