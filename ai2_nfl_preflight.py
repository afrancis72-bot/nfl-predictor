"""Fail-closed NFL preflight checks. No simulation or export is performed here.

These checks are intentionally stricter than legacy eligibility flags.
"""
from dataclasses import dataclass
from typing import Mapping, Sequence

BLOCKED = frozenset({"OUT", "O", "IR", "INACTIVE", "SUSPENDED", "PUP", "NFI", "DOUBTFUL", "D"})
REQUIRED = ("Name", "Position", "TeamAbbrev", "Salary", "proj",
            "projection_source", "weekly_projection_valid", "opportunity_verified")

@dataclass(frozen=True)
class Preflight:
    ready: bool
    reasons: tuple[str, ...]
    eligible_count: int

def validate_pool(rows: Sequence[Mapping], *, source_fresh: bool,
                  historical_validation_passed: bool, lineup_feasibility_passed: bool = False) -> Preflight:
    reasons = []
    if not source_fresh:
        reasons.append("Current-week data freshness not verified")
    if not historical_validation_passed:
        reasons.append("Historical engine validation not passed")
    if not lineup_feasibility_passed:
        reasons.append('Legal lineup feasibility not independently verified')
    eligible = 0
    for i, row in enumerate(rows):
        missing = [k for k in REQUIRED if k not in row or row[k] is None]
        if missing:
            reasons.append(f"Player row {i}: missing {', '.join(missing)}")
            continue
        if str(row.get("dk_status", "")).upper().strip() in BLOCKED:
            continue
        if row.get("weekly_projection_valid") is not True:
            continue
        if row.get("opportunity_verified") is not True:
            continue
        if str(row["projection_source"]).strip().lower() in {"dk ppg fallback", "unknown", ""}:
            continue
        try:
            salary = float(row["Salary"])
            projection = float(row["proj"])
        except (ValueError, TypeError):
            continue
        import math
        if salary <= 0 or not math.isfinite(projection) or projection <= 0:
            continue
        eligible += 1
    if eligible < 9:
        reasons.append(f"Only {eligible} verified players; minimum 9 required for NFL Classic")
    return Preflight(not reasons, tuple(reasons), eligible)
