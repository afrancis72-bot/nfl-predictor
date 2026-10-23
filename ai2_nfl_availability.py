"""Conservative current-week NFL availability screening.

DraftKings status is a useful warning, never an independent official injury
verification. Unknown and questionable statuses require explicit review.
"""
import pandas as pd

BLOCKED = {"OUT", "O", "IR", "INJURED RESERVE", "PUP", "SUSP", "SUSPENDED", "DOUBTFUL", "D"}
REVIEW = {"Q", "QUESTIONABLE", "GTD", "GAME TIME DECISION", "UNKNOWN", ""}
READY_LABEL = "No DK status restriction; official participation still unverified"


def availability_report(pool):
    if not isinstance(pool, pd.DataFrame):
        raise ValueError("Expected player pool DataFrame")
    if not {"Name", "TeamAbbrev", "Position"}.issubset(pool.columns):
        raise ValueError("Pool requires player, team and position columns")
    status_col = next((c for c in ("dk_status", "Status", "status") if c in pool.columns), None)
    rows = pool.copy()
    if status_col is None:
        rows["availability_status"] = "MISSING"
        rows["availability_gate"] = "REVIEW"
        rows["availability_reason"] = "No current-week DK status field; verify official injury report"
        return rows
    statuses = rows[status_col].astype("string").fillna("").str.upper().str.strip()
    rows["availability_status"] = statuses.replace("", "UNKNOWN")
    rows["availability_gate"] = statuses.map(
        lambda s: "BLOCK" if s in BLOCKED else "REVIEW" if s in REVIEW else "REVIEW"
    )
    rows["availability_reason"] = statuses.map(
        lambda s: "Reported unavailable; exclude until verified" if s in BLOCKED
        else "Status uncertain; check official injury report" if s in REVIEW
        else READY_LABEL
    )
    # All non-blocked statuses remain REVIEW: a salary export cannot certify
    # active-game participation or expected touches.
    return rows
