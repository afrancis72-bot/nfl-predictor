"""Evidence gate for inexpensive NFL Classic WR/RB selections.

A supplied boolean is not independent proof of opportunity. For salary-saving
players require concrete projected or recent volume, or a documented injury role.
Missing evidence fails closed; never manufacture metrics.
"""
from math import isfinite

def _number(row, names):
    for key in names:
        value = row.get(key)
        try:
            result = float(value)
        except (ValueError, TypeError):
            continue
        if isfinite(result):
            return result
    return None

def player_opportunity_evidence(row, *, value_ceiling=5000):
    position = str(row.get("Position", "")).upper().strip()
    try:
        salary = float(row.get("Salary", 0))
    except (ValueError, TypeError):
        return False, "Missing salary"
    if position not in {"WR", "RB"} or salary >= value_ceiling:
        return True, "Not a low-cost RB/WR"
    if row.get("opportunity_verified") is not True:
        return False, "Weekly opportunity not verified"
    if position == "WR":
        projection = _number(row, ("projected_targets", "proj_targets"))
        recent = _number(row, ("recent_targets_per_game", "targets_per_game_last_3"))
        routes = _number(row, ("route_participation", "routes_pct"))
        if projection is not None and projection >= 4:
            return True, f"Projected targets {projection:.1f}"
        if recent is not None and recent >= 4:
            return True, f"Recent targets/game {recent:.1f}"
        if routes is not None and routes >= 0.65:
            return True, f"Route participation {routes:.0%}"
    else:
        touches = _number(row, ("projected_touches", "proj_touches"))
        recent = _number(row, ("recent_touches_per_game", "touches_per_game_last_3"))
        snaps = _number(row, ("snap_share", "recent_snap_share"))
        if touches is not None and touches >= 10:
            return True, f"Projected touches {touches:.1f}"
        if recent is not None and recent >= 10:
            return True, f"Recent touches/game {recent:.1f}"
        if snaps is not None and snaps >= 0.40:
            return True, f"Snap share {snaps:.0%}"
    injury = row.get("injury_opportunity_verified") is True
    reason = str(row.get("injury_opportunity_reason") or "").strip()
    if injury and len(reason) >= 15:
        return True, f"Documented injury role: {reason}"
    return False, "Insufficient numeric role evidence; verify targets/routes/touches or documented injury role"
