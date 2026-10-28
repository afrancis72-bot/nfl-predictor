"""Conservative, auditable workload adjustment for officially unavailable teammates.

This is a pregame opportunity estimate, NOT certification of an individual role.
All deltas depend on observed historical rates, never manufactured depth charts.
"""
import math
import pandas as pd

UNAVAILABLE = {"OUT", "O", "INACTIVE", "IR", "PUP", "NFI", "DOUBTFUL", "D", "SUSPENDED"}
VOLUMES = {
    "RB": {"carries_pg": 0.75, "targets_pg": 0.45},
    "WR": {"targets_pg": 0.60},
    "TE": {"targets_pg": 0.45},
}

def redistribute_opportunities(rates, statuses, *, max_multiplier=1.60):
    """Return adjusted observed rates and an explainable change log.

    statuses: {normalized (player, team): status}; only an explicit unavailable
    game status frees opportunity. Allocate to same-team eligible positions with
    observed touches, applying a multiplier cap. Undistributed volume is lost,
    rather than invented. Does not authorize roster eligibility.
    """
    if not isinstance(rates, pd.DataFrame):
        raise TypeError("rates must be a pandas DataFrame")
    required = {"player", "team", "position", "carries_pg", "targets_pg"}
    missing = required - set(rates.columns)
    if missing:
        raise ValueError("Missing history fields: " + ", ".join(sorted(missing)))
    if not 1 <= float(max_multiplier) <= 3:
        raise ValueError("Invalid workload multiplier")
    result = rates.copy(deep=True).reset_index(drop=True)
    result["projection_opportunity_adjusted"] = False
    for field in ("carries_pg", "targets_pg"):
        result[field] = pd.to_numeric(result[field], errors="coerce")
    def norm(v):
        return "".join(ch for ch in str(v).lower() if ch.isalnum())
    keys = {(norm(p), norm(t)): str(s).upper().strip()
            for (p, t), s in statuses.items()}
    unavailable = [
        i for i, row in result.iterrows()
        if keys.get((norm(row["player"]), norm(row["team"])), "") in UNAVAILABLE
    ]
    changes = []
    for i in unavailable:
        out = result.loc[i]
        position = str(out["position"]).upper()
        for field, fraction in VOLUMES.get(position, {}).items():
            volume = out[field]
            if not pd.notna(volume) or not math.isfinite(float(volume)) or volume <= 0:
                continue
            freed = float(volume) * fraction
            # RB carries remain within the backfield. Receiving targets may be
            # allocated among all eligible pass catchers according to observed usage.
            eligible_positions = {"RB"} if field == "carries_pg" else {"RB", "WR", "TE"}
            indices = [
                j for j, row in result.iterrows()
                if j != i
                and norm(row["team"]) == norm(out["team"])
                and str(row["position"]).upper() in eligible_positions
                and keys.get((norm(row["player"]), norm(row["team"])), "") not in UNAVAILABLE
                and pd.notna(row[field]) and float(row[field]) > 0
            ]
            # Never allocate to a player with no observed relevant opportunity.
            remaining = freed
            weights = {j: float(result.at[j, field]) for j in indices}
            for _ in range(len(indices)):
                active = [j for j in indices if weights[j] > 0
                          and result.at[j, field] < rates.iloc[j][field] * max_multiplier - 1e-8]
                if not active or remaining <= 1e-8:
                    break
                total = sum(weights[j] for j in active)
                given = 0.0
                for j in active:
                    base = float(rates.iloc[j][field])
                    room = max(0.0, base * max_multiplier - float(result.at[j, field]))
                    amount = min(room, remaining * weights[j] / total)
                    if amount > 0:
                        result.at[j, field] += amount
                        result.at[j, "projection_opportunity_adjusted"] = True
                        changes.append({"injured_player": str(out["player"]),
                                        "beneficiary": str(result.at[j, "player"]),
                                        "team": str(out["team"]), "stat": field,
                                        "added_per_game": round(amount, 3),
                                        "basis": "Observed team usage, capped redistribution"})
                        given += amount
                remaining -= given
                if given <= 1e-8:
                    break
    if unavailable:
        result.loc[unavailable, "optimizer_excluded_for_injury"] = True
    return result, pd.DataFrame(changes)
