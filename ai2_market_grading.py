"""Grade frozen simulation and sportsbook props against realized player statistics.

This module never changes projections or places wagers. A single game is
diagnostic evidence, not proof of a profitable betting strategy.
"""
from math import isfinite
from ai2_market_challenge import FrozenSimulation

def grade_market_challenge(frozen, props, actuals):
    if not isinstance(frozen, FrozenSimulation):
        raise TypeError("Frozen simulation required")
    if not isinstance(props, list) or not isinstance(actuals, list):
        raise TypeError("Props and actuals must be lists")
    actual_by_key = {}
    for row in actuals:
        key = (str(row["player_id"]), str(row["market_key"]))
        if key in actual_by_key:
            raise ValueError("Duplicate actual player/market")
        value = float(row["actual"])
        if not isfinite(value) or value < 0:
            raise ValueError("Invalid actual statistic")
        actual_by_key[key] = value
    simulated = {(pid, market): sum(samples) / len(samples)
                 for pid, market, samples in frozen.observations}
    output = []
    seen = set()
    for row in props:
        if str(row.get("period", "")).upper() != "FULL":
            continue
        key = (str(row.get("player_id", "")), str(row.get("market_key", "")))
        if key in seen:
            raise ValueError("Duplicate prop player/market; select one bookmaker first")
        seen.add(key)
        if key not in simulated or key not in actual_by_key:
            continue
        line = float(row["line"])
        if not isfinite(line) or line < 0:
            raise ValueError("Invalid prop line")
        actual = actual_by_key[key]
        projection = simulated[key]
        output.append({
            "run_id": frozen.run_id, "simulation_digest": frozen.digest,
            "player_id": key[0], "market_key": key[1],
            "simulation_mean": projection, "market_line": line,
            "actual": actual,
            "simulation_absolute_error": abs(projection - actual),
            "market_line_absolute_error": abs(line - actual),
            "simulation_closer": abs(projection - actual) < abs(line - actual),
            "market_closer": abs(line - actual) < abs(projection - actual),
        })
    return output

def summarize_market_challenge(rows):
    if not rows:
        return {"observations": 0, "status": "INSUFFICIENT_MATCHED_RESULTS"}
    n = len(rows)
    return {
        "observations": n,
        "simulation_mae": sum(r["simulation_absolute_error"] for r in rows) / n,
        "market_line_mae": sum(r["market_line_absolute_error"] for r in rows) / n,
        "simulation_closer_count": sum(r["simulation_closer"] for r in rows),
        "market_closer_count": sum(r["market_closer"] for r in rows),
        "tie_count": sum(not r["simulation_closer"] and not r["market_closer"] for r in rows),
        "status": "DIAGNOSTIC_ONLY",
    }
