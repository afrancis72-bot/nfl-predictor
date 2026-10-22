"""Leakage-aware historical NFL Classic scoring and calibration utilities.

Research-only: does not authorize lineup generation or export.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite, sqrt
from statistics import mean
from ai2_nfl_lineup import check_classic_lineup

@dataclass(frozen=True)
class BacktestMetrics:
    observations: int
    mae: float
    rmse: float
    bias: float

def _timestamp(value):
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("Timestamp must be ISO-8601")
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Timestamp must include timezone")
    return result.astimezone(timezone.utc)

def score_projection_history(records):
    """Score historical predictions without accepting post-lock predictions.

    Each record must contain slate_id, player_id, kickoff_at, prediction_asof,
    projected_dk_points, and actual_dk_points. Duplicated slate/player keys fail.
    """
    errors = []
    seen = set()
    for record in records:
        key = (str(record["slate_id"]).strip(), str(record["player_id"]).strip())
        if not all(key) or key in seen:
            raise ValueError("Missing or duplicate slate/player identifier")
        seen.add(key)
        if _timestamp(record["prediction_asof"]) >= _timestamp(record["kickoff_at"]):
            raise ValueError("Prediction timestamp must precede kickoff")
        try:
            prediction = float(record["projected_dk_points"])
            actual = float(record["actual_dk_points"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid fantasy point value") from exc
        if not isfinite(prediction) or not isfinite(actual) or prediction < 0:
            raise ValueError("Nonfinite or negative projected points")
        errors.append(prediction - actual)
    if not errors:
        raise ValueError("At least one historical observation required")
    return BacktestMetrics(
        observations=len(errors),
        mae=mean(abs(e) for e in errors),
        rmse=sqrt(mean(e * e for e in errors)),
        bias=mean(errors),
    )

def evaluate_lineup_history(lineup, actual_by_id, *, salary_floor=47500, salary_cap=50000):
    """Return realized DK points for a legal lineup, without optimizing it."""
    problems = check_classic_lineup(lineup, salary_floor=salary_floor, salary_cap=salary_cap)
    if problems:
        raise ValueError("Invalid lineup: " + "; ".join(problems))
    total = 0.0
    for player in lineup:
        identifier = str(player["ID"]).strip()
        if identifier not in actual_by_id:
            raise ValueError("Missing historical actual for player " + identifier)
        score = float(actual_by_id[identifier])
        if not isfinite(score):
            raise ValueError("Nonfinite historical actual")
        total += score
    return total

def evaluate_portfolio_profit(lineup_scores, payouts, entry_fee):
    """Compute realized net profit from supplied contest payout data only."""
    if len(lineup_scores) != len(payouts) or not lineup_scores:
        raise ValueError("Scores and payouts must be nonempty and aligned")
    fee = float(entry_fee)
    if not isfinite(fee) or fee <= 0:
        raise ValueError("Entry fee must be positive")
    amounts = [float(p) for p in payouts]
    if any(not isfinite(p) or p < 0 for p in amounts):
        raise ValueError("Payouts must be finite and nonnegative")
    cost = fee * len(amounts)
    winnings = sum(amounts)
    return {"entries": len(amounts), "entry_cost": cost,
            "winnings": winnings, "net_profit": winnings - cost,
            "roi": (winnings - cost) / cost,
            "cash_rate": sum(p > 0 for p in amounts) / len(amounts)}
