"""Out-of-time diagnostic: player-adjusted defense rate by football mechanism.

Only a conditional efficiency study: historical ACTUAL target/carry/attempt
counts are denominators in grading, NOT pregame opportunity forecasts. It
cannot validate full stat-driven DFS forecasts or bankroll outcomes.
"""
import json
import numpy as np

from ai2_nfl_defense_components import (
    defense_component_profiles, component_pregame_grading_rows)
from props_data import fetch_weekly

WEIGHTS = (0., .25, .5, 1.)


def rows_for_period(history, season, weeks):
    results = []
    for week in weeks:
        p = defense_component_profiles(history, season, week)
        lookups = {
            (str(x["defense"]), str(x["position"]), str(x["channel"])): x
            for x in p.to_dict("records")
        }
        for row in component_pregame_grading_rows(
                history, season, week).to_dict("records"):
            opps = float(row["actual_opportunities"])
            actual = float(row["actual_yards"]) / opps
            prior = float(row["prior_yards_per_opp"])
            x = lookups.get((row["defense"], row["position"], row["channel"]))
            factor = float(x["shrunk_efficiency_pct"])/100 if x else 0.
            if not np.isfinite([actual, prior, factor]).all():
                continue
            results.append({
                "position": row["position"], "channel": row["channel"],
                "actual_efficiency": actual,
                "baseline_efficiency": prior,
                "defensive_pct": factor,
                "matched": bool(x),
            })
    if not results:
        raise ValueError("No pregame component rates and eventual actuals")
    return results


def metrics(rows, strength):
    a = np.array([z["actual_efficiency"] for z in rows], float)
    p = np.array([z["baseline_efficiency"] * (
        1. + strength*z["defensive_pct"]) for z in rows], float)
    return {
        "n": len(rows), "mae_yards_per_opp": round(
            float(np.abs(p-a).mean()), 5),
        "rmse_yards_per_opp": round(
            float(np.sqrt(np.mean((p-a)**2))), 5),
        "signed_error_yards_per_opp": round(float(np.mean(p-a)), 5),
        "matched_pct": round(
            100*sum(z["matched"] for z in rows)/len(rows), 2),
    }


def compare(history):
    train = rows_for_period(history, 2025, (9,10,11,12))
    holdout = rows_for_period(history, 2026, (4,5,6,7,8))
    train_results = {str(w):metrics(train,w) for w in WEIGHTS}
    choice = min(WEIGHTS, key=lambda w:(train_results[str(w)]["mae_yards_per_opp"],w))
    test_results = {str(w):metrics(holdout,w) for w in WEIGHTS}
    by_channel = {}
    for channel in sorted(set(r["channel"]+":"+r["position"]
                              for r in holdout)):
        records = [r for r in holdout
                   if r["channel"]+":"+r["position"] == channel]
        by_channel[channel] = {
            "neutral": metrics(records,0.),
            "train_selected": metrics(records,choice)}
    return {
        "train": "2025 Weeks 9-12",
        "untouched_holdout": "2026 Weeks 4-8",
        "training_selected_weight": choice,
        "training": train_results,
        "holdout": test_results,
        "per_channel": by_channel,
        "holdout_mae_improved": (
            test_results[str(choice)]["mae_yards_per_opp"]
            < test_results["0.0"]["mae_yards_per_opp"]),
        "method": ("Per-position QB pass YPA, RB run YPC and RB/WR/TE "
                   "receiving YPT against prior SAME-player/team averages. "
                   "Source pregame nflverse, no current-week target data. "
                   "Historical actual volume used ONLY for conditional "
                   "efficiency grading; not a pregame touch or game simulation."),
        "status": "Research only. Historical sample efficiency is not calibrated player-DFS scoring."
    }


if __name__ == "__main__":
    print(json.dumps(compare(fetch_weekly([2024,2025,2026])), indent=2))
