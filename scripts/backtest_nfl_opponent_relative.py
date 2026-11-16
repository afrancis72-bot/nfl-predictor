"""Blind 2026 benchmark of defense-versus-opponent-normal output.

Training weeks only choose a weight; 2026 weeks 4-8 never choose it.
These are own-player historical partial-DK baselines, not the full AI2
game-engine forecasts. Do not enable in production from this result alone.
"""
import json
import numpy as np

from ai2_nfl_opponent_relative import (
    opponent_relative_profiles, prior_player_baselines_for_grading)
from props_data import fetch_weekly

WEIGHTS = (0., .25, .50, 1.0)


def matched_rows(history, season, weeks, *, max_offset=6.0):
    rows = []
    for week in weeks:
        profiles = opponent_relative_profiles(history, season, week)
        lookup = {
            (str(p["defense"]), str(p["position"])): p
            for p in profiles.to_dict("records")
        }
        targets = prior_player_baselines_for_grading(history, season, week)
        for player in targets.to_dict("records"):
            baseline = float(player["pregame_player_baseline_dk"])
            actual = float(player["actual_partial_dk"])
            if not np.isfinite(baseline) or not np.isfinite(actual):
                continue
            profile = lookup.get((player["defense"], player["position"]))
            delta = (float(profile["shrunk_excess_dk"])
                     if profile is not None else 0.)
            rows.append({
                "season": season, "week": week, "player": player["player"],
                "position": player["position"], "defense": player["defense"],
                "baseline": baseline, "actual": actual,
                "relative_signal": float(np.clip(delta, -max_offset, max_offset)),
                "defender_observations": (int(profile["observations"])
                                          if profile is not None else 0)
            })
    if not rows:
        raise ValueError("No pregame eligible player observations for evaluation")
    return rows


def measure(rows, weight):
    y = np.array([x["actual"] for x in rows], float)
    base = np.array([x["baseline"] for x in rows], float)
    delta = np.array([x["relative_signal"] for x in rows], float)
    pred = base + weight * delta
    errors = pred - y
    return {
        "n": len(rows),
        "mae": round(float(np.abs(errors).mean()), 4),
        "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 4),
        "bias": round(float(errors.mean()), 4),
        "signal_coverage_pct": round(float(100 * np.mean(
            np.array([x["defender_observations"] for x in rows]) > 0)), 1),
        "defense_matched_players": int(sum(
            x["defender_observations"] > 0 for x in rows))
    }


def compare(history, *, train_weeks=(9,10,11,12),
            holdout_weeks=(4,5,6,7,8)):
    train = matched_rows(history, 2025, train_weeks)
    holdout = matched_rows(history, 2026, holdout_weeks)
    training = {str(w): measure(train, w) for w in WEIGHTS}
    selected = min(WEIGHTS, key=lambda w: (training[str(w)]["mae"], w))
    testing = {str(w): measure(holdout, w) for w in WEIGHTS}
    by_position = {}
    for pos in ("QB", "RB", "WR", "TE"):
        sample = [r for r in holdout if r["position"] == pos]
        if sample:
            by_position[pos] = {
                "neutral": measure(sample, 0),
                "trained_choice": measure(sample, selected)}
    neutral = testing["0.0"]
    chosen = testing[str(selected)]
    return {
        "method": (
            "Source nflverse; rolling prior-three-to-eight SAME-player/team "
            "completed-game DK baseline; defense residual is actual less "
            "player pregame baseline, winsorized to 25 DK then shrunk with "
            "30 neutral observations. Ten latest completed defense games. "
            "Partial DK excludes fumbles and returns. No game week leaks."),
        "training": "2025 weeks 9-12",
        "holdout": "2026 weeks 4-8; frozen choice from 2025",
        "training_scores": training,
        "holdout_scores": testing,
        "training_selected_weight": selected,
        "holdout_selected_mae_improved": (
            chosen["mae"] < neutral["mae"]),
        "holdout_mean_absolute_error_change": round(
            chosen["mae"] - neutral["mae"], 4),
        "by_position": by_position,
        "status": ("RESEARCH ONLY: even an improved baseline does not validate "
                   "the full football-stat model, fantasy ceiling or ROI"),
    }


if __name__ == "__main__":
    historical = fetch_weekly([2024, 2025, 2026])
    print(json.dumps(compare(historical), indent=2))
