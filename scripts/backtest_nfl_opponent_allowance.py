"""Chronological opponent-allowance ablation for NFL research.

For each 2026 test week, both the player baseline and defending-team allowance
use only earlier game observations. No opponent injury uplift is applied: its
role/route impact is not calibrated. A lower MAE alone isn't contest ROI.
"""
import json
import numpy as np
from ai2_nfl_matchup_diagnostics import defensive_allowance
from ai2_nfl_matchups import canonical_team
from props_data import (fetch_weekly, clean_weekly, make_rates, dk_classic_points,
                        COUNT_COLS)


def run_walkforward(history, season=2026, weeks=range(4, 8)):
    h = clean_weekly(history)
    records = []
    for week in weeks:
        prior_rates = make_rates(history, season, week)
        prior_defense = defensive_allowance(history, season, week)
        lookups = {
            (str(x["defense"]), str(x["position"]).upper()):
            (float(x["multiplier"]), int(x["player_game_observations"]))
            for x in prior_defense.to_dict("records")}
        actual = h[h["season"].eq(season) & h["week"].eq(week)]
        for row in prior_rates.to_dict("records"):
            pos, team = str(row["position"]).upper(), str(row["team"]).upper()
            observed = actual[
                actual["player_name"].eq(str(row["player"])) &
                actual["recent_team"].eq(team) &
                actual["position"].str.upper().eq(pos)]
            if len(observed) != 1:
                continue
            fact = observed.iloc[0]
            opponent = canonical_team(fact.get("opponent_team") or "")
            signal = lookups.get((opponent, pos))
            if not signal or signal[1] < 20:
                continue
            prior_stats = {source: row.get(dest) for dest, source in COUNT_COLS.items()}
            predicted = dk_classic_points(prior_stats, pos)
            actual_points = dk_classic_points(fact.to_dict(), pos)
            if not np.isfinite(predicted) or not np.isfinite(actual_points):
                continue
            # Hypothesis only: 50%-shrunk historical opponent adjustment,
            # bounded +/-10%. This is NOT a live engine setting.
            factor = float(np.clip(1 + .5 * (signal[0] - 1), .9, 1.1))
            adjusted = float(predicted * factor)
            records.append({
                "season": int(season), "week": int(week),
                "player": row["player"], "position": pos,
                "opposing_defense": opponent,
                "baseline": float(predicted), "adjusted": adjusted,
                "actual": float(actual_points), "factor": factor})
    if not records:
        raise ValueError("No valid pregame holdout data to score")
    baseline_error = np.array([abs(z["baseline"] - z["actual"]) for z in records])
    adjusted_error = np.array([abs(z["adjusted"] - z["actual"]) for z in records])
    per_week = []
    for w in sorted({r["week"] for r in records}):
        items = [r for r in records if r["week"] == w]
        per_week.append({
            "week": w, "n": len(items),
            "baseline_mae": round(float(np.mean([abs(x["baseline"]-x["actual"]) for x in items])), 4),
            "adjusted_mae": round(float(np.mean([abs(x["adjusted"]-x["actual"]) for x in items])), 4)})
    report = {
        "season": season, "sample_count": len(records),
        "baseline_mae": round(float(baseline_error.mean()), 4),
        "adjusted_mae": round(float(adjusted_error.mean()), 4),
        "mae_improvement_pct": round(float(100*(baseline_error.mean()-adjusted_error.mean())
                                           / baseline_error.mean()), 2),
        "weeks": per_week,
        "recommendation": "Do not deploy uncalibrated adjustments based on this exploratory test alone",
        "method": "week-by-week pregame historical nflverse partial DK; excludes any post-kickoff info"}
    return report


if __name__ == "__main__":
    h = fetch_weekly([2025, 2026])
    print(json.dumps(run_walkforward(h), indent=2))
