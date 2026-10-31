"""Chronological NFL *experimental* role/matchup signal ablation.

This does not affect the production NFL simulator. Every prediction uses weeks
strictly before the scored game. Opponent identity is taken from the known
schedule field in the observed row; no game outcomes are used as forecast input.

A better partial-DK MAE is NOT proof of GPP profitability. Factors here are
hypotheses to disprove, not production coefficients.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from ai2_nfl_input_integrity import player_role_trends
from ai2_nfl_matchup_diagnostics import defensive_allowance
from ai2_nfl_matchups import canonical_team
from props_data import clean_weekly, fetch_weekly, make_rates, dk_classic_points, COUNT_COLS


def compare_pregame_signals(history, season=2026, weeks=(4, 5, 6, 7, 8)):
    h = clean_weekly(history)
    scored = []
    for week in weeks:
        rates = make_rates(history, season, week)
        if rates.empty:
            continue
        defense = defensive_allowance(history, season, week)
        lookups = {(canonical_team(r["defense"]), str(r["position"]).upper()):
                   (float(r["multiplier"]), int(r["player_game_observations"]))
                   for r in defense.to_dict("records")}
        salary_proxy = [{"Name": p["player"], "TeamAbbrev": p["team"], "Position": p["position"]}
                        for p in rates.to_dict("records")]
        trend = player_role_trends(history, salary_proxy, season, week)
        trends = {(r["Player"].casefold(), r["Team"], r["Position"]): r for r in trend}
        actual = h[(h["season"].eq(int(season))) & (h["week"].eq(int(week)))].copy()
        actual["_team"] = actual["recent_team"].map(canonical_team)
        for p in rates.to_dict("records"):
            position = str(p["position"]).upper()
            team = canonical_team(p["team"])
            observed = actual[
                (actual["player_name"].astype(str).str.casefold() == str(p["player"]).casefold()) &
                actual["_team"].eq(team) & actual["position"].astype(str).str.upper().eq(position)]
            if len(observed) != 1:
                continue
            truth = observed.iloc[0]
            predicted = dk_classic_points({src: p.get(dest) for dest, src in COUNT_COLS.items()}, position)
            actual_dk = dk_classic_points(truth.to_dict(), position)
            if not np.isfinite(predicted) or not np.isfinite(actual_dk):
                continue
            player_trend = trends.get((str(p["player"]).casefold(), team, position))
            # Deliberately modest role hypothesis: 20% of measured role delta.
            # Cut off at +/-15%, only after same-team 3+2 observed games.
            role_factor = 1.
            if player_trend and player_trend.get("Change %") is not None:
                role_factor = float(np.clip(1. + .20 * player_trend["Change %"] / 100., .85, 1.15))
            defense_name = canonical_team(truth.get("opponent_team") or "")
            signal = lookups.get((defense_name, position))
            match_factor = (float(np.clip(1. + .5 * (signal[0]-1.), .90, 1.10))
                            if signal and signal[1] >= 20 else 1.)
            scored.append({"week": int(week), "position": position,
                           "actual": float(actual_dk),
                           "history_only": float(predicted),
                           "recent_role": float(predicted * role_factor),
                           "opponent": float(predicted * match_factor),
                           "both": float(predicted * role_factor * match_factor),
                           "role_adjusted": role_factor != 1.,
                           "opponent_adjusted": match_factor != 1.})
    if not scored:
        raise ValueError("No complete out-of-sample pregame predictions for evaluation")
    df = pd.DataFrame(scored)
    comparisons = []
    for (pos, w), group in df.groupby(["position", "week"], dropna=False):
        for model in ("history_only", "recent_role", "opponent", "both"):
            comparisons.append({"position": pos, "week": int(w), "model": model,
                                "players": len(group),
                                "mae": round(float((group[model]-group["actual"]).abs().mean()), 4)})
    overall = []
    for model in ("history_only", "recent_role", "opponent", "both"):
        overall.append({"model": model, "players": len(df),
                        "mae": round(float((df[model]-df["actual"]).abs().mean()), 4),
                        "rmse": round(float(np.sqrt(np.mean((df[model]-df["actual"])**2))), 4)})
    return {"season": int(season), "weeks": sorted(df["week"].unique().tolist()),
            "samples": len(df), "role_coverage": int(df["role_adjusted"].sum()),
            "opponent_coverage": int(df["opponent_adjusted"].sum()),
            "overall": overall, "by_position_week": comparisons,
            "production_change": False,
            "warning": "Partial DK score test only; no sportsbook field, weather or GPP ownership. Do not deploy without per-slate/per-position holdouts."}


if __name__ == "__main__":
    raw = fetch_weekly([2025, 2026])
    print(json.dumps(compare_pregame_signals(raw), indent=2))
