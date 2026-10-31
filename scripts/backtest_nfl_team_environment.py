"""Evaluate NFL team-specific pace and passing opportunity before adopting in sims.

Source: pregame nflverse weekly individual QB passing/sack and all-player carries.
Actual plays proxy = sum(team QB attempts + QB sacks + all-player carries).
This proxy is NOT official full team plays. No sportsbook totals or weather used.
Train weights on 2025, evaluate separately on untouched 2026 weeks.
"""
import json
import numpy as np
import pandas as pd
from props_data import fetch_weekly, clean_weekly
from ai2_nfl_matchups import canonical_team


def observed_team_volumes(history):
    h = clean_weekly(history).copy()
    h["team"] = h["recent_team"].map(canonical_team)
    h["opp"] = h["opponent_team"].map(canonical_team) if "opponent_team" in h else ""
    out = []
    for (season, week, team), g in h.groupby(["season", "week", "team"]):
        if not team:
            continue
        q = g[g["position"].astype(str).str.upper().eq("QB")]
        if q.empty:
            continue
        attempts = pd.to_numeric(q.get("attempts"), errors="coerce")
        sacks = pd.to_numeric(q.get("sacks_suffered"), errors="coerce")
        carries = pd.to_numeric(g.get("carries"), errors="coerce")
        if not attempts.notna().any() or attempts.sum() < 8 or not sacks.notna().any():
            continue
        att = float(attempts.fillna(0).sum())
        sack = float(sacks.fillna(0).sum())
        rush = float(carries.fillna(0).sum())
        # No attempt to synthesize unobserved plays or reconstruct penalties.
        out.append({"season": int(season), "week": int(week), "team": team,
                    "opponent": next((v for v in g["opp"].dropna() if str(v)), ""),
                    "pass_attempts": att, "rush_attempts": rush, "sacks": sack,
                    "plays_proxy": att + sack + rush})
    return pd.DataFrame(out)


def _estimate(df, season, week, team, metric, weight, use_opponent=False, opponent=None):
    # NO target week or later appears anywhere in the estimate.
    prior = df[(df["season"] < season) |
               (df["season"].eq(season) & df["week"].lt(week))]
    if prior.empty:
        return np.nan
    league = float(prior[metric].mean())
    own = prior[prior["team"].eq(team)].sort_values(["season", "week"]).tail(6)
    if own.empty:
        return league
    values = own[metric].to_numpy(float)
    w = np.power(.85, np.arange(len(values)-1, -1, -1))
    observed = float(np.average(values, weights=w))
    # Empirical Bayes: sparse team samples shrink toward league prior.
    reliability = len(own) / (len(own) + 4.)
    player = league + weight * reliability * (observed - league)
    if use_opponent and opponent:
        other = prior[prior["team"].eq(opponent)].sort_values(["season", "week"]).tail(6)
        if len(other) >= 3:
            v = other[metric].to_numpy(float)
            wt = np.power(.85, np.arange(len(v)-1, -1, -1))
            their = float(np.average(v, weights=wt))
            their_reliability = len(other)/(len(other)+4.)
            # Symmetric opponent pace hypothesis, not a production adjustment.
            player += .25 * weight * their_reliability * (their - league)
    return float(player)


def evaluate_volume_model(games, season=2026, weeks=range(4, 9), weight=1.0):
    records = []
    targets = games[games["season"].eq(season) & games["week"].isin(list(weeks))]
    for r in targets.to_dict("records"):
        prior = games[(games["season"] < season) |
                      (games["season"].eq(season) & games["week"].lt(r["week"]))]
        if len(prior) < 50:
            continue
        for metric in ("plays_proxy", "pass_attempts"):
            for kind, w, use_opp in (("generic65_32", None, False),
                                    ("pregame_league", 0, False),
                                    ("team_prior", weight, False),
                                    ("team_and_opponent", weight, True)):
                if w is None:
                    pred = 65. if metric == "plays_proxy" else 32.
                else:
                    pred = _estimate(games, season, r["week"], r["team"], metric,
                                     w, use_opponent=use_opp, opponent=r["opponent"])
                if np.isfinite(pred):
                    records.append({"week": int(r["week"]), "metric": metric, "model": kind,
                                    "actual": float(r[metric]), "pred": float(pred)})
    if not records:
        raise ValueError("Not enough historical pregame team observations")
    out = pd.DataFrame(records)
    return [{"metric": key[0], "model": key[1], "count": len(g),
             "mae": round(float((g["actual"]-g["pred"]).abs().mean()), 3),
             "rmse": round(float(np.sqrt(((g["actual"]-g["pred"])**2).mean())), 3)}
            for key,g in out.groupby(["metric", "model"])]


def train_2025_test_2026(history):
    games = observed_team_volumes(history)
    if len(games) < 100:
        raise ValueError("NFL box score history insufficient to estimate team game volume")
    scores = [(w, evaluate_volume_model(games, season=2025,
                weeks=range(8, 19), weight=w)) for w in (.25, .5, .75, 1.0)]
    chosen = {}
    for metric in ("plays_proxy", "pass_attempts"):
        chosen[metric] = min(scores, key=lambda item: next(
            r["mae"] for r in item[1] if r["metric"] == metric and r["model"] == "team_prior"))[0]
    test = {metric: evaluate_volume_model(games, season=2026,
            weeks=range(4, 9), weight=weight)
            for metric,weight in chosen.items()}
    summary = []
    for metric, rows in test.items():
        summary += [r for r in rows if r["metric"] == metric]
    return {"observed_team_game_proxies": len(games),
            "training_season": 2025, "heldout_season": 2026,
            "heldout_weeks": [4,5,6,7,8],
            "team_prior_weights_chosen_without_2026": chosen,
            "heldout": summary,
            "production_change": False,
            "limitations": "Box-score play proxy, not certified snaps or actual possession pace. No scoring model change without further testing."}


if __name__ == "__main__":
    print(json.dumps(train_2025_test_2026(fetch_weekly([2025, 2026])), indent=2))
