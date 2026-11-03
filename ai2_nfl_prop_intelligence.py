"""Stat-first NFL player forecasting, compatible with prop-bet comparisons.

Compute each player's source-verified, same-world football-stat distribution
BEFORE constructing DraftKings lineups. All projections and over probabilities
come from simulated actual game events; fantasy-point bonuses are already
included in the independent full-world DraftKings scoring matrix.

No sportsbook line or odds = NO betting edge/recommendation. A modeled
probability must NEVER be described as empirically calibrated until it is
validated on real out-of-time games.
"""
from collections import Counter
from math import isfinite
import numpy as np


STAT_FIELDS = {
    "QB": ("passing_yards", "passing_tds", "interceptions", "rushing_yards",
           "rushing_tds"),
    "RB": ("carries", "rushing_yards", "targets", "receptions",
           "receiving_yards", "rushing_tds", "receiving_tds"),
    "WR": ("targets", "receptions", "receiving_yards", "receiving_tds",
           "carries", "rushing_yards"),
    "TE": ("targets", "receptions", "receiving_yards", "receiving_tds",
           "carries", "rushing_yards", "rushing_tds"),
    "DST": ("sacks", "interceptions", "fumble_recoveries",
            "defensive_tds", "safeties", "points_allowed"),
}
DISPLAY = {
    "passing_yards": "Passing yards",
    "passing_tds": "Passing TDs",
    "interceptions": "Interceptions",
    "rushing_yards": "Rushing yards",
    "rushing_tds": "Rushing TDs",
    "carries": "Carries",
    "targets": "Targets",
    "receptions": "Receptions",
    "receiving_yards": "Receiving yards",
    "receiving_tds": "Receiving TDs",
    "sacks": "Sacks",
    "fumble_recoveries": "Fumble recoveries",
    "defensive_tds": "Defensive TDs",
    "safeties": "Safeties",
    "points_allowed": "Points allowed",
}
MARKET_NAMES = {
    "pass_yds": "passing_yards",
    "rush_yds": "rushing_yards",
    "rec_yds": "receiving_yards",
    "recs": "receptions",
    "pass_tds": "passing_tds",
    "rush_tds": "rushing_tds",
}
MIN_PROP_WORLDS = 100


def _series(detail, name, size):
    if not isinstance(detail, dict) or name not in detail:
        return None
    result = np.asarray(detail[name], dtype=float)
    # Stats are produced in 10,000 shared worlds; the optimizer uses only
    # the first 7,000, reserving the last 3,000 for an untouched holdout.
    # Never use held-out worlds to inform prebuild player decisions.
    if result.ndim != 1 or result.size < size or not np.isfinite(result).all():
        raise ValueError(f"Invalid model prop simulation series: {name}")
    return result[:size]


def _summarize(series):
    return {
        "mean": round(float(series.mean()), 2),
        "median": round(float(np.median(series)), 2),
        "p10": round(float(np.percentile(series, 10)), 2),
        "p90": round(float(np.percentile(series, 90)), 2),
    }


def _scoring_breakdown(position, detail, points):
    """Derive points from the SAME scenario stats; fail on mismatches.

    Separate receptions/yardage, TDs and bonuses rather than applying TD
    odds to an already TD-included fantasy projection (double counting).
    """
    n = len(points)
    if not isinstance(detail, dict):
        return {}
    required = {
        "QB": ("passing_yards", "passing_tds", "interceptions",
               "rushing_yards", "rushing_tds"),
        "RB": ("receptions", "receiving_yards", "receiving_tds",
               "rushing_yards", "rushing_tds"),
        "WR": ("receptions", "receiving_yards", "receiving_tds",
               "rushing_yards", "rushing_tds"),
        "TE": ("receptions", "receiving_yards", "receiving_tds",
               "rushing_yards", "rushing_tds"),
    }.get(position)
    if position == "DST":
        defense_tds = _series(detail, "defensive_tds", n)
        if defense_tds is None:
            return {}
        td_points = 6.0 * defense_tds
        non_td = points - td_points
        if np.any(~np.isfinite(non_td)):
            raise ValueError("Invalid DST scenario score breakdown")
        return {
            "Model non-TD DK": round(float(non_td.mean()), 2),
            "Model expected TD DK": round(float(td_points.mean()), 2),
            "Model chance 1+ TD involvement %": round(
                100 * float((defense_tds >= 1).mean()), 1),
            "Model chance 10+ DK without TD %": round(
                100 * float((non_td >= 10).mean()), 1),
            "Scoring breakdown": "Same-world DST TD points isolated",
        }
    if required is None:
        return {}
    series = {key: _series(detail, key, n) for key in required}
    if any(v is None for v in series.values()):
        return {}
    if position == "QB":
        pass_y = series["passing_yards"]
        rush_y = series["rushing_yards"]
        base = (.04 * pass_y + .1 * rush_y - series["interceptions"])
        td_points = 4 * series["passing_tds"] + 6 * series["rushing_tds"]
        bonuses = 3 * (pass_y >= 300) + 3 * (rush_y >= 100)
        total = None
        touchdown_count = series["passing_tds"] + series["rushing_tds"]
    else:
        rush_y = series["rushing_yards"]
        rec_y = series["receiving_yards"]
        receptions = series["receptions"]
        total = rush_y + rec_y
        base = .1 * total + receptions
        td_points = 6 * (series["rushing_tds"] + series["receiving_tds"])
        bonuses = 3 * (rush_y >= 100) + 3 * (rec_y >= 100)
        touchdown_count = series["rushing_tds"] + series["receiving_tds"]
    reconstructed = base + td_points + bonuses
    if not np.allclose(reconstructed, points, atol=.02, rtol=0):
        raise ValueError("Model stat projection does not reconcile to DK points")
    non_td = base + bonuses
    result = {
        "Model yards/receptions DK": round(float(base.mean()), 2),
        "Model bonus DK": round(float(bonuses.mean()), 2),
        "Model non-TD DK": round(float(non_td.mean()), 2),
        "Model expected TD DK": round(float(td_points.mean()), 2),
        "Model chance 1+ TD involvement %": round(
            100 * float((touchdown_count >= 1).mean()), 1),
        "Model chance 10+ DK without TD %": round(
            100 * float((non_td >= 10).mean()), 1),
        "Scoring breakdown": "Reconciled: yardage/receptions + TDs + bonuses",
    }
    if total is not None:
        result["Total scrimmage yards projection"] = round(
            float(total.mean()), 2)
        result["Total scrimmage yards median"] = round(
            float(np.median(total)), 2)
    return result


def player_prop_decisions(players, stat_draws, point_draws, pregame_role=None,
                          market_lines=None):
    """Return forecasts and market comparison with source-aware confidence.

    market_lines optionally maps (DraftKings ID, stat name) -> {'line': x,
    'over_american_odds': y}. Only when a real external line is supplied can
    this function label 'over' / 'under' versus a sportsbook number.
    Source / time verification of market observations belongs in ingestion.
    """
    n = len(point_draws)
    if (n < MIN_PROP_WORLDS or point_draws.ndim != 2
            or point_draws.shape[1] != len(players)
            or not np.isfinite(point_draws).all()):
        raise ValueError("Invalid NFL same-world player points simulation")
    pregame_role = pregame_role or {}
    market_lines = market_lines or {}
    name_count = Counter(str(x) for x in players["Name"])
    result = []
    for col, player in players.reset_index(drop=True).iterrows():
        name = str(player["Name"])
        position = str(player["Position"]).upper()
        ident = str(player["ID"])
        points = np.asarray(point_draws[:, col], dtype=float)
        evidence = pregame_role.get(ident, {})
        role = str(evidence.get("Role gate", "NOT_ASSESSED"))
        individual = (stat_draws.get(name)
                      if name_count[name] == 1 else None)
        row = {
            "Player": name,
            "Team": str(player["TeamAbbrev"]),
            "Position": position,
            "DraftKings ID": ident,
            "Salary": int(player["Salary"]),
            "Historical role validation": role,
            "Role evidence": str(evidence.get("Role rationale", "Not assessed")),
            "Model mean DK": round(float(points.mean()), 2),
            "Model median DK": round(float(np.median(points)), 2),
            "Model P10 DK": round(float(np.percentile(points, 10)), 2),
            "Model P90 DK": round(float(np.percentile(points, 90)), 2),
            "Model chance 10+ DK %": round(100 * float((points >= 10).mean()), 1),
            "Model chance 15+ DK %": round(100 * float((points >= 15).mean()), 1),
            "Model chance 20+ DK %": round(100 * float((points >= 20).mean()), 1),
            "Model chance <=5 DK %": round(100 * float((points <= 5).mean()), 1),
            "Model mean DK / $1K": round(
                float(points.mean()) / (float(player["Salary"]) / 1000), 2),
            "Stat evidence": "Verified in same-world football outcomes" if
               isinstance(individual, dict) and position in STAT_FIELDS else
               "Unavailable / ambiguous - no stat-specific confidence",
            "Market comparison": "NO VERIFIED SPORTSBOOK LINE",
            "Predictive calibration": "UNVALIDATED: modeled probabilities, not observed hit rates",
        }
        row.update(_scoring_breakdown(position, individual, points))
        for stat in STAT_FIELDS.get(position, ()):
            vals = _series(individual, stat, n)
            if vals is None:
                row[f"{DISPLAY[stat]} projection"] = None
                continue
            summary = _summarize(vals)
            row[f"{DISPLAY[stat]} projection"] = summary["mean"]
            row[f"{DISPLAY[stat]} median"] = summary["median"]
            row[f"{DISPLAY[stat]} P10"] = summary["p10"]
            row[f"{DISPLAY[stat]} P90"] = summary["p90"]
            market = market_lines.get((ident, stat))
            if not market:
                continue
            line = float(market["line"])
            if not isfinite(line) or line < 0:
                raise ValueError(f"Invalid supplied market prop line for {name} {stat}")
            p_over = float((vals > line).mean())
            p_under = float((vals < line).mean())
            p_push = float((vals == line).mean())
            row[f"{DISPLAY[stat]} market line"] = line
            row[f"{DISPLAY[stat]} model over %"] = round(100 * p_over, 1)
            row[f"{DISPLAY[stat]} model under %"] = round(100 * p_under, 1)
            row[f"{DISPLAY[stat]} model push %"] = round(100 * p_push, 1)
            row["Market comparison"] = "MODELED PROBABILITY ONLY - NOT CALIBRATED"
            if "over_american_odds" in market:
                odds = float(market["over_american_odds"])
                if not isfinite(odds) or abs(odds) < 100:
                    raise ValueError("Invalid actual American odds for over prop")
                win_profit = 100 / abs(odds) if odds < 0 else odds / 100
                row[f"{DISPLAY[stat]} estimated unit EV"] = round(
                    p_over * win_profit - p_under, 3)
                # Push is a refund; not a win. This is uncalibrated
                # model-implied EV, never a confident real-world bet.
        result.append(row)
    return result
