"""Auditable pregame historical-stat baseline; never certifies current-week opportunity."""
import re
import numpy as np
import pandas as pd
from props_data import make_rates, dk_classic_points

ALIASES = {"JAC": "JAX", "LA": "LAR", "WAS": "WSH"}
def _norm_name(value):
    s = re.sub(r"[^a-z0-9 ]", " ", str(value).lower())
    return " ".join(x for x in s.split() if x not in {"jr", "sr", "ii", "iii", "iv"})

def _team(value):
    v = str(value).strip().upper()
    return ALIASES.get(v, v)

def build_historical_baselines(salary_rows, history, season, week):
    """Match observed pre-kickoff player rates to DK IDs and compute transparent baselines.

    Baselines omit fumbles and return TDs, and do not incorporate matchup,
    injuries, projected snaps, or verified opportunity. Never mark as eligible.
    """
    rates = make_rates(history, season, week)
    if rates.empty:
        raise ValueError("No historical rates for requested pregame week")
    by_identity = {}
    for _, r in rates.iterrows():
        key = (_norm_name(r["player"]), _team(r["team"]), str(r["position"]).upper())
        by_identity.setdefault(key, []).append(r)
    out = []
    for s in salary_rows:
        pos = str(s.get("Position", "")).upper().replace("D/ST", "DST")
        key = (_norm_name(s.get("Name", "")), _team(s.get("TeamAbbrev", "")), pos)
        matches = by_identity.get(key, [])
        row = {"ID": str(s.get("ID", "")), "Name": s.get("Name", ""),
               "Team": s.get("TeamAbbrev", ""), "Position": pos,
               "Salary": s.get("Salary", ""), "DK Status": s.get("Status", ""),
               "Game Info": s.get("Game Info", ""),
               "historical_partial_dk": np.nan, "proj": np.nan, "games_observed": 0,
               "latest_observed_season": np.nan, "latest_observed_week": np.nan,
               "pass_attempts_pg": np.nan, "carries_pg": np.nan, "targets_pg": np.nan,
               "match_status": "no unique historical match",
               "projection_source": "nflverse historical weighted stats (incomplete baseline)",
               "weekly_projection_valid": False, "opportunity_verified": False}
        if len(matches) == 1 and pos in {"QB", "RB", "WR", "TE"}:
            r = matches[0]
            stats = {stat_name: r[rate_name] for stat_name, rate_name in (
                ("passing_yards", "pass_yards_pg"),
                ("passing_tds", "passing_tds_pg"),
                ("interceptions", "interceptions_pg"),
                ("rushing_yards", "rushing_yards_pg"),
                ("rushing_tds", "rushing_tds_pg"),
                ("receiving_yards", "receiving_yards_pg"),
                ("receptions", "receptions_pg"),
                ("receiving_tds", "receiving_tds_pg"))}
            score = dk_classic_points(stats, pos)
            row["historical_partial_dk"] = float(score) if np.isfinite(score) else np.nan
            row["games_observed"] = int(r["games_played"])
            row["latest_observed_season"] = int(r["latest_observed_season"])
            row["latest_observed_week"] = int(r["latest_observed_week"])
            for field in ("pass_attempts_pg", "carries_pg", "targets_pg"):
                row[field] = r[field]
            row["proj"] = row["historical_partial_dk"]
            row["match_status"] = "matched" if np.isfinite(score) else "incomplete historical stats"
        status = str(s.get("Status", "")).strip().upper()
        if status in {"IR", "OUT", "O", "D", "DOUBTFUL", "PUP", "NFI", "SUSPENDED", "INACTIVE"}:
            row["readiness_reason"] = "blocked by DraftKings injury designation"
        elif status in {"Q", "QUESTIONABLE"}:
            row["readiness_reason"] = "questionable: official availability and role review required"
        elif pos == "DST":
            row["readiness_reason"] = "DST needs separate defensive-stat projection"
        elif row["match_status"] != "matched":
            row["readiness_reason"] = "insufficient or unmatched historical statistics"
        elif row["games_observed"] < 3:
            row["readiness_reason"] = "sparse historical sample: role review required"
        else:
            row["readiness_reason"] = "historical estimate only: current-week role and matchup unverified"
        out.append(row)
    return pd.DataFrame(out)
