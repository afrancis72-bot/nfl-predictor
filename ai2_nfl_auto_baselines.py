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
               "historical_partial_dk": np.nan, "games_observed": 0,
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
            row["match_status"] = "matched" if np.isfinite(score) else "incomplete historical stats"
        out.append(row)
    return pd.DataFrame(out)
