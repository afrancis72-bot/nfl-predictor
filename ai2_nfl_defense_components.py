"""Talent-adjusted defense efficiency by football mechanism, research only.

Pass YPA, RB run YPC and RB/WR/TE receiving yards per target are observed
against each opponent's OWN earlier efficiency. Weights and recent windows are
bounded and inspectable. In contrast to raw fantasy points allowed, a star's
normal yards per opportunity do not register as a weak defender matchup.

No touchdowns, future box scores, individual CB alignments, or new touches are
predicted here. No factor from this module is scored by the live simulator.
"""
from __future__ import annotations

from collections import defaultdict, deque
import numpy as np
import pandas as pd

from props_data import clean_weekly
from ai2_nfl_matchups import canonical_team
from ai2_nfl_opponent_relative import _identity

SOURCE = "nflverse pregame own-player rate vs completed opponent_team"
# channel -> numerator, denominator, minimum previous total opps,
# minimum observed opps in the single result game, max absolute
# residual yards per observed opportunity for outlier resistance.
METRICS = {
    ("QB", "pass_yards_per_attempt"): ("passing_yards", "attempts", 45., 12., 5.),
    ("RB", "rush_yards_per_carry"): ("rushing_yards", "carries", 18., 4., 5.),
    ("RB", "receive_yards_per_target"): ("receiving_yards", "targets", 8., 2., 9.),
    ("WR", "receive_yards_per_target"): ("receiving_yards", "targets", 12., 3., 9.),
    ("TE", "receive_yards_per_target"): ("receiving_yards", "targets", 8., 2., 9.),
}
COLUMNS = ["defense", "position", "channel", "observations", "defense_games",
           "opportunities", "expected_yards_per_opp", "actual_yards_per_opp",
           "shrunk_efficiency_pct", "source", "target_season", "target_week"]


def _observations(history, season, week, *, include_week=False, window=8):
    h = clean_weekly(history)
    if "opponent_team" not in h.columns:
        raise ValueError("Missing historical opponent_team for statistical matchups")
    h = h[(h["season"] >= int(season)-1) &
          ((h["season"] < int(season)) |
           ((h["season"] == int(season)) &
            ((h["week"] <= int(week)) if include_week
             else (h["week"] < int(week)))))].copy()
    h = h.sort_values(["season", "week", "recent_team", "player_name"],
                      kind="stable")
    prior = defaultdict(lambda: deque(maxlen=window))
    seen = set()
    out = []
    for row in h.to_dict("records"):
        player_key = _identity(row)
        unique = (int(row["season"]), int(row["week"]), player_key)
        if unique in seen:
            raise ValueError("Duplicate player and week in efficiency history")
        seen.add(unique)
        pos = player_key[2]
        raw_defense = row.get("opponent_team")
        defense = (canonical_team(raw_defense)
                   if pd.notna(raw_defense) else "")
        if defense in {"", "NAN", "NONE", "<NA>", "UNKNOWN"}:
            defense = ""
        history_rows = prior[player_key]
        for (channel_pos, channel), (
                numerator, denominator, min_prev, min_current, max_per_opp) in METRICS.items():
            if pos != channel_pos or not defense or len(history_rows) < 3:
                continue
            try:
                actual_n = float(row.get(numerator, np.nan))
                actual_d = float(row.get(denominator, np.nan))
            except (TypeError, ValueError):
                continue
            if (not np.isfinite([actual_n, actual_d]).all()
                    or actual_d < min_current):
                continue
            available = []
            for old in history_rows:
                try:
                    n = float(old.get(numerator, np.nan))
                    d = float(old.get(denominator, np.nan))
                except (TypeError, ValueError):
                    continue
                if np.isfinite([n, d]).all() and d > 0:
                    available.append((n, d))
            if len(available) < 2:
                continue
            previous_count = sum(d for n, d in available)
            if previous_count < min_prev:
                continue
            baseline_efficiency = sum(n for n, d in available) / previous_count
            expected = baseline_efficiency * actual_d
            residual = float(np.clip(
                actual_n - expected, -max_per_opp*actual_d,
                max_per_opp*actual_d))
            out.append({
                "season": int(row["season"]), "week": int(row["week"]),
                "player": str(row["player_name"]), "position": pos,
                "defense": defense, "channel": channel,
                "actual_opportunities": actual_d,
                "actual_yards": actual_n,
                "expected_yards_at_actual_volume": expected,
                "bounded_excess_yards": residual,
                "prior_yards_per_opp": baseline_efficiency,
            })
        prior[player_key].append(row)
    return pd.DataFrame(out)


def defense_component_profiles(history, season, week, *,
                               defense_window=10, min_observations=7,
                               min_opportunities=28, shrink_opportunities=110.):
    if defense_window <= 0 or min_observations < 1:
        raise ValueError("Invalid defense observation constraints")
    if min_opportunities <= 0 or shrink_opportunities <= 0:
        raise ValueError("Invalid opportunity prior")
    observations = _observations(history, season, week)
    if observations.empty:
        return pd.DataFrame(columns=COLUMNS)
    output = []
    for defense, team in observations.groupby("defense"):
        weeks = sorted(set(zip(team["season"].astype(int),
                               team["week"].astype(int))))
        recent = set(weeks[-int(defense_window):])
        active = team[[ (int(x["season"]), int(x["week"])) in recent
                        for x in team.to_dict("records")]]
        for (pos, channel), group in active.groupby(["position", "channel"]):
            opportunities = float(group["actual_opportunities"].sum())
            if len(group) < min_observations or opportunities < min_opportunities:
                continue
            expected = float(group["expected_yards_at_actual_volume"].sum())
            actual = float(group["actual_yards"].sum())
            bounded = float(group["bounded_excess_yards"].sum())
            if expected <= 0:
                continue
            raw_pct = bounded / expected
            reliability = opportunities / (opportunities + shrink_opportunities)
            factor_pct = float(np.clip(raw_pct * reliability, -.15, .15))
            output.append({
                "defense": defense, "position": pos, "channel": channel,
                "observations": int(len(group)),
                "defense_games": int(group[["season","week"]].drop_duplicates().shape[0]),
                "opportunities": int(opportunities),
                "expected_yards_per_opp": round(expected / opportunities, 4),
                "actual_yards_per_opp": round(actual / opportunities, 4),
                "shrunk_efficiency_pct": round(factor_pct * 100, 3),
                "source": SOURCE, "target_season": int(season),
                "target_week": int(week),
            })
    return pd.DataFrame(output, columns=COLUMNS)


def component_pregame_grading_rows(history, season, week):
    """Results become available after the game. Expected rates are pregame."""
    out = _observations(history, season, week, include_week=True)
    if out.empty:
        return out
    return out[(out["season"] == int(season)) &
               (out["week"] == int(week))].reset_index(drop=True)
