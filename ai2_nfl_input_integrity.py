"""Pregame NFL input-to-output audit and observable role-trend research.

NEVER certify a current-week snap/route role from historical box-score samples.
This module changes no simulated player scoring or existing approved lineups.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from props_data import clean_weekly
from ai2_nfl_matchups import canonical_team
from ai2_nfl_auto_baselines import _norm_name


def player_role_trends(history, salary_rows, season, week):
    """Compare last three *observed pregame appearances* to earlier five.

    A missing appearance is NOT filled with zero; players with too little
    same-team, current-season evidence are explicitly unassessed.
    """
    h = clean_weekly(history)
    h = h[(h["season"].eq(int(season)) & h["week"].lt(int(week)))].copy()
    if h.empty:
        return []
    h["_name"] = h["player_name"].map(_norm_name)
    h["_team"] = h["recent_team"].map(canonical_team)
    h["_position"] = h["position"].astype(str).str.upper().str.strip()
    rows = []
    for player in salary_rows:
        name = str(player.get("Name") or "")
        position = str(player.get("Position") or "").upper().replace("D/ST", "DST")
        team = canonical_team(player.get("TeamAbbrev") or player.get("Team") or "")
        if position not in {"QB", "RB", "WR", "TE"}:
            continue
        metric = ("pass_attempts" if position == "QB" else
                  "rush_attempts_plus_targets" if position == "RB" else "targets")
        matching = h[(h["_name"].eq(_norm_name(name))) &
                     (h["_team"].eq(team)) &
                     (h["_position"].eq(position))].sort_values("week").tail(8)
        values = []
        for _, g in matching.iterrows():
            if position == "QB":
                raw = (g.get("attempts"),)
            elif position == "RB":
                raw = (g.get("carries"), g.get("targets"))
            else:
                raw = (g.get("targets"),)
            parsed = pd.to_numeric(pd.Series(raw), errors="coerce")
            values.append(float(parsed.sum()) if parsed.notna().all() else float("nan"))
        observed = len(matching)
        record = {"Player": name, "Team": team, "Position": position,
                  "Role metric": metric, "Observed current-season games": observed,
                  "Recent 3 per game": None, "Earlier up-to-5 per game": None,
                  "Change %": None, "Role trend": "INSUFFICIENT_HISTORY",
                  "Interpretation": "Not enough same-team box-score games to compare recent and earlier workloads"}
        # Require three recent + at least two distinct older observations.
        if observed >= 5 and np.isfinite(values).all():
            earlier = float(np.mean(values[:-3]))
            recent = float(np.mean(values[-3:]))
            record["Recent 3 per game"] = round(recent, 2)
            record["Earlier up-to-5 per game"] = round(earlier, 2)
            if earlier >= 1:
                change = (recent / earlier - 1) * 100
                record["Change %"] = round(change, 1)
                if change <= -30:
                    record["Role trend"] = "DECLINING"
                    record["Interpretation"] = ("Observed opportunity is at least 30% lower recently; "
                                                "review current snaps/routes/injury before trusting historical forecasts")
                elif change >= 35:
                    record["Role trend"] = "RISING"
                    record["Interpretation"] = ("Recent opportunity increased; verify whether the role change "
                                                "is sustainable or driven by temporary absences")
                else:
                    record["Role trend"] = "NO_LARGE_CHANGE"
                    record["Interpretation"] = "No large change in this box-score opportunity proxy"
            else:
                record["Interpretation"] = "Prior opportunity too small for a stable percentage comparison"
        elif observed >= 5:
            record["Role trend"] = "INCOMPLETE_STATS"
            record["Interpretation"] = "Historical opportunity fields contain missing values; no trend inferred"
        rows.append(record)
    return rows


def input_lineage_rows(research):
    """Describe actual 2026 NFL Classic score pathway, not marketing claims."""
    allowance = research.get("opponent_allowances")
    available_allowance = isinstance(allowance, pd.DataFrame) and not allowance.empty
    injuries = research.get("injury_meta") or {}
    injury_available = injuries.get("source") not in (None, "", "unavailable")
    role_rows = research.get("role_trends") or []
    return [
        {"Information": "Historical pass attempts, passing efficiency and TD/INT", "Data use": "SCORED",
         "Why": "Historical nflverse QB rates enter game-level scenarios"},
        {"Information": "Historical carries, targets, catches, rushing/receiving efficiency", "Data use": "SCORED",
         "Why": "Historical nflverse rushing and receiving rates allocate scenarios"},
        {"Information": "Confirmed unavailable offensive players", "Data use": "SCORED" if injury_available else "UNAVAILABLE",
         "Why": "Confirmed unavailability removes players and redistributes historical opportunity once"},
        {"Information": "Practice report and questionable designation",
         "Data use": "ELIGIBILITY_ONLY" if injury_available else "UNAVAILABLE",
         "Why": "Questionable players are held out; no verified workload estimate for players playing hurt"},
        {"Information": "Opponent fantasy points allowed by position",
         "Data use": "RESEARCH_ONLY" if available_allowance else "UNAVAILABLE",
         "Why": "Calculated for the defensive research panel but not fed into simulated offensive scoring"},
        {"Information": "Opposing defender absences and coverage questions",
         "Data use": "RESEARCH_ONLY" if injury_available else "UNAVAILABLE",
         "Why": "Visible in opponent report; no measured matchup or automatic offensive boost"},
        {"Information": "Last-3 versus earlier same-team workload",
         "Data use": "RESEARCH_ONLY" if role_rows else "UNAVAILABLE",
         "Why": "Pregame box-score trend is a warning, not yet a historically calibrated projection adjustment"},
        {"Information": "DraftKings salary, roster constraints and diversification", "Data use": "OPTIMIZER_ONLY",
         "Why": "Filters eligible lineups and assigns exposures; does not set player scoring probabilities"},
        {"Information": "Verified weekly snaps, routes and depth-chart roles", "Data use": "NOT_CONNECTED",
         "Why": "Historical carry/target rates do not certify this week's role"},
        {"Information": "Team- and opponent-specific pace and market totals/spreads",
         "Data use": "NOT_CONNECTED",
         "Why": "Simulator uses generalized team-play priors and historical QB rates"},
        {"Information": "Weather and wind effects", "Data use": "NOT_CONNECTED",
         "Why": "No verified weather-conditioned statistical model in this NFL Classic pipeline"},
        {"Information": "Projected GPP ownership, duplication and payout model",
         "Data use": "NOT_CONNECTED",
         "Why": "Portfolio prioritizes simulated mean/P99 and exposure, not opponent field behavior"},
    ]


def lineup_role_warnings(lineups, trends):
    """Flag observed role declines for selected golfers—err, NFL players only.

    Keep this as review evidence: do not substitute warnings for player vetoes.
    """
    by_player = {(r["Player"].casefold(), r["Team"]): r for r in trends}
    flagged = []
    for lineup_index, item in enumerate(lineups, 1):
        candidate = item.get("candidate") or {}
        for player in candidate.get("players") or []:
            key = (str(player.get("Name") or "").casefold(),
                   canonical_team(player.get("TeamAbbrev") or ""))
            reading = by_player.get(key)
            if reading and reading["Role trend"] in {"DECLINING", "RISING", "INCOMPLETE_STATS"}:
                flagged.append({"Lineup": lineup_index, **reading})
    return flagged
