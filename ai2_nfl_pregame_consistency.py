"""Evidence-first NFL prebuild role consistency and chronological value study.

Uses only pre-kickoff NFL weekly observations to decide if a low-salary
RB/WR/TE has documented opportunity. A single spike is not enough.
The replay function separately joins *future results* for POSTGAME evaluation
without ever feeding those results into pregame eligibility.

This module does not identify historical tournament winners: those require
actual contest standings exports and historical DK salaries.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np
import pandas as pd

from props_data import clean_weekly, dk_classic_points
from ai2_nfl_auto_baselines import _norm_name
from ai2_nfl_matchups import canonical_team

LOW_PRICE = {"RB": 6000, "WR": 5500, "TE": 4200}
MEANINGFUL_OPPORTUNITY = {"RB": 7.0, "WR": 5.0, "TE": 4.0}
POSITIONS = frozenset(LOW_PRICE)


def _numeric(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x >= 0 else None


def _key(name, team, position):
    return (_norm_name(name), canonical_team(team), str(position).strip().upper())


def _salary(row):
    try:
        return float(row["Salary"])
    except (ValueError, TypeError, KeyError):
        return 0.0


def build_pregame_role_evidence(history, salary_rows, season, week):
    """Return source-auditable per-player role findings, strictly before week.

    Only individual real appearances are counted as games with observed
    opportunity. Missing team games are shown separately and never assigned
    synthetic zero stats. A player can be a useful GPP punt, but unverified
    one-week opportunity cannot enter the low-priced pool automatically.
    """
    season, week = int(season), int(week)
    h = clean_weekly(history)
    h = h[h["season"].eq(season) & h["week"].lt(week)].copy()
    h["_name"] = h["player_name"].map(_norm_name)
    h["_team"] = h["recent_team"].map(canonical_team)
    h["_position"] = h["position"].astype(str).str.upper()
    groups = defaultdict(list)
    for row in h.to_dict("records"):
        if row["_position"] in POSITIONS:
            groups[(row["_name"], row["_team"], row["_position"])].append(row)
    team_games = {}
    for team, group in h.groupby("_team"):
        team_games[str(team)] = sorted(set(int(w) for w in group["week"].dropna()))
    # Same-position competition is observed from the same prior team games.
    # A teammate missing a recent box score is an observed absence, NOT proof
    # of injury or clearance. Never substitute this for official game status.
    group_by_team_position = defaultdict(list)
    for group in groups.values():
        for r in group:
            group_by_team_position[(r["_team"], r["_position"])].append(r)

    def teammate_context(team, position, person_key):
        weeks = team_games.get(team, [])[-5:]
        if len(weeks) < 3:
            return ""
        last_two = set(weeks[-2:])
        earlier = set(weeks[:-2])
        details = []
        peer_names = {}
        for r in group_by_team_position.get((team, position), []):
            key = r["_name"]
            if key == person_key or int(r["week"]) not in weeks:
                continue
            peer_names.setdefault(key, r["player_name"])
        for peer, name in peer_names.items():
            games = [r for r in group_by_team_position[(team, position)]
                     if r["_name"] == peer and int(r["week"]) in weeks]
            def count_opportunities(r):
                c = _numeric(r.get("carries"))
                t = _numeric(r.get("targets"))
                if t is None or (position == "RB" and c is None):
                    return None
                return ((c or 0) + t) if position == "RB" else t
            before = [count_opportunities(r) for r in games
                      if int(r["week"]) in earlier]
            after = [count_opportunities(r) for r in games
                     if int(r["week"]) in last_two]
            before = [v for v in before if v is not None]
            after = [v for v in after if v is not None]
            if not before:
                continue
            first_avg = float(np.mean(before))
            recent_avg = float(np.mean(after)) if after else None
            # Only surface materially involved peers; omit a backup with
            # negligible prior opportunity so the report remains focused.
            if first_avg < (4.0 if position == "RB" else 3.0):
                continue
            if not after:
                trend = "NO RECENT RECORDED APPEARANCE"
            elif recent_avg < .60 * first_avg:
                trend = "DECLINING OBSERVED ROLE"
            elif recent_avg > 1.45 * first_avg:
                trend = "INCREASING OBSERVED ROLE"
            else:
                trend = "SIMILAR OBSERVED ROLE"
            if trend != "SIMILAR OBSERVED ROLE":
                details.append((first_avg, f"{name}: {trend} "
                                f"({first_avg:.1f} earlier to "
                                f"{recent_avg:.1f} recent)" if recent_avg is not None
                                else f"{name}: {trend} ({first_avg:.1f} prior opportunities/game)"))
        return "; ".join(x[1] for x in sorted(details, reverse=True)[:3])

    findings = {}
    for player in salary_rows:
        position = str(player.get("Position") or "").strip().upper().replace("D/ST", "DST")
        if position not in POSITIONS:
            continue
        team = canonical_team(player.get("TeamAbbrev") or player.get("Team") or "")
        ident = str(player.get("ID") or "").strip()
        if not ident:
            continue
        name = str(player.get("Name") or "")
        rows = sorted(groups.get(_key(name, team, position), []),
                      key=lambda r: int(r["week"]))
        # Duplicate player-week identities cannot be resolved by guessing.
        seen_weeks = [int(r["week"]) for r in rows]
        ambiguous = len(seen_weeks) != len(set(seen_weeks))
        recent = rows[-5:] if not ambiguous else []
        observed = []
        for r in recent:
            carries = _numeric(r.get("carries"))
            targets = _numeric(r.get("targets"))
            # Some position feeds have null inapplicable carry columns;
            # targets and carries are relevant across RB/WR/TE when observed.
            if targets is None or (position == "RB" and carries is None):
                observed.append(None)
                continue
            observed.append((carries or 0.0) + targets if position == "RB" else targets)
        available = [v for v in observed if v is not None]
        threshold = MEANINGFUL_OPPORTUNITY[position]
        meaningful = sum(v >= threshold for v in available)
        latest_two = sum(v is not None and v >= threshold for v in observed[-2:])
        team_last5 = team_games.get(team, [])[-5:]
        played_team_games = len(set(seen_weeks) & set(team_last5))
        scoring = []
        for r in recent:
            score = dk_classic_points(r, position)
            scoring.append(float(score) if np.isfinite(score) else None)
        double_digit = sum(x is not None and x >= 10 for x in scoring)
        spike = (len(available) >= 4 and meaningful <= 2 and
                 max(available) >= max(threshold * 1.6,
                                      2.3 * float(np.median(available))))
        # Direct opportunity (not touchdowns) is the primary route.
        sustained = (not ambiguous and len(available) >= 4 and
                     played_team_games >= 4 and meaningful >= 3 and latest_two >= 1)
        # Backups can produce consistently with smaller touch totals; at
        # least two meaningful-use games must support an output-based override.
        sustained_scores = (not ambiguous and len(scoring) >= 4 and
                            double_digit >= 3 and meaningful >= 2 and
                            played_team_games >= 4 and latest_two >= 1)
        emerging = (not ambiguous and len(available) == 3 and
                    meaningful == 3 and latest_two == 2 and
                    len(set(seen_weeks) & set(team_games.get(team, [])[-3:])) == 3)
        stable = bool(sustained or sustained_scores or emerging)
        below = _salary(player) < LOW_PRICE[position]
        if ambiguous:
            status, reason = "BLOCK", "Ambiguous duplicate player-week statistics"
        elif not below:
            status, reason = "NOT_LOW_SALARY", "Full-price player; low-price gate not applicable"
        elif stable:
            status = "PASS"
            reason = ("Multiple prior games of verified opportunities" if sustained else
                      "Repeated double-digit scoring with recurring touches/targets"
                      if sustained_scores else "Three consecutive recent team games with verified opportunity")
        elif spike:
            status, reason = "BLOCK", "Single-week volume spike without sustained role"
        elif len(available) < 3:
            status, reason = "BLOCK", "Fewer than three pregame observations with usable opportunity"
        else:
            status, reason = "BLOCK", "Insufficient recurring targets/carries or recent role continuity"
        findings[ident] = {
            "Player": name, "Team": team, "Position": position,
            "Salary": _salary(player), "Role gate": status,
            "Role rationale": reason, "Pregame source": "nflverse prior-week player stats",
            "Prior observed games": len(recent),
            "Last five team games": len(team_last5),
            "Played in last five team games": played_team_games,
            "Meaningful opportunities": meaningful,
            "Meaningful opportunity threshold": threshold,
            "Meaningful opportunities in last two appearances": latest_two,
            "Prior games with 10+ partial DK points": double_digit,
            "Recent individual opportunities": [round(v, 2) if v is not None else None
                                                 for v in observed],
            "Recent individual weeks": seen_weeks[-5:],
            "Ambiguous historical identity": ambiguous,
            "Same-position teammate pregame trends": teammate_context(
                team, position, _norm_name(name)),
            "Price tier evaluated": below,
        }
    return findings


def historical_value_replay(history, historical_salary_rows, season, target_week,
                            *, min_score=15.0, min_points_per_1000=3.0):
    """POSTGAME study all cheap players, including failures, using same-week results.

    Salaries MUST be from the historical week's actual DK slate, not current
    salaries; the pregame evidence deliberately excludes the played week.
    This is a value-breakout study, NOT a reconstructed winning contest lineup.
    """
    rows = list(historical_salary_rows)
    pre = build_pregame_role_evidence(history, rows, season, target_week)
    data = clean_weekly(history)
    actual = data[data["season"].eq(int(season)) &
                  data["week"].eq(int(target_week))]
    lookup = defaultdict(list)
    for row in actual.to_dict("records"):
        key = _key(row.get("player_name"), row.get("recent_team"), row.get("position"))
        lookup[key].append(row)
    results = []
    for player in rows:
        position = str(player.get("Position") or "").upper()
        if position not in POSITIONS or not _salary(player):
            continue
        ident = str(player.get("ID") or "").strip()
        evidence = pre.get(ident)
        if not evidence or not evidence["Price tier evaluated"]:
            continue
        matched = lookup.get(_key(player.get("Name"), player.get("TeamAbbrev") or
                                  player.get("Team"), position), [])
        score = None
        if len(matched) == 1:
            points = dk_classic_points(matched[0], position)
            if np.isfinite(points):
                score = round(float(points), 2)
        ratio = (score / (_salary(player) / 1000) if score is not None else None)
        breakout = (score is not None and score >= min_score and
                    ratio >= min_points_per_1000)
        results.append({
            "Player": evidence["Player"], "Team": evidence["Team"],
            "Position": position, "Historical salary": _salary(player),
            "Pregame role gate": evidence["Role gate"],
            "Pregame rationale": evidence["Role rationale"],
            "Prior meaningful games": evidence["Meaningful opportunities"],
            "Prior 10+ DK games": evidence["Prior games with 10+ partial DK points"],
            "Pregame teammate role changes": evidence["Same-position teammate pregame trends"],
            "Actual partial DK points": score,
            "Actual points per $1K": round(ratio, 2) if ratio is not None else None,
            "Value breakout": breakout,
            "Outcome available": score is not None,
            "Note": "Hindsight outcome is separated from prior-week signals; no game outcome enters the gate",
        })
    return results
