"""Research-only pregame QB/RB/WR/TE defensive role matchup evidence.

Previous opponents are categorized by their OWN *pregame* usage rank within
their offense (WR1/WR2/WR3, RB1/RB2, TE1/TE2, QB1). We compare each player's
partial DraftKings fantasy score to that player's own prior-game expected
score. No current game's usage determines historical rank or baseline.

Usage ranks are observed box-score proxies, not official depth charts,
individual coverage, slot/outside roles, or verified player availability.
Never use these averages as direct fantasy-scoring multipliers without a
separate chronological holdout and player-specific opportunity calibration.
"""
from __future__ import annotations

from collections import defaultdict, deque
import math
import re

import numpy as np
import pandas as pd
from props_data import clean_weekly, dk_classic_points
from ai2_nfl_matchups import canonical_team, opponent_from_game_info
from ai2_nfl_opponent_relative import _identity

SOURCE = "nflverse pregame own-player DK vs defense; teammate prior-usage rank"
ROLE_LIMIT = {"QB": 1, "RB": 2, "WR": 3, "TE": 2}
ROLE_USAGE = {"QB": "attempts", "RB": "carries+targets",
              "WR": "targets", "TE": "targets"}
INVALID_TEAM = {"", "NAN", "NONE", "UNKNOWN", "<NA>", "NULL"}
INVALID_STATUS = {"OUT", "O", "IR", "INACTIVE", "PUP", "NFI",
                  "SUSPENDED", "DOUBTFUL", "D"}
OBS_COLUMNS = ("season", "week", "defense", "offense", "player",
               "position", "role", "baseline_dk", "actual_dk", "excess_dk",
               "prior_games", "usual_usage", "sample_note")
PROFILE_COLUMNS = ("defense", "role", "position", "n", "games",
                   "baseline_dk_avg", "actual_dk_avg", "raw_excess_dk",
                   "shrunk_excess_dk", "observed_positive_pct",
                   "source", "target_season", "target_week")


def _canon(value):
    if pd.isna(value):
        return ""
    v = canonical_team(str(value))
    return "" if v in INVALID_TEAM else v


def _name(value):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(value).lower()).split())


def _usage(row, pos):
    try:
        if pos == "QB":
            value = float(row.get("attempts", np.nan))
        elif pos == "RB":
            carry = float(row.get("carries", np.nan))
            target = float(row.get("targets", np.nan))
            value = carry + target
        else:
            value = float(row.get("targets", np.nan))
    except (ValueError, TypeError):
        return np.nan
    return value if np.isfinite(value) and value >= 0 else np.nan


def _weighted(vals):
    a = np.asarray(vals, float)
    weights = .87 ** np.arange(len(a)-1, -1, -1)
    return float(np.average(a, weights=weights))


def _recent(past, season, week):
    # Last known active within three weeks; early season permits the previous
    # year's last games but does not treat that as verified current availability.
    if not past:
        return False
    yr, wk = past[-1][0], past[-1][1]
    return bool((yr == season and week-wk <= 3) or
                (yr == season-1 and week <= 3 and wk >= 15))


def _role_ranks(past, season, week, *, min_rank_games=2):
    groups = defaultdict(list)
    for key, games in past.items():
        if len(games) < min_rank_games or not _recent(games, season, week):
            continue
        pos = key[2]
        if pos not in ROLE_LIMIT:
            continue
        vals = [z[3] for z in games if np.isfinite(z[3])]
        if len(vals) < min_rank_games:
            continue
        groups[(key[1], pos)].append((key, _weighted(vals)))
    lookup = {}
    for (team, pos), values in groups.items():
        ordered = sorted(values, key=lambda z: (-z[1], z[0][0]))
        for rank, (key, usage) in enumerate(ordered, start=1):
            if rank <= ROLE_LIMIT[pos]:
                lookup[key] = (f"{pos}{rank}", float(usage))
    return lookup


def role_observations(history, season, week, *,
                      include_week=False, prior_seasons=1,
                      min_baseline_games=3, window=8):
    """Each game's opponent, role and expected points are frozen before kickoff.

    include_week=True exists exclusively for retrospective grading. A week's
    entire box score is added to prior history AFTER its rows are processed.
    """
    if min_baseline_games < 2 or window < min_baseline_games:
        raise ValueError("Invalid rolling baseline history requirements")
    h = clean_weekly(history)
    if "opponent_team" not in h.columns:
        raise ValueError("Missing opponent_team in observed NFL history")
    end = h["week"].le(int(week)) if include_week else h["week"].lt(int(week))
    h = h[h["season"].ge(int(season)-int(prior_seasons)) &
          ((h["season"].lt(int(season))) |
           (h["season"].eq(int(season)) & end))].copy()
    h = h.sort_values(["season", "week", "recent_team", "player_name"],
                      kind="stable")
    previous = defaultdict(lambda: deque(maxlen=int(window)))
    seen = set()
    result = []
    for (yr, wk), frame in h.groupby(["season", "week"], sort=True):
        yr, wk = int(yr), int(wk)
        ranks = _role_ranks(previous, yr, wk)
        pending = []
        for row in frame.to_dict("records"):
            pos = str(row["position"]).upper()
            if pos not in ROLE_LIMIT:
                continue
            ident = _identity(row)
            unique = (yr, wk, ident)
            if unique in seen:
                raise ValueError("Duplicate player/team/position in same NFL week")
            seen.add(unique)
            score = dk_classic_points(row, pos)
            use = _usage(row, pos)
            if not np.isfinite(score):
                continue
            past = previous[ident]
            opponent = _canon(row.get("opponent_team"))
            role_info = ranks.get(ident)
            if (opponent and role_info and
                    len(past) >= min_baseline_games):
                score_baseline = _weighted([z[2] for z in past])
                role, prior_usage = role_info
                result.append({
                    "season": yr, "week": wk, "defense": opponent,
                    "offense": ident[1], "player": str(row["player_name"]),
                    "position": pos, "role": role,
                    "baseline_dk": score_baseline,
                    "actual_dk": float(score),
                    "excess_dk": float(score-score_baseline),
                    "prior_games": len(past),
                    "usual_usage": prior_usage,
                    "sample_note": "usage-ranked from pregame teammate history",
                })
            pending.append((ident, (yr, wk, float(score), float(use))))
        for key, record in pending:
            previous[key].append(record)
    return pd.DataFrame(result, columns=OBS_COLUMNS)


def defense_role_profiles(history, season, week, *,
                          defense_window=10, shrink_games=6.,
                          max_abs_delta=30., min_observations=2):
    """Pregame defense x WR1/WR2/WR3/RB1/RB2/TE1/TE2/QB1 comparison.

    An explicit sample, raw per-player-game excess and shrunk estimate are
    returned. If historical observations are absent, omit the signal.
    """
    if defense_window <= 0 or shrink_games <= 0 or max_abs_delta <= 0:
        raise ValueError("Invalid role matchup regularization")
    observed = role_observations(history, season, week)
    if observed.empty:
        return pd.DataFrame(columns=PROFILE_COLUMNS), observed
    profiles = []
    details = []
    for defense, group in observed.groupby("defense"):
        games = sorted({(int(r["season"]), int(r["week"]))
                        for r in group.to_dict("records")})
        allowed = set(games[-int(defense_window):])
        recent = group[[ (int(r["season"]),int(r["week"])) in allowed
                         for r in group.to_dict("records")]].copy()
        details.append(recent)
        for role, sample in recent.groupby("role"):
            n = len(sample)
            if n < min_observations:
                continue
            raw = float(sample["excess_dk"].mean())
            bounded = sample["excess_dk"].clip(-max_abs_delta,max_abs_delta)
            delta = float(bounded.mean()*n/(n+shrink_games))
            profiles.append({
                "defense": defense, "role": role, "position": role[:2]
                    if role.startswith(("QB", "RB", "WR", "TE")) else "",
                "n": n,
                "games": len(sample[["season","week"]].drop_duplicates()),
                "baseline_dk_avg": round(float(sample["baseline_dk"].mean()), 2),
                "actual_dk_avg": round(float(sample["actual_dk"].mean()), 2),
                "raw_excess_dk": round(raw, 2),
                "shrunk_excess_dk": round(delta, 2),
                "observed_positive_pct": round(float(
                    (sample["excess_dk"] > 0).mean())*100, 1),
                "source": SOURCE, "target_season": int(season),
                "target_week": int(week),
            })
    out = (pd.concat(details,ignore_index=True)
           if details else pd.DataFrame(columns=OBS_COLUMNS))
    return pd.DataFrame(profiles,columns=PROFILE_COLUMNS), out


def slate_role_report(salary_rows, history, season, week, profiles, *, official_status_by_id=None):
    """Show EVERY salary-file player: plausible pregame usage rank + matchup.

    Live role rank is among non-OUT salary rows with at least two observed
    games; it is a proxy, NOT verified snaps, first-team depth or WR alignment.
    """
    if not isinstance(profiles, pd.DataFrame):
        raise ValueError("Role profiles must be sourced")
    official_status_by_id = official_status_by_id or {}
    h = clean_weekly(history)
    h = h[h["season"].ge(int(season)-1) &
          ((h["season"].lt(int(season))) |
           (h["season"].eq(int(season)) & h["week"].lt(int(week))))].copy()
    h = h.sort_values(["season","week"])
    by_name = defaultdict(list)
    for r in h.to_dict("records"):
        pos = str(r["position"]).upper()
        team = _canon(r.get("recent_team"))
        key = (_name(r["player_name"]), team, pos)
        score = dk_classic_points(r, pos)
        if np.isfinite(score):
            by_name[key].append(r)
    eligibility = []
    by_team_pos = defaultdict(list)
    for item in salary_rows:
        pos = str(item.get("Position", "")).upper().replace("D/ST","DST")
        if pos not in ROLE_LIMIT:
            continue
        team = _canon(item.get("TeamAbbrev") or item.get("Team") or "")
        opponent = opponent_from_game_info(team, str(item.get("Game Info", "")))
        name = str(item.get("Name", "")).strip()
        dk_status = str(item.get("Status", "") or "").upper().strip()
        official = str(official_status_by_id.get(str(item.get("ID","")), "") or "").upper().strip()
        # Either explicit authoritative OUT or a blocked DK status removes a
        # player from the present-slate role rank; stale DK statuses must not
        # silently override verified official OUT designations.
        status = (dk_status if dk_status in INVALID_STATUS else
                  official if official else dk_status)
        relevant = by_name.get((_name(name),team,pos),[])[-8:]
        # Do not give a current-season WR1/RB1 label based solely on the
        # previous season's production or a player absent for a month.
        fresh = bool(relevant and _recent(
            [(int(z["season"]), int(z["week"]), 0., 0.)
             for z in relevant], int(season), int(week)))
        scores = [dk_classic_points(z,pos) for z in relevant]
        usages = [_usage(z,pos) for z in relevant]
        valid_uses = [z for z in usages if np.isfinite(z)]
        rate = _weighted(valid_uses) if fresh and len(valid_uses)>=2 else None
        own = _weighted(scores) if fresh and len(scores)>=3 else None
        row = {
            "Player": name, "Team": team, "Opponent": opponent or "",
            "Position": pos, "Role": "",
            "Salary": item.get("Salary", ""),
            "Slate status": status or "not designated",
            "Player pregame DK baseline": round(own, 2) if own is not None else None,
            "Player history games": len(relevant),
            "Previous role usage": round(rate, 2) if rate is not None else None,
            "Defense same-role prior games": 0,
            "Defense versus role (raw DK above own average)": None,
            "Defense versus role (shrunk DK)": None,
            "Research only": True,
            "Role evidence": "",
        }
        eligibility.append(row)
        if status not in INVALID_STATUS and rate is not None and team and opponent:
            by_team_pos[(team,pos)].append(row)
    lookup = {(str(z["defense"]),str(z["role"])):z
              for z in profiles.to_dict("records")}
    for (team,pos), candidates in by_team_pos.items():
        candidates.sort(key=lambda z: (-z["Previous role usage"],_name(z["Player"])))
        for rank,row in enumerate(candidates,1):
            if rank>ROLE_LIMIT[pos]:
                row["Role evidence"]="Outside tracked top usage ranks"
                continue
            row["Role"]=f"{pos}{rank}"
            row["Role evidence"]=("Pregame within-team usage rank; "
                                  "not official lineup or coverage")
            z=lookup.get((row["Opponent"],row["Role"]))
            if z:
                row["Defense same-role prior games"]=int(z["n"])
                row["Defense versus role (raw DK above own average)"]=z["raw_excess_dk"]
                row["Defense versus role (shrunk DK)"]=z["shrunk_excess_dk"]
    for row in eligibility:
        if not row["Role evidence"]:
            row["Role evidence"]=("Insufficient pregame usage or unavailable; "
                                  "no implied positive matchup")
    return eligibility
