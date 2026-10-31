"""Pregame injury context: count only newly freed target/carry opportunity.

Historical nflverse per-player rates are *conditional on appearing in a game*,
while active teammates' recent weighted averages already incorporate games
where an absent teammate did not play.  Crediting the full missing player's
conditional rate again double-counts some of those opportunities.

This module estimates how much of that unavailable player's participation
was represented in the ACTIVE teammates' historical lookback.  A multi-game
absence reduces the additional reallocation. It does NOT infer unavailable
players from a missing box-score row or an incomplete DraftKings file.
"""
from __future__ import annotations
import numpy as np
from props_data import clean_weekly
from ai2_nfl_auto_baselines import _norm_name
from ai2_nfl_matchups import canonical_team

SCORABLE_POSITIONS = frozenset(("RB", "WR", "TE", "QB"))
ALLOWED_OFFICIAL_OUT = frozenset(("OUT",))


def official_off_slate_out(practice_rows, salary_rows, rates):
    """Recognize explicitly ruled-OUT players absent from DK's price slate.

    A same-team historical identity and a genuine NFL game designation are
    required. IR players not on the injury report remain unknown and are
    NEVER silently inferred OUT from a missing salary or box-score row.
    """
    salary_keys = {(_norm_name(r.get("Name", "")),
                    canonical_team(r.get("TeamAbbrev", "")))
                   for r in salary_rows}
    slate_teams = {canonical_team(r.get("TeamAbbrev", "")) for r in salary_rows}
    rate_ids = {
        (_norm_name(r["player"]), canonical_team(r["team"]),
         str(r["position"]).upper())
        for r in rates.to_dict("records")
    }
    confirmed = []
    seen = set()
    for r in practice_rows or []:
        status = str(r.get("game_status") or "").strip().upper()
        name = str(r.get("name") or "")
        team = canonical_team(r.get("team") or "")
        pos = str(r.get("position") or "").upper()
        key = (_norm_name(name), team)
        if (status not in ALLOWED_OFFICIAL_OUT or not team or team not in slate_teams
                or not key[0] or key in salary_keys or key in seen):
            continue
        # Some official tables omit position, but team+name can still identify
        # exactly one current-team NFL weekly-stat player unambiguously.
        matches = [p for n, t, p in rate_ids if (n,t) == key
                   and (not pos or pos == p) and p in SCORABLE_POSITIONS]
        if len(matches) != 1:
            continue
        confirmed.append({"player": name, "team": team,
                          "position": matches[0],
                          "source": "NFL official game-status OUT; absent from DK salary slate"})
        seen.add(key)
    return confirmed


def estimate_injury_presence(history, out_players, season, week, *, window=8):
    """Pregame team-game weighted OUT-player appearance fraction.

    Last eight *completed current-season* team games are the denominator,
    not the injured player's last eight personal appearances.  Therefore
    several consecutive missed games reduce the incremental redistribution.

    Weight: 65% all-game presence, 35% last-three-game presence, matching
    the current simulator's approximate historical/snap role information mix.
    The adjustment remains a conservative heuristic, not trained to ROI.
    """
    h = clean_weekly(history)
    h = h[(h["season"].eq(int(season))) & h["week"].lt(int(week))].copy()
    if h.empty:
        return {}, []
    h["_team"] = h["recent_team"].map(canonical_team)
    h["_name"] = h["player_name"].map(_norm_name)
    team_games = {
        team: sorted(set(group["week"].astype(int)))
        for team, group in h.groupby("_team")
    }
    appearances = {
        (team, name): set(group["week"].astype(int))
        for (team, name), group in h.groupby(["_team", "_name"])
    }
    roles, report = {}, []
    processed = set()
    for absent in out_players:
        name = str(absent.get("player") or "")
        key = (_norm_name(name), canonical_team(absent.get("team") or ""))
        if not key[0] or not key[1] or key in processed:
            continue
        processed.add(key)
        games = team_games.get(key[1], [])[-int(window):]
        if not games:
            share = 0.0
            recent = 0.0
            latest = None
            status = "NO_RELIABLE_TEAM_HISTORY"
        else:
            present = appearances.get((key[1], key[0]), set())
            def weighted_appearance(sample):
                weights = np.power(.87, np.arange(len(sample)-1, -1, -1))
                return float(np.average([int(w in present) for w in sample],
                                        weights=weights))
            total = weighted_appearance(games)
            recent = weighted_appearance(games[-3:])
            share = float(np.clip(.65*total+.35*recent, 0., 1.))
            latest = games[-1]
            status = ("ALREADY_ABSENT_IN_RECENT_GAMES"
                      if latest not in present else "PLAYED_LAST_TEAM_GAME")
        roles[key] = {
            "share": share,
            "team_games": len(games),
            "last_team_week": latest,
            "source": "nflverse prior-game same-team active-week share",
            "status": status
        }
        report.append({
            "Unavailable player": name, "Team": key[1],
            "Team games examined": len(games),
            "Last team game week": latest,
            "Was active last team game": status == "PLAYED_LAST_TEAM_GAME",
            "Historical share not yet reflected": round(share,3),
            "Already reflected share": round(1-share,3),
            "Additional redistribution %": round(100*share,1),
            "Analysis status": status,
            "Evidence": roles[key]["source"],
        })
    return roles, report
