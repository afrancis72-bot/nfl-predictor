"""Auditable pregame opposing-defense research (no inferred coverage assignments).

Observed position allowances and opponent defender absences can generate hypotheses.
Neither route-level coverage, shadow CBs, snap losses nor a validated fantasy-point
boost can be inferred from team injury lists. No unverified multiplier is fed into
the DFS simulation. The model must earn the right to modify probabilities with
chronological, out-of-sample backtesting.
"""
import math
import pandas as pd
from ai2_nfl_matchups import canonical_team, opponent_from_game_info

PASS_COVERAGE = frozenset({"CB", "DB", "S", "FS", "SS"})
PASS_PRESSURE = frozenset({"DE", "EDGE", "OLB", "LB", "DT", "NT"})
RUN_FRONT = frozenset({"DT", "NT", "DE", "EDGE", "LB", "ILB", "OLB"})
ABSENT = frozenset({"OUT"})
UNCERTAIN = frozenset({"QUESTIONABLE", "DOUBTFUL", ""})


def _relevant_defenders(practice_rows, team):
    out, watch = [], []
    seen = set()
    for p in practice_rows or []:
        if canonical_team(p.get("team", "")) != team:
            continue
        name = str(p.get("name", "")).strip()
        position = str(p.get("position", "")).strip().upper()
        if position not in PASS_COVERAGE | PASS_PRESSURE | RUN_FRONT or not name:
            continue
        key = (name.casefold(), position)
        if key in seen:
            continue
        seen.add(key)
        status = str(p.get("game_status", "")).upper().strip()
        practice = str(p.get("practice_status", "")).lower().strip()
        item = {
            "player": name, "position": position, "game_status": status or "Not designated",
            "practice_status": str(p.get("practice_status") or ""),
            "coverage_relevant": position in PASS_COVERAGE,
            "pressure_relevant": position in PASS_PRESSURE,
            "run_front_relevant": position in RUN_FRONT,
            "source": "official NFL team injury report",
        }
        if status in ABSENT:
            out.append(item)
        elif status == "DOUBTFUL" or status == "QUESTIONABLE" or (
                "did not participate" in practice or "limited" in practice):
            # Practicing status alone is uncertainty, NOT a confirmed absence.
            watch.append(item)
    return out, watch


def make_opponent_report(salary_rows, allowances, injury_meta):
    """Build a pregame, team-scoped defensive research desk.

    Every listed player has an explicit offense + opponent from the DK slate.
    Allowances reflect PREVIOUS games only, are shrunk toward a league prior,
    and require adequate sample size to be considered evidence of a tendency.
    Missing defender team/position or coverage alignment means NO matchup boost.
    """
    if not isinstance(allowances, pd.DataFrame):
        raise ValueError("Expected observed opponent allowance DataFrame")
    required = {"defense", "position", "multiplier", "player_game_observations"}
    if not required.issubset(allowances.columns):
        raise ValueError("Opponent report lacks history: " + str(sorted(required - set(allowances.columns))))
    lookups = {}
    for row in allowances.to_dict("records"):
        defense, position = canonical_team(row["defense"]), str(row["position"]).upper()
        factor, count = float(row["multiplier"]), int(row["player_game_observations"])
        if not math.isfinite(factor) or not .75 <= factor <= 1.25 or count < 1:
            continue
        lookups[(defense, position)] = (factor, count)
    teams = {}
    for row in salary_rows:
        team = canonical_team(row.get("TeamAbbrev") or row.get("Team") or "")
        opponent = opponent_from_game_info(team, row.get("Game Info", ""))
        if team and opponent:
            teams[team] = opponent

    source = str((injury_meta or {}).get("source") or "")
    check_time = (injury_meta or {}).get("checked_utc")
    rows = (injury_meta or {}).get("practice_rows", [])
    output = []
    for team, opponent in sorted(teams.items()):
        confirmed, watch = _relevant_defenders(rows, opponent)
        for position in ("QB", "RB", "WR", "TE"):
            reading = lookups.get((opponent, position))
            factor, games = reading if reading is not None else (None, 0)
            reliable = factor is not None and games >= 20
            tendency = ("higher historical allowance" if reliable and factor >= 1.08
                        else "lower historical allowance" if reliable and factor <= .92
                        else "no strong historical signal")
            relevant_out = [p for p in confirmed
                            if (position in {"QB", "WR", "TE"} and
                                (p["coverage_relevant"] or p["pressure_relevant"]))
                            or (position == "RB" and p["run_front_relevant"])]
            relevant_watch = [p for p in watch
                              if (position in {"QB", "WR", "TE"} and
                                  (p["coverage_relevant"] or p["pressure_relevant"]))
                              or (position == "RB" and p["run_front_relevant"])]
            output.append({
                "Offense": team, "Opposing defense": opponent, "Position": position,
                "Historical allowance index": round(factor, 3) if reliable else None,
                "Observations": games,
                "Observed tendency": tendency,
                "Defenders ruled out": ", ".join(f"{x['player']} ({x['position']})"
                                                 for x in relevant_out) or "None verified",
                "Defenders to monitor": ", ".join(f"{x['player']} ({x['position']})"
                                                 for x in relevant_watch) or "None",
                "Coverage assignment": "Unknown — no verified CB/receiver route map",
                "Game-script hypothesis": (
                    "Investigate potential passing-game changes" if position in {"QB", "WR", "TE"}
                    and (relevant_out or relevant_watch) else
                    "Investigate rushing-front changes" if position == "RB"
                    and (relevant_out or relevant_watch) else
                    "Neutral — insufficient confirmed injury evidence"),
                "Applied simulation boost": 1.0,
                "Evidence": ("Observed pregame position aggregate + official team/position report"
                             if (reliable or relevant_out or relevant_watch)
                             else "Insufficient; neutral prior"),
                "Injury checked UTC": check_time,
                "Injury source": source,
            })
    return output


def report_summary(report):
    return {
        "offensive_teams": len({r["Offense"] for r in report}),
        "sampled_matchups": sum(r["Historical allowance index"] is not None for r in report),
        "opponent_injury_flags": sum(
            r["Defenders ruled out"] != "None verified" or
            r["Defenders to monitor"] != "None" for r in report),
        "coverage_matchup_status": "not verified: route/CB alignment and shadow coverage not supplied",
        "game_script_status": "hypothesis only; no automatic projections adjustment until chronological validation",
    }
