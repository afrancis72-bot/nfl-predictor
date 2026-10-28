"""Build a conservative NFL Classic simulation pool from *observed* public data.

Not a third-party projections upload. Opportunity is evidenced by NFL historical
play volume and available game designations. Forecasts remain experimental:
depth-chart, coverage and defensive models are not fully calibrated.
"""
import math
import pandas as pd
from ai2_nfl_matchups import MATCHUP, canonical_team, opponent_from_game_info
from ai2_nfl_auto_baselines import _norm_name

BLOCKED = frozenset({"OUT", "O", "IR", "INACTIVE", "SUSPENDED", "PUP",
                     "NFI", "DOUBTFUL", "D", "Q", "QUESTIONABLE"})

def _norm_team(value):
    return canonical_team(str(value).upper().strip())

def _num(value, fallback=0.0):
    try:
        number = float(value)
        return number if math.isfinite(number) else fallback
    except (TypeError, ValueError):
        return fallback

def create_observed_model_pool(research):
    """Return (pool, unmodified rates, confirmed out players, diagnostics).

    The game-level simulator redistributes OUT players *once* using raw rates.
    The separate capped workload estimator is used for evidence and diagnostics
    only. Never feed its adjusted rates into the game simulator as well.
    """
    baseline = research["players"]
    raw_rates = research["raw_rates"]
    adjusted = research["adjusted_rates"]
    if not isinstance(baseline, pd.DataFrame) or not isinstance(raw_rates, pd.DataFrame):
        raise ValueError("Automatic research returned malformed data")
    if research.get("injury_meta", {}).get("source") in (None, "", "unavailable"):
        raise ValueError("Official injury status source unavailable; cannot validate lineup readiness")
    season, week = int(research["target_season"]), int(research["target_week"])
    lookup = {}
    for _, rate in raw_rates.iterrows():
        key = (_norm_name(rate["player"]), _norm_team(rate["team"]),
               str(rate["position"]).upper())
        if key in lookup:
            # Ambiguous identity is not usable in a financial contest.
            lookup[key] = None
        else:
            lookup[key] = rate
    adjusted_lookup = {}
    for _, rate in adjusted.iterrows():
        key = (_norm_name(rate["player"]), _norm_team(rate["team"]),
               str(rate["position"]).upper())
        adjusted_lookup[key] = rate
    excluded, prepared, outs = [], [], []
    for _, row in baseline.iterrows():
        ident = str(row["ID"]).strip()
        name = str(row["Name"]).strip()
        pos = str(row["Position"]).upper().strip().replace("D/ST", "DST")
        team = _norm_team(row["Team"])
        status = str(row["current_status"]).upper().strip()
        if status in {"OUT", "O", "IR", "INACTIVE", "SUSPENDED", "PUP", "NFI", "DOUBTFUL", "D"}:
            if pos in {"QB", "RB", "WR", "TE"}:
                outs.append({"player": name, "team": team})
            excluded.append((name, "confirmed unavailable"))
            continue
        if status in {"Q", "QUESTIONABLE"}:
            excluded.append((name, "questionable without verified availability"))
            continue
        match = MATCHUP.match(str(row["Game Info"] or ""))
        opponent = opponent_from_game_info(team, row["Game Info"])
        if not ident or not name or not match or not opponent:
            excluded.append((name, "missing roster identity or game matchup"))
            continue
        game = "@".join(_norm_team(x) for x in match.groups())
        salary = _num(row["Salary"])
        if salary <= 0:
            continue
        if pos == "WR" and salary < 4000 or pos == "RB" and salary < 4500:
            continue
        key = (_norm_name(name), team, pos)
        r = lookup.get(key)
        a = adjusted_lookup.get(key)
        if pos == "DST":
            # DST fantasy outcomes are generated from simulated opposing
            # turnovers/sacks/points allowed, not fabricated player history.
            proj = 6.5
            touches = targets = 0.0
            games = 0
            source = "Team game-script simulation, neutral DST prior (uncalibrated)"
        else:
            if row["match_status"] != "matched" or r is None or a is None:
                excluded.append((name, "no unique observed NFL statistical identity"))
                continue
            last_season = int(_num(r.get("latest_observed_season")))
            last_week = int(_num(r.get("latest_observed_week")))
            games = int(_num(r.get("games_played")))
            # Do not rank a player on stale previous-season usage.
            if last_season != season or week - last_week > 3 or games < 2:
                excluded.append((name, "insufficient recent in-season observations"))
                continue
            carries = _num(a.get("carries_pg"))
            targets = _num(a.get("targets_pg"))
            touches = carries + targets
            passing = _num(r.get("pass_attempts_pg"))
            if pos == "QB" and passing < 18:
                excluded.append((name, "not a regular observed primary passer"))
                continue
            if pos == "RB" and touches < 6:
                excluded.append((name, "insufficient observed or redistributed RB volume"))
                continue
            if pos == "WR" and targets < 3.5:
                excluded.append((name, "insufficient observed or redistributed WR targets"))
                continue
            if pos == "TE" and targets < 2.3:
                excluded.append((name, "insufficient observed or redistributed TE targets"))
                continue
            proj = _num(row["historical_partial_dk"])
            if proj <= 0:
                excluded.append((name, "incomplete observed fantasy-stat baseline"))
                continue
            source = (f"Own nflverse pregame statistical model; {games} observed games, "
                      f"{last_season} W{last_week}; injury-conditioned opportunities")
        prepared.append({
            "ID": ident, "Name": name, "Position": pos, "TeamAbbrev": team,
            "Salary": int(salary), "game": game, "opponent": opponent,
            "proj": proj, "dk_status": status, "weekly_projection_valid": True,
            "opportunity_verified": True, "optimizer_eligible": True,
            "projection_source": source, "projected_targets": targets,
            "projected_touches": touches, "recent_targets_per_game": targets,
            "recent_touches_per_game": touches,
            "observed_games": games, "projection_type": "AUTO_STAT_MODEL",
        })
    # In the simulator, exactly one quarterback may represent each team.
    best_qbs = {}
    for r in prepared:
        if r["Position"] != "QB":
            continue
        key = (_norm_name(r["Name"]), r["TeamAbbrev"], "QB")
        passing = _num(lookup[key].get("pass_attempts_pg"))
        if r["TeamAbbrev"] not in best_qbs or passing > best_qbs[r["TeamAbbrev"]][0]:
            best_qbs[r["TeamAbbrev"]] = (passing, r["ID"])
    qb_ids = {v[1] for v in best_qbs.values()}
    prepared = [r for r in prepared if r["Position"] != "QB" or r["ID"] in qb_ids]
    # Reject incomplete contests rather than simulating zero scoring for a side.
    complete = set()
    frame = pd.DataFrame(prepared)
    if not frame.empty:
        for game, group in frame.groupby("game"):
            sides = set(game.split("@"))
            offensive = set(group[group["Position"].ne("DST")]["TeamAbbrev"])
            if sides == offensive:
                complete.add(game)
        frame = frame[frame["game"].isin(complete)].reset_index(drop=True)
    if frame.empty or not {"QB", "RB", "WR", "TE", "DST"}.issubset(set(frame["Position"])):
        raise ValueError("Not enough observed, currently eligible players to simulate this slate")
    return frame, raw_rates, outs, {
        "eligible_player_count": len(frame),
        "excluded_player_count": len(excluded),
        "excluded_examples": excluded[:12],
        "complete_games": len(complete),
        "research_source": research["source"],
        "limitations": ("Projection outputs are experimental; no independent "
                        "depth-chart and team-defense calibration yet."),
    }
