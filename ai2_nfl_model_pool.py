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
    pregame_role = research.get("pregame_role_evidence") or {}
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
    excluded_by_id={}
    def reject(ident,name,reason):
        excluded.append((name,reason))
        excluded_by_id[str(ident)]=reason
    for _, row in baseline.iterrows():
        ident = str(row["ID"]).strip()
        name = str(row["Name"]).strip()
        pos = str(row["Position"]).upper().strip().replace("D/ST", "DST")
        team = _norm_team(row["Team"])
        status = str(row["current_status"]).upper().strip()
        if status in {"OUT", "O", "IR", "INACTIVE", "SUSPENDED", "PUP", "NFI", "DOUBTFUL", "D"}:
            if pos in {"QB", "RB", "WR", "TE"}:
                outs.append({"player": name, "team": team})
            reject(ident,name, "confirmed unavailable")
            continue
        if status in {"Q", "QUESTIONABLE"}:
            reject(ident,name, "questionable without verified availability")
            continue
        match = MATCHUP.match(str(row["Game Info"] or ""))
        opponent = opponent_from_game_info(team, row["Game Info"])
        if not ident or not name or not match or not opponent:
            reject(ident,name, "missing roster identity or game matchup")
            continue
        game = "@".join(_norm_team(x) for x in match.groups())
        salary = _num(row["Salary"])
        if salary <= 0:
            reject(ident,name,'Invalid salary')
            continue
        if pos == "WR" and salary < 4000 or pos == "RB" and salary < 4500:
            reject(ident,name,"Below configured optimizer salary floor")
            continue
        # Cheap players need repeated REAL opportunity before they can even
        # enter the football simulator; inflated simulated targets cannot
        # retroactively make a one-game breakout sustainable.
        evidence = pregame_role.get(ident)
        # Fail closed for *missing* individual evidence too. An unnamed
        # backup must not bypass the gate because one CSV ID failed to join.
        from ai2_nfl_pregame_consistency import LOW_PRICE
        cheap = pos in LOW_PRICE and salary < LOW_PRICE[pos]
        if cheap and (not evidence or evidence.get("Role gate") not in ("PASS","EMERGING_LIMITED")):
            reject(ident,name, "prebuild recurring-opportunity gate: " +
                            str((evidence or {}).get("Role rationale",
                                                   "No matching individual pregame usage evidence")))
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
                reject(ident,name, "no unique observed NFL statistical identity")
                continue
            last_season = int(_num(r.get("latest_observed_season")))
            last_week = int(_num(r.get("latest_observed_week")))
            games = int(_num(r.get("games_played")))
            # Do not rank a player on stale previous-season usage.
            if last_season != season or week - last_week > 3 or games < 2:
                reject(ident,name, "insufficient recent in-season observations")
                continue
            carries = _num(a.get("carries_pg"))
            targets = _num(a.get("targets_pg"))
            touches = carries + targets
            passing = _num(r.get("pass_attempts_pg"))
            if pos == "QB" and passing < 18:
                reject(ident,name, "not a regular observed primary passer")
                continue
            if pos == "RB" and touches < 6:
                reject(ident,name, "insufficient observed or redistributed RB volume")
                continue
            if pos == "WR" and targets < 3.5:
                reject(ident,name, "insufficient observed or redistributed WR targets")
                continue
            if pos == "TE" and targets < 2.3:
                reject(ident,name, "insufficient observed or redistributed TE targets")
                continue
            proj = _num(row["historical_partial_dk"])
            if proj <= 0:
                reject(ident,name, "incomplete observed fantasy-stat baseline")
                continue
            source = (f"Own nflverse pregame statistical model; {games} observed games, "
                      f"{last_season} W{last_week}; injury-conditioned opportunities")
        prepared.append({
            "ID": ident, "Name": name, "Position": pos, "TeamAbbrev": team,
            "pregame_role_gate": (evidence or {}).get("Role gate", "UNAVAILABLE"),
            "emerging_role_limited": (evidence or {}).get("Role gate") == "EMERGING_LIMITED",
            "emerging_role_evidence": ((evidence or {}).get("Role rationale", "")
                                      if (evidence or {}).get("Role gate") == "EMERGING_LIMITED"
                                      else ""),
            "pregame_role_rationale": (evidence or {}).get("Role rationale", ""),
            "Salary": int(salary), "game": game, "opponent": opponent,
            "proj": proj, "dk_status": status, "weekly_projection_valid": True,
            "opportunity_verified": True, "optimizer_eligible": True,
            "projection_source": source, "projected_targets": targets,
            "projected_touches": touches, "recent_targets_per_game": targets,
            "recent_touches_per_game": touches,
            "observed_games": games,
            # Raw, pre-redistribution role evidence, separate from the
            # simulated touches inflated by a teammate's temporary absence.
            "observed_carries_pg": (_num(r.get("carries_pg")) if r is not None else None),
            "observed_targets_pg": (_num(r.get("targets_pg")) if r is not None else None),
            "observed_rushing_yards_pg": (_num(r.get("rushing_yards_pg")) if r is not None else None),
            "projection_type": "AUTO_STAT_MODEL",
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
    for r in prepared:
        if r["Position"]=="QB" and r["ID"] not in qb_ids:
            reject(r["ID"],r["Name"],"Another same-team QB has stronger observed passing role")
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
        for r in prepared:
            if r["game"] not in complete:
                reject(r["ID"],r["Name"],"Incomplete simulated game: one side has no observed offense")
        frame = frame[frame["game"].isin(complete)].reset_index(drop=True)
    if frame.empty or not {"QB", "RB", "WR", "TE", "DST"}.issubset(set(frame["Position"])):
        raise ValueError("Not enough observed, currently eligible players to simulate this slate")
    included=set(frame["ID"].astype(str))
    coverage=[]
    for row in baseline.to_dict("records"):
        ident=str(row["ID"]).strip()
        evidence=pregame_role.get(ident) or {}
        def numeric(key):
            value=_num(row.get(key),float("nan"))
            return value if math.isfinite(value) else None
        coverage.append({"ID":ident,"Player":str(row["Name"]),"Team":str(row["Team"]),
                         "Position":str(row["Position"]),"Salary":numeric("Salary"),
                         "DK status":str(row.get("DK Status","")),"Official status":str(row.get("official_game_status","")),
                         "Effective status":str(row.get("current_status","")),
                         "Historical identity":str(row.get("match_status","")),
                         "Observed games":numeric("games_observed"),"Last observed week":numeric("latest_observed_week"),
                         "Observed carries/game":numeric("carries_pg"),"Observed targets/game":numeric("targets_pg"),
                         "Observed pass attempts/game":numeric("pass_attempts_pg"),
                         "Role gate":str(evidence.get("Role gate","")),"Role rationale":str(evidence.get("Role rationale","")),
                         "In simulation pool":ident in included,
                         "Pool reason":"Included" if ident in included else excluded_by_id.get(ident,"Unclassified exclusion - investigate")})
    return frame, raw_rates, outs, {
        "eligible_player_count": len(frame),
        "pregame_consistency_blocked": sum(
            "prebuild recurring-opportunity gate" in str(r) for _, r in excluded),
        "excluded_player_count": len(excluded),
        "excluded_examples": excluded[:12],
        "player_pool_coverage":coverage,
        "complete_games": len(complete),
        "research_source": research["source"],
        "limitations": ("Projection outputs are experimental; no independent "
                        "depth-chart and team-defense calibration yet."),
    }
