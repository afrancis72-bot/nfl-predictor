"""Single-input NFL Classic model.

Upload a DraftKings salary slate, optionally change the seed, and generate
20 review-ready candidate lineups from observed NFL statistics. No secondary
projection upload, synthetic players, or hidden user-side research controls.
This is a research-stage model, not an independently calibrated profit engine.
"""
import re
from datetime import date
import numpy as np
from ai2_nfl_inputs import parse_csv_payload
from ai2_nfl_auto_intelligence import collect_automatic_baselines
from ai2_nfl_model_pool import create_observed_model_pool
from ai2_nfl_simulation_adapter import run_verified_stat_simulation
from ai2_nfl_player_audit import summarize_simulated_players
from ai2_nfl_candidate_builder import build_classic_candidates
from ai2_nfl_sim_portfolio import select_simulated_portfolio
from ai2_nfl_export import export_classic_candidates
from ai2_nfl_opponent_intelligence import make_opponent_report, report_summary
from ai2_nfl_input_integrity import input_lineage_rows, lineup_role_warnings
from ai2_nfl_projection_review import projection_review_flags
import pandas as pd

def _slate_week(saved_salaries):
    rows = parse_csv_payload(saved_salaries[1])
    dates = []
    for row in rows:
        match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})", str(row.get("Game Info", "")))
        if match:
            month, day, year = (int(x) for x in match.groups())
            dates.append(date(year + 2000 if year < 100 else year, month, day))
    if not dates:
        raise ValueError("No DraftKings slate game dates found in Game Info")
    common = max(set(dates), key=dates.count)
    season = common.year
    # Schedule anchors are explicit: no assumptions about future NFL seasons.
    regular_season_start = {2026: date(2026, 9, 10)}.get(season)
    if regular_season_start is None:
        raise ValueError(f"Automated regular-season week mapping not configured for {season}")
    week = (common - regular_season_start).days // 7 + 1
    if not 1 <= week <= 18:
        raise ValueError("The salary file does not match a regular-season NFL slate")
    return season, week

def build_oneclick_portfolio(salaries, projections=None, *, seed=42, count=20):
    """Build from the salary file; legacy projections argument is ignored."""
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        raise ValueError("Simulation seed must be an integer from 0 to 2147483647")
    if count not in (15, 20, 50, 150):
        raise ValueError("Choose 15, 20, 50, or 150 lineups")
    if not salaries:
        raise ValueError("Upload a DraftKings salary CSV to build lineups")
    season, week = _slate_week(salaries)
    research = collect_automatic_baselines(salaries, season, week)
    if any("Official NFL injury feed unavailable" in issue for issue in research["blockers"]):
        raise ValueError("Official game-status data cannot be retrieved; please retry when the source is available")
    pool, raw_rates, outs, audit = create_observed_model_pool(research)
    # Confirmed officially OUT players can be absent from DK salaries.
    # Ensure a specific player's historical targets are redistributed at most once.
    already_out = {(str(z["player"]).casefold(),str(z["team"])) for z in outs}
    for z in research.get("confirmed_off_slate_out", []):
        k=(str(z["player"]).casefold(),str(z["team"]))
        if k not in already_out:
            outs.append({"player":z["player"],"team":z["team"]})
            already_out.add(k)
    # Player eligibility is evaluated for the current slate. Past human vetoes
    # are audit feedback, not permanent or automatically carried-forward bans.
    # Explicit current-slate exclusions would need to be confirmed separately.
    audit["user_vetoed"] = []
    # This is a CURRENT WEEK 8, $3-tier-only human veto. It must not
    # exclude Gray from regular entries or carry forward to other weeks.
    premium_exclusion_names = {(2026, 8): {"noah gray"}}.get((season, week), set())
    premium_excluded_ids = frozenset(
        str(p["ID"]) for p in pool.to_dict("records")
        if str(p.get("Name", "")).strip().casefold() in premium_exclusion_names)
    audit["premium_only_exclusions"] = sorted(premium_exclusion_names)

    # The game-level simulator already redistributes OUT-player touches and
    # targets. Use unmodified rates with outs to avoid double counting.
    players, draws, stat_draws = run_verified_stat_simulation(
        pool, raw_rates, n_sims=10000, seed=seed, out_players=outs,
        team_environment=research.get("team_environment"),
        snap_profiles=research.get("snap_profiles"),
        injury_presence=research.get("injury_presence", {}))
    if draws.shape != (10000, len(players)) or not np.isfinite(draws).all():
        raise ValueError("Incomplete NFL simulation; refusing to export lineups")
    try:
        simulation_component_audit = summarize_simulated_players(players, draws, stat_draws)
        simulation_component_audit_error = None
    except (ValueError, KeyError, TypeError) as exc:
        # Diagnostic-only feature must never invalidate otherwise legal lineups.
        simulation_component_audit = []
        simulation_component_audit_error = str(exc)
    high_mean_flags = projection_review_flags(simulation_component_audit, pool)
    means = dict(zip(players["ID"].astype(str), draws.mean(axis=0).tolist()))
    prepared = []
    from collections import Counter
    repeated_names = Counter(str(n) for n in pool["Name"])
    for player in pool.to_dict(orient="records"):
        ident = str(player["ID"])
        if ident not in means or not np.isfinite(means[ident]) or means[ident] <= 0:
            continue
        player["proj"] = float(means[ident])
        # Prefer *simulated* post-injury touch/target estimates over the
        # pre-simulation screening heuristic. Duplicate names are skipped:
        # the legacy stats adapter keys its detail by display name.
        name = str(player["Name"])
        detail = stat_draws.get(name) if repeated_names[name] == 1 else None
        if isinstance(detail, dict) and player["Position"] in ("RB", "WR", "TE"):
            targets = np.asarray(detail.get("targets", []), dtype=float)
            carries = np.asarray(detail.get("carries", []), dtype=float)
            if (targets.shape == (10000,) and carries.shape == (10000,)
                    and np.isfinite(targets).all() and np.isfinite(carries).all()):
                player["projected_targets"] = float(targets.mean())
                player["projected_touches"] = float((targets + carries).mean())
        player["projection_source"] += "; 10,000 football-stat scenarios"
        prepared.append(player)
    candidates = build_classic_candidates(
        prepared, count=(1200 if count <= 20 else 2000), attempts=200000, seed=seed,
        salary_floor=49600, salary_cap=50000, max_exposure=0.35,
        min_wr_salary=4000, min_rb_salary=4500, require_opportunity=True)
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} legal candidates under role and salary rules. "
            "No partial portfolio exported.")
    errors = []
    for efficiency in (0.95, 0.90, 0.85):
        try:
            ranked = select_simulated_portfolio(
                candidates, players, draws, count=count,
                max_exposure=0.35, upside_share=0.25,
                premium_count=(5 if count == 20 else 0), min_wr_salary=4000, min_rb_salary=4500,
                value_salary_ceiling=5000, value_max_appearances=(2 if count <= 20 else max(2, int(count*0.35))),
                premium_value_max_appearances=1, require_opportunity=True,
                p90_efficiency_retention=efficiency,
                premium_excluded_ids=premium_excluded_ids)
            break
        except ValueError as exc:
            errors.append(str(exc))
    else:
        raise ValueError("The model could not produce qualifying lineups: " + errors[-1])
    # Display the exact official injury source and practice participation
    # inspected during this build. Neither an empty game-status cell nor a
    # full practice automatically certifies a player's Sunday workload.
    meta = research.get("injury_meta") or {}
    practice = {}
    duplicate_names = set()
    for row in meta.get("practice_rows", []):
        key = str(row.get("normalized_name") or "").strip()
        if not key:
            continue
        if key in practice:
            duplicate_names.add(key)
        practice[key] = row
    injury_review = []
    for _, row in research["players"].iterrows():
        name = str(row["Name"])
        key = "".join(ch for ch in name.casefold() if ch.isalnum())
        matched = practice.get(key) if key not in duplicate_names else None
        status = str(row.get("current_status") or "").strip().upper()
        if status in {"<NA>", "NAN", "NONE"}:
            status = ""
        if matched or status:
            injury_review.append({
                "Player": name,
                "Team": str(row.get("Team", "")),
                "Injury": str((matched or {}).get("injury") or ""),
                "Practice": str((matched or {}).get("practice_status") or "Not reported"),
                "Game status": status or str((matched or {}).get("game_status") or "Not yet designated"),
                "DraftKings status": str(row.get("DK Status") or "").strip(),
                "Source": "NFL report" if matched else "DK slate status only"})
    allowance_data = research.get("opponent_allowances")
    if allowance_data is None:
        allowance_data = pd.DataFrame(columns=[
            "defense", "position", "multiplier", "player_game_observations"])
    opposing_report = make_opponent_report(
        parse_csv_payload(salaries[1]), allowance_data, meta)
    opposing_summary = report_summary(opposing_report)
    opposing_summary["historical_data_issue"] = research.get("opponent_history_issue")
    pregame_report = {
        "official_injury_url": str(meta.get("source") or ""),
        "checked_utc": meta.get("checked_utc"),
        "injuries": injury_review,
        "questionable_count": int(research.get("questionable_count", 0)),
        "excluded_count": int(research.get("excluded_count", 0)),
        "news_feed": "No verified late-breaking news feed connected. Official game designations and published practice reports are checked; Sunday inactives and workload limits require a fresh check."}
    summary = []
    for index, entry in enumerate(ranked, 1):
        roster = entry["candidate"]["players"]
        summary.append({
            "Lineup": index, "Contest": entry["contest_tier"],
            "Salary": entry["candidate"]["salary"],
            "Mean": round(entry["simulated_mean"], 2),
            "P90": round(entry["simulated_p90"], 2),
            "P99": round(entry["simulated_p99"], 2),
            "P90/$1000": entry["p90_per_1000"],
            "Players": ", ".join(str(p["Name"]) for p in roster),
        })
    role_trends = research.get("role_trends") or []
    return {
        "team_environment": research.get("team_environment") or {},
        "snap_role_report": [
            {"Player": p[0], "Team": p[1], "Position": p[2],
             "Last observed week": v["last_observed_week"],
             "Offensive snaps expected": v["expected_offensive_snaps"],
             "Recent offense snap share %": v["recent_snap_pct"],
             "Earlier offense snap share %": v["earlier_snap_pct"],
             "Snap-based target estimate": v.get("expected_targets"),
             "Snap-based carry estimate": v.get("expected_carries"),
             "Observed snap games": v["snap_games"],
             "Source": v["source"], "Routes available": False}
            for p, v in sorted((research.get("snap_profiles") or {}).items())],
        "snap_feed_issue": research.get("snap_issue"),
        "injury_context_report": research.get("injury_context_report") or [],
        "off_slate_confirmed_out": research.get("confirmed_off_slate_out") or [],
        "reserve_status_issue": research.get("reserve_status_issue"),
        "input_lineage": input_lineage_rows(research),
        "role_trends": role_trends,
        "lineup_role_warnings": lineup_role_warnings(ranked, role_trends),
        "summary": summary, "ranked": ranked,
        "csv": export_classic_candidates([x["candidate"] for x in ranked]),
        "audit": audit, "seed": seed, "pregame_report": pregame_report,
        "player_simulation_audit": simulation_component_audit,
        "projection_review_flags": high_mean_flags,
        "player_simulation_audit_error": simulation_component_audit_error,
        "opponent_matchup_report": opposing_report,
        "opponent_matchup_summary": opposing_summary,
        "research_warning": (
            "Experimental, uncalibrated own-model forecasts based on observed play "
            "rates and official injury designations. Depth-chart changes, late "
            "inactives, matchups and defensive projections require review."),
    }
