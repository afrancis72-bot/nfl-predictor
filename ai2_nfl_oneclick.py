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
from ai2_nfl_player_audit import summarize_simulated_players, require_reconciled_player_audit
from ai2_nfl_candidate_builder import build_classic_candidates
from ai2_nfl_sim_portfolio import select_simulated_portfolio
from ai2_nfl_qb_audit import qb_selection_report
from ai2_nfl_export import export_classic_candidates
from ai2_nfl_opponent_intelligence import make_opponent_report, report_summary
from ai2_nfl_defensive_efficiency import build_defense_profiles
from ai2_nfl_input_integrity import input_lineage_rows, lineup_role_warnings
from ai2_nfl_projection_review import projection_review_flags
from ai2_nfl_role_plausibility import compare_simulated_role_to_history
from ai2_nfl_projection_accountability import uncertain_role_exposures
from ai2_nfl_role_intelligence import returning_rb_competition
from ai2_nfl_prop_intelligence import player_prop_decisions
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

def build_oneclick_portfolio(salaries, projections=None, *, seed=42, count=20,
                             simulation_mode="existing", availability_context=None,
                             draft_manual_review=False, draft_role_defense_strength=0.0):
    """Build from the salary file; legacy projections argument is ignored."""
    from ai2_nfl_draft_matchup import review_strength, paired_projection_rows, draft_exposure_limit
    role_defense_strength=review_strength(draft_role_defense_strength,draft_manual_review)
    if not isinstance(draft_manual_review,bool):
        raise ValueError("Draft manual review flag must be boolean")
    if simulation_mode not in {"existing","availability_aware"}:
        raise ValueError("Unknown NFL simulator. Select existing or availability-aware.")
    if simulation_mode=="availability_aware" and not isinstance(availability_context,dict):
        raise ValueError("Availability-aware NFL builds require the verified official-source prebuild check")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        raise ValueError("Simulation seed must be an integer from 0 to 2147483647")
    if count not in (14, 15, 20, 50, 150):
        raise ValueError("Choose 14, 15, 20, 50, or 150 lineups")
    exposure_limit=draft_exposure_limit(count)
    if not salaries:
        raise ValueError("Upload a DraftKings salary CSV to build lineups")
    season, week = _slate_week(salaries)
    research = collect_automatic_baselines(salaries, season, week)
    missing_official_status=any(
        "Official NFL injury feed unavailable" in issue
        for issue in research["blockers"])
    if missing_official_status and not draft_manual_review:
        raise ValueError(
            "Official game-status data cannot be retrieved; please retry "
            "when the source is available")
    # In human-review DRAFT mode, draft-only projections may proceed using
    # explicit DK salary status and historical role data. Missing official
    # injuries stay UNKNOWN and require manual injury verification before use.
    if not research.get("pregame_role_evidence"):
        raise ValueError("Pregame consistency evidence unavailable; refusing to build from historical averages alone")
    pool, raw_rates, outs, audit = create_observed_model_pool(research)
    # A stale source week or an uncertain returning high-volume teammate must
    # not let the optimization treat a reserve RB as the unquestioned starter.
    # Review only documented pregame rates + currently surfaced designations.
    returning_risks, returning_watch = returning_rb_competition(
        research["players"], raw_rates, season=season, week=week)
    # Confirmed officially OUT players can be absent from DK salaries.
    # Ensure a specific player's historical targets are redistributed at most once.
    already_out = {(str(z["player"]).casefold(),str(z["team"])) for z in outs}
    for z in research.get("confirmed_off_slate_out", []):
        k=(str(z["player"]).casefold(),str(z["team"]))
        if k not in already_out:
            outs.append({"player":z["player"],"team":z["team"]})
            already_out.add(k)
    # Research-only official game-day source claims are integrated into THIS
    # same NFL simulator and portfolio builder. The existing scoring path is
    # unchanged unless the user explicitly enables the availability mode.
    football_extras=None
    source_snapshots={}
    availability_report={
        "mode":"EXISTING_NFL_GAME_ENGINE",
        "effective_scorer":"EXISTING_PLAYER_UNIVERSE",
        "source_coverage":[],
        "game_snapshot_count":0,
        "verified_active_players":0,"verified_out_players":0,
        "football_only_players_added":0,
        "predictive_accuracy_validated":False,
    }
    if simulation_mode=="availability_aware":
        from ai2_nfl_unified_availability import apply_status_sources_to_pool
        source_snapshots=availability_context.get("snapshots_by_game",{})
        pool,football_extras,official_status_out,availability_report=(
            apply_status_sources_to_pool(
                pool,raw_rates,context=availability_context,
                season=season,week=week))
        present={(str(x["player"]).casefold(),str(x["team"]).upper())
                 for x in outs}
        newly_identified_out=[]
        for item in official_status_out:
            key=(str(item["player"]).casefold(),str(item["team"]).upper())
            if key not in present:
                outs.append(item)
                present.add(key)
                newly_identified_out.append(item)
        availability_report["newly_applied_verified_out"]=len(newly_identified_out)
        availability_report["newly_applied_verified_out_names"]=[
            str(x["player"])+" ("+str(x["team"])+")"
            for x in newly_identified_out]
        # A matchup with no eligible simulated DK players is not part of the
        # scoring run; do not smuggle its unrelated article status into the
        # multi-game evidence validator.
        live_games=set(pool.loc[
            pool["optimizer_eligible"].eq(True),"game"].astype(str))
        source_snapshots={
            game:snap for game,snap in source_snapshots.items()
            if game in live_games
        }
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

    # Opponent records are strictly PRE-kickoff, source checked and capped.
    # This first-stage adjustment changes observed yard efficiency only;
    # targets, carries, touchdown opportunities and coverage assignments remain
    # independently modeled. Do not claim this heuristic is calibrated yet.
    allowance_data = research.get("opponent_allowances")
    if allowance_data is None:
        allowance_data = pd.DataFrame(columns=[
            "defense", "position", "multiplier", "player_game_observations", "source"])
    if not isinstance(allowance_data, pd.DataFrame):
        raise ValueError("Opponent allowance source must be a DataFrame")
    defense_profiles = (build_defense_profiles(allowance_data, season, week)
                        if not allowance_data.empty else {})
    # Independent 2026 W4-W8 holdout (117 player-games) showed neutral scoring
    # MAE 6.904 vs 6.911 at strength .25. Keep matchup evidence visible, but
    # reject production activation until a revised signal beats the baseline.
    defense_weight = 0.0
    # Manual-review drafts must not silently enable the EXPERIMENTAL
    # role-defense strength just because draft generation is available.
    # We can display measured matchup evidence independently while the
    # unvalidated scoring coefficient stays neutral (0). Research-specific
    # validation can explicitly use the experimental adapter elsewhere.
    # Explicit draft-only strength; neutral remains the certified/default API path.
    if role_defense_strength and (not isinstance(research.get('opponent_role_stat_profiles'),pd.DataFrame) or research['opponent_role_stat_profiles'].empty):
        raise ValueError('Role-adjusted draft requested but dated role/stat evidence is missing. Select neutral benchmark or refresh sources.')

    # The game-level simulator already redistributes OUT-player touches and
    # targets. Use unmodified rates with outs to avoid double counting.
    availability_options=(
        dict(
            football_only_players=football_extras,
            football_universe_target_season=season,
            football_universe_target_week=week,
            football_availability_snapshots=source_snapshots,
            football_availability_as_of_utc=availability_report.get("as_of_utc"),
        ) if simulation_mode=="availability_aware" else {})
    sim_kwargs=dict(n_sims=10000, seed=seed, out_players=outs,
        team_environment=research.get("team_environment"),
        snap_profiles=research.get("snap_profiles"),
        injury_presence=research.get("injury_presence", {}),
        defense_profiles=defense_profiles, defense_weight=defense_weight,
        role_defense_profiles=research.get("opponent_role_stat_profiles"),
        role_defense_assignments=research.get("opponent_role_slate"),
        role_defense_target_season=season,
        role_defense_target_week=week,
        **availability_options)
    players, draws, stat_draws = run_verified_stat_simulation(pool,raw_rates,role_defense_strength=role_defense_strength,**sim_kwargs)
    paired_matchup_rows=[]
    if role_defense_strength:
        neutral_players,neutral_draws,neutral_stats=run_verified_stat_simulation(pool,raw_rates,role_defense_strength=0.,**sim_kwargs)
        require_reconciled_player_audit(neutral_players,neutral_draws,summarize_simulated_players(neutral_players,neutral_draws,neutral_stats))
        paired_matchup_rows=paired_projection_rows(players,draws,neutral_players,neutral_draws,seed=seed,season=season,week=week)
    if simulation_mode=="availability_aware":
        simulation_ledger=stat_draws.get("__FOOTBALL_PLAYER_UNIVERSE__") or {}
        actual_added=int(simulation_ledger.get("historical_football_only_players",0))
        if actual_added!=int(availability_report["football_only_players_added"]):
            raise ValueError("Football-only teammate source audit does not match simulated world")
        availability_report["simulation_internal_teammates"]=actual_added
        availability_report["simulation_model_label"]=(
            "EXPERIMENTAL_SOURCE_GATED_FULL_FOOTBALL_UNIVERSE"
            if actual_added else
            "EXISTING_SCORING_WITH_CONFIRMED_OUT_FILTER_IF_ANY")
        availability_report["last_build_seed"]=seed
        availability_report["source_data_reverified_by_app"]=bool(
            availability_report.get("game_snapshot_count",0))
        availability_report["statistical_improvement_proven"]=False
    if draws.shape != (10000, len(players)) or not np.isfinite(draws).all():
        raise ValueError("Incomplete NFL simulation; refusing to export lineups")
    # Required scoring safety check, not an optional diagnostic. A high-mean
    # player must never reach the optimizer with missing DK components.
    try:
        simulation_component_audit = summarize_simulated_players(players, draws, stat_draws)
        require_reconciled_player_audit(players, draws, simulation_component_audit)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise ValueError(
            "NFL fantasy scoring components could not be verified for every "
            "eligible player; no lineup export was generated. " + str(exc)
        ) from exc
    simulation_component_audit_error = None
    # Strictly disjoint, seed-deterministic football worlds: the first 70%
    # select players and lineups; the last 30% NEVER enter optimization.
    # This tests simulation overfitting but is not independent historical truth.
    selection_draws=draws[:7000]
    evaluation_draws=draws[7000:]
    # Stat-first decision layer runs BEFORE candidate generation. Bookmaker
    # lines are NOT invented: absent verified lines, only our own predicted
    # statistics, uncertainty and expected DraftKings outcomes are reported.
    player_prop_forecasts = player_prop_decisions(
        players, stat_draws, selection_draws,
        pregame_role=research["pregame_role_evidence"])
    high_mean_flags = projection_review_flags(simulation_component_audit, pool)
    observed_role_flags = compare_simulated_role_to_history(
        simulation_component_audit, raw_rates,
        research.get("injury_context_report") or [])
    means = dict(zip(players["ID"].astype(str), selection_draws.mean(axis=0).tolist()))
    role_trends = research.get("role_trends") or []
    recent_role_lookup = {
        (str(r.get("Player","")).casefold(), str(r.get("Team","")).upper()): r
        for r in role_trends if r.get("Position") == "RB"
    }
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
                player["projected_targets"] = float(targets[:7000].mean())
                player["projected_touches"] = float((targets[:7000] + carries[:7000]).mean())
        if player.get("Position") == "RB" and returning_risks.get(ident):
            impacts = returning_risks[ident]
            player["return_competition_reason"] = "; ".join(
                f"{r['returning_player']} ({r['current_status']}), "
                f"{r['observed_carries_pg']:.1f} historical carries/game, "
                f"last observed W{r['last_observed_week']}"
                for r in impacts)
        # A source-driven RB cap is only possible with pregame observed
        # games, not a fabricated current-week depth-chart assumption.
        if player.get("Position") == "RB":
            recent = recent_role_lookup.get((
                str(player.get("Name","")).casefold(),
                str(player.get("TeamAbbrev","")).upper()), {})
            for source_key,target_key in (
                ("Last 4 observed games","observed_recent_four_games"),
                ("Last 4 rushing yards per game","observed_recent_four_rush_yd_pg"),
                ("Last 4 carries per game","observed_recent_four_carries_pg"),
                ("Last 4 targets per game","observed_recent_four_targets_pg")):
                if recent.get(source_key) is not None:
                    player[target_key] = recent[source_key]
        player["projection_source"] += "; 10,000 football-stat scenarios"
        prepared.append(player)
    candidates = build_classic_candidates(
        prepared, count=(1200 if count <= 20 else 2000), attempts=200000, seed=seed,
        salary_floor=49600, salary_cap=50000, max_exposure=exposure_limit,
        min_wr_salary=4000, min_rb_salary=4500, require_opportunity=True)
    mip_report={'enabled':False,'added_candidates':0}
    if draft_manual_review:
        from ai2_nfl_claude_candidates import supplemental_candidates
        candidates,mip_report=supplemental_candidates(prepared,candidates,players,selection_draws,seed=seed)
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} legal candidates under role and salary rules. "
            "No partial portfolio exported.")
    errors = []
    qb_selection_diagnostics = {}
    for efficiency in (0.95, 0.90, 0.85):
        qb_selection_diagnostics.clear()
        try:
            ranked = select_simulated_portfolio(
                candidates, players, selection_draws, count=count,
                max_exposure=exposure_limit, upside_share=0.25,
                premium_count=(5 if count == 20 else 0), min_wr_salary=4000, min_rb_salary=4500,
                value_salary_ceiling=5000, value_max_appearances=(2 if count <= 20 else max(2, int(count*0.35))),
                premium_value_max_appearances=1, require_opportunity=True,
                p90_efficiency_retention=efficiency,
                premium_excluded_ids=premium_excluded_ids,
                diagnostics=qb_selection_diagnostics,
                validation_draws=evaluation_draws)
            break
        except ValueError as exc:
            errors.append(str(exc))
    else:
        raise ValueError("The model could not produce qualifying lineups: " + errors[-1])
    # Explain every verified QB's journey from salary slate / eligible WR-TE
    # partners to legal candidates, simulated P90 filter, and FINAL exposure.
    # This report is a diagnosis, not a request to force a particular QB.
    qb_audit=qb_selection_report(
        prepared,candidates,ranked,diagnostics=qb_selection_diagnostics,
        team_environment=research.get("team_environment"))
    qb_missing_stacks=[
        r for r in qb_audit
        if r["Eligible passing partners"] and not r["Stacked QB candidates"]]
    qb_selected_stacks=sum(r["Final QB/WR-TE stacks"] for r in qb_audit)
    qb_selection_diagnostics["final_stacked_lineups"]=qb_selected_stacks
    qb_selection_diagnostics["final_unstacked_lineups"]=len(ranked)-qb_selected_stacks
    qb_selection_diagnostics["qb_with_verified_receiver_but_no_candidate_stack"]=len(qb_missing_stacks)
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
    opposing_report = make_opponent_report(
        parse_csv_payload(salaries[1]), allowance_data, meta,
        defense_profiles=defense_profiles, defense_weight=defense_weight)
    # Compare previous opponents with their own rolling pregame baselines,
    # rather than assuming that facing a star automatically made a defense weak.
    # Research-only: the 2026 holdout did not improve total historical MAE.
    relative_data = research.get("opponent_relative_profiles")
    relative_rows = (relative_data.to_dict("records")
                     if isinstance(relative_data, pd.DataFrame) else [])
    relative_lookup = {
        (str(z["defense"]), str(z["position"])): z
        for z in relative_rows}
    for r in opposing_report:
        relative = relative_lookup.get(
            (r["Opposing defense"], r["Position"]))
        r["Opponent-relative DK excess (shrunk)"] = (
            relative["shrunk_excess_dk"] if relative else None)
        r["Opponent-relative samples"] = (
            relative["observations"] if relative else 0)
        r["Previous opponent baseline DK"] = (
            relative["mean_player_baseline_dk"] if relative else None)
        r["Previous opponent actual DK"] = (
            relative["mean_actual_dk"] if relative else None)
        r["Opponent-relative adjustment applied"] = False
    # Distinguish matchup mechanism from raw DK scoring. These historical
    # per-opportunity rates are measured against past opponents' own baselines,
    # and are RESEARCH ONLY until an independently tested stat simulator earns
    # an improvement. Never apply them to P90, carries, targets or TDs here.
    component_data = research.get("opponent_component_profiles")
    component_lookup = (component_data.to_dict("records")
                        if isinstance(component_data, pd.DataFrame) else [])
    offense_for_defense = {
        (str(r["Offense"]), str(r["Opposing defense"]))
        for r in opposing_report}
    opponent_components = []
    for z in component_lookup:
        for offense, opponent in offense_for_defense:
            if str(z["defense"]) != opponent:
                continue
            opponent_components.append({
                "Offense": offense,
                "Opposing defense": opponent,
                "Position": z["position"],
                "Mechanism": z["channel"],
                "Opponents usual yards per opportunity":
                    z["expected_yards_per_opp"],
                "Actual yards per opportunity allowed":
                    z["actual_yards_per_opp"],
                "Quality-adjusted efficiency delta %":
                    z["shrunk_efficiency_pct"],
                "Previous player-games": z["observations"],
                "Prior opportunities": z["opportunities"],
                "Effect on current simulation": "None - research only",
            })
    opponent_components.sort(key=lambda x: (
        x["Offense"], x["Position"], x["Mechanism"]))
    # Role-specific DK defensive tendencies are for human projection review.
    # Roles are based on each prior opponent's historical target/carry/pass
    # usage BEFORE facing the defense, not the target game's target ranks.
    # Expose all DK salary rows, including players outside the model pool.
    role_matchup_players = research.get("opponent_role_slate") or []
    role_defense_ledger = (stat_draws.get("__ROLE_DEFENSE_STAT_CHANNELS__")
                           or {})
    role_defense_issue = research.get("opponent_role_stat_issue")
    if role_defense_issue:
        role_defense_ledger["source_issue"] = role_defense_issue
    role_defense_ledger["release_gate"] = "NOT READY FOR ENTRY"
    role_profile_data = research.get("opponent_role_profiles")
    role_detail_data = research.get("opponent_role_details")
    relevant_defenses = {opp for _, opp in offense_for_defense}
    role_profile_report = (
        role_profile_data[role_profile_data["defense"].isin(
            relevant_defenses)].to_dict("records")
        if isinstance(role_profile_data, pd.DataFrame)
        and not role_profile_data.empty else [])
    role_detail_report = (
        role_detail_data[role_detail_data["defense"].isin(
            relevant_defenses)].to_dict("records")
        if isinstance(role_detail_data, pd.DataFrame)
        and not role_detail_data.empty else [])
    opposing_summary = report_summary(opposing_report)
    opposing_summary["role_matchup_issue"] = research.get("opponent_role_issue")
    opposing_summary["role_matchup_note"] = (
        "Previous QB1, RB1/RB2, WR1/WR2/WR3 and TE1/TE2 opponents "
        "classified by THEIR OWN pregame usage (QB attempts, RB carries + "
        "targets, WR/TE targets). Their actual partial DraftKings fantasy "
        "points are compared with THEIR OWN rolling pregame averages. "
        "A +4.00 reading means four MORE points than those players normally "
        "produced, not four total points. Raw and sample-shrunk values both "
        "shown. Unavailable, unmatched and low-sample players have no boost. "
        "This is NOT verified depth-chart/coverage. No direct DK residual "
        "bonus is used; separate source-backed stat-channel role adjustments "
        "are experimental in the locked simulator and shown in its audit.")
    opposing_summary["component_data_issue"] = research.get("opponent_component_issue")
    opposing_summary["component_note"] = (
        "Observed passing yards/attempt, rushing yards/carry and receiving "
        "yards/target against previous opponents' own pregame efficiency. "
        "The RB rushing and receiving channels are independent. These rates "
        "DO NOT adjust the current simulation, QB/WR touchdown odds, injury "
        "status or coverage assignments.")
    opposing_summary["historical_data_issue"] = research.get("opponent_history_issue")
    opposing_summary["relative_evidence_issue"] = research.get("opponent_relative_issue")
    opposing_summary["relative_evidence_note"] = (
        "Research only. Prior opponents' partial DraftKings points are "
        "compared with THEIR OWN pregame rolling baseline, then shrunk "
        "toward no effect. This helps remove player-talent confounding "
        "but does not identify WR-CB assignments, injuries or pass/run mechanisms. "
        "An independent 2026 holdout did NOT improve overall mean-error. "
        "These differences are NOT applied to lineup scoring.")
    opposing_summary["active_yard_efficiency_weight"] = defense_weight
    opposing_summary["eligible_defense_position_profiles"] = len(defense_profiles)
    opposing_summary["yards_only_warning"] = (
        "Opponent-position yard adjustment is DISABLED (neutral 1.0) after "
        "independent 2026 W4-W8 player holdout: MAE 6.904 neutral versus "
        "6.911 with weight 0.25 (n=117). The signal uses fantasy points "
        "allowed as a yards-efficiency proxy and needs redesign. "
        "Defender absences and WR-CB coverage remain research-only.")
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
    role_risk_exposure=uncertain_role_exposures(observed_role_flags,ranked)
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
        "priced_confirmed_reserve_out": research.get("confirmed_priced_reserve_out") or [],
        "reserve_status_issue": research.get("reserve_status_issue"),
        "input_lineage": input_lineage_rows(research),
        "role_trends": role_trends,
        "pregame_role_evidence": list(research["pregame_role_evidence"].values()),
        "player_prop_forecasts": player_prop_forecasts,
        "prop_model_method": (
            "Player-by-player simulated football-stat predictions from 7,000 "
            "selection worlds BEFORE lineup construction; same-world simulated "
            "DraftKings points include catches, yards, touchdowns and bonuses. "
            "No current sportsbook lines, odds or historical calibration are "
            "automatically connected."),
        "pregame_role_blocked_count": sum(
            x.get("Role gate") == "BLOCK"
            for x in research["pregame_role_evidence"].values()),
        "pregame_role_review_method": (
            "Last five observed PRE-kickoff games by same-team player, with "
            "repeat targets/carries and 10+ partial-DK scoring history; "
            "one-game spikes cannot independently qualify cheap players."),
        "returning_rb_watch": returning_watch,
        "returning_rb_affected_ids": {
            ident: [r["returning_player"] for r in matches]
            for ident,matches in returning_risks.items()},
        "lineup_role_warnings": lineup_role_warnings(ranked, role_trends),
        "qb_selection_audit": qb_audit,
        "qb_selection_diagnostics": qb_selection_diagnostics,
        "counterfactual_holdout": qb_selection_diagnostics.get("independent_scenario_audit",{}),
        "simulation_mode":simulation_mode,
        "draft_manual_review_only":bool(draft_manual_review),
        "official_injury_status_coverage_incomplete":bool(missing_official_status),
        "statistical_engine_version":str(stat_draws.get("__MODEL_VERSION__","UNKNOWN")),
        "availability_intelligence":availability_report,
        "summary": summary, "ranked": ranked,
        "csv": export_classic_candidates([x["candidate"] for x in ranked]),
        "audit": audit, "seed": seed, "pregame_report": pregame_report,
        "player_simulation_audit": simulation_component_audit,
        "projection_review_flags": high_mean_flags,
        "role_projection_anomalies": observed_role_flags,
        "role_risk_exposure": role_risk_exposure,
        "player_simulation_audit_error": simulation_component_audit_error,
        "opponent_matchup_report": opposing_report,
        "opponent_matchup_summary": opposing_summary,
        "opponent_component_report": opponent_components,
        "opponent_role_player_report": role_matchup_players,
        "opponent_role_defense_report": role_profile_report,
        "opponent_role_prior_opponents": role_detail_report,
        "opponent_role_stat_scoring_audit": role_defense_ledger,
        "draft_role_defense_strength": role_defense_strength,
        "paired_matchup_projections": paired_matchup_rows,
        "claude_mip_candidate_report": mip_report,
        "opponent_role_stat_profiles": (
            research.get("opponent_role_stat_profiles").to_dict("records")
            if isinstance(research.get("opponent_role_stat_profiles"),pd.DataFrame)
            else []),
        "research_warning": (
            "DRAFT ONLY: uncalibrated player projections, manual review mandatory. "
            "Check the official NFL game-status list and every late inactive; "
            "depth charts, routes and individual coverage may be incomplete. "
            "No automatic DraftKings submission. "
            "Legacy opponent-position yard multiplier is OFF. " +
            ("Role-defense draft adjustments are ON; same-seed neutral comparison included. " if role_defense_strength else "Role-defense projection adjustments are OFF. ") +
            "Bentley certification has NOT passed."+
            (" Official injury feed could NOT be retrieved; "
             "all unsourced statuses require manual verification."
             if missing_official_status else "")+
            (" Experimental official-source player universe opted in; "
             "predictive accuracy and P90 improvement NOT YET validated."
             if simulation_mode=="availability_aware" else "")),
    }
