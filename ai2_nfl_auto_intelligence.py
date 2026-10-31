"""Automatic pregame research from DraftKings salaries and observed NFL sources.

Collects independently sourced pregame history and explicit NFL game statuses.
Historical averages are not quietly certified as current-week forecasts.
"""
from ai2_nfl_inputs import parse_csv_payload
from ai2_nfl_auto_baselines import build_historical_baselines
from ai2_nfl_official_injuries import fetch_official_injury_statuses, match_injury_statuses
from props_data import fetch_weekly, make_rates
from ai2_nfl_workload import redistribute_opportunities
from ai2_nfl_matchup_diagnostics import defensive_allowance
from ai2_nfl_input_integrity import player_role_trends
from ai2_nfl_team_environment import predict_team_environment
from ai2_nfl_snap_roles import fetch_snap_counts, predict_snap_roles
from ai2_nfl_injury_context import official_off_slate_out, estimate_injury_presence
import pandas as pd

UNAVAILABLE = frozenset({"OUT", "O", "IR", "INACTIVE", "SUSPENDED", "PUP", "NFI", "DOUBTFUL", "D"})

def collect_automatic_baselines(salaries, season, week, *, history=None, injury_fetcher=None,
                                snap_history=None, snap_fetcher=None):
    salary_rows = parse_csv_payload(salaries[1])
    if not salary_rows:
        raise ValueError("Salary slate contains no players")
    history = fetch_weekly([season - 1, season]) if history is None else history
    baseline = build_historical_baselines(salary_rows, history, season, week)
    fetcher = injury_fetcher or fetch_official_injury_statuses
    overrides = {}
    injury_meta = {"source": "unavailable", "checked_utc": None}
    injury_issue = None
    try:
        official, injury_meta = fetcher()
        overrides = match_injury_statuses(salary_rows, official)
    except (ValueError, OSError, TimeoutError) as exc:
        injury_issue = str(exc)
    baseline = baseline.copy()
    baseline["official_game_status"] = baseline["ID"].astype(str).map(overrides).astype("string").fillna("")
    baseline["current_status"] = baseline["official_game_status"].where(
        baseline["official_game_status"].ne(""),
        baseline["DK Status"].fillna("").astype(str).str.upper().str.strip())
    baseline["excluded_for_status"] = baseline["current_status"].isin(UNAVAILABLE)
    baseline["requires_injury_review"] = baseline["current_status"].isin({"QUESTIONABLE", "Q"})
    baseline["model_candidate"] = (baseline["match_status"].eq("matched")
                                   & ~baseline["excluded_for_status"]
                                   & ~baseline["requires_injury_review"])
    rates = make_rates(history, season, week)
    team_environment = predict_team_environment(history, season, week)
    snap_profiles = {}
    snap_issue = None
    try:
        if snap_history is None:
            fetch_snap = snap_fetcher or fetch_snap_counts
            snap_history = fetch_snap([season])
        snap_profiles = predict_snap_roles(
            history, snap_history, season, week, team_environment=team_environment)
    except (OSError, ValueError, TypeError, KeyError, TimeoutError) as exc:
        # Don't pretend historical targets were adjusted when the snap source
        # is unreachable. Keep the old scoring path, and disclose the reason.
        snap_issue = str(exc)
    # Observable pregame usage changes must be visible in the review, not
    # silently confused with independent current-week role verification.
    role_trends = player_role_trends(history, salary_rows, season, week)
    statuses = {(str(row["Name"]), str(row.get("Team", ""))): str(row["current_status"])
                for _, row in baseline.iterrows()}
    adjusted_rates, opportunity_changes = redistribute_opportunities(rates, statuses)
    # Official OUT reports can identify a teammate who is missing from the
    # DraftKings salary file; never infer IR or OUT merely from that absence.
    off_slate_out = official_off_slate_out(
        injury_meta.get("practice_rows") or [], salary_rows, rates)
    recorded_out = [
        {"player": row["Name"], "team": row["Team"]}
        for _, row in baseline.iterrows()
        if bool(row["excluded_for_status"]) and
           str(row["Position"]).upper() in {"QB","RB","WR","TE"}
    ]
    injury_presence, injury_context_report = estimate_injury_presence(
        history, recorded_out + off_slate_out, season, week)
    opponent_issue = None
    try:
        # Every game used here predates the target slate week.
        opponent_allowances = defensive_allowance(history, season, week)
    except (KeyError, ValueError, TypeError) as exc:
        # Missing data must be neutral, never silently fabricated.
        opponent_allowances = pd.DataFrame(columns=[
            "defense", "position", "multiplier", "player_game_observations"])
        opponent_issue = str(exc)
    missing = baseline[baseline["match_status"].ne("matched")]
    blockers = ["Workload redistribution is estimated from observed rates, not validated against current depth charts",
                "Independent DST projections not yet connected",
                "Historical rates are not a verified current-week role forecast"]
    if injury_issue:
        blockers.append("Official NFL injury feed unavailable: " + injury_issue)
    return {
        "players": baseline, "raw_rates": rates, "adjusted_rates": adjusted_rates,
        "opponent_allowances": opponent_allowances, "opponent_history_issue": opponent_issue,
        "opportunity_changes": opportunity_changes, "role_trends": role_trends,
        "team_environment": team_environment,
        "snap_profiles": snap_profiles,
        "snap_issue": snap_issue,
        "injury_presence": injury_presence,
        "injury_context_report": injury_context_report,
        "confirmed_off_slate_out": off_slate_out,
        "source": "nflverse historical stats and NFL official game designations",
        "target_season": season, "target_week": week,
        "matched_count": int(baseline["match_status"].eq("matched").sum()),
        "candidate_count": int(baseline["model_candidate"].sum()),
        "excluded_count": int(baseline["excluded_for_status"].sum()),
        "questionable_count": int(baseline["requires_injury_review"].sum()),
        "missing_count": len(missing), "missing_examples": missing["Name"].astype(str).head(6).tolist(),
        "injury_meta": injury_meta, "release_ready": False, "blockers": blockers,
    }
