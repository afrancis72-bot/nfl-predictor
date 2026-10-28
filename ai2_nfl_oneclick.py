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
from ai2_nfl_candidate_builder import build_classic_candidates
from ai2_nfl_sim_portfolio import select_simulated_portfolio
from ai2_nfl_export import export_classic_candidates

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
    if count != 20:
        raise ValueError("NFL Classic one-click build requires 20 lineups")
    if not salaries:
        raise ValueError("Upload a DraftKings salary CSV to build lineups")
    season, week = _slate_week(salaries)
    research = collect_automatic_baselines(salaries, season, week)
    if any("Official NFL injury feed unavailable" in issue for issue in research["blockers"]):
        raise ValueError("Official game-status data cannot be retrieved; please retry when the source is available")
    pool, raw_rates, outs, audit = create_observed_model_pool(research)
    # The game-level simulator already redistributes OUT-player touches and
    # targets. Use unmodified rates with outs to avoid double counting.
    players, draws, stat_draws = run_verified_stat_simulation(
        pool, raw_rates, n_sims=10000, seed=seed, out_players=outs)
    if draws.shape != (10000, len(players)) or not np.isfinite(draws).all():
        raise ValueError("Incomplete NFL simulation; refusing to export lineups")
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
        prepared, count=1200, attempts=200000, seed=seed,
        salary_floor=49600, salary_cap=50000, max_exposure=1.0,
        min_wr_salary=4000, min_rb_salary=4500, require_opportunity=True)
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} legal candidates under role and salary rules. "
            "No partial portfolio exported.")
    errors = []
    for efficiency in (0.95, 0.90, 0.85):
        try:
            ranked = select_simulated_portfolio(
                candidates, players, draws, count=20,
                max_exposure=0.35, upside_share=0.25,
                premium_count=5, min_wr_salary=4000, min_rb_salary=4500,
                value_salary_ceiling=5000, value_max_appearances=2,
                premium_value_max_appearances=1, require_opportunity=True,
                p90_efficiency_retention=efficiency)
            break
        except ValueError as exc:
            errors.append(str(exc))
    else:
        raise ValueError("The model could not produce 20 qualifying lineups: " + errors[-1])
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
    return {
        "summary": summary, "ranked": ranked,
        "csv": export_classic_candidates([x["candidate"] for x in ranked]),
        "audit": audit, "seed": seed,
        "research_warning": (
            "Experimental, uncalibrated own-model forecasts based on observed play "
            "rates and official injury designations. Depth-chart changes, late "
            "inactives, matchups and defensive projections require review."),
    }
