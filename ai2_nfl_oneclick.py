"""One action NFL build: salary slate + optional seed.

No synthetic fallback. Current verified weekly projections are still required
from a real source; fail closed rather than silently promote historic baselines.
"""
import re
from datetime import date, datetime
import numpy as np
import pandas as pd
from ai2_nfl_inputs import inspect_saved_nfl_inputs, parse_csv_payload
from ai2_nfl_eligibility import eligible_nfl_rows
from ai2_nfl_candidate_builder import build_classic_candidates
from ai2_nfl_simulation_adapter import run_verified_stat_simulation
from ai2_nfl_sim_portfolio import select_simulated_portfolio
from ai2_nfl_export import export_classic_candidates

def _slate_week(saved_salaries):
    rows = parse_csv_payload(saved_salaries[1])
    dates = []
    for row in rows:
        match = re.search(r"(\\d{1,2})/(\\d{1,2})/(\\d{4}|\\d{2})", str(row.get("Game Info", "")))
        if match:
            m, d, year = (int(x) for x in match.groups())
            if year < 100:
                year += 2000
            dates.append(date(year, m, d))
    if not dates:
        raise ValueError("Slate game dates not found. Use the current DraftKings NFL Classic salary CSV.")
    season = max(set(dates), key=dates.count).year
    start = date(season, 9, 10) if season == 2026 else None
    if start is None:
        raise ValueError("Automatic NFL week mapping is not configured for this season.")
    week = ((max(set(dates), key=dates.count) - start).days // 7) + 1
    if not 1 <= week <= 18:
        raise ValueError("The salary file does not appear to be a regular-season NFL slate.")
    return season, week

def build_oneclick_portfolio(salaries, projections, *, seed=42, count=20):
    if not 0 <= seed <= 2147483647:
        raise ValueError("Seed must be a nonnegative 32-bit integer.")
    if count != 20:
        raise ValueError("This screen builds exactly 20 Classic lineups.")
    season, week = _slate_week(salaries)
    if not projections:
        raise ValueError(
            "Build cannot complete: automated VERIFIED Week %s player projections "
            "and opportunity data are not connected yet. Your salary file is saved. "
            "Historical data alone cannot justify current-week picks. "
            "No misleading lineups have been generated." % week)
    rows, issues = inspect_saved_nfl_inputs(salaries, projections)
    if issues:
        raise ValueError("Slate validation failed: " + "; ".join(issues))
    eligible = eligible_nfl_rows(rows, min_wr_salary=4000, min_rb_salary=4500, require_opportunity=True)
    if len(eligible) < 9:
        raise ValueError("Insufficient verified current-week player opportunities; no lineups generated.")
    from props_data import fetch_weekly, make_rates
    history = fetch_weekly([season - 1, season])
    rates = make_rates(history, season, week)
    if rates.empty:
        raise ValueError("Observed historical rates unavailable for this slate.")
    pool = pd.DataFrame(eligible)
    pool["optimizer_eligible"] = True
    players, draws, _ = run_verified_stat_simulation(pool, rates, n_sims=10000, seed=seed)
    if draws.shape != (10000, len(players)) or not np.isfinite(draws).all():
        raise ValueError("Stat simulation incomplete; no lineup export.")
    candidates = build_classic_candidates(eligible, count=1000, attempts=200000, seed=seed,
                                         max_exposure=1.0, salary_floor=49600)
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} eligible legal candidates were found.")
    ranked = select_simulated_portfolio(
        candidates, players, draws, count=count, max_exposure=0.35, upside_share=0.25,
        premium_count=5, min_wr_salary=4000, min_rb_salary=4500,
        value_salary_ceiling=5000, value_max_appearances=2,
        premium_value_max_appearances=1, require_opportunity=True)
    summary = []
    for index, entry in enumerate(ranked, 1):
        names = ", ".join(str(p.get("Name") or p["ID"]) for p in entry["candidate"]["players"])
        summary.append({"Lineup": index, "Contest": entry["contest_tier"],
                        "Salary": entry["candidate"]["salary"],
                        "Mean": round(entry["simulated_mean"], 2),
                        "P90": round(entry["simulated_p90"], 2),
                        "P99": round(entry["simulated_p99"], 2),
                        "P90/$1000": entry["p90_per_1000"], "Players": names})
    return {"summary": summary, "ranked": ranked,
            "csv": export_classic_candidates([x["candidate"] for x in ranked])}
