"""Leakage-safe Week 4-6 replay preflight; no claim of full simulator validation.

Archived DraftKings salary slates and dated injury snapshots are required
before a complete historical DFS lineup replay can be certified.
"""
import pandas as pd
from props_data import clean_weekly, make_rates, walkforward_dk_offense

REQUIRED = {"ID", "Name", "Position", "TeamAbbrev", "Salary", "game"}

def replay_preflight(history, season=2026, weeks=(4, 5, 6), salary_snapshots=None):
    h = clean_weekly(history)
    salary_snapshots = salary_snapshots or {}
    rows = []
    for week in weeks:
        week = int(week)
        prior = h[(h.season < season) | ((h.season == season) & (h.week < week))]
        actual = h[(h.season == season) & (h.week == week)]
        blockers = []
        if prior.empty:
            blockers.append("No pre-week historical stats")
        if actual.empty:
            blockers.append("No target-week results")
        count = 0
        if not prior.empty:
            rates = make_rates(prior, season, week)
            count = len(rates)
            if ((rates.latest_observed_season > season) | ((rates.latest_observed_season == season) & (rates.latest_observed_week >= week))).any():
                blockers.append("Future-week leakage")
        salary = salary_snapshots.get(week)
        if salary is None:
            blockers.append("Missing archived pre-kickoff DraftKings salaries")
        elif REQUIRED - set(salary.columns):
            blockers.append("Salary columns missing: " + ", ".join(sorted(REQUIRED - set(salary.columns))))
        rows.append(dict(week=week, pregame_player_rates=count, observed_rows=len(actual),
                         full_replay_ready=not blockers, blockers="; ".join(blockers)))
    return pd.DataFrame(rows)

def baseline_replay(history, season=2026, weeks=(4, 5, 6)):
    """Only baseline partial-DK scoring, NOT complete simulator/lineup ROI."""
    h = clean_weekly(history)
    detail = pd.concat([walkforward_dk_offense(h, season, first_week=w, last_week=w).assign(replay_week=w)
                        for w in weeks], ignore_index=True)
    summary = detail.groupby("replay_week", as_index=False).agg(
        player_weeks=("absolute_error", "size"),
        mae=("absolute_error", "mean"), bias=("error", "mean"))
    return detail, summary

if __name__ == "__main__":
    from props_data import fetch_weekly
    history = fetch_weekly([2025, 2026])
    print(replay_preflight(history).to_string(index=False))
    detail, summary = baseline_replay(history)
    print(summary.to_string(index=False))
    detail.to_csv("nfl_replay_baseline_detail.csv", index=False)
    summary.to_csv("nfl_replay_baseline_summary.csv", index=False)

def audit_simulation_outputs(players, draws, ranked=None, *, expected_sims=10000,
                             max_exposure=0.35):
    """Validate simulated fantasy points and optional selected lineup portfolio.

    These are structural/integrity tests, not statistical calibration or ROI.
    """
    import numpy as np
    from collections import Counter
    from ai2_nfl_lineup import check_classic_lineup
    issues = []
    matrix = np.asarray(draws, dtype=float)
    if matrix.ndim != 2 or matrix.shape != (expected_sims, len(players)):
        issues.append("Simulation matrix shape/count mismatch")
    elif not np.isfinite(matrix).all():
        issues.append("Nonfinite fantasy-point outcomes")
    if "ID" not in players or players["ID"].astype(str).duplicated().any():
        issues.append("Missing or duplicate simulation player IDs")
    lineup_count = len(ranked) if ranked is not None else 0
    exposure = Counter()
    if ranked is not None:
        seen = set()
        index = {str(pid): i for i, pid in enumerate(players["ID"].astype(str))} if "ID" in players else {}
        for item in ranked:
            roster = item["candidate"]["players"]
            issues.extend(check_classic_lineup(roster))
            ids = [str(p["ID"]) for p in roster]
            signature = tuple(sorted(ids))
            if signature in seen:
                issues.append("Duplicate lineup")
            seen.add(signature)
            exposure.update(ids)
            if matrix.ndim == 2 and matrix.shape[1] == len(players) and all(k in index for k in ids):
                totals = matrix[:, [index[k] for k in ids]].sum(axis=1)
                if abs(float(totals.mean()) - float(item["simulated_mean"])) > 0.05:
                    issues.append("Lineup mean differs from underlying draws")
                if abs(float(np.percentile(totals, 90)) - float(item["simulated_p90"])) > 0.05:
                    issues.append("Lineup P90 differs from underlying draws")
            else:
                issues.append("Lineup player missing from simulation matrix")
        if lineup_count:
            excessive = {k: n for k, n in exposure.items() if n / lineup_count > max_exposure + 1e-9}
            if excessive:
                issues.append("Player exposure exceeds limit: " + str(excessive))
    return {"status": "PASS" if not issues else "FAIL",
            "simulations": matrix.shape[0] if matrix.ndim == 2 else 0,
            "players": len(players), "lineups": lineup_count,
            "max_exposure": max(exposure.values(), default=0) / max(lineup_count, 1),
            "issues": sorted(set(issues)),
            "scope": "Structural integrity only; no forecast accuracy or profitability claim"}
