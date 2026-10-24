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
