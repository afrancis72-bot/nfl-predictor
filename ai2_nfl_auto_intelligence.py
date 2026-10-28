"""Automatic nflverse-backed pregame research from the saved DraftKings slate.

This stage is a historical baseline, not a certified current-week prediction.
It returns provenance and blockers explicitly. No outside projections upload.
"""
from ai2_nfl_inputs import parse_csv_payload
from ai2_nfl_auto_baselines import build_historical_baselines
from props_data import fetch_weekly

def collect_automatic_baselines(salaries, season, week):
    salary_rows = parse_csv_payload(salaries[1])
    if not salary_rows:
        raise ValueError("Salary slate contains no players")
    history = fetch_weekly([season - 1, season])
    baselines = build_historical_baselines(salary_rows, history, season, week)
    eligible = baselines[baselines["match_status"].eq("matched")]
    unsupported = baselines[~baselines["match_status"].eq("matched")]
    return {
        "players": baselines,
        "source": "nflverse stats_player historical weekly observations",
        "target_season": season,
        "target_week": week,
        "matched_count": len(eligible),
        "missing_count": len(unsupported),
        "missing_examples": unsupported["Name"].astype(str).head(6).tolist(),
        "release_ready": False,
        "blockers": [
            "Automatic injury-aware role redistribution is not yet validated",
            "DST projections need an independent defensive-stat model",
            "Historical usage is not a verified current-week opportunity forecast",
        ],
    }
