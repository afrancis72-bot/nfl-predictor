"""Real DraftKings slate preview using observed pregame history, not verified projections.

Never certify these rows for contest use. This is a diagnostic bridge to the
actual football-statistics simulation engine, not the verified production path.
"""
import numpy as np
import pandas as pd
from ai2_nfl_auto_baselines import build_historical_baselines
from ai2_nfl_matchups import MATCHUP, canonical_team, opponent_from_game_info
from props_data import make_rates

BLOCKED = {"OUT", "O", "IR", "INACTIVE", "SUSPENDED", "PUP", "NFI", "D", "DOUBTFUL"}

def prepare_real_slate_preview(salary_rows, history, season=2026, week=7):
    baselines = build_historical_baselines(salary_rows, history, season, week)
    rates = make_rates(history, season, week)
    pool = []
    for _, row in baselines.iterrows():
        pos = str(row["Position"]).upper()
        if pos not in {"QB", "RB", "WR", "TE", "DST"}:
            continue
        if str(row["DK Status"]).upper().strip() in BLOCKED:
            continue
        match = MATCHUP.match(str(row["Game Info"] or ""))
        if not match:
            continue
        game = "@".join(canonical_team(t) for t in match.groups())
        team = canonical_team(row["Team"])
        opponent = opponent_from_game_info(team, row["Game Info"])
        if not opponent:
            continue
        try:
            salary = float(row["Salary"])
            historical = float(row["historical_partial_dk"])
        except (ValueError, TypeError):
            continue
        if pos == "DST":
            # The observed offensive history does not include complete DST scoring.
            # Explicit provisional prior is only for workflow diagnostics.
            historical = 6.0
        if not np.isfinite(historical) or historical <= 0 or salary <= 0:
            continue
        # Remove near-zero-volume historical skill players BEFORE the engine
        # allocates team touchdowns. These are not credible DFS candidates.
        if pos in {"RB", "WR", "TE"} and historical < 1.0:
            continue
        pool.append(dict(ID=str(row["ID"]), Name=str(row["Name"]), Position=pos,
                         TeamAbbrev=team, Salary=salary, game=game,
                         opponent=opponent, proj=historical,
                         optimizer_eligible=True, weekly_projection_valid=False,
                         opportunity_verified=False,
                         projection_source="PROVISIONAL historical partial DK baseline"))
    if not pool:
        raise ValueError("No real DraftKings players matched observed historical statistics")
    frame = pd.DataFrame(pool)
    # The statistical engine requires both teams in every simulated matchup.
    complete = set()
    for game, group in frame.groupby("game"):
        if set(group["TeamAbbrev"]) == set(game.split("@")):
            complete.add(game)
    frame = frame[frame["game"].isin(complete)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No complete two-team matchups with matched real-player history")
    return frame, rates
