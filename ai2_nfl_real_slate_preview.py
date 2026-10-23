"""Real DraftKings slate preview using observed pregame history, not verified projections.

Never certify these rows for contest use. This is a diagnostic bridge to the
actual football-statistics simulation engine, not the verified production path.
"""
import numpy as np
import pandas as pd
from ai2_nfl_auto_baselines import build_historical_baselines
from ai2_nfl_matchups import MATCHUP, canonical_team, opponent_from_game_info
from props_data import make_rates

BLOCKED = {"OUT", "O", "IR", "INACTIVE", "SUSPENDED", "PUP", "NFI", "D", "DOUBTFUL", "RESERVE", "PRACTICE SQUAD"}
MIN_OFFENSIVE_BASELINE_DK = 4.0
MIN_SKILL_OPPORTUNITIES_PG = {"RB": 3.0, "WR": 2.0, "TE": 2.0}
MIN_QB_PASS_ATTEMPTS_PG = 15.0

def prepare_real_slate_preview(salary_rows, history, season=2026, week=7, current_statuses=None):
    """Historical preview only; optional verified status overrides keyed by DK ID.

    Missing current status never means that a player is confirmed active.
    """
    current_statuses = current_statuses or {}
    baselines = build_historical_baselines(salary_rows, history, season, week)
    rates = make_rates(history, season, week)
    pool = []
    for _, row in baselines.iterrows():
        pos = str(row["Position"]).upper()
        if pos not in {"QB", "RB", "WR", "TE", "DST"}:
            continue
        current_status = str(current_statuses.get(str(row["ID"]), row["DK Status"])).upper().strip()
        if current_status in BLOCKED:
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
        if pos in {"QB", "RB", "WR", "TE"} and historical < MIN_OFFENSIVE_BASELINE_DK:
            continue
        if pos in {"RB", "WR", "TE"}:
            # A positive partial fantasy score can still come almost entirely
            # from rare touchdowns. Require observed non-TD football volume.
            observed = rates[
                rates["player"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
                .eq(str(row["Name"]).lower().replace(" ", "").replace(".", "").replace("'", "").replace("-", ""))
                & rates["team"].map(canonical_team).eq(team)
                & rates["position"].astype(str).str.upper().eq(pos)
            ]
            if len(observed) != 1:
                continue
            obs = observed.iloc[0]
            carries = float(pd.to_numeric(obs.get("carries_pg"), errors="coerce") or 0)
            targets = float(pd.to_numeric(obs.get("targets_pg"), errors="coerce") or 0)
            if not np.isfinite(carries):
                carries = 0.0
            if not np.isfinite(targets):
                targets = 0.0
            if (carries + targets) < MIN_SKILL_OPPORTUNITIES_PG[pos]:
                continue
        pool.append(dict(ID=str(row["ID"]), Name=str(row["Name"]), Position=pos,
                         TeamAbbrev=team, Salary=salary, game=game,
                         opponent=opponent, proj=historical,
                         dk_status=current_status,
                         optimizer_eligible=True, weekly_projection_valid=False,
                         opportunity_verified=False,
                         projection_source="PROVISIONAL historical partial DK baseline"))
    if not pool:
        raise ValueError("No real DraftKings players matched observed historical statistics")
    frame = pd.DataFrame(pool)
    # The engine simulates exactly one QB per team; allowing several DK
    # quarterbacks creates unsimulated zero-point rows and aborts the run.
    # Select the observed primary passer, not every historical backup.
    quarterbacks = frame[frame["Position"].eq("QB")].copy()
    if not quarterbacks.empty:
        qb_rates = rates[rates["position"].astype(str).str.upper().eq("QB")].copy()
        qb_rates["_team"] = qb_rates["team"].map(canonical_team)
        qb_rates["_name"] = qb_rates["player"].astype(str).str.lower().str.replace(
            r"[^a-z0-9]", "", regex=True)
        qb_rates["_pass"] = pd.to_numeric(qb_rates["pass_attempts_pg"], errors="coerce").fillna(0)
        qb_lookup = {(str(r["_team"]), str(r["_name"])): float(r["_pass"])
                     for _, r in qb_rates.iterrows()}
        quarterbacks["_pass"] = quarterbacks.apply(
            lambda r: qb_lookup.get((r["TeamAbbrev"],
                "".join(c for c in r["Name"].lower() if c.isalnum())), 0.0), axis=1)
        # Do not nominate a low-volume backup as the sole simulated QB.
        quarterbacks = quarterbacks[quarterbacks["_pass"] >= MIN_QB_PASS_ATTEMPTS_PG]
        primary = quarterbacks.sort_values(
            ["_pass", "proj"], ascending=False).drop_duplicates("TeamAbbrev")
        keep_ids = set(primary["ID"])
        frame = frame[(frame["Position"].ne("QB")) | (frame["ID"].isin(keep_ids))].copy()
    # The statistical engine requires both teams in every simulated matchup.
    complete = set()
    for game, group in frame.groupby("game"):
        if set(group["TeamAbbrev"]) == set(game.split("@")):
            complete.add(game)
    frame = frame[frame["game"].isin(complete)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No complete two-team matchups with matched real-player history")
    return frame, rates
