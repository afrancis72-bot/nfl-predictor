"""Multi-matchup snap-level research runner (no calibrated DST projection).

Runs the existing two-team engine independently for each DK classic matchup.
Strictly requires a confirmed starting QB and verified pregame usage; it never
substitutes DK salary for player opportunity. No look-ahead information.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from classic_slate_research import prepare_classic_slate
from historical_usage import build_usage_priors, apply_usage_priors
from snap_level_research import simulate_snaps, summarize_player_props

def run_classic_snap_slate(salaries: pd.DataFrame, weekly_stats: pd.DataFrame,
                           *, season: int, week: int, starting_qbs: dict,
                           excluded_names=(), n_games: int=200, seed: int=316,
                           min_history_games: int=2):
    """Return projections, per-game simulated outcomes, audit and excluded players.

    Raises rather than silently simulating with incomplete personnel.
    QB replacement can be verified by explicit caller-provided history overrides
    in future versions. The output is offensive-only; DST remains unmodeled.
    """
    if not 1 <= int(n_games) <= 100000:
        raise ValueError("n_games outside supported range")
    eligible, offense, audit, excluded = prepare_classic_slate(salaries, ())
    # DFS exclusions are not simulation exclusions: a real starter still affects teammates.
    excluded = eligible.loc[eligible["Name"].isin(set(excluded_names))].copy()
    priors=build_usage_priors(weekly_stats, as_of_week=week, season=season)
    if priors.empty: raise ValueError("No matching pregame usage priors; inspect season/week and data source")
    all_proj=[]; all_stats=[]
    for game_index, (game, raw) in enumerate(offense.groupby("matchup",sort=True)):
        teams=game.split("@")
        missing=[t for t in teams if t not in starting_qbs]
        if missing: raise ValueError(f"{game}: missing confirmed starting QB for {missing}")
        # Keep exactly the confirmed starter for each team; other QBs excluded.
        starters={t:starting_qbs[t] for t in teams}
        g=raw.loc[(raw.Position!="QB")|
                  raw.apply(lambda r:r["Name"]==starters.get(r.TeamAbbrev),axis=1)].copy()
        for t in teams:
            q=g[(g.TeamAbbrev==t)&(g.Position=="QB")]
            if len(q)!=1: raise ValueError(f"{game}: starter {starters[t]!r} missing or excluded")
        g=apply_usage_priors(g,priors,min_games=min_history_games)
        # Fail closed on missing historical opportunity, including new starters.
        missing_usage=g.loc[~g.usage_verified,["Name","TeamAbbrev","games"]]
        if len(missing_usage):
            names=", ".join(f"{r.Name} ({r.TeamAbbrev}; games={r.games if pd.notna(r.games) else 0})" for r in missing_usage.head(12).itertuples())
            raise ValueError(f"{game}: unverified usage ({names}); resolve before simulation")
        points,stats=simulate_snaps(g,teams,n_games=n_games,seed=seed+game_index)
        summary=summarize_player_props(stats)
        summary=summary.merge(g[["Name","TeamAbbrev","ID","Salary","Position"]],
                              left_on=["Name","Team"],right_on=["Name","TeamAbbrev"],
                              validate="one_to_one")
        summary["matchup"]=game
        all_proj.append(summary)
        stats["matchup"]=game
        stats["scenario"]=stats["game"]
        stats["dk_id"]=g.iloc[stats["player_index"].to_numpy()]["ID"].to_numpy()
        all_stats.append(stats)
    if not all_proj: raise ValueError("No matchups found")
    return (pd.concat(all_proj,ignore_index=True),
            pd.concat(all_stats,ignore_index=True),audit,excluded)
