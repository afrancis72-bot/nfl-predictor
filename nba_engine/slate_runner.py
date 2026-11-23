"""Independent NBA per-game execution; unavailable fixtures never poison slate.

Selected matchups with missing official team reports are skipped *before* this
routine. Every eligible game gets its own complete rotation, baseline, 10K
team-constrained scenarios and source-tagged audit. A data/roster failure in
one eligible game doesn't invalidate other games, but the failure is exposed
by game ID; it is not silently treated as healthy or successfully simulated.
"""
from __future__ import annotations

from datetime import date
from hashlib import sha256
import pandas as pd

from nba_engine.predictor import build_projections
from nba_engine.injury_report import (
    injury_exclusions,match_excluded_ids,normalize_person)
from nba_engine.game_conservation import (
    normalize_team_minutes,calibrate_team_scoring_baselines,
    simulate_conserved_nba_games)


def run_verified_slate(logs:pd.DataFrame,team_opponents:dict,
                       official_status:pd.DataFrame,verified_games,
                       *,when:date,trials:int,seed:int):
    """Return complete successes; list independently failed eligible games."""
    verified=set(verified_games)
    if not verified:
        raise ValueError("No NBA matchups have complete official team reports")
    frames=[]
    minute_frames=[]
    baseline_frames=[]
    score_frames=[]
    samples={}
    excluded_ids=set()
    unmatched_out=set()
    completed=[]
    failed={}
    available_keys=set()
    for game in sorted(verified):
        try:
            parts=game.split("@")
            if len(parts)!=2:
                raise ValueError("Game ID is not AWAY@HOME")
            away,home=parts
            if (team_opponents.get(away)!=home or
                    team_opponents.get(home)!=away):
                raise ValueError("Schedule disagrees with league game ID")
            pair={away:home,home:away}
            if len(available_keys.intersection(pair)):
                raise ValueError("Same player team listed in multiple games")
            available_rows=official_status.loc[
                official_status["game"].eq(game)].copy()
            excluded,uncertain=injury_exclusions(available_rows,pair)
            omissions,unmatched=match_excluded_ids(
                logs,excluded,pair,when)
            forecast=build_projections(
                logs,pair,when,min_minutes=8,excluded=omissions)
            forecast,mins=normalize_team_minutes(forecast)
            forecast,baselines=calibrate_team_scoring_baselines(
                forecast,logs,when)
            forecast["availability_status"]=[
                uncertain.get((str(row.team),normalize_person(row.player)),
                              "Not listed (active unverified)")
                for row in forecast.itertuples(index=False)]
            # A stable game-specific seed is reproducible regardless of
            # which other matchups were selected with this game.
            game_hash=int.from_bytes(
                sha256(game.encode("ascii")).digest()[:4],"big")
            game_seed=(int(seed)+game_hash)%2147483647
            draws,score=simulate_conserved_nba_games(
                forecast,logs,when=when,trials=trials,seed=game_seed)
            duplicate=set(samples).intersection(draws)
            if duplicate:
                raise ValueError("Player simulation IDs collide across games")
            samples.update(draws)
            frames.append(forecast)
            minute_frames.append(mins)
            baseline_frames.append(baselines)
            score_frames.append(score)
            excluded_ids.update(omissions)
            unmatched_out.update(unmatched)
            available_keys.update(pair)
            completed.append(game)
        except (ValueError,KeyError,TypeError,IndexError,
                ArithmeticError) as exc:
            failed[game]=f"{type(exc).__name__}: {str(exc)[:240]}"
    if not completed:
        raise ValueError(
            "None of the selected NBA games could be simulated with "
            "the official report and complete roster data: "+
            "; ".join(f"{game}: {msg}" for game,msg in sorted(failed.items())))
    projections=pd.concat(frames,ignore_index=True)
    if projections["player_id"].astype(str).duplicated().any():
        raise ValueError("Duplicated NBA player IDs across game simulations")
    return {
        "projections":projections,
        "samples":samples,
        "game_audit":pd.concat(score_frames,ignore_index=True),
        "minute_audit":pd.concat(minute_frames,ignore_index=True),
        "baseline_audit":pd.concat(baseline_frames,ignore_index=True),
        "teams_map":{team:team_opponents[team]
                     for team in available_keys},
        "games_completed":completed,
        "games_failed":failed,
        "official_excluded_ids":sorted(excluded_ids),
        "official_out_not_in_history":sorted(unmatched_out),
    }
