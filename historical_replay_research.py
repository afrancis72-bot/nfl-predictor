"""Reproducible historical game replay using strictly pregame inputs.

Provide a matchup manifest of independently verified historical starting rosters.
Actuals are joined ONLY after the simulator has generated its distributions.
"""
from __future__ import annotations
import pandas as pd
from snap_level_research import simulate_snaps
from defensive_priors_research import build_defense_profiles
from backtest_research import evaluate_predictions

STATS=("pass_yd","rush_yd","rec_yd","receptions","targets","rush_att","pass_att")

def replay_game(matchup_id, season, week, teams, pregame_players,
                prior_play_by_play, heldout_actuals, n_games=1000, seed=316):
    if len(teams)!=2 or len(set(teams))!=2:
        raise ValueError("Exactly two distinct teams required")
    if not isinstance(matchup_id,str) or not matchup_id.strip():
        raise ValueError("matchup_id required")
    if not {"Name","TeamAbbrev","Position","usage_verified"}.issubset(pregame_players):
        raise ValueError("Verified pregame player pool required")
    if not pregame_players.usage_verified.fillna(False).all():
        raise ValueError("Unverified player in pregame pool")
    defense,audit=build_defense_profiles(prior_play_by_play,season,week)
    missing=set(teams)-set(defense)
    if missing:
        raise ValueError("Missing historical defensive profiles: "+", ".join(sorted(missing)))
    _,long=simulate_snaps(pregame_players,teams,n_games=n_games,seed=seed,
                          defense_profiles=defense)
    melted=long.melt(id_vars=["game","name"],value_vars=list(STATS),
                     var_name="stat",value_name="value")
    predictions=melted.rename(columns={"game":"simulation","name":"player"})
    predictions.insert(0,"matchup_id",matchup_id)
    observed=heldout_actuals.copy()
    if "matchup_id" not in observed:
        observed.insert(0,"matchup_id",matchup_id)
    if not observed["matchup_id"].eq(matchup_id).all():
        raise ValueError("Held-out actuals include a different matchup")
    details,metrics=evaluate_predictions(predictions,observed)
    return {"predictions":predictions,"details":details,
            "metrics":metrics,"defense_audit":audit}

def replay_batch(games):
    """Each games entry is a dict of replay_game arguments."""
    reports=[replay_game(**game) for game in games]
    if not reports: raise ValueError("At least one historical game required")
    details=pd.concat([r["details"] for r in reports],ignore_index=True)
    metrics=details.groupby("stat").agg(
        observations=("actual","size"),mae=("abs_error","mean"),
        bias=("error","mean"),coverage_80=("covered_80","mean"),
        coverage_90=("covered_90","mean"),
        fraction_below_median=("actual_below_median","mean")).reset_index()
    return details,metrics
