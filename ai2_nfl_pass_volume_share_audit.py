"""Research-only diagnostic: split WR target error into passes vs target shares.

The "oracle volume" counterfactual uses target-game ACTUAL passing attempts
*after* the game solely to explain errors. It is never a pregame predictor.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ai2_nfl_matchups import canonical_team
from ai2_nfl_pbp_drives import engine_team

TEAM_KEY=["season","week","game","team"]
WR_KEY=TEAM_KEY+["player"]

def team_pass_attempt_row(*, season, week, game, team, players, stat_draws,
                          actual_week, n_sims):
    """Capture simulated team pass attempts, then observe actual attempts."""
    team=canonical_team(team)
    qb=players[(players["TeamAbbrev"].map(canonical_team).eq(team)) &
               (players["Position"].eq("QB"))]
    if len(qb)!=1:
        raise ValueError(f"Need exactly one simulated QB for {team}, got {len(qb)}")
    qb_name=str(qb.iloc[0]["Name"])
    sample=stat_draws.get(qb_name,{}).get("pass_attempts")
    if sample is None:
        sample=stat_draws.get("__OTHER_QB_PASSING__|"+engine_team(team),{}).get("pass_attempts")
    draw=np.asarray(sample,dtype=float)
    if draw.shape!=(int(n_sims),) or not np.isfinite(draw).all() or (draw<0).any():
        raise ValueError("Invalid team pass-attempt simulation")
    actual_rows=actual_week[actual_week["recent_team"].map(canonical_team).eq(team)]
    # Include QB and non-QB throwers; WR gadget passes are real team attempts.
    actual_values=pd.to_numeric(actual_rows.get("attempts"),errors="coerce")
    if actual_values.empty or actual_values.isna().all():
        raise ValueError("Missing actual team pass attempt observations")
    actual_attempts=float(actual_values.fillna(0).sum())
    if not np.isfinite(actual_attempts) or actual_attempts<=0:
        raise ValueError("Invalid actual team pass attempts")
    prior=stat_draws.get("__TEAM_PASS_BASELINE__|"+engine_team(team),{})
    if not prior or not np.isfinite(float(prior.get("team_pass_attempts_baseline",np.nan))):
        raise ValueError("Missing original pregame team pass volume baseline")
    return dict(season=int(season),week=int(week),game=str(game),team=team,
        actual_pass_attempts=actual_attempts,
        model_pass_attempts=float(draw.mean()),
        historical_team_pass_baseline=float(prior["team_pass_attempts_baseline"]),
        passing_qb=qb_name)

def decompose_receiver_errors(receiver_rows,team_pass_rows):
    """Exact accounting identity for each matched receiver-game.
    predicted_targets - actual_targets =
      (model_passes - actual_passes) * predicted_target_share
      + actual_passes * (predicted_target_share - actual_target_share).
    """
    wr=pd.DataFrame(receiver_rows)
    tm=pd.DataFrame(team_pass_rows)
    if wr.empty or tm.empty:
        raise ValueError("Both receiver and pass observations required")
    if wr.duplicated(WR_KEY).any() or tm.duplicated(TEAM_KEY).any():
        raise ValueError("Duplicate player or team-game in decomposition")
    paired=wr.merge(tm,on=TEAM_KEY,how="left",validate="many_to_one",indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("Missing team pass forecast for WR target")
    paired=paired.drop(columns="_merge")
    numeric=("actual_targets","predicted_targets","prior_targets_pg",
             "actual_pass_attempts","model_pass_attempts")
    for col in numeric:
        paired[col]=pd.to_numeric(paired[col],errors="coerce")
        if not np.isfinite(paired[col].to_numpy(float)).all():
            raise ValueError("Nonfinite input for WR decomposition")
    if ((paired[["actual_targets","predicted_targets"]]<0).any().any() or
        (paired[["actual_pass_attempts","model_pass_attempts"]]<=0).any().any()):
        raise ValueError("Invalid target/attempt counts")
    if (paired["actual_targets"]>paired["actual_pass_attempts"]+0.01).any():
        raise ValueError("Receiver targets exceed actual team attempts")
    paired["predicted_target_share"]=paired["predicted_targets"]/paired["model_pass_attempts"]
    paired["observed_target_share"]=paired["actual_targets"]/paired["actual_pass_attempts"]
    if (paired["predicted_target_share"]>1.00001).any():
        raise ValueError("Receiver simulated target share exceeds 100%")
    paired["volume_error_component"]=(
        (paired["model_pass_attempts"]-paired["actual_pass_attempts"])*
        paired["predicted_target_share"])
    paired["allocation_error_component"]=paired["actual_pass_attempts"]*(
        paired["predicted_target_share"]-paired["observed_target_share"])
    paired["total_target_error"]=paired["predicted_targets"]-paired["actual_targets"]
    paired["oracle_actual_volume_prediction"]=(
        paired["actual_pass_attempts"]*paired["predicted_target_share"])
    if not np.allclose(paired["total_target_error"],
            paired["volume_error_component"]+paired["allocation_error_component"],
            rtol=0,atol=1e-8):
        raise ValueError("WR target error decomposition did not reconcile")
    return paired

def summary(wr,team):
    if wr.empty or team.empty:
        raise ValueError("No completed WR/team decomposition")
    out={}
    actual=team["actual_pass_attempts"].to_numpy(float)
    model=team["model_pass_attempts"].to_numpy(float)
    history=team["historical_team_pass_baseline"].to_numpy(float)
    out["team_passes"]={
        "team_games":int(len(team)),
        "sim_mean":round(float(model.mean()),3),
        "actual_mean":round(float(actual.mean()),3),
        "sim_bias_attempts":round(float(np.mean(model-actual)),3),
        "sim_mae_attempts":round(float(np.mean(np.abs(model-actual))),3),
        "pregame_baseline_mae_attempts":round(float(np.mean(np.abs(history-actual))),3),
        "underprojected_team_games":int(np.sum(model<actual)),
    }
    for role in ("WR1","WR2","WR3","WR4+"):
        a=wr[wr["role"].eq(role)]
        if a.empty:
            out[role]={"n":0};continue
        err=a["total_target_error"].to_numpy(float)
        vol=a["volume_error_component"].to_numpy(float)
        alloc=a["allocation_error_component"].to_numpy(float)
        oracle=a["oracle_actual_volume_prediction"].to_numpy(float)
        y=a["actual_targets"].to_numpy(float)
        out[role]={
            "n":int(len(a)),
            "target_bias":round(float(np.mean(err)),3),
            "target_mae":round(float(np.mean(np.abs(err))),3),
            "volume_component_bias":round(float(np.mean(vol)),3),
            "allocation_component_bias":round(float(np.mean(alloc)),3),
            "volume_component_mean_abs":round(float(np.mean(np.abs(vol))),3),
            "allocation_component_mean_abs":round(float(np.mean(np.abs(alloc))),3),
            "oracle_actual_volume_target_mae":round(float(np.mean(np.abs(oracle-y))),3),
            "oracle_actual_volume_target_bias":round(float(np.mean(oracle-y)),3),
            "mean_predicted_target_share_pct":round(100*float(np.mean(a["predicted_target_share"])),2),
            "mean_actual_target_share_pct":round(100*float(np.mean(a["observed_target_share"])),2),
        }
    return out
