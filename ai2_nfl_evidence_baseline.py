"""Evidence-weighted partial DraftKings mean forecast — research only.

This is deliberately a SMALL, pre-specified experiment. Historical player
means and current simulator means come from the same pregame frozen
backtest, not reconstructed using target-game outcomes. Measured nflverse
prior-game counts and PFR snap-match availability affect blend weight;
unverified live availability gives NO special boosts. This is NOT a
play-by-play or target-stat correction, and MUST NOT change simulated P90.

A candidate qualifies ONLY if it meaningfully beats BOTH existing means on
training data, has no major position regression and survives a separate
retrospective comparison. The 2025 / 2026 windows have already informed
other projects and are NOT blind prospective evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS=("season","week","game","player","team","position")
MODES=("equal","evidence","guardrail")
REQUIRED_PRED=set(KEYS)|{
    "history_mean_partial_dk","simulated_mean_dk","actual_partial_dk",
    "simulated_p10_dk","simulated_p90_dk",
}
REQUIRED_PROOF=set(KEYS)|{
    "prior_player_games","last_player_season","last_player_week",
    "has_four_plus_snap_games","snap_games","routes_verified",
    "game_day_active_verified",
}


def _validate_and_join(predictions, evidence):
    if not isinstance(predictions,pd.DataFrame) or not isinstance(evidence,pd.DataFrame):
        raise ValueError("Blending requires paired pregame forecast and evidence frames")
    if not REQUIRED_PRED.issubset(predictions) or not REQUIRED_PROOF.issubset(evidence):
        raise ValueError("Blending requires all pregame forecast and evidence columns")
    keys=list(KEYS)
    if predictions.empty or evidence.empty or (predictions.duplicated(keys).any()
        or evidence.duplicated(keys).any()):
        raise ValueError("Empty or duplicate player-game forecast/evidence inputs")
    # Anti-join blocks artificially favorable results from dropping rows.
    joined=predictions.merge(evidence,on=keys,how="outer",
                             validate="one_to_one",indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Pregame evidence coverage differs from graded player cohort")
    joined=joined.drop(columns=["_merge"])
    for field in ("history_mean_partial_dk","simulated_mean_dk",
                  "simulated_p10_dk","simulated_p90_dk",
                  "prior_player_games","last_player_season","last_player_week",
                  "snap_games"):
        vals=pd.to_numeric(joined[field],errors="coerce")
        if not np.isfinite(vals).all():
            raise ValueError(f"Invalid nonfinite pregame predictor: {field}")
    if ((joined["last_player_season"]>joined["season"]) |
        ((joined["last_player_season"]==joined["season"]) &
         (joined["last_player_week"]>=joined["week"]))).any():
        raise ValueError("Future player-stat season/week in pregame evidence")
    if (joined["prior_player_games"]<1).any() or (joined["prior_player_games"]>8).any():
        raise ValueError("Invalid number of pregame player games")
    if not joined["routes_verified"].eq(False).all() or not joined["game_day_active_verified"].eq(False).all():
        raise ValueError("Study cannot treat routes/official live availability as verified")
    if not joined["has_four_plus_snap_games"].isin((True,False)).all():
        raise ValueError("Snap coverage evidence must be explicit")
    if (joined["has_four_plus_snap_games"] & joined["snap_games"].lt(4)).any():
        raise ValueError("Insufficient snap games misclassified as sufficient")
    if (joined["simulated_p10_dk"]>joined["simulated_p90_dk"]).any():
        raise ValueError("Invalid existing simulator quantiles")
    return joined


def evidence_blend(predictions,evidence,mode):
    """Produce one pregame-calculable partial-DK mean and transparent trust data.

    No actuals are used to determine the blend; actual score remains only for
    after-game grading in the returned joined frame.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown precommitted evidence mode: {mode}")
    f=_validate_and_join(predictions,evidence)
    history=f["history_mean_partial_dk"].to_numpy(float)
    sim=f["simulated_mean_dk"].to_numpy(float)
    games=f["prior_player_games"].to_numpy(float)
    snap=f["has_four_plus_snap_games"].to_numpy(bool)
    recent=(f["last_player_season"].to_numpy(int)==
            f["season"].to_numpy(int)) & (
            f["week"].to_numpy(int)-f["last_player_week"].to_numpy(int)<=2)
    divergence=np.abs(sim-history)/np.maximum(5.,np.abs(history))
    if mode=="equal":
        history_weight=np.full(len(f),.5)
    elif mode=="evidence":
        # Reliable prior mean with current-season, frequently observed roles
        # warrants more trust; scarce history/snap coverage shrinks back to the
        # simulator. Coefficients fixed BEFORE evaluating historical outcomes.
        history_weight=np.clip(.20+.28*(games/8.)+
                               .12*snap.astype(float)+
                               .08*recent.astype(float),.20,.70)
    else:
        # A *different* idea: if the two models diverge widely, cap the risk of
        # using an unexplained large simulator override in an unverified role.
        history_weight=np.clip(.35+.16*(games/8.)+
                               .08*snap.astype(float)+
                               .16*(divergence>.4),.35,.75)
    # Never target-game-fit the weight; the labels are not consulted here.
    f=f.copy()
    f["historical_weight"]=history_weight
    f["simulator_weight"]=1-history_weight
    f["blended_mean_partial_dk"]=history_weight*history+(1-history_weight)*sim
    f["blend_mode"]=mode
    f["history_is_pregame"]=True
    f["current_week_active_verified"]=False
    f["partial_dk_only"]=True
    f["p90_not_recalibrated"]=True
    return f
