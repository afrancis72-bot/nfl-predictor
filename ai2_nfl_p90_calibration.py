"""Historical, leakage-controlled calibration for reported NFL player P90.

This corrects the *reported predictive quantile*, NOT simulated statistics.
The underlying football-world draws and their DraftKings scoring stay intact.
Never fit this on current week, never fit on the held-out 2026 season, and
never claim a corrected quantile is an actual football outcome scenario.

Fit on 2025 completed player-games, validate on later 2025 games and test
independently on 2026 games.  Use positive-only safety corrections: a P90
already too high is not reduced by untested finite-sample noise.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

POSITIONS = ("QB","RB","WR","TE")


def pinball(actual,predicted,quantile=.9):
    a=np.asarray(actual,dtype=float)
    p=np.asarray(predicted,dtype=float)
    if a.shape!=p.shape or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Quantile evaluation requires paired finite numbers")
    d=a-p
    return float(np.mean(np.maximum(quantile*d,(quantile-1.)*d)))


def fit_p90_margins(training, *, min_group=30, prior_weight=60.):
    """Use only explicitly supplied pregame + subsequently completed training rows.

    A pooled finite-sample 90th residual is the conservative default; role-
    specific P90 residuals shrink toward the pooled correction for stability.
    """
    required={"position","actual_partial_dk","simulated_p90_dk","season","week"}
    if not isinstance(training,pd.DataFrame) or not required.issubset(training.columns):
        raise ValueError("Calibrating P90 needs dated, scored training observations")
    if training.empty or (training["season"]!=2025).any() or (
            training["week"]>14).any() or (training["week"]<5).any():
        raise ValueError("P90 calibration is permitted only on historic 2025 weeks 5-14")
    if len(training)<80:
        raise ValueError("Too few real prior-season player-games to fit P90")
    actual=training["actual_partial_dk"].to_numpy(float)
    raw=training["simulated_p90_dk"].to_numpy(float)
    if not np.isfinite(actual).all() or not np.isfinite(raw).all():
        raise ValueError("Missing real fantasy-score evidence")
    errors=actual-raw
    pooled=max(0.,float(np.quantile(errors,.9,method="higher")))
    by_position={}
    counts={}
    for pos in POSITIONS:
        group=training[training["position"]==pos]
        n=len(group)
        counts[pos]=n
        if n<int(min_group):
            continue
        group_errors=(group["actual_partial_dk"]-group["simulated_p90_dk"]).to_numpy(float)
        own=max(0.,float(np.quantile(group_errors,.9,method="higher")))
        share=n/(n+float(prior_weight))
        by_position[pos]=round(float(share*own+(1.-share)*pooled),6)
    return {
        "pooled_margin_dk":round(pooled,6),
        "role_margins_dk":by_position,
        "role_counts":counts,
        "training_rows":len(training),
        "training_season":2025,
        "training_final_week":int(training["week"].max()),
        "model":"2025 pregame residual 90th quantile, positive-only, role shrinkage",
        "notes":"Marginal coverage for players who appear; not lineup P90 or actual stat draws"
    }


def apply_p90_margin(p90,position,calibration,*,mode="pooled"):
    """Defensive calibrated display value; never mutate game-stat worlds."""
    value=float(p90)
    if not np.isfinite(value):
        raise ValueError("Invalid P90 forecast")
    if not isinstance(calibration,dict) or calibration.get("training_season")!=2025:
        raise ValueError("Missing sourced prior-season P90 calibration")
    if mode=="pooled":
        margin=float(calibration["pooled_margin_dk"])
    elif mode=="role":
        margin=float(calibration["role_margins_dk"].get(str(position).upper(),
                      calibration["pooled_margin_dk"]))
    else:
        raise ValueError("Unsupported calibration mode")
    if not np.isfinite(margin) or margin<0.:
        raise ValueError("Invalid positive P90 correction")
    return value+margin


def measure(frame,calibration=None,mode="pooled"):
    if frame.empty:
        raise ValueError("Missing actual completed player-games")
    actual=frame["actual_partial_dk"].to_numpy(float)
    p90=frame["simulated_p90_dk"].to_numpy(float)
    if calibration is None:
        adjusted=p90
    else:
        adjusted=np.array([
            apply_p90_margin(z,pos,calibration,mode=mode)
            for z,pos in zip(p90,frame["position"])])
    return {
        "n":len(frame),
        "actual_below_reported_p90_pct":round(100*float(np.mean(actual<=adjusted)),2),
        "p90_pinball":round(pinball(actual,adjusted),4),
        "reported_p90_mean_dk":round(float(adjusted.mean()),3),
        "average_added_p90_points":round(float((adjusted-p90).mean()),3),
        "undercount":int(np.sum(actual>adjusted)),
    }
