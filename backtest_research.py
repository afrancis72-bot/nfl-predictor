"""Out-of-sample research calibration: compare predicted distributions to actuals.

Inputs are long-form simulated outcomes and held-out observed outcomes, joined
by matchup_id, player, and stat. Never train or tune using the holdout set.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def evaluate_predictions(simulations: pd.DataFrame, actuals: pd.DataFrame):
    """Sim columns: matchup_id, player, stat, simulation, value.
    Actual columns: matchup_id, player, stat, actual.
    Return calibration diagnostics per player/stat with interval coverage.
    """
    sim_required={"matchup_id","player","stat","simulation","value"}
    actual_required={"matchup_id","player","stat","actual"}
    if not sim_required.issubset(simulations):
        raise ValueError("Missing simulation fields: "+str(sorted(sim_required-set(simulations))))
    if not actual_required.issubset(actuals):
        raise ValueError("Missing actual fields: "+str(sorted(actual_required-set(actuals))))
    keys=["matchup_id","player","stat"]
    if actuals.duplicated(keys).any():
        raise ValueError("Duplicate observed player/stat outcomes")
    if simulations.duplicated(keys+["simulation"]).any():
        raise ValueError("Duplicate simulated player/stat outcomes")
    if simulations.empty or actuals.empty:
        raise ValueError("Simulation and actual inputs cannot be empty")
    grouped=simulations.groupby(keys,sort=False)["value"]
    summary=grouped.agg(mean="mean",median="median",n_simulations="count").reset_index()
    for q,label in ((.05,"p05"),(.10,"p10"),(.90,"p90"),(.95,"p95")):
        summary[label]=grouped.quantile(q).to_numpy()
    joined=summary.merge(actuals[keys+["actual"]],on=keys,how="inner",validate="one_to_one")
    if joined.empty: raise ValueError("No overlapping predictions and actuals")
    for col in ["mean","median","p05","p10","p90","p95","actual"]:
        joined[col]=pd.to_numeric(joined[col],errors="coerce")
    if joined[["mean","actual"]].isna().any().any():
        raise ValueError("Non-numeric outcomes")
    joined["error"]=joined["mean"]-joined["actual"]
    joined["abs_error"]=joined["error"].abs()
    joined["covered_80"]=(joined.actual>=joined.p10)&(joined.actual<=joined.p90)
    joined["covered_90"]=(joined.actual>=joined.p05)&(joined.actual<=joined.p95)
    joined["actual_below_median"]=(joined.actual<=joined["median"])
    metrics=joined.groupby("stat").agg(
        observations=("actual","size"),mae=("abs_error","mean"),
        bias=("error","mean"),coverage_80=("covered_80","mean"),
        coverage_90=("covered_90","mean"),
        fraction_below_median=("actual_below_median","mean")).reset_index()
    return joined,metrics
