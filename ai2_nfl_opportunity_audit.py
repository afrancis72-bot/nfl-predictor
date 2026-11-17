"""Offline NFL player opportunity / efficiency / TD diagnostic.

This is NOT an optimizer or an individual performance-guarantee gate.
It extracts pre-kickoff simulator *football stats* (not DK points) and
compares them to same-player subsequent nflverse box scores.  Missing
observations are reported, not invented as zeros.  All model outcomes are
measured against a separate prior-week historical mean on the same cohort.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

# (simulator statistic, actual nflverse field, prior-week historical rate,
#  domain). These are diagnostic categories, not an additive decomposition.
STATS = {
    "QB": [
        ("pass_attempts", "attempts", "pass_attempts_pg", "opportunity"),
        ("completions", "completions", "completions_pg", "opportunity"),
        ("passing_yards", "passing_yards", "pass_yards_pg", "efficiency_output"),
        ("passing_tds", "passing_tds", "passing_tds_pg", "touchdown"),
        ("interceptions", "interceptions", "interceptions_pg", "turnover"),
        ("carries", "carries", "carries_pg", "opportunity"),
        ("rushing_yards", "rushing_yards", "rushing_yards_pg", "efficiency_output"),
        ("rushing_tds", "rushing_tds", "rushing_tds_pg", "touchdown"),
    ],
    "RB": [
        ("carries", "carries", "carries_pg", "opportunity"),
        ("targets", "targets", "targets_pg", "opportunity"),
        ("receptions", "receptions", "receptions_pg", "opportunity"),
        ("rushing_yards", "rushing_yards", "rushing_yards_pg", "efficiency_output"),
        ("receiving_yards", "receiving_yards", "receiving_yards_pg", "efficiency_output"),
        ("rushing_tds", "rushing_tds", "rushing_tds_pg", "touchdown"),
        ("receiving_tds", "receiving_tds", "receiving_tds_pg", "touchdown"),
    ],
    "WR": [
        ("carries", "carries", "carries_pg", "opportunity"),
        ("targets", "targets", "targets_pg", "opportunity"),
        ("receptions", "receptions", "receptions_pg", "opportunity"),
        ("rushing_yards", "rushing_yards", "rushing_yards_pg", "efficiency_output"),
        ("receiving_yards", "receiving_yards", "receiving_yards_pg", "efficiency_output"),
        ("rushing_tds", "rushing_tds", "rushing_tds_pg", "touchdown"),
        ("receiving_tds", "receiving_tds", "receiving_tds_pg", "touchdown"),
    ],
    "TE": [
        ("carries", "carries", "carries_pg", "opportunity"),
        ("targets", "targets", "targets_pg", "opportunity"),
        ("receptions", "receptions", "receptions_pg", "opportunity"),
        ("rushing_yards", "rushing_yards", "rushing_yards_pg", "efficiency_output"),
        ("receiving_yards", "receiving_yards", "receiving_yards_pg", "efficiency_output"),
        ("rushing_tds", "rushing_tds", "rushing_tds_pg", "touchdown"),
        ("receiving_tds", "receiving_tds", "receiving_tds_pg", "touchdown"),
    ],
}
DIRECT_DK_WEIGHTS = {
    "passing_yards":0.04,
    "passing_tds":4.0,
    "interceptions":-1.0,
    "rushing_yards":0.1,
    "rushing_tds":6.0,
    "receiving_yards":0.1,
    "receptions":1.0,
    "receiving_tds":6.0,
}
REQUIRED = {
    "season","week","game","player","team","position","stat",
    "category","actual","prior_baseline","predicted_mean",
    "predicted_p10","predicted_p90","n_sims","model",
}


def _finite(value):
    try:
        v=float(value)
    except (TypeError,ValueError):
        return None
    return v if math.isfinite(v) else None


def extract_player_stats(*, player, team, position, game, season, week,
                         observed, prior, simulated_stats, n_sims,
                         model="neutral"):
    """Return long-form rows; only matched position-applicable observed stats.

    Refuse malformed simulated draws. Do not impute missing actuals.
    Call only after the upstream pregame identity match has been established.
    """
    if position not in STATS:
        return []
    if not isinstance(simulated_stats,dict):
        raise ValueError(f"Missing simulator stat draws for {player} ({position})")
    if not (int(prior.get("latest_observed_season",0))<int(season) or
            (int(prior.get("latest_observed_season",0))==int(season) and
             int(prior.get("latest_observed_week",99))<int(week))):
        raise ValueError("Prior week player baseline contains the target game")
    result=[]
    for stat,field,baseline,category in STATS[position]:
        y=_finite(observed.get(field))
        old=_finite(prior.get(baseline))
        if y is None or old is None:
            continue
        if stat not in simulated_stats:
            raise ValueError(f"Missing simulator {stat} for {player}")
        samples=np.asarray(simulated_stats[stat],dtype=float)
        if (samples.ndim!=1 or len(samples)!=int(n_sims)
                or not np.isfinite(samples).all()):
            raise ValueError(f"Invalid {stat} simulation samples for {player}")
        if stat in {"pass_attempts","completions","passing_tds",
                    "interceptions","carries","targets","receptions",
                    "rushing_tds","receiving_tds"} and (samples < 0).any():
            raise ValueError(f"Negative simulated opportunity/touchdown: {stat}")
        result.append({
            "season":int(season),"week":int(week),"game":str(game),
            "player":str(player),"team":str(team),"position":str(position),
            "stat":stat,"category":category,"model":str(model),
            "actual":float(y),"prior_baseline":float(old),
            "predicted_mean":float(np.mean(samples)),
            "predicted_p10":float(np.percentile(samples,10)),
            "predicted_p90":float(np.percentile(samples,90)),
            "n_sims":int(n_sims),
            "direct_dk_weight":DIRECT_DK_WEIGHTS.get(stat,0.0),
            "absolute_error":float(abs(np.mean(samples)-y)),
            "prior_absolute_error":float(abs(old-y)),
        })
    return result


def stat_summary(rows):
    if isinstance(rows,list):
        rows=pd.DataFrame(rows)
    if rows.empty:
        return {"n":0}
    missing=REQUIRED-set(rows)
    if missing:
        raise ValueError(f"Missing stat audit columns: {sorted(missing)}")
    y=rows["actual"].to_numpy(float)
    pred=rows["predicted_mean"].to_numpy(float)
    prior=rows["prior_baseline"].to_numpy(float)
    p10=rows["predicted_p10"].to_numpy(float)
    p90=rows["predicted_p90"].to_numpy(float)
    if not all(np.isfinite(x).all() for x in (y,pred,prior,p10,p90)):
        raise ValueError("Missing/invalid football stat observation or forecast")
    if (p10>p90).any():
        raise ValueError("P10 above P90")
    sim_err=pred-y
    base_err=prior-y
    sim_mae=float(np.mean(np.abs(sim_err)))
    base_mae=float(np.mean(np.abs(base_err)))
    lift=(100*(base_mae-sim_mae)/base_mae if base_mae>0 else None)
    # For integer/zero-inflated stats, coverage has discreteness effects:
    # approximate expected 90% for P90 and 80% for P10-P90, not exact per row.
    return {
        "n":int(len(rows)),
        "sim_mae":round(sim_mae,4),
        "history_mae":round(base_mae,4),
        "sim_mae_improvement_pct":round(lift,3) if lift is not None else None,
        "sim_bias_predicted_minus_actual":round(float(np.mean(sim_err)),4),
        "within_one_unit_pct":round(100*float(np.mean(np.abs(sim_err)<=1)),2),
        "below_p90_pct":round(100*float(np.mean(y<=p90)),2),
        "inside_p10_p90_pct":round(
            100*float(np.mean((y>=p10)&(y<=p90))),2),
    }


def _cluster_lift_ci(rows, *, resamples=1000, seed=42):
    if rows.empty:
        return None
    grouped=rows.copy()
    grouped["_game"]=(grouped["season"].astype(str)+"-"+
                      grouped["week"].astype(str)+"-"+
                      grouped["game"].astype(str))
    grouped["_old"]=(grouped["actual"]-grouped["prior_baseline"]).abs()
    grouped["_new"]=(grouped["actual"]-grouped["predicted_mean"]).abs()
    sums=grouped.groupby("_game")[["_old","_new"]].sum().to_numpy(float)
    if len(sums)<5:
        return None
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(sums),size=(resamples,len(sums)))
    picked=sums[idx].sum(axis=1)
    lift=100*(picked[:,0]-picked[:,1])/np.maximum(picked[:,0],1e-12)
    return [round(float(x),3) for x in np.percentile(lift,[2.5,97.5])]


def opportunity_report(rows, *, skipped_games=0):
    if isinstance(rows,list):
        rows=pd.DataFrame(rows)
    if rows.empty:
        raise ValueError("No matched pregame opportunity forecasts")
    all_metrics={}
    for (position,stat),grp in rows.groupby(["position","stat"]):
        m=stat_summary(grp)
        m["game_cluster_lift_95_ci"]=_cluster_lift_ci(grp)
        m["category"]=str(grp.iloc[0]["category"])
        m["direct_DK_points_per_unit"]=float(
            DIRECT_DK_WEIGHTS.get(stat,0))
        # Metric is a *diagnostic priority*, NOT additive attribution of
        # observed fantasy point misses (bonus thresholds/correlation matter).
        m["mean_abs_DK_weighted_stat_error"]=round(
            m["sim_mae"]*abs(m["direct_DK_points_per_unit"]),4)
        all_metrics[f"{position}.{stat}"]=m
    by_category={
        category:stat_summary(grp)
        for category,grp in rows.groupby("category")
    }
    priorities=sorted(
        [{"role_stat":key,**value} for key,value in all_metrics.items()
         if value["n"]>=20 and value["direct_DK_points_per_unit"]!=0],
        key=lambda v:v["mean_abs_DK_weighted_stat_error"],reverse=True)
    worst=(rows.assign(dk_weighted_abs_error=
                        (rows["actual"]-rows["predicted_mean"]).abs()*
                        rows["stat"].map(DIRECT_DK_WEIGHTS).fillna(0).abs())
           .sort_values("dk_weighted_abs_error",ascending=False)
           [["season","week","game","player","position","stat",
             "actual","prior_baseline","predicted_mean",
             "predicted_p90","dk_weighted_abs_error"]]
           .head(30).round(3).to_dict("records"))
    return {
        "summary_by_position_and_stat":all_metrics,
        "summary_by_category_non_comparable_units":by_category,
        "high_error_priority_weighted_DK_approx":priorities[:30],
        "worst_30_single_stat_misses":worst,
        "stat_observations":len(rows),
        "distinct_player_games":int(rows[[
            "season","week","game","player","team","position"
        ]].drop_duplicates().shape[0]),
        "simulated_games":int(rows[[
            "season","week","game"
        ]].drop_duplicates().shape[0]),
        "skipped_games":int(skipped_games),
        "research_only":True,
        "production_approved":False,
        "limitations":[
            "Categories contain different units: DO NOT compare MAE directly across counts and yards",
            "Weighted stat error ranks diagnostic work only; it is not additive causal attribution of DK losses",
            "An absence of available actuals is NOT a zero stat",
            "2026 data previously inspected; no genuinely unseen holdout",
            "Matched active players from reconstructed prior-week pool only",
            "No historical salary/status or opponent injury context",
            "Discrete/zero-inflated TD/turnover stats have non-Gaussian P10/P90 calibration behavior",
            "No full historical DraftKings contest earnings reconstructed",
        ],
    }
