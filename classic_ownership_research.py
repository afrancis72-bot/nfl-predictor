"""Slate-aware heuristic ownership estimates; uncalibrated until actual ownership audits."""
import numpy as np
import pandas as pd

def estimate_ownership(salaries, projections, dst_points, excluded_names=()):
    """Estimate classic DK ownership with positional roster-mass constraints.

    Every position's expected lineup selections are normalized:
    QB=1, RB=2.4, WR=3.4, TE=1.2, DST=1.0 (total 9).
    FLEX mass is distributed between RB/WR/TE.
    """
    pool=salaries.copy()
    pool["ID"]=pool["ID"].astype(str).str.replace(r"\.0$","",regex=True)
    pool["Position"]=pool["Position"].astype(str).str.upper().str.strip().replace({"D":"DST"})
    pool["Salary"]=pd.to_numeric(pool["Salary"],errors="coerce")
    pool=pool[pool.Position.isin(["QB","RB","WR","TE","DST"]) & ~pool.Name.isin(excluded_names)].copy()
    proj=projections[["ID","dk_points_mean"]].copy()
    proj["ID"]=proj["ID"].astype(str).str.replace(r"\.0$","",regex=True)
    pool=pool.merge(proj,on="ID",how="left",validate="one_to_one")
    pool["projected_points"]=np.where(pool.Position.eq("DST"),pool.TeamAbbrev.map(dst_points),pool.dk_points_mean)
    pool=pool[pool.projected_points.notna() & pool.Salary.gt(0)].copy()
    if pool.empty: raise ValueError("No projected eligible players for ownership")
    pool["value"]=pool.projected_points/(pool.Salary/1000)
    pool["ownership_pct"]=0.0
    pool["ownership_confidence"]="low — heuristic, not calibrated"
    # Softmax of relative position-specific scoring and value, with a
    # salary/popularity proxy. Calibrated totals, NOT calibrated player rates.
    masses={"QB":100.0,"RB":240.0,"WR":340.0,"TE":120.0,"DST":100.0}
    for position,mass in masses.items():
        part=pool[pool.Position.eq(position)]
        if part.empty: raise ValueError("Missing eligible position "+position)
        pts=part.projected_points.to_numpy(dtype=float)
        val=part.value.to_numpy(dtype=float)
        sal=part.Salary.to_numpy(dtype=float)
        def z(x):
            spread=max(float(np.std(x)),1.0)
            return np.clip((x-np.mean(x))/spread,-3,3)
        strength=0.90*z(pts)+0.65*z(val)+0.18*z(sal)
        weights=np.exp(np.clip(strength-strength.max(),-30,0))
        # Water-fill so no individual can exceed 100% while preserving
        # each position's expected selection mass.
        remaining=np.ones(len(part),dtype=bool)
        allocation=np.zeros(len(part),dtype=float)
        left=mass
        while remaining.any() and left>1e-8:
            share=left*weights[remaining]/weights[remaining].sum()
            saturated=share>=100
            ids=np.flatnonzero(remaining)
            if not saturated.any():
                allocation[ids]=share
                break
            allocation[ids[saturated]]=100.0
            left-=100.0*int(saturated.sum())
            remaining[ids[saturated]]=False
        pool.loc[part.index,"ownership_pct"]=allocation
    return pool[["ID","Name","Position","TeamAbbrev","Salary","projected_points","ownership_pct","ownership_confidence"]].sort_values("ownership_pct",ascending=False).reset_index(drop=True)
