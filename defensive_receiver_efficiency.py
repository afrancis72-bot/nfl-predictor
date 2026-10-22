"""Research-only, receiver-specific matchup efficiency transforms.

This module never changes target volume, touchdowns, or DFS scores on its own.
Apply only to verified observed coverage exposure; default is neutral.
"""
import numpy as np
import pandas as pd

def adjust_receiving_efficiency(targets, baseline_catch_probability, baseline_yards_per_reception,
                                exposure_factor, max_shift=.12):
    """Return adjusted catch probability and yards/reception without adding targets.

    A bounded, positive matchup factor splits its effect between catch probability
    and conditional yards per reception. This is a testable *hypothesis*, not a
    calibrated causal injury estimate. Does not mutate or resample targets.
    """
    t=np.asarray(targets)
    if np.any(~np.isfinite(t)) or np.any(t<0):
        raise ValueError("Invalid target counts")
    if not np.isfinite(baseline_catch_probability) or not 0<=baseline_catch_probability<=1:
        raise ValueError("Invalid baseline catch probability")
    if not np.isfinite(baseline_yards_per_reception) or baseline_yards_per_reception<0:
        raise ValueError("Invalid baseline yards per reception")
    if not np.isfinite(exposure_factor) or not 1-max_shift<=exposure_factor<=1+max_shift:
        raise ValueError("Unverified or out-of-range matchup factor")
    shift=float(exposure_factor)-1.
    catch=float(np.clip(baseline_catch_probability*(1.+.5*shift),0.,1.))
    ypr=float(baseline_yards_per_reception*(1.+.5*shift))
    return dict(targets=t.copy(),catch_probability=catch,yards_per_reception=ypr,
                expected_receptions=t*catch,expected_receiving_yards=t*catch*ypr,
                research_only=True)

def apply_verified_receiver_factor(targets, catch_probability, yards_per_reception,
                                   receiver_id, exposure):
    """Fail neutral if receiver evidence is missing, unsupported or duplicated."""
    factor=1.
    if isinstance(exposure,pd.DataFrame) and not exposure.empty:
        if {"player_id","supported","efficiency_factor"}.issubset(exposure.columns):
            rows=exposure[exposure.player_id.astype(str)==str(receiver_id)]
            if len(rows)==1 and bool(rows.iloc[0].supported):
                factor=float(rows.iloc[0].efficiency_factor)
    return adjust_receiving_efficiency(targets,catch_probability,yards_per_reception,factor)
