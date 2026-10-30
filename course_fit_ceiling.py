from __future__ import annotations
import numpy as np
import pandas as pd

# Transparent, coverage-aware course-fit ceiling layer.  Weights are applied only
# to columns present in the weekly model input; missing stats are not treated as zero.
COURSE_PROFILES = {
    "Balanced": {
        "sg_approach": .18, "recent5_sg_approach": .14, "recent_ball_striking": .12,
        "sg_ott": .10, "driving_accuracy": .08, "driving_distance": .07,
        "sg_putting": .10, "sg_arg": .07, "course_history_z_v8": .07,
        "model_data_confidence_v9": .07,
    },
    "Approach / Scoring": {
        "sg_approach": .25, "recent5_sg_approach": .25, "recent_ball_striking": .20,
        "driving_fit": .12, "course_history_z_v8": .08, "sg_putting": .10,
    },
    "Accuracy / Positioning": {
        "sg_approach": .20, "recent5_sg_approach": .15, "driving_accuracy": .22,
        "sg_ott": .10, "recent_ball_striking": .10, "sg_putting": .08,
        "sg_arg": .08, "course_history_z_v8": .07,
    },
    "Distance / Birdie": {
        "sg_approach": .20, "recent5_sg_approach": .15, "driving_distance": .18,
        "sg_ott": .15, "recent_ball_striking": .12, "sg_putting": .10,
        "course_history_z_v8": .10,
    },
}

ALIASES = {
    "course_history_z_v8": ["course_history_z_v8", "course_history_z", "course_history_score"],
    "model_data_confidence_v9": ["model_data_confidence_v10", "model_data_confidence_v9", "model_data_confidence_v8", "model_data_confidence"],
    "recent5_sg_approach": ["recent5_sg_approach", "form_sg_approach"],
    "recent_ball_striking": ["recent_ball_striking"],
    "driving_fit": ["driving_fit"],
}

def _z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (x - x.mean()) / sd

def _resolve(df: pd.DataFrame, key: str) -> str | None:
    for c in ALIASES.get(key, [key]):
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return None

def add_course_fit_ceiling(df: pd.DataFrame, profile: str = "Balanced") -> pd.DataFrame:
    out = df.copy()
    weights = COURSE_PROFILES.get(profile, COURSE_PROFILES["Balanced"])
    numerator = pd.Series(0.0, index=out.index)
    denom = pd.Series(0.0, index=out.index)
    components = []
    for key, weight in weights.items():
        col = _resolve(out, key)
        if col is None:
            continue
        z = _z(out[col])
        valid = z.notna()
        numerator.loc[valid] += weight * z.loc[valid]
        denom.loc[valid] += weight
        components.append(col)
    out["course_fit_ceiling"] = (numerator / denom.replace(0, np.nan)).fillna(0.0)
    out["course_fit_coverage"] = denom / max(sum(weights.values()), 1e-9)
    out["course_fit_profile"] = profile
    out.attrs["course_fit_components"] = components
    return out
