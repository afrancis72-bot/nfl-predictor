"""Validate and normalize RickRunGood Last-50-rounds strokes-gained CSV.

This file contains *observed* per-round golf statistics, not projected fantasy
points. It is an input to Team AI²'s own model. RickRunGood's separate
course-fit FIT ranking is an external methodology comparator, not a model
input. Normalization is exact except for an explicit four-name alias table.

Small samples are shrinkage-adjusted toward neutral zero SG. Missing samples
stay NaN and never become fabricated SG estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rrg_course_fit import name_key

_REQUIRED = ("name", "salary", "rounds", "sg ott", "sg app", "sg arg", "sg putt", "sg tot")
_MEASUREMENTS = {
    "sg ott": "sg_ott",
    "sg app": "sg_approach",
    "sg arg": "sg_arg",
    "sg putt": "sg_putting",
    "sg tot": "sg_total",
    "sg t2g": "sg_tee_to_green",
    "driving distance": "driving_distance",
    "driving accuracy %": "driving_accuracy",
    "gir %": "gir_pct",
}
_SG_KEYS = {"sg_ott", "sg_approach", "sg_arg", "sg_putting", "sg_total", "sg_tee_to_green"}
_CORE = ("sg_ott", "sg_approach", "sg_arg", "sg_putting")


def normalize_rrg_strokes_gained(
    raw: pd.DataFrame,
    dk_field: pd.DataFrame,
    prior_rounds: float = 12.0,
) -> tuple[pd.DataFrame, dict]:
    """Return DK-name-keyed observed SG data and a conservative coverage audit."""
    if raw is None or raw.empty:
        raise ValueError("RickRunGood strokes-gained CSV is empty.")
    if dk_field is None or dk_field.empty or "player" not in dk_field:
        raise ValueError("A normalized DraftKings PGA field is required.")
    lookup = {str(col).strip().casefold(): col for col in raw.columns}
    missing = [name for name in _REQUIRED if name not in lookup]
    if missing:
        raise ValueError("RickRunGood SG export missing: " + ", ".join(missing))
    rows = raw.copy()
    rows["__player_key"] = rows[lookup["name"]].map(name_key)
    if rows["__player_key"].eq("").any() or rows["__player_key"].duplicated().any():
        raise ValueError("RickRunGood SG export has missing or duplicate golfer names.")
    salary = pd.to_numeric(rows[lookup["salary"]], errors="coerce")
    rounds = pd.to_numeric(rows[lookup["rounds"]], errors="coerce")
    if salary.isna().any() or (salary <= 0).any():
        raise ValueError("RickRunGood SG export has missing or invalid salaries.")
    if rounds.isna().any() or (rounds < 0).any():
        raise ValueError("RickRunGood SG export has invalid round counts.")
    rows["rrg_sg_salary"] = salary.astype(int)
    rows["rrg_sg_rounds"] = rounds
    fields = ["__player_key", "rrg_sg_salary", "rrg_sg_rounds"]
    for source_name, target in _MEASUREMENTS.items():
        if source_name not in lookup:
            continue
        value = pd.to_numeric(rows[lookup[source_name]], errors="coerce")
        if not np.isfinite(value.dropna()).all():
            raise ValueError("RickRunGood SG export has non-finite values in " + source_name)
        rows["rrg_raw_" + target] = value
        fields.append("rrg_raw_" + target)
    # Recent tournament finishes are not equivalent to round-level strokes gained.
    # Keep them for display/quality audit only, not as invented sg/form features.
    recent_cols = [name for name in raw.columns if str(name).startswith("Recent -")]
    rows["rrg_recent_finishes"] = raw[recent_cols].fillna("").astype(str).agg(
        lambda items: " / ".join(v for v in items if v), axis=1
    ) if recent_cols else ""
    fields.append("rrg_recent_finishes")
    dk = dk_field[["player", "salary"]].drop_duplicates("player").copy()
    dk["__player_key"] = dk["player"].map(name_key)
    if dk["__player_key"].duplicated().any():
        raise ValueError("DraftKings golfer identity collision after approved aliases.")
    matched = dk.merge(rows[fields], on="__player_key", how="left", validate="one_to_one")
    if matched["rrg_sg_salary"].notna().any():
        disagreement = matched.loc[
            matched["rrg_sg_salary"].notna() &
            matched["salary"].ne(matched["rrg_sg_salary"]),
            ["player", "salary", "rrg_sg_salary"],
        ]
        if not disagreement.empty:
            examples = disagreement.head(4).to_dict("records")
            raise ValueError(f"RickRunGood/DraftKings salary mismatch; wrong slate? {examples}")
    n = pd.to_numeric(matched["rrg_sg_rounds"], errors="coerce").fillna(0).clip(lower=0)
    shrink = n / (n + float(prior_rounds))
    for key in _SG_KEYS:
        raw_col = "rrg_raw_" + key
        if raw_col in matched:
            matched[key] = matched[raw_col] * shrink
    for key in ("driving_distance", "driving_accuracy", "gir_pct"):
        raw_col = "rrg_raw_" + key
        if raw_col in matched:
            matched[key] = matched[raw_col]  # Different units, no SG shrinkage.
    core_valid = matched[["rrg_raw_" + name for name in _CORE]].notna().sum(axis=1)
    matched["model_data_confidence"] = (shrink * core_valid / len(_CORE)).clip(0, 1)
    matched["rrg_sg_sample_weight"] = shrink
    matched["model_skill_source"] = "RickRunGood last-50 SG (sample shrunk)"
    # The primary model merges the returned player stats and never imports salary
    # from this vendor. DK salary remains the source of truth.
    out_cols = ["player", "rrg_sg_rounds", "rrg_sg_sample_weight", "model_data_confidence",
                "model_skill_source", "rrg_recent_finishes"]
    out_cols += [name for name in _MEASUREMENTS.values() if name in matched]
    out_cols += ["rrg_raw_" + name for name in _MEASUREMENTS.values()
                 if "rrg_raw_" + name in matched]
    normalized = matched[out_cols].copy()
    present_core = int(core_valid.eq(len(_CORE)).sum())
    valid_total = int(matched.get("rrg_raw_sg_total", pd.Series(dtype=float)).notna().sum())
    rounds12 = int((n >= 12).sum())
    return normalized, {
        "source": "RickRunGood last-50 SG CSV, sample shrunk",
        "field_size": int(len(dk)), "matched": int(matched["rrg_sg_rounds"].notna().sum()),
        "full_sg_components": present_core,
        "sg_total_available": valid_total,
        "at_least_12_rounds": rounds12,
        "no_sample_count": int(n.eq(0).sum()),
        "mean_rounds": float(n.mean()),
        "external_projection_supplied": False,
        "shrinkage_prior_rounds": float(prior_rounds),
    }
