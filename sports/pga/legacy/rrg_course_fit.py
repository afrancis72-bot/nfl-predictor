"""RickRunGood course-fit import and independent source comparison.

The RRG FIT value and component contributions are NOT strokes gained,
calibrated fantasy projections, or a replacement for Team AI²'s own model.
Only compare scores/ranks; never silently inject RRG into a model run.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

# Verified common naming variants, audited rather than fuzzy matched.
_NAME_ALIASES = {
    "camdavis": "camerondavis",
    "zachbauchou": "zacharybauchou",
    "richardhoey": "ricohoey",
    "jordansmith": "jordanlsmith",
}


def name_key(value: object) -> str:
    basic = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    key = re.sub(r"[^a-z0-9]", "", basic.casefold())
    return _NAME_ALIASES.get(key, key)


def normalize_rrg_course_fit(raw: pd.DataFrame, dk_field: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize FIT and components, matching the DK field by safely normalized names."""
    if raw is None or raw.empty:
        return pd.DataFrame(), {"source": "not supplied", "matched": 0, "total": len(dk_field)}
    required = {"player", "fit"}
    lookup = {str(c).strip().casefold(): c for c in raw.columns}
    if not required.issubset(lookup):
        raise ValueError("RRG course-fit CSV requires Player and FIT columns.")
    df = raw.rename(columns={
        lookup["player"]: "rrg_source_player",
        lookup["fit"]: "rrg_fit",
    }).copy()
    optional = {
        "rank": "rrg_source_rank",
        "driving": "rrg_driving",
        "approach": "rrg_approach",
        "around green": "rrg_around_green",
        "putting": "rrg_putting",
        "scoring": "rrg_scoring",
    }
    rename = {lookup[k]: v for k, v in optional.items() if k in lookup}
    df = df.rename(columns=rename)
    cols = ["rrg_source_player", "rrg_fit"] + list(rename.values())
    df = df[cols].copy()
    for col in cols:
        if col not in {"rrg_source_player"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["rrg_fit"]).copy()
    df["_key"] = df["rrg_source_player"].map(name_key)
    df = df[df["_key"].ne("")].drop_duplicates("_key")
    dk = dk_field[["player"]].drop_duplicates().copy()
    dk["_key"] = dk["player"].map(name_key)
    if dk["_key"].duplicated().any():
        raise ValueError("DK names produce ambiguous matches; verify golfer identities.")
    matched = dk.merge(df, on="_key", how="left").drop(columns="_key")
    count = int(matched["rrg_fit"].notna().sum())
    matched["rrg_rank"] = matched["rrg_fit"].rank(method="min", ascending=False)
    details = {
        "source": "RickRunGood course-fit (not OTIS)",
        "matched": count,
        "total": int(len(dk)),
        "source_rows": int(len(df)),
        "coverage": count / len(dk) if len(dk) else 0.0,
    }
    return matched, details


def attach_source_comparison(
    model: pd.DataFrame, rrg: pd.DataFrame | None,
    otis: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge external ranks for audit only. Model projections remain unchanged."""
    out = model.copy()
    if "model_rank" not in out.columns:
        raise ValueError("Model must have been run before comparison.")
    if rrg is not None and not rrg.empty:
        cols = [c for c in rrg.columns if c == "player" or c.startswith("rrg_")]
        out = out.merge(rrg[cols].drop_duplicates("player"), on="player", how="left", validate="one_to_one")
    if otis is not None and not otis.empty:
        src = otis[["player"] + [c for c in ("otis_rank", "otis_true_skill", "otis_course_fit", "otis_form") if c in otis]].drop_duplicates("player")
        # OTIS is a separate audit rank. It is never synthesized from model strength.
        out = out.merge(src, on="player", how="left", validate="one_to_one", suffixes=("", "_audit"))
    if "rrg_rank" not in out:
        out["rrg_rank"] = np.nan
    if "otis_rank" not in out:
        out["otis_rank"] = np.nan
    out["model_vs_rrg_rank_delta"] = pd.to_numeric(out["model_rank"], errors="coerce") - pd.to_numeric(out["rrg_rank"], errors="coerce")
    out["model_vs_otis_rank_delta"] = pd.to_numeric(out["model_rank"], errors="coerce") - pd.to_numeric(out["otis_rank"], errors="coerce")
    return out
