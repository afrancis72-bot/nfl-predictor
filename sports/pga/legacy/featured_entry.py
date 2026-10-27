"""Rank a PGA single-entry feature build from an already generated portfolio.

A transparent, heuristic ranking of MODEL ESTIMATES — not a validated contest
win probability and not a statement that a golfer is safe to bet on.
External RRG/OTIS rankings do not enter this calculation.
"""
from __future__ import annotations

import pandas as pd
from typing import Sequence


def choose_featured_lineup(summary: pd.DataFrame, portfolio: pd.DataFrame,
                           predictions: pd.DataFrame,
                           feature_exclusions: Sequence[str] | None = None) -> tuple[pd.DataFrame, int]:
    if summary is None or summary.empty or portfolio is None or portfolio.empty:
        raise ValueError("Generate a nonempty PGA portfolio first.")
    required = {"lineup", "sum_dk_proxy", "avg_make_cut_pct", "sum_top10_pct"}
    if not required.issubset(summary.columns):
        raise ValueError("Portfolio summary is missing core ranking columns.")
    if not {"lineup", "player"}.issubset(portfolio.columns):
        raise ValueError("Portfolio golfer identities are missing.")
    if not {"player", "v11_p90_dk"}.issubset(predictions.columns):
        raise ValueError("Run the golf-first V11 simulation before selecting a feature lineup.")

    src = predictions[["player", "v11_p90_dk"]].copy()
    if src["player"].duplicated().any():
        raise ValueError("Duplicate golfer projections; feature ranking is ambiguous.")
    details = portfolio[["lineup", "player"]].merge(src, on="player", how="left", validate="many_to_one")
    if details["v11_p90_dk"].isna().any():
        raise ValueError("Cannot rank feature lineup with missing simulation tails.")
    tails = details.groupby("lineup")["v11_p90_dk"].sum().rename("sum_player_p90_proxy")

    d = summary.copy().merge(tails, left_on="lineup", right_index=True, how="left")
    metrics = {
        "sum_dk_proxy": 0.35,
        "avg_make_cut_pct": 0.25,
        "sum_top10_pct": 0.25,
        "sum_player_p90_proxy": 0.15,
    }
    d["feature_index"] = 0.0
    for metric, weight in metrics.items():
        x = pd.to_numeric(d[metric], errors="coerce")
        if x.isna().any():
            raise ValueError(f"Missing feature ranking metric: {metric}")
        pct = x.rank(method="average", pct=True)
        d["feature_index"] += weight * pct

    # A feature-only veto never removes the golfer from the remaining GPP lineups.
    # First score ALL model lineups consistently; then filter eligibility
    # ONLY when assigning the higher-stakes contest.
    veto = {str(v).strip().casefold() for v in (feature_exclusions or []) if str(v).strip()}
    if veto:
        disqualified = set(
            details.loc[
                details["player"].astype(str).str.strip().str.casefold().isin(veto),
                "lineup",
            ]
        )
        d["feature_eligible"] = ~d["lineup"].isin(disqualified)
    else:
        d["feature_eligible"] = True
    eligible = d.loc[d["feature_eligible"]].copy()
    if eligible.empty:
        raise ValueError(
            "No eligible lineup for the $5 entry after feature-only exclusions. "
            "Adjust the feature veto or rebuild the 11-lineup portfolio; "
            "no automatic override is allowed."
        )
    # Stable deterministic tiebreaker; not optimized with hindsight.
    d = d.sort_values(["feature_index", "sum_dk_proxy", "lineup"],
                      ascending=[False, False, True])
    eligible = eligible.sort_values(["feature_index", "sum_dk_proxy", "lineup"],
                                    ascending=[False, False, True])
    featured_id = int(eligible.iloc[0]["lineup"])
    d["contest_allocation"] = d["lineup"].apply(
        lambda value: "$5 SINGLE ENTRY" if int(value) == featured_id else "$0.50 GPP"
    )
    return d, featured_id
