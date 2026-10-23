"""Aggregate multiple frozen pregame NFL simulation audits without future leakage.

Each batch entry is (players, draws, actuals), from one distinct historical slate.
This layer does not run the simulator or invent historical DK salaries.
"""
import pandas as pd
from ai2_nfl_distribution_audit import distribution_diagnostics


def audit_historical_slates(slates):
    """Return per-slate and overall distribution diagnostics.

    Each item is a dict with slate_id, frozen_before_kickoff (bool),
    players, draws, and actuals. Reject missing freeze attestations,
    duplicate slate IDs, and reused player IDs within one slate.
    """
    if not isinstance(slates, (list, tuple)) or not slates:
        raise ValueError("At least one frozen historical slate is required")
    all_rows = []
    seen = set()
    for item in slates:
        if not isinstance(item, dict):
            raise ValueError("Historical slate entries must be dictionaries")
        key = str(item.get("slate_id", "")).strip()
        if not key or key in seen:
            raise ValueError("Historical slate IDs must be present and unique")
        seen.add(key)
        if item.get("frozen_before_kickoff") is not True:
            raise ValueError(f"Slate {key} lacks explicit pregame freeze attestation")
        try:
            diagnostics = distribution_diagnostics(item["players"], item["draws"], item["actuals"])
        except KeyError as exc:
            raise ValueError(f"Slate {key} missing required simulation data: {exc}") from exc
        diagnostics.insert(0, "slate_id", key)
        all_rows.append(diagnostics)
    combined = pd.concat(all_rows, ignore_index=True)
    per_slate = (combined.groupby("slate_id", as_index=False)
                 .agg(players=("ID", "size"),
                      mean_absolute_error=("absolute_error", "mean"),
                      central_50_coverage=("covered_50", "mean"),
                      central_80_coverage=("covered_80", "mean"),
                      mean_pit=("pit", "mean")))
    overall = {
        "slates": len(seen),
        "players": int(len(combined)),
        "mean_absolute_error": float(combined["absolute_error"].mean()),
        "central_50_coverage": float(combined["covered_50"].mean()),
        "central_80_coverage": float(combined["covered_80"].mean()),
        "mean_pit": float(combined["pit"].mean()),
    }
    return per_slate, combined, overall
