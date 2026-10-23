"""Auditable confirmed-out opportunity redistribution; diagnostic, not certification.

Uses DraftKings status as a provisional availability signal. Current official
injury and depth-chart verification is still required before lineup export.
"""
import numpy as np
import pandas as pd
from ai2_nfl_matchups import canonical_team
from injury_opportunity import redistribute

OUT = frozenset({"OUT", "O", "IR", "INACTIVE", "SUSPENDED", "PUP", "NFI"})
ROLE = {"RB": "rush", "WR": "targets", "TE": "targets"}

def opportunity_redistribution(baselines, *, other_fraction=0.15):
    """Return diagnostic player rows and a team-level conservation audit.

    Targets are redistributed across WR/TE/RB; rushes only across RB.
    Missing/unmatched injured volume is NOT invented. Other/unmodeled
    opportunity is an explicit nonplayer recipient, not a hidden projection.
    """
    if not 0 <= other_fraction < 1:
        raise ValueError("other_fraction must be in [0,1)")
    x = baselines.copy()
    required = {"Team", "Position", "DK Status", "targets_pg", "carries_pg", "match_status"}
    if not required.issubset(x.columns):
        raise ValueError("Missing baseline fields: " + ", ".join(sorted(required - set(x.columns))))
    x["team_key"] = x["Team"].map(canonical_team)
    x["adjusted_targets_pg"] = np.nan
    x["adjusted_carries_pg"] = np.nan
    x["redistribution_note"] = "No reliable observed role available"
    audit = []
    for team, group in x.groupby("team_key"):
        for field, positions, output in (
            ("targets_pg", {"RB", "WR", "TE"}, "adjusted_targets_pg"),
            ("carries_pg", {"RB"}, "adjusted_carries_pg"),
        ):
            eligible = group[group["Position"].isin(positions)
                             & group["match_status"].eq("matched")].copy()
            if eligible.empty:
                continue
            volumes = pd.to_numeric(eligible[field], errors="coerce")
            eligible = eligible.loc[volumes.notna() & volumes.ge(0)].copy()
            if eligible.empty:
                continue
            weights = eligible[field].astype(float).to_numpy()
            status = eligible["DK Status"].fillna("").astype(str).str.upper().str.strip()
            removed = [i for i, flag in enumerate(status) if flag in OUT]
            other = float(weights.sum() * other_fraction / (1 - other_fraction))
            try:
                redistributed, other_after = redistribute(weights, other, removed_indices=removed)
            except ValueError:
                continue
            x.loc[eligible.index, output] = redistributed
            x.loc[eligible.index, "redistribution_note"] = (
                "DK-status-based provisional opportunity allocation; role unverified"
            )
            audit.append({
                "team": team, "opportunity": field, "observed_players": len(eligible),
                "out_players": len(removed), "removed_observed_volume": float(weights[removed].sum()),
                "player_volume_before": float(weights.sum()),
                "player_volume_after": float(redistributed.sum()),
                "other_before": other, "other_after": other_after,
                "conservation_delta": float(redistributed.sum() + other_after - weights.sum() - other),
                "source": "nflverse observed opportunity + DK salary status (not official availability)",
            })
    # A redistribution is NOT proof of injury status, snaps or weekly projection validity.
    return x, pd.DataFrame(audit)
