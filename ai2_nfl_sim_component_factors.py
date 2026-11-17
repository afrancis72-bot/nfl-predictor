"""Research-only validated pregame opponent rate factors for football worlds.

Never takes a raw fantasy-point residual and adds it to a player's score.
The underlying defense profiles are measured against PREgame per-opportunity
efficiency.  This module only extracts bounded, provenance-checked factors.

Production one-click scoring does not pass component_strength.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ai2_nfl_defense_components import SOURCE, METRICS
from ai2_nfl_matchups import canonical_team

VALID_CHANNELS = frozenset(METRICS)


def validated_component_lookup(profiles, *, season, week, strength=0.0):
    """Reject future/unknown/unsourced records instead of accepting fake boosts."""
    if not np.isfinite(strength) or not 0. <= strength <= 1.:
        raise ValueError("Matchup component strength must be between 0 and 1")
    if strength == 0:
        return {}
    if season is None or week is None:
        raise ValueError("Component effect requires an explicit pregame season/week")
    if profiles is None:
        raise ValueError("Active component strength requires pregame defense profiles")
    if not isinstance(profiles, pd.DataFrame):
        raise ValueError("Defensive component profiles must be measured DataFrame rows")
    if profiles.empty:
        return {}
    required = {"defense", "position", "channel", "observations",
                "opportunities", "shrunk_efficiency_pct", "source",
                "target_season", "target_week"}
    if not required.issubset(profiles.columns):
        raise ValueError("Incomplete opponent component evidence")
    result = {}
    for row in profiles.to_dict("records"):
        if (str(row["source"]) != SOURCE or
                int(row["target_season"]) != int(season) or
                int(row["target_week"]) != int(week)):
            raise ValueError("Unsourced or wrong-week defensive component evidence")
        position, channel = str(row["position"]), str(row["channel"])
        if (position, channel) not in VALID_CHANNELS:
            raise ValueError("Unknown defensive football-efficiency channel")
        sample, opportunities = float(row["observations"]), float(row["opportunities"])
        delta = float(row["shrunk_efficiency_pct"])
        if (not np.isfinite([sample, opportunities, delta]).all() or
                sample < 7 or opportunities < 28 or abs(delta) > 15.01):
            raise ValueError("Defensive component effect lacks valid minimum evidence")
        defense = canonical_team(row["defense"])
        key = (defense, position, channel)
        if defense in ("", "NONE", "NAN", "UNKNOWN") or key in result:
            raise ValueError("Invalid or duplicated defensive component")
        result[key] = float(np.clip(1.0 + strength*delta/100, 0.85, 1.15))
    return result


def component_factor(lookup, opponent, position, channel):
    return lookup.get((canonical_team(opponent), position, channel), 1.0)
