"""Source-checked, bounded opponent-specific football YARD efficiency.

Uses pregame nflverse player-week partial DK allowed by defense/position.
Adjusts ONLY receiving or rushing yards per observed opportunity; never
creates new targets, carries, scores or extra team plays.

Research hypothesis until 2025 training / independent 2026 validation.
Player-specific defender injury assumptions and unverified WR-CB shadow
matchups do not enter this model.
"""
from __future__ import annotations
import math
import pandas as pd
from ai2_nfl_matchups import canonical_team

SUPPORTED_POS=frozenset(("QB","RB","WR","TE"))
SOURCE_PREFIX="nflverse observed pregame opponent_team"

def build_defense_profiles(allowances,season,week,*,min_observations=20):
    """Translate sourced partial-DK defense allowances into team+position records."""
    if not isinstance(allowances,pd.DataFrame):
        raise ValueError("Opponent allowance source must be a DataFrame")
    fields={"defense","position","multiplier","player_game_observations","source"}
    if not fields.issubset(allowances.columns):
        raise ValueError("Missing nflverse defensive allowance evidence")
    results={}
    seen=set()
    for entry in allowances.to_dict("records"):
        defense=canonical_team(entry["defense"])
        position=str(entry["position"]).upper()
        key=(defense,position)
        if not defense or position not in SUPPORTED_POS or key in seen:
            if key in results:
                del results[key]
            continue
        seen.add(key)
        try:
            mult=float(entry["multiplier"])
            sample=int(entry["player_game_observations"])
        except (TypeError,ValueError):
            continue
        if (not math.isfinite(mult) or not .75<=mult<=1.25 or
            sample<int(min_observations) or
            not str(entry["source"]).startswith(SOURCE_PREFIX)):
            continue
        results[key]={"factor":mult,"sample":sample,
                      "source":str(entry["source"]),
                      "season":int(season),"week":int(week)}
    return results


def matchup_yard_multiplier(profiles,opposing_team,position,*,strength=0.):
    if not math.isfinite(strength) or not 0.<=strength<=1.:
        raise ValueError("Invalid opponent yard-efficiency weight")
    if not strength:
        return 1.
    record=profiles.get((canonical_team(opposing_team),str(position).upper()))
    if not record:
        return 1.
    if not str(record.get("source","")).startswith(SOURCE_PREFIX):
        raise ValueError("Scoring cannot use unsourced defensive estimates")
    if int(record.get("sample",0))<20:
        return 1.
    factor=float(record["factor"])
    if not math.isfinite(factor) or not .75<=factor<=1.25:
        raise ValueError("Unreasonable opposing-defense multiplier")
    return float(max(.88,min(1.12,1.+strength*(factor-1.))))
