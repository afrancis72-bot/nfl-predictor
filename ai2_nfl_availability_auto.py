"""One-button NFL mode selection; do not imply experimental status is a new predictor.

Official capture collection is performed by the existing pregame source bridge.
Use the SAME existing engine when there are no verified snapshots. A verified
snapshot can introduce sourced OUT exclusions and positive ACTIVE historical
teammates, but does NOT imply a different projection architecture.
"""
from __future__ import annotations


def select_nfl_engine_from_sources(context):
    """(engine, reason) for a point-in-time verified-source context.

    No status archive -> exactly the existing model, rather than calling
    an experimental mode whose simulated output would be identical.
    """
    if not isinstance(context, dict):
        raise ValueError("NFL availability context must be source-backed")
    snapshots=context.get("snapshots_by_game")
    if not isinstance(snapshots,dict):
        raise ValueError("NFL availability snapshots must be a game map")
    if snapshots:
        return ("availability_aware","VERIFIED_GAME_DAY_SOURCE_PRESENT")
    return ("existing","NO_VERIFIED_GAME_DAY_SOURCE")


def report_nfl_availability_effect(report,engine):
    """Describe actual model-input differences, not an unproved scoring edge."""
    if engine not in ("existing","availability_aware"):
        raise ValueError("Unknown NFL game engine")
    if not isinstance(report,dict):
        raise ValueError("NFL build must provide source report")
    added=int(report.get("football_only_players_added",0) or 0)
    excluded=int(report.get("newly_applied_verified_out",0) or 0)
    priced=int(report.get("applied_verified_out_to_priced_players",0) or 0)
    if min(added,excluded,priced)<0:
        raise ValueError("Invalid negative official-source effect")
    has_effect=bool(added or excluded or priced)
    if engine=="existing":
        status="EXISTING_MODEL_NO_EXTRA_VERIFIED_INPUTS"
        explanation=(
            "No verified additional game-day player information was available. "
            "The original NFL simulation and lineup selection were used.")
    elif has_effect:
        status="VERIFIED_STATUS_CHANGED_SIMULATION_INPUTS"
        explanation=(
            f"Verified game-day sources affected the modeled player data: "
            f"{added} internal non-DK teammates, {excluded} newly identified "
            f"OUT players for simulation, and {priced} newly excluded "
            "salary-listed players. Resulting lineups may or may not differ.")
    else:
        status="VERIFIED_SOURCE_WITH_NO_NEW_SIMULATION_INPUTS"
        explanation=(
            "A verified source was found, but it added no previously unknown "
            "OUT status or eligible internal teammate. The same player model "
            "inputs were used; different lineup results are not expected.")
    return {
        "engine_input_effect":status,
        "changed_model_player_inputs":has_effect,
        "engine_input_effect_explanation":explanation,
        "newly_applied_verified_out":excluded,
        "applied_verified_out_to_priced_players":priced,
        "football_only_players_added":added,
        "independent_predictive_accuracy_proven":False,
    }
