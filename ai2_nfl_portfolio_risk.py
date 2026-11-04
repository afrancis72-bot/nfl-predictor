"""Evidence-sensitive NFL Classic portfolio risk policy.

This operates AFTER simulating players; it does not fabricate scores or infer a
depth chart. NFL Classic portfolios must diversify meaningful uncertainty:
a cheap reserve/backup-like RB can have upside without being treated as a
stable starting runner. User approval and injury verification remain required.
"""
from itertools import combinations
from math import floor, isfinite

DEFAULT_MAX_EXPOSURE = 0.28
UNCERTAIN_RB_EXPOSURE = 0.12
RETURN_COMPETITION_EXPOSURE = 0.08
MAX_SHARED_PLAYERS = 6


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def uncertain_running_back_reason(player):
    """Conservative evidence-based workload flag, not a depth-chart claim.

    Raw observed pregame data must be present; simulated injury-driven
    opportunity cannot on its own establish a sustainable workload.
    """
    if str(player.get("Position", "")).upper() != "RB":
        return None
    if player.get("emerging_role_limited") is True:
        return ("Emerging two-game role: needs official current-week confirmation; "
                + str(player.get("emerging_role_evidence") or ""))
    returning = str(player.get("return_competition_reason") or "").strip()
    if returning:
        return "Returning starter/teammate workload uncertainty: " + returning
    games = _finite(player.get("observed_games"))
    carries = _finite(player.get("observed_carries_pg"))
    targets = _finite(player.get("observed_targets_pg"))
    yards = _finite(player.get("observed_rushing_yards_pg"))
    modeled = _finite(player.get("projected_touches"))
    if games is None or games < 3 or any(x is None for x in (carries, targets, yards)):
        return None
    observed_touches = carries + targets
    recent_n = _finite(player.get("observed_recent_four_games"))
    recent_yards = _finite(player.get("observed_recent_four_rush_yd_pg"))
    recent_carries = _finite(player.get("observed_recent_four_carries_pg"))
    recent_targets = _finite(player.get("observed_recent_four_targets_pg"))
    # Unweighted recent game volume provides a cross-check when one spike
    # distorts recency-weighted historical projections.
    if (recent_n is not None and recent_n >= 4
            and all(x is not None for x in (recent_yards,recent_carries,recent_targets))
            and recent_yards < 35 and recent_carries < 12
            and recent_targets < 4):
        return (f"Limited last-four-game rushing role: {recent_yards:.1f} "
                f"rushing yards, {recent_carries:.1f} carries and "
                f"{recent_targets:.1f} targets per game; "
                "backup/rotation workload needs current-week verification")
    # A player who historically carries infrequently is not automatically a
    # featured back after one injury-dependent spike.
    if carries < 8 and yards < 40 and targets < 4:
        return (f"Limited observed RB rushing role: {carries:.1f} carries, "
                f"{yards:.1f} rushing yards, {targets:.1f} targets per game "
                f"across {int(games)} weighted source games")
    if modeled is not None and observed_touches < 12 and modeled > observed_touches * 1.60:
        return (f"Projected {modeled:.1f} touches exceeds observed "
                f"{observed_touches:.1f} touches by >60% without a verified "
                "current-week lead-back role")
    return None


def maximum_player_appearances(player, lineup_count, overall_fraction=DEFAULT_MAX_EXPOSURE):
    """Strict whole-count exposure ceiling; special RB limit when evidenced."""
    if lineup_count < 1 or not 0 < overall_fraction <= 1:
        raise ValueError("Invalid NFL portfolio exposure parameters")
    overall = max(1, floor(lineup_count * overall_fraction))
    if player.get("return_competition_reason"):
        return min(overall, max(1, floor(lineup_count * RETURN_COMPETITION_EXPOSURE)))
    # Emerging two-game role is an opportunity candidate, not a validated
    # full-game workhorse. Retain only limited exploratory GPP exposure.
    if player.get("emerging_role_limited") is True:
        return min(overall, max(1, floor(lineup_count * RETURN_COMPETITION_EXPOSURE)))
    if uncertain_running_back_reason(player):
        return min(overall, max(1, floor(lineup_count * UNCERTAIN_RB_EXPOSURE)))
    return overall


def overlap_conflicts(roster, selected_rosters, *, max_shared=MAX_SHARED_PLAYERS):
    keys = {str(p.get("ID")) for p in roster}
    return sum(len(keys & set(other)) > max_shared for other in selected_rosters)


def overlap_groups(scored, *, max_shared=MAX_SHARED_PLAYERS):
    """Groups of candidate indexes that cannot jointly be selected.

    Two 9-player rosters sharing >=7 players necessarily contain a repeated
    7-ID subset. One sparse set-packing constraint per repeated subset
    avoids pairwise quadratic constraints.
    """
    required = max_shared + 1
    shared = {}
    for i, item in enumerate(scored):
        ids = sorted({str(p["ID"]) for p in item["candidate"]["players"]})
        if len(ids) < required:
            continue
        for sig in combinations(ids, required):
            shared.setdefault(sig, []).append(i)
    return [indexes for indexes in shared.values() if len(indexes) > 1]
