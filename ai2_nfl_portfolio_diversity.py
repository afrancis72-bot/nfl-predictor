"""Explain NFL DFS lineup overlap without mistaking concentration for an error.

Portfolio concentration is a conscious risk budget, not something to eliminate.
This module reports overlapping nine-player rosters and distinct QB/game
scripts so the user can decide when a concentrated high-ceiling core is worth it.
"""
from collections import Counter
from itertools import combinations


def portfolio_overlap_audit(ranked, *, alert_shared=7):
    if not isinstance(alert_shared, int) or not 2 <= alert_shared <= 9:
        raise ValueError("Overlap alert must be between 2 and 9 players")
    lineups = []
    qb_exposure = Counter()
    for number, entry in enumerate(ranked, 1):
        roster = entry["candidate"]["players"]
        ids = {str(p["ID"]) for p in roster}
        if len(roster) != 9 or len(ids) != 9:
            raise ValueError("Cannot score overlap for non-unique nine-player roster")
        qbs = [p for p in roster if str(p.get("Position", "")).upper() == "QB"]
        if len(qbs) != 1:
            raise ValueError("Exactly one quarterback needed to identify script")
        qb = qbs[0]
        qb_exposure[str(qb.get("Name", qb["ID"]))] += 1
        team = str(qb.get("TeamAbbrev", "")).upper()
        partners = [str(p.get("Name", p["ID"])) for p in roster
                    if p.get("TeamAbbrev", "").upper() == team
                    and p["Position"] in {"WR", "TE"}]
        lineups.append({
            "lineup": number,
            "contest": str(entry["contest_tier"]),
            "ids": ids,
            "QB": str(qb.get("Name", qb["ID"])),
            "QB team": team,
            "Stack partners": ", ".join(partners) if partners else "No direct receiver",
            "Mean": round(float(entry["simulated_mean"]), 2),
            "P99": round(float(entry["simulated_p99"]), 2),
        })
    overlaps = []
    histogram = Counter()
    for left, right in combinations(lineups, 2):
        count = len(left["ids"] & right["ids"])
        histogram[count] += 1
        if count >= alert_shared:
            overlaps.append({
                "Lineup A": left["lineup"], "Tier A": left["contest"],
                "Lineup B": right["lineup"], "Tier B": right["contest"],
                "Shared players": count,
                "Same QB": left["QB"] == right["QB"],
                "Shared-player percent": round(100 * count / 9, 1)
            })
    overlaps.sort(key=lambda item: (-item["Shared players"],
                                    item["Lineup A"], item["Lineup B"]))
    total_pairs = len(lineups) * (len(lineups) - 1) // 2
    return {
        "summary": {
            "lineups": len(lineups),
            "distinct_qbs": len(qb_exposure),
            "qb_counts": dict(qb_exposure),
            "pairs": total_pairs,
            "pairs_at_least_alert": len(overlaps),
            "exactly_eight_shared_pairs": histogram[8],
            "concentration_policy": (
                "Preserve strong repeated cores when justified by projections. "
                "Near-duplicate lineups expose the same game script; review "
                "alternative QB/bring-back paths when their modeled upside "
                "is competitive. Do not force unsupported players."
            ),
            "market_game_total_coverage": "Not automatically integrated",
        },
        "near_duplicate_pairs": overlaps,
        "lineup_scripts": [{key: value for key, value in x.items() if key != "ids"}
                           for x in lineups],
    }
