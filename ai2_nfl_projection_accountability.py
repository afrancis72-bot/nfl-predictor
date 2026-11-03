"""Link measured NFL workload warnings to actual lineup exposures for pre-entry review.

Not a fitted injury model. These are alerts only: do not automatically change
the historical source data, touchdown scoring ledger or the user's final say.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from ai2_nfl_matchups import canonical_team


def uncertain_role_exposures(anomalies, ranked):
    """Which source-vs-simulation discrepancies actually dominate this portfolio?"""
    if not ranked:
        return []
    n=len(ranked)
    appearances=Counter()
    eligible_names={}
    for row in ranked:
        names=set()
        for p in row["candidate"]["players"]:
            key=(str(p.get("Name","")).strip().casefold(),
                 canonical_team(p.get("TeamAbbrev","")))
            names.add(key)
            eligible_names[key]=str(p.get("Name",""))
        appearances.update(names)
    by_player=defaultdict(list)
    for issue in anomalies or []:
        key=(str(issue.get("Player","")).strip().casefold(),
             canonical_team(issue.get("Team","")))
        if key in appearances:
            by_player[key].append(issue)
    rows=[]
    for key,issues in by_player.items():
        rises=[]
        for issue in issues:
            try: rises.append(float(issue.get("Increase %")))
            except (ValueError,TypeError): pass
        most=max(rises) if rises else None
        count=appearances[key]
        rows.append({
            "Player":eligible_names[key],
            "Team":key[1],
            "Lineups using flagged player":count,
            "Portfolio exposure %":round(count/n*100,2),
            "Observed vs simulation concerns":"; ".join(
                str(r.get("Concern","")) for r in issues),
            "Largest observed discrepancy %":round(most,1) if most is not None else None,
            "Prior observed player-games":max(
                int(float(r.get("Prior player-games") or 0)) for r in issues),
            "Review priority":("HIGH - exposed >=20% with 75%+ discrepancy"
                               if count/n>=.20 and most is not None and most>=75
                               else "Review"),
            "Status":"Experimental projection discrepancy; inspect injury and recent-role evidence",
        })
    return sorted(rows,key=lambda x:(
        x["Review priority"]!="HIGH - exposed >=20% with 75%+ discrepancy",
        -x["Portfolio exposure %"],x["Player"]))
