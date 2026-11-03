"""Explain QB/WR/TE stack consideration across the ACTUAL unified NFL pipeline.

This reports evidence and optimizer decisions; it never assigns a favorite QB,
invents betting totals or alters a user's approved portfolio.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from ai2_nfl_matchups import canonical_team
from ai2_nfl_role_evidence import player_opportunity_evidence


def qb_and_partners(roster):
    """Return (QB, direct WR/TE pass catchers); RB does not count as a pass stack."""
    qbs=[p for p in roster if str(p.get("Position", "")).upper()=="QB"]
    if len(qbs)!=1:
        return None,[]
    qb=qbs[0]
    team=canonical_team(qb.get("TeamAbbrev",""))
    others=[p for p in roster if (str(p.get("Position", "")).upper() in {"WR","TE"}
            and canonical_team(p.get("TeamAbbrev",""))==team
            and str(p.get("ID"))!=str(qb.get("ID")))]
    return qb,others


def _eligible_pass_partner(p, *, min_wr_salary=4000):
    pos=str(p.get("Position","")).upper()
    try:
        salary=float(p.get("Salary",0))
    except (TypeError,ValueError):
        return False
    return (pos in {"WR","TE"}
            and (pos!="WR" or salary>=min_wr_salary)
            and p.get("opportunity_verified") is True
            and player_opportunity_evidence(p)[0])


def qb_selection_report(pool,candidates,ranked,*,diagnostics=None,
                        team_environment=None):
    """One row per source-eligible quarterback: from field to final 20/150.

    All reports identify why a QB disappears; they never imply selection
    should occur just because a player or team ultimately scored points.
    """
    if diagnostics is None:
        diagnostics={}
    team_environment=team_environment or {}
    records=list(pool)
    qbs=[p for p in records if (str(p.get("Position","")).upper()=="QB"
          and p.get("opportunity_verified") is True
          and player_opportunity_evidence(p)[0])]
    teammates=defaultdict(list)
    for p in records:
        if _eligible_pass_partner(p):
            teammates[canonical_team(p.get("TeamAbbrev",""))].append(p)
    candidate_counts=Counter()
    candidate_stacks=Counter()
    final_counts=Counter()
    final_stacks=Counter()
    final_upside=Counter()
    final_premium=Counter()
    for entry in candidates:
        qb,partners=qb_and_partners(entry["players"])
        if qb is None:
            continue
        ident=str(qb["ID"])
        candidate_counts[ident]+=1
        candidate_stacks[ident]+=bool(partners)
    for entry in ranked:
        qb,partners=qb_and_partners(entry["candidate"]["players"])
        if qb is None:
            continue
        ident=str(qb["ID"])
        final_counts[ident]+=1
        final_stacks[ident]+=bool(partners)
        final_upside[ident]+=entry.get("portfolio_role")=="upside"
        final_premium[ident]+=entry.get("contest_tier")=="$3"
    before=diagnostics.get("scored_qbs_before_efficiency",{})
    after=diagnostics.get("scored_qbs_after_efficiency",{})
    stacked_after=diagnostics.get("stacked_qbs_after_efficiency",{})
    best_p99=diagnostics.get("qb_best_simulated_p99",{})
    rows=[]
    for qb in sorted(qbs,key=lambda r:(-float(r.get("proj",0)),str(r.get("Name","")))):
        ident=str(qb["ID"])
        team=canonical_team(qb.get("TeamAbbrev",""))
        partners=teammates[team]
        observed=team_environment.get(team,{})
        if not partners:
            reason="No verified eligible WR/TE passing partner in uploaded slate"
        elif not candidate_counts[ident]:
            reason="No legal QB roster survived salary/roster/role candidate search"
        elif not candidate_stacks[ident]:
            reason="No legal direct receiver stack reached candidate shortlist"
        elif not stacked_after.get(ident,0):
            reason="Stacked candidates failed simulated P90 salary-efficiency filter and credible ceiling gate"
        elif not final_counts[ident]:
            reason="Simulated portfolio, salary efficiency, value and exposure favored alternatives"
        else:
            reason="Selected in final portfolio; no QB is guaranteed exposure"
        rows.append({
            "QB":str(qb.get("Name",ident)),
            "QB ID":ident,
            "Team":team,
            "QB simulated mean DK":round(float(qb.get("proj",0)),2),
            "Eligible passing partners":len(partners),
            "Best available partners":", ".join(str(x.get("Name",x["ID"])) for x in
                                                sorted(partners,key=lambda x:-float(x.get("proj",0)))[:3]),
            "Legal QB candidates":candidate_counts[ident],
            "Stacked QB candidates":candidate_stacks[ident],
            "Simulated candidates before filter":int(before.get(ident,0)),
            "After efficiency filter":int(after.get(ident,0)),
            "Stacked candidates after efficiency":int(stacked_after.get(ident,0)),
            "Best candidate simulated P99":(
                round(float(best_p99[ident]),2) if ident in best_p99 else None),
            "Final QB lineups":final_counts[ident],
            "Final QB/WR-TE stacks":final_stacks[ident],
            "Final upside QB lineups":final_upside[ident],
            "Final premium QB lineups":final_premium[ident],
            "Prior-game pass attempts expected":observed.get("expected_pass_attempts"),
            "Team environment source":observed.get("source","not available"),
            "QB selection status":reason,
        })
    return rows
