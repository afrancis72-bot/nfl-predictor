"""Find strongest legal NFL Classic QB stacks using the ACTUAL eligible slate.

This is a candidate-generation cross-check, not a mandate to roster a QB.
It uses the same salary, opportunity, roster, RB-team and QB-vs-DST rules as
the live Classic builder. A missing feasible solution is reported as missing,
never as zero projected upside or proof that a team is unattractive.
"""
from __future__ import annotations

import numpy as np

from ai2_nfl_lineup import check_classic_lineup
from ai2_nfl_matchups import canonical_team


def anchored_qb_stack(pools, quarterback, *, min_receivers=1,
                      salary_floor=49600, salary_cap=50000, time_limit=.55):
    """MILP maximize current projected DK mean conditional on a named QB and a
    minimum number of genuine same-team WR/TE teammates.

    Returns a LEGAL lineup or None (which may indicate solver timeout); does
    not generate a hypothetical unsupported or zero-touch player.
    """
    try:
        from scipy.optimize import milp, Bounds, LinearConstraint
    except ImportError as exc:
        raise ValueError("SciPy MILP required for reliable QB stack search") from exc
    if min_receivers not in (1,2):
        raise ValueError("Stack receiver count must be one or two")
    qb_team=canonical_team(quarterback.get("TeamAbbrev",""))
    qb_id=str(quarterback.get("ID",""))
    if not qb_id or not qb_team:
        return None
    skill=[]
    for pos in ("RB","WR","TE","DST"):
        skill.extend(pools.get(pos,[]))
    roster=[quarterback]+sorted(skill,key=lambda p:(str(p.get("Position")),str(p.get("ID"))))
    n=len(roster)
    team_receivers=[i for i,p in enumerate(roster)
                    if p.get("Position") in ("WR","TE")
                    and canonical_team(p.get("TeamAbbrev",""))==qb_team]
    if len(team_receivers)<min_receivers:
        return None
    if sum(p.get("Position")=="RB" for p in roster)<2 or sum(p.get("Position")=="WR" for p in roster)<3:
        return None
    try:
        salaries=np.asarray([float(p["Salary"]) for p in roster],dtype=float)
        projected=np.asarray([float(p["proj"]) for p in roster],dtype=float)
    except (TypeError,ValueError,KeyError) as exc:
        raise ValueError("QB stack candidate lacks salary/projection evidence") from exc
    if not (np.isfinite(salaries).all() and np.isfinite(projected).all()
            and np.all(salaries>0)):
        raise ValueError("QB stack candidate salaries/projections are invalid")
    c=-projected
    lb=np.zeros(n)
    ub=np.ones(n)
    lb[0]=ub[0]=1.
    # Do not permit Classic QB against his opposing DST.
    opponent=canonical_team(quarterback.get("opponent",""))
    if opponent:
        for i,p in enumerate(roster):
            if p.get("Position")=="DST" and canonical_team(p.get("TeamAbbrev",""))==opponent:
                ub[i]=0.
    rows=[];lows=[];highs=[]
    def constrain(indices,lo,hi):
        row=np.zeros(n)
        row[list(indices)]=1.
        rows.append(row);lows.append(lo);highs.append(hi)
    constrain(range(n),9,9)
    for pos,lo,hi in (("QB",1,1),("RB",2,3),("WR",3,4),("TE",1,2),("DST",1,1)):
        constrain([i for i,p in enumerate(roster) if p.get("Position")==pos],lo,hi)
    constrain(team_receivers,min_receivers,len(team_receivers))
    teams={canonical_team(p.get("TeamAbbrev","")) for p in roster if p.get("Position")=="RB"}
    for team in teams:
        constrain([i for i,p in enumerate(roster) if p.get("Position")=="RB"
                   and canonical_team(p.get("TeamAbbrev",""))==team],0,1)
    rows.append(salaries);lows.append(salary_floor);highs.append(salary_cap)
    result=milp(c=c, integrality=np.ones(n,dtype=int),
                bounds=Bounds(lb,ub),
                constraints=LinearConstraint(np.asarray(rows),lows,highs),
                options={"time_limit":float(time_limit),"mip_rel_gap":.015})
    if result.x is None:
        return None
    chosen=[roster[i] for i,x in enumerate(result.x) if x>.5]
    if len(chosen)!=9 or check_classic_lineup(chosen,salary_floor=salary_floor,
                                              salary_cap=salary_cap):
        return None
    num=sum(p.get("Position") in ("WR","TE")
            and canonical_team(p.get("TeamAbbrev",""))==qb_team for p in chosen)
    return chosen if num>=min_receivers else None
