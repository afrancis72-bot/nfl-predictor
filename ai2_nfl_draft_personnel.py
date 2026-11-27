"""Uncalibrated, draft-only personnel sensitivity using observed rate components.
No implied medical probability: equally weighted active/inactive stress worlds.
"""
import numpy as np

def shrink_efficiency(yards, attempts, games, prior, prior_attempts=40.):
    count=max(0.,float(attempts))*max(0.,float(games))
    raw=float(yards)/max(float(attempts),.1)
    return (raw*count+prior*prior_attempts)/(count+prior_attempts)

def personnel_factors(active, absent, *, strength=.5):
    """Rows are (targets, catches, receiving yards, receiving TDs, games).
    Compare observed available receiving efficiency with a full-unit prior.
    Absence weights must already reflect prior participation.
    """
    def efficiency(rows):
        targets=0.; y=0.; td=0.
        for t,c,yd,s,g in rows:
            if t<=0: continue
            targets+=t
            y+=t*shrink_efficiency(yd,t,g,7.,40.)
            td+=t*shrink_efficiency(s,t,g,.045,60.)
        return targets,y/max(targets,.001),td/max(targets,.001)
    av,ay,at=efficiency(active)
    allv,fy,ft=efficiency(list(active)+list(absent))
    if av<=0 or allv<=av: return 1.,1.
    return (float(np.clip(1+strength*(ay/max(fy,.001)-1),.80,1.15)),
            float(np.clip(1+strength*(at/max(ft,.001)-1),.75,1.15)))

def availability_masks(questionable, n, rng):
    """Balanced stress cases per player, not a medical forecast."""
    masks=np.ones((n,len(questionable)),dtype=bool)
    for j,q in enumerate(questionable):
        if q:
            order=rng.permutation(n)
            masks[order[:n//2],j]=False
    return masks

def scenario_allocation(rng, totals, base, masks, allocator, concentration=None):
    """Renormalize the existing budget among available players and OTHER.
    Group identical availability states, preserving shared scenario totals.
    """
    totals=np.asarray(totals)
    out=np.zeros((len(totals),len(base)),dtype=np.int32)
    for mask in np.unique(masks,axis=0):
        select=np.all(masks==mask,axis=1)
        weights=np.asarray(base,float).copy()
        weights[:len(mask)]*=mask
        if weights.sum()<=0: weights[-1]=1.
        out[select]=allocator(rng,totals[select],weights/weights.sum(),concentration=concentration)
    return out
