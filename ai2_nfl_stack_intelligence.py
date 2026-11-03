"""Same-world QB/pass-catcher upper-tail association for NFL Classic.

The score measures the extra frequency of a QB and receiver both clearing
their simulated 80th percentile versus independent outcomes. It is not
historical hit rate, contest cash probability, or ownership-adjusted ROI.
"""
from math import isfinite
import numpy as np
from ai2_nfl_qb_audit import qb_and_partners


def score_stack_lift(roster, index, draws, *, percentile=80.):
    qb, partners = qb_and_partners(roster)
    if qb is None or not partners:
        return 0.
    draws = np.asarray(draws)
    if draws.ndim != 2 or draws.shape[0] < 100:
        raise ValueError("Verified same-world simulation required for stack lift")
    qkey = str(qb["ID"])
    if qkey not in index:
        return 0.
    q = draws[:,index[qkey]]
    if float(np.std(q)) < .001:
        return 0.
    q_high = q >= np.percentile(q,percentile)
    lifts = []
    for p in partners:
        key=str(p["ID"])
        if key not in index:
            continue
        v=draws[:,index[key]]
        if float(np.std(v)) < .001:
            continue
        p_high=v>=np.percentile(v,percentile)
        pq=float(q_high.mean())
        pp=float(p_high.mean())
        joint=float(np.mean(q_high & p_high))
        possible=pq*(1.-pp)
        if possible <= .0001:
            continue
        lifts.append(float(np.clip((joint-pq*pp)/possible,0.,1.)))
    return round(max(lifts,default=0.),4)
