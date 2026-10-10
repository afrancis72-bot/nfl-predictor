"""Conservative pre-simulation injury redistribution; no sportsbook/injury-feed claims."""
import numpy as np

def redistribute(weights, other, removed_indices=(), missing_volume=0., role_weights=None):
    """Conserve existing opportunity. Missing off-pool usage is taken from OTHER first.

    weights: eligible-player baseline target/carry rates; other: baseline OTHER.
    removed_indices: eligible players ruled OUT; missing_volume: historical usage of
    explicitly OUT players not in eligible pool (already contained in OTHER).
    role_weights: optional eligibility multipliers; defaults to proportional shares.
    """
    w=np.maximum(np.asarray(weights,dtype=float),0).copy()
    other=max(float(other),0.)
    missing=max(float(missing_volume),0.)
    if missing>other+1e-8:
        raise ValueError(f'OUT player usage {missing:.2f} exceeds OTHER {other:.2f}; cannot safely redistribute')
    removed=set(int(i) for i in removed_indices)
    freed=missing
    other-=missing
    for i in removed:
        if i<0 or i>=len(w): raise ValueError('invalid injury index')
        freed+=w[i]; w[i]=0.
    eligible=np.ones(len(w),dtype=bool)
    for i in removed: eligible[i]=False
    if role_weights is not None:
        role=np.asarray(role_weights,dtype=float)
        if len(role)!=len(w): raise ValueError('role weight length mismatch')
        role=np.maximum(role,0)
    else: role=np.ones(len(w))
    shares=w*role*eligible
    total=shares.sum()+other
    if freed>0 and total<=0: raise ValueError('No available players or OTHER to receive redistributed volume')
    if total>0:
        w+=freed*shares/total
        other+=freed*other/total
    return w,other
