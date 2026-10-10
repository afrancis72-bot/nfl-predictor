"""Pre-simulation injury opportunity redistribution.

Confirmed-OUT opportunity is reallocated across the remaining modeled players
and the genuine OTHER bucket in proportion to their existing opportunity shares.
The function conserves total opportunity and never creates fantasy points directly.
"""
import numpy as np

def redistribute(weights, other, removed_indices=(), missing_volume=0., role_weights=None):
    """Redistribute confirmed-OUT opportunity while conserving the opportunity budget.

    Parameters
    ----------
    weights : array-like
        Baseline target/carry rates for modeled active players.
    other : float
        Genuine unmodeled/OTHER baseline opportunity.
    removed_indices : iterable[int]
        Optional indices of players still present in ``weights`` who are OUT.
        Their baseline opportunity is zeroed and added to the amount redistributed.
    missing_volume : float
        Historical opportunity belonging to explicitly OUT players that have already
        been removed from the modeled player array.
    role_weights : array-like, optional
        Multipliers for recipient suitability. Defaults to 1.0, which means the
        missing opportunity follows each remaining player's normal opportunity share.

    Returns
    -------
    (new_weights, new_other)
        Redistributed active-player and OTHER opportunity. Their sum equals the
        original active + OTHER + confirmed-OUT opportunity budget.
    """
    w=np.maximum(np.asarray(weights,dtype=float),0.0).copy()
    other=max(float(other),0.0)
    missing=max(float(missing_volume),0.0)

    removed=set(int(i) for i in removed_indices)
    freed=missing
    for i in removed:
        if i<0 or i>=len(w):
            raise ValueError('invalid injury index')
        freed+=w[i]
        w[i]=0.0

    eligible=np.ones(len(w),dtype=bool)
    for i in removed:
        eligible[i]=False

    if role_weights is not None:
        role=np.maximum(np.asarray(role_weights,dtype=float),0.0)
        if len(role)!=len(w):
            raise ValueError('role weight length mismatch')
    else:
        role=np.ones(len(w),dtype=float)

    # Recipient shares are based on NORMAL remaining opportunity. OTHER participates
    # as its own baseline bucket; it is not required to "contain" the injured usage.
    player_basis=w*role*eligible
    other_basis=other
    recipient_total=float(player_basis.sum()+other_basis)

    if freed>0 and recipient_total<=0:
        raise ValueError('No available players or OTHER to receive redistributed volume')

    before=float(w.sum()+other+freed)
    if freed>0:
        w += freed*(player_basis/recipient_total)
        other += freed*(other_basis/recipient_total)

    after=float(w.sum()+other)
    if not np.isclose(before,after,rtol=1e-10,atol=1e-8):
        raise ValueError(f'Opportunity conservation failed: before={before:.8f}, after={after:.8f}')

    return w,other
