"""Conservative scoring-compatible team-to-player passing-yard reconciliation.

The game engine simulates yards for modeled eligible receivers and an explicit
accounting-only OTHER receiver. Previously, when the sum was below the QB's
observed passing-yard prior, EVERY receiver's yards were scaled upward (up to
35%). That can overcredit a high-target player if a teammate is simply omitted
from the DK player pool or excluded for limited role evidence.

Experiment: let OTHER absorb part of the unexplained passing-yard gap first,
but only on passes that were actually caught by OTHER, capped at a plausible
per-catch yard figure. The remaining correction uses the original shared
rescale, conserving quarterback/team yards exactly.
No new attempts, catches, TDs or roster-eligible players.
"""
from __future__ import annotations
import numpy as np


def reconcile_other_yards(other_yards, other_receptions,
                          generated_team_yards, desired_team_yards,
                          *, strength=0., max_other_ypr=22., min_other_ypr=3.):
    if not np.isfinite(strength) or not 0.<=strength<=1.:
        raise ValueError("OTHER receiving-yard reconciliation strength must be 0..1")
    other=np.asarray(other_yards,dtype=float)
    catches=np.asarray(other_receptions,dtype=float)
    generated=np.asarray(generated_team_yards,dtype=float)
    desired=np.asarray(desired_team_yards,dtype=float)
    if (other.shape!=catches.shape or other.shape!=generated.shape or
            generated.shape!=desired.shape or
            not all(np.isfinite(v).all() for v in (other,catches,generated,desired)) or
            (other<0).any() or (catches<0).any() or (generated<0).any()):
        raise ValueError("Reconciliation requires valid same-shape scoring vectors")
    if not 0.<=min_other_ypr<max_other_ypr<=40.:
        raise ValueError("Unsupported OTHER per-reception yard bounds")
    if strength==0.:
        return other.copy(),generated.copy(),np.zeros_like(other)
    gap=desired-generated
    upward=np.maximum(0.,max_other_ypr*catches-other)
    downward=np.maximum(0.,other-min_other_ypr*catches)
    shift=np.where(gap>=0.,
                   np.minimum(gap*strength,upward),
                   -np.minimum((-gap)*strength,downward))
    # A zero-catch OTHER receiver cannot be assigned fantasy yards, and no
    # negative receiving yards are created by the controlled adjustment.
    shift=np.where(catches>0,shift,0.)
    return other+shift,generated+shift,shift
