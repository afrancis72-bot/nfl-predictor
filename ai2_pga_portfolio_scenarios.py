"""Joint-world PGA DFS portfolio selection and risk reporting.

All probabilities are model estimates from one tournament simulation, not
historically calibrated or market-implied payout probabilities. No actual
contest ROI is inferred from simulated scores without a payout distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned_worlds(players, draws, cuts):
    names=[str(x) for x in players]
    if len(names)!=len(set(names)):
        raise ValueError("PGA same-world audit cannot align duplicate golfer names.")
    scores=np.asarray(draws,dtype=np.float32)
    raw_cut=np.asarray(cuts)
    if (scores.ndim!=2 or scores.shape[1]!=len(names)
            or scores.shape[0]<50 or raw_cut.shape!=scores.shape):
        raise ValueError("PGA player scores/cut worlds are missing or not aligned.")
    if not np.isfinite(scores).all():
        raise ValueError("PGA player scenario scores contain nonfinite values.")
    if not np.issubdtype(raw_cut.dtype,np.bool_):
        raise ValueError("PGA cut event scenarios must be boolean.")
    return names,scores,raw_cut


def audit_portfolio_worlds(portfolio,predictions,draws,cuts,*,strategy=None):
    """Audit individual and JOINT lineups in the same simulated tournament worlds.

    'at least one 6/6' uses the union of simulated cut events across lineups,
    never the sum of independent golfer or lineup cut probabilities.
    """
    names,draws,cuts=_aligned_worlds(predictions["player"],draws,cuts)
    ids={name:i for i,name in enumerate(names)}
    if "lineup" not in portfolio or "player" not in portfolio or portfolio.empty:
        raise ValueError("PGA joint portfolio audit needs six golfers per lineup.")
    score_columns=[]
    cut_columns=[]
    summaries=[]
    for no,g in portfolio.groupby("lineup",sort=True):
        picked=g["player"].astype(str).tolist()
        if len(picked)!=6 or len(set(picked))!=6 or any(x not in ids for x in picked):
            raise ValueError("PGA lineup contains missing or duplicate golfers.")
        index=[ids[name] for name in picked]
        ls=draws[:,index].sum(axis=1)
        cut_event=np.all(cuts[:,index],axis=1)
        score_columns.append(ls)
        cut_columns.append(cut_event)
        summaries.append({
            "lineup":int(no),
            "sim_lineup_mean":round(float(ls.mean()),2),
            "sim_lineup_p10":round(float(np.quantile(ls,.10)),2),
            "sim_lineup_p90":round(float(np.quantile(ls,.90)),2),
            "sim_lineup_p95":round(float(np.quantile(ls,.95)),2),
            "sim_lineup_p99":round(float(np.quantile(ls,.99)),2),
            "sim_lineup_6of6_pct":round(float(cut_event.mean())*100,2),
        })
    per_world_scores=np.stack(score_columns,axis=1)
    per_world_cut=np.stack(cut_columns,axis=1)
    best=per_world_scores.max(axis=1)
    count6=per_world_cut.sum(axis=1)
    overall={
        "strategy":str(strategy or "not specified"),
        "lineup_count":int(per_world_scores.shape[1]),
        "simulation_worlds":int(len(best)),
        "sim_portfolio_at_least_one_6of6_pct":round(float((count6>=1).mean())*100,2),
        "sim_portfolio_expected_6of6_count":round(float(count6.mean()),3),
        "sim_portfolio_best_score_mean":round(float(best.mean()),2),
        "sim_portfolio_best_score_p50":round(float(np.quantile(best,.50)),2),
        "sim_portfolio_best_score_p90":round(float(np.quantile(best,.90)),2),
        "sim_portfolio_best_score_p95":round(float(np.quantile(best,.95)),2),
        "sim_portfolio_best_score_p99":round(float(np.quantile(best,.99)),2),
        "sim_portfolio_avg_lineup_mean":round(float(per_world_scores.mean()),2),
    }
    return pd.DataFrame(summaries),overall


def choose_joint_scenario_portfolio(
        ranked,players,draws,cuts,*,lineup_count,max_appearances,min_unique,
        roster_size,strategy,seed):
    """Choose a constrained portfolio using marginal *joint scenario* value.

    Cut Equity: cover additional six-for-six worlds (not simply high individual
    cut rates). GPP Ceiling: increase upside of the highest-scoring entry in the
    portfolio. Balanced: consider mean, cut coverage and upper-score outcomes.

    Candidate construction remains in the existing salary/role-aware optimizer.
    We cap scenario-evaluated candidates for predictable memory/runtime, while
    including deterministic diverse tail samples to avoid a top-score bottleneck.
    If the capped pool cannot satisfy every hard constraint, refuse export.
    """
    names,draws,cuts=_aligned_worlds(players,draws,cuts)
    if not ranked:
        raise ValueError("No legal PGA lineup candidates were generated.")
    if not 1<=int(lineup_count)<=30:
        raise ValueError("Invalid PGA lineup count.")
    lookup={name:i for i,name in enumerate(names)}
    cap=min(len(ranked),max(1600,int(lineup_count)*185))
    chosen_candidates=list(ranked[:cap])
    if len(ranked)>cap:
        rng=np.random.default_rng(int(seed)+4177)
        rest=np.arange(cap,len(ranked))
        chosen_candidates.extend(ranked[int(i)] for i in
                                 rng.choice(rest,size=min(550,len(rest)),replace=False))
    candidate_sets=[item[2] for item in chosen_candidates]
    n=len(candidate_sets)
    nworld=draws.shape[0]
    world_scores=np.empty((n,nworld),dtype=np.float32)
    world_cuts=np.empty((n,nworld),dtype=np.bool_)
    for i,playerset in enumerate(candidate_sets):
        if len(playerset)!=roster_size or not playerset.issubset(lookup):
            raise ValueError("PGA candidate lineup missing players or roster spots.")
        indices=[lookup[p] for p in sorted(playerset)]
        world_scores[i]=draws[:,indices].sum(axis=1)
        world_cuts[i]=cuts[:,indices].all(axis=1)

    mean=world_scores.mean(axis=1)
    p90=np.quantile(world_scores,.90,axis=1)
    p99=np.quantile(world_scores,.99,axis=1)
    cut_rate=world_cuts.mean(axis=1)
    threshold=float(np.median(p90))
    def z(a,floor=1.):
        return (a-np.median(a))/max(float(np.std(a)),floor)
    mean_z=z(mean,3.)
    cut_z=z(cut_rate,0.01)
    p99_z=z(p99,8.)
    selected=[]
    used=np.zeros(n,dtype=bool)
    exposure={name:0 for name in names}
    max_overlap=int(roster_size)-int(min_unique)
    covered=np.zeros(nworld,dtype=bool)
    best_world=None
    objective=str(strategy).lower()

    for slot in range(int(lineup_count)):
        legal=np.ones(n,dtype=bool)
        for j,playerset in enumerate(candidate_sets):
            if (used[j] or any(exposure[name]>=max_appearances for name in playerset)
                    or any(len(playerset & old)>max_overlap for old in selected)):
                legal[j]=False
        if not legal.any():
            raise RuntimeError(
                f"Scenario portfolio could build only {len(selected)}/{lineup_count} legal lineups. "
                "Lower min-unique, relax exposure, or increase available golfers; "
                "no incomplete portfolio will be exported."
            )
        # Vectorize the same-world marginal value of each potential addition.
        new_cut_worlds=world_cuts & ~covered[None,:]
        added_cut_pct=100*new_cut_worlds.mean(axis=1)
        if best_world is None:
            added_best=mean-float(np.median(mean))
            added_tail=np.maximum(world_scores-threshold,0).mean(axis=1)
        else:
            new_best=np.maximum(world_scores,best_world[None,:])
            added_best=new_best.mean(axis=1)-float(best_world.mean())
            added_tail=(np.maximum(new_best-threshold,0).mean(axis=1)-
                        float(np.maximum(best_world-threshold,0).mean()))
        # These weights are explicitly exploratory portfolio preferences, not
        # calibrated financial utility or guarantees of future contest results.
        if "cut" in objective:
            utility=(1.20*added_cut_pct + .90*cut_z + .20*mean_z +
                     .035*added_best)
        elif "gpp" in objective or "ceiling" in objective:
            utility=(.92*added_tail + .18*added_best +
                     .95*p99_z + .18*mean_z)
        else:
            utility=(.40*added_cut_pct + .17*added_best + .25*added_tail +
                     .30*cut_z + .38*mean_z)
        penalty=np.asarray([
            sum((exposure[p]/max_appearances)**2 for p in playerset)
            for playerset in candidate_sets
        ],dtype=float)
        utility=utility-.16*penalty
        utility[~legal]=-np.inf
        selected_index=int(np.argmax(utility))
        selected_set=candidate_sets[selected_index]
        selected.append(selected_set)
        used[selected_index]=True
        for p in selected_set:
            exposure[p]+=1
        covered|=world_cuts[selected_index]
        best_world=(world_scores[selected_index].copy() if best_world is None
                    else np.maximum(best_world,world_scores[selected_index]))
    return selected
