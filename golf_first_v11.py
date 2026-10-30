"""PGA V11 experimental golf-first DraftKings Classic simulator.

Pre-tournament only. Models 72 actual hole outcomes per golfer, then awards
DraftKings points. Uses a documented generic par-71 hole template because
verified hole-by-hole course inputs were not in the supplied projections.
No-cut is explicit for Baycurrent; other events must specify cut policy.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import logsumexp

# DK Classic, not Pick6 single-round scoring.
HOLE_DK = np.array([13., 8., 3., .5, -.5, -1., -1.])
REL = np.array([-3, -2, -1, 0, 1, 2, 3])
FINISH = [(1,30),(2,20),(3,18),(4,16),(5,14),(6,12),(7,10),(8,9),(9,8),(10,7),
          (15,6),(20,5),(25,4),(30,3),(40,2),(50,1)]

def simulate_golf_first(projections:pd.DataFrame, sims:int=5000, seed:int=42,
                        course_par:int=71, no_cut:bool=True, cut_size:int=65,
                        hole_pars:list[int]|None=None,
                        round_field_mean_to_par:float=-1.5):
    """Return player summary, full DK outcomes (sim x player), and diagnostics.

    Assumptions: 'expected_round' in V10 is field-centered, NOT absolute strokes.
    Generic hole pars are provisional and must be replaced with verified course holes.
    Baseline probabilities are illustrative, NOT fitted to real hole-level data.
    """
    if sims < 100 or sims > 100000: raise ValueError('sims must be 100..100000')
    if not no_cut and cut_size < 1: raise ValueError('cut_size must be positive')
    d=projections.copy().reset_index(drop=True)
    for c in ('player','salary','expected_round','round_sd'):
        if c not in d: raise ValueError(f'Missing {c}')
    if d.player.duplicated().any(): raise ValueError('Duplicate golfers')
    n=len(d)
    if hole_pars is None:
        # Generic hole layout, provisional until verified pars are uploaded.
        n3,n5=4,3
        n4=18-n3-n5
        if 3*n3+4*n4+5*n5!=course_par:
            n5=course_par-72+n3
            n4=18-n3-n5
        if min(n3,n4,n5)<0: raise ValueError('Provide hole pars for unusual course par')
        hole_pars=[4]*n4+[3]*n3+[5]*n5
    hp=np.asarray(hole_pars,int)
    if len(hp)!=18 or hp.sum()!=course_par: raise ValueError('18 hole pars must total course par')
    rng=np.random.default_rng(seed)
    # Outcome categories: albatross, eagle, birdie, par, bogey, double, triple+.
    base={3:[.00001,.0002,.105,.715,.155,.022,.00279],
          4:[.00001,.008,.235,.580,.147,.026,.00499],
          5:[.0001,.055,.405,.425,.098,.014,.0029]}
    # Re-center probabilities so average modeled golfer matches the supplied
    # baseline field scoring; then golfer-specific tilts target expected_round.
    def expected_rel(logits):
        z=np.exp(logits-logsumexp(logits,axis=-1,keepdims=True))
        return (z*REL).sum(axis=-1)
    raw=np.stack([np.log(np.array(base[int(par)])) for par in hp])
    target=round_field_mean_to_par/18
    lo,hi=-4.,4.
    for _ in range(55):
        mid=(lo+hi)/2
        val=expected_rel(raw-mid*REL).sum()
        if val>round_field_mean_to_par: lo=mid
        else: hi=mid
    common=(lo+hi)/2
    birdie_proxy=pd.to_numeric(d.get('otis_fit_app',pd.Series(50.,index=d.index)),errors='coerce').fillna(50.)
    birdie_proxy=(birdie_proxy-birdie_proxy.mean())/max(birdie_proxy.std(ddof=0),1)
    # A bounded extra birdie-vs-par volatility component; uncalibrated.
    birdie_bias=np.clip(.10*birdie_proxy.to_numpy(),-.2,.2)
    golfer_mu=pd.to_numeric(d.expected_round,errors='coerce').fillna(0).to_numpy()+round_field_mean_to_par
    golfer_sd=pd.to_numeric(d.round_sd,errors='coerce').fillna(2.7).clip(1.7,4.5).to_numpy()
    # Solve each golfer's probability tilt against desired strokes to par.
    shifts=np.zeros(n)
    for i in range(n):
        left,right=-5.,5.
        for _ in range(38):
            m=(left+right)/2
            logit=raw-common*REL-m*REL
            logit[:,2]+=birdie_bias[i]
            logit[:,3]-=birdie_bias[i]*.5
            if expected_rel(logit).sum()>golfer_mu[i]:left=m
            else:right=m
        shifts[i]=(left+right)/2
    # Scenarios preserve shared day effects and golfer-specific tournament form.
    scores=np.zeros((sims,n),np.float32)
    round_dk=np.zeros((sims,n,4),np.float32)
    total_to_par=np.zeros((sims,n),np.int16)
    round_strokes=np.zeros((sims,n,4),np.int16)
    birdies=np.zeros((sims,n),np.int16)
    eagles=np.zeros((sims,n),np.int16)
    bogeys=np.zeros((sims,n),np.int16)
    streaks=np.zeros((sims,n),np.int8)
    bogeyfree=np.zeros((sims,n),np.int8)
    day_shock=rng.normal(0,.36,size=(sims,4))
    latent=rng.normal(0,.27,size=(sims,n))
    # Use chunks to limit peak memory. One golfer and day at a time gives
    # reproducible outcomes and avoids allocating a sims*n*72*7 tensor.
    for j in range(n):
        for r in range(4):
            logits=raw[None,:,:]-(common+shifts[j]+latent[:,j,None]+day_shock[:,r,None])[:,:,None]*REL[None,None,:]
            logits=np.broadcast_to(logits,(sims,18,7)).copy()
            logits[:,:,2]+=birdie_bias[j]
            logits[:,:,3]-=birdie_bias[j]*.5
            probs=np.exp(logits-logsumexp(logits,axis=2,keepdims=True))
            u=rng.random((sims,18))
            outcomes=(u[:,:,None]>np.cumsum(probs,axis=2)).sum(axis=2).clip(0,6)
            rel=REL[outcomes]
            round_strokes[:,j,r]=course_par+rel.sum(axis=1)
            total_to_par[:,j]+=rel.sum(axis=1)
            round_dk[:,j,r]+=HOLE_DK[outcomes].sum(axis=1)
            # A par-3 eagle is a hole-in-one: additional +5 DK Classic bonus.
            round_dk[:,j,r]+=5*((hp[None,:]==3)&(outcomes==1)).sum(axis=1)
            birdies[:,j]+=(outcomes==2).sum(axis=1)
            eagles[:,j]+=(outcomes<=1).sum(axis=1)
            bogeys[:,j]+=(outcomes>=4).sum(axis=1)
            hot=(outcomes<=2)
            has_streak=(hot[:,:-2]&hot[:,1:-1]&hot[:,2:]).any(axis=1)
            round_dk[:,j,r]+=3*has_streak
            streaks[:,j]+=has_streak.astype(np.int8)
            free=(outcomes<4).all(axis=1)
            round_dk[:,j,r]+=3*free
            bogeyfree[:,j]+=free.astype(np.int8)
        scores[:,j]+=5*(round_strokes[:,j,:]<70).all(axis=1)
    if no_cut:
        made_cut=np.ones((sims,n),dtype=bool)
    else:
        two_rounds=round_strokes[:,:,:2].sum(axis=2)
        k=min(n,int(cut_size))
        threshold=np.partition(two_rounds,k-1,axis=1)[:,k-1]
        made_cut=two_rounds<=threshold[:,None]  # ties at cutline make the cut
        scores[~made_cut]=0.0  # no four-round bonus for missed cuts
    scores+=round_dk[:,:,:2].sum(axis=2)
    scores+=np.where(made_cut,round_dk[:,:,2:].sum(axis=2),0.)
    # Competition ranks (ties share finish points based on tied place).
    order=np.argsort(np.where(made_cut,total_to_par,9999),axis=1,kind='stable')
    ranks=np.empty((sims,n),dtype=np.int16)
    rows=np.arange(sims)[:,None]
    ranks[rows,order]=np.arange(1,n+1,dtype=np.int16)
    # Tournament finishing ties: rank = first tied position.
    sorted_scores=np.take_along_axis(np.where(made_cut,total_to_par,9999),order,axis=1)
    first=np.ones_like(sorted_scores,dtype=bool)
    first[:,1:]=sorted_scores[:,1:]!=sorted_scores[:,:-1]
    start=np.maximum.accumulate(np.where(first,np.arange(1,n+1)[None,:],0),axis=1)
    ranks[rows,order]=start
    finish_pts=np.zeros((sims,n),np.float32)
    lower=0
    for upper,points in FINISH:
        finish_pts[(ranks>lower)&(ranks<=upper)]=points
        lower=upper
    scores+=np.where(made_cut,finish_pts,0.)
    d['v11_mean_dk']=scores.mean(axis=0)
    d['v11_median_dk']=np.median(scores,axis=0)
    d['v11_p75_dk']=np.percentile(scores,75,axis=0)
    d['v11_p90_dk']=np.percentile(scores,90,axis=0)
    d['v11_p95_dk']=np.percentile(scores,95,axis=0)
    d['v11_sd_dk']=scores.std(axis=0)
    d['v11_win_pct']=(ranks==1).mean(axis=0)*100
    d['v11_top10_pct']=(ranks<=10).mean(axis=0)*100
    d['v11_make_cut_pct']=made_cut.mean(axis=0)*100.
    d['v11_avg_birdies']=birdies.mean(axis=0)
    d['v11_avg_eagles']=eagles.mean(axis=0)
    d['v11_avg_bogeys']=bogeys.mean(axis=0)
    d['v11_avg_strokes']=round_strokes.mean(axis=(0,2))
    d['v11_avg_streak_rounds']=streaks.mean(axis=0)
    d['v11_avg_bogeyfree_rounds']=bogeyfree.mean(axis=0)
    d['v11_salary_value']=d.v11_mean_dk/(pd.to_numeric(d.salary)/1000)
    return d,scores,{'sims':sims,'no_cut':bool(no_cut),'cut_size':None if no_cut else int(cut_size),'par':course_par,
                     'field_mean_to_par_assumption':round_field_mean_to_par,
                     'hole_pars_source':'provided course hole pars' if hole_pars is not None else 'generic provisional template',
                     'probability_source':'illustrative baseline, not calibrated to Yokohama hole-level stats',
                     'golfer_count':n}
