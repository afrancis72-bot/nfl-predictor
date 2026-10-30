"""PGA round Showdown portfolio builder; no invented player statistics.

Scores are transparent round-rating fantasy proxies, not DraftKings scoring predictions.
"""
import math
import numpy as np
import pandas as pd


def _key(x):
    import re
    return re.sub(r'[^a-z0-9]+',' ',str(x).casefold()).strip()


def build_showdown_pool(dk, ratings, round_no):
    d=dk.copy(); r=ratings.copy()
    d['match_key']=d.player.map(_key); r['match_key']=r.player.map(_key)
    if 'Position' in d.columns:
        positions=d.Position.astype(str).str.upper()
        d=d[positions.isin(['G','GOLF','GOLFER','UTIL','FLEX'])].copy() if positions.isin(['G','GOLF','GOLFER','UTIL','FLEX']).any() else d
    if 'Roster Position' in d.columns:
        # PGA Showdown is six golfers, not NFL-style CPT/FLEX.
        d=d[~d['Roster Position'].astype(str).str.upper().str.contains('CPT')].copy()
    x=d.merge(r[['match_key','pre_round_rating','live_form_adj','live_rounds','round_rating','course_fit_signal']],on='match_key',how='left',validate='many_to_one')
    x=x.dropna(subset=['round_rating']).copy()
    if len(x)<6: raise ValueError('Fewer than six DraftKings golfers match OTIS ratings.')
    x['salary']=pd.to_numeric(x.salary,errors='coerce')
    x=x[x.salary.between(3000,20000)].copy()
    # Strokes-to-points proxy, anchored at ~70 points with 0.055 strokes per rating point.
    # A high rating implies more birdie opportunities; not a calibrated DK distribution.
    x['showdown_proxy']=((x.round_rating-50)*0.22+55).round(2)
    x['proxy_ceiling']=(x.showdown_proxy+14).round(2)
    x['round']=int(round_no)
    return x[['player','salary','round','round_rating','pre_round_rating','live_form_adj','live_rounds','course_fit_signal','showdown_proxy','proxy_ceiling']].reset_index(drop=True)


def optimize_showdown(pool,n_lineups=4,max_exposure=.5,salary_floor=46500,seed=42):
    if len(pool)<6: raise ValueError('Need at least six golfers.')
    n_lineups=int(n_lineups); rng=np.random.default_rng(seed)
    salaries=pool.salary.to_numpy(dtype=int)
    rating=pool.showdown_proxy.to_numpy(float)
    # Generate a candidate bank with diverse strength/salary preferences.
    candidates={}
    for i in range(35000):
        exponent=0.6+(i%9)*0.12
        weight=np.exp(np.clip((rating-rating.mean())/max(3,rating.std())*exponent,-6,6))
        if i%4==0: weight*=np.clip(salaries.mean()/salaries,.5,2.0)
        prob=weight/weight.sum()
        ids=tuple(sorted(rng.choice(len(pool),size=6,replace=False,p=prob)))
        total=int(salaries[list(ids)].sum())
        if total<salary_floor or total>50000: continue
        score=float(rating[list(ids)].sum())
        if ids not in candidates or score>candidates[ids][0]: candidates[ids]=(score,total)
    if not candidates:
        raise ValueError('No legal 6-golfer combinations found. Try lowering the salary floor.')
    ordered=sorted(candidates.items(),key=lambda item:(item[1][0],item[1][1]),reverse=True)
    cap=max(1,int(math.floor(n_lineups*max_exposure+1e-9)))
    if cap*len(pool)<n_lineups*6:
        raise ValueError('Exposure cap is mathematically infeasible for this field and lineup count.')
    counts=np.zeros(len(pool),dtype=int); chosen=[]
    for _ in range(n_lineups):
        best=None; best_val=-1e9
        for ids,(score,salary) in ordered:
            if ids in chosen or any(counts[j]>=cap for j in ids): continue
            # Strong portfolio diversification without suppressing best projected players.
            overlap=max((len(set(ids)&set(old)) for old in chosen),default=0)
            value=score-1.75*overlap
            if value>best_val: best=(ids,score,salary); best_val=value
        if best is None:
            raise ValueError(f'Could construct only {len(chosen)} of {n_lineups} lineups under exposure cap; increase cap or reduce lineups.')
        ids,score,salary=best; chosen.append(ids)
        for j in ids: counts[j]+=1
    rows=[]
    for i,ids in enumerate(chosen,1):
        record={'Lineup':i,'Salary':int(salaries[list(ids)].sum()),'Round proxy total':round(float(rating[list(ids)].sum()),2)}
        for j,idx in enumerate(ids,1): record[f'G{j}']=pool.iloc[idx].player
        rows.append(record)
    exposure=pd.DataFrame({'Player':pool.player,'Lineups':counts,'Exposure %':np.round(100*counts/n_lineups,1)}).query('Lineups>0').sort_values(['Lineups','Player'],ascending=[False,True])
    return pd.DataFrame(rows),exposure.reset_index(drop=True)
