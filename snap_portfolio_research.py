"""Research-only diversified Showdown portfolio construction.

Uses simulated player outcomes and salary constraints. Does not estimate contest ROI,
ownership, or real-world win probabilities.
"""
import numpy as np
import pandas as pd

def build_research_portfolio(pool, points, count=5, seed=316, min_salary=45000,
                             min_flex_salary=2000, max_shared=4, trials=30000):
    p=pool.reset_index(drop=True)
    sal=pd.to_numeric(p["FLEX Salary"],errors="coerce").fillna(0).to_numpy(dtype=int)
    teams=p["TeamAbbrev"].to_numpy()
    pos=p["Position"].to_numpy()
    n=len(p)
    if points.ndim!=2 or points.shape[1]!=n: raise ValueError("Simulation/player dimension mismatch")
    # Historical opportunity is required; QBs qualify via confirmed starter selection.
    opportunity=(p["rush_weight"].fillna(0).to_numpy(dtype=float)+
                 p["target_weight"].fillna(0).to_numpy(dtype=float))
    eligible=np.where((sal>=min_flex_salary)&
                      ((pos=="QB")|(opportunity>=0.75)))[0]
    if len(eligible)<6: raise ValueError("Fewer than six opportunity-qualified players")
    rng=np.random.default_rng(seed)
    mean=points.mean(axis=0)
    p90=np.quantile(points,.9,axis=0)
    upside=np.maximum(p90,0)
    candidates={}
    for trial in range(trials):
        # Sample game-specific outcomes to explore distinct winning game scripts.
        game=int(rng.integers(points.shape[0]))
        score=.55*points[game]+.25*mean+.20*upside
        if trial%3==0: score=.70*mean+.30*upside
        if trial%3==1: score=.25*mean+.75*upside
        # Vary captain and flex choices; never use salary as opportunity.
        cap_weights=np.maximum(score[eligible],.25)**2
        cap=int(rng.choice(eligible,p=cap_weights/cap_weights.sum()))
        budget=50000-int(round(sal[cap]*1.5))
        flex_ids=eligible[eligible!=cap]
        flex_scores=np.maximum(score[flex_ids],.1)
        flex_weights=flex_scores**2
        chosen=rng.choice(flex_ids,size=5,replace=False,p=flex_weights/flex_weights.sum())
        selected=np.concatenate(([cap],chosen))
        spent=50000-budget+int(sal[chosen].sum())
        if not min_salary<=spent<=50000: continue
        if len(set(teams[selected]))!=2: continue
        key=(cap,tuple(sorted(map(int,chosen))))
        if key in candidates: continue
        realized=points[:,cap]*1.5+points[:,chosen].sum(axis=1)
        # Mean and upper tail, with a moderate diversification incentive.
        value=float(.65*realized.mean()+.35*np.quantile(realized,.9))
        candidates[key]=(value,int(spent),tuple(map(int,selected)))
    ranked=sorted(candidates.values(),key=lambda x:x[0],reverse=True)
    selected_lineups=[]
    for cap_limit in (1,2,5):
        for shared_limit in (max_shared,5):
            chosen=list(selected_lineups)
            for value,cost,ids in ranked:
                if any(ids==old[2] for old in chosen):continue
                if sum(1 for old in chosen if old[2][0]==ids[0])>=cap_limit:continue
                if any(len(set(ids)&set(old[2]))>shared_limit for old in chosen):continue
                chosen.append((value,cost,ids))
                if len(chosen)>=count:break
            if len(chosen)>=count:
                selected_lineups=chosen[:count]
                break
        if len(selected_lineups)>=count:break
        if len(chosen)>len(selected_lineups):selected_lineups=chosen
    rows=[]
    for value,cost,ids in selected_lineups:
        rows.append({"CPT":p.iloc[ids[0]]["Name"],
                     **{f"FLEX{i}":p.iloc[j]["Name"] for i,j in enumerate(ids[1:],1)},
                     "Salary":cost,"Portfolio research score":round(value,2),
                     "Simulated mean DK":round(float((1.5*points[:,ids[0]]+points[:,list(ids[1:])].sum(axis=1)).mean()),2)})
    return pd.DataFrame(rows)
