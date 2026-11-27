"""Supplemental MILP candidate search, adapted from user-supplied Claude
optimizer.py (Nov 23). Use Team AI2's real scenario matrix, never Claude's
synthetic or split-normal scores. Existing portfolio selection remains final.
"""
from collections import Counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from ai2_nfl_lineup import check_classic_lineup
from ai2_nfl_matchups import canonical_team
from ai2_nfl_correlation_guard import _opponent
from ai2_nfl_role_evidence import player_opportunity_evidence

SLOTS=(('QB',('QB',),1),('RB',('RB',),2),('WR',('WR',),3),
       ('TE',('TE',),1),('FLEX',('RB','WR','TE'),1),('DST',('DST',),1))

def optimize_slot_lineup(pool,score,*,previous=(),time_limit=.8):
    """Adapt Claude's player-by-slot binary variables and linear constraints."""
    if not pool: return None
    pos=np.array([p['Position'] for p in pool])
    team=np.array([canonical_team(p.get('TeamAbbrev','')) for p in pool])
    opp=np.array([_opponent(p) or '' for p in pool])
    sal=np.array([float(p['Salary']) for p in pool])
    pairs=[(i,j) for j,(_,eligible,_) in enumerate(SLOTS) for i in range(len(pool)) if pos[i] in eligible]
    if not pairs or not np.isfinite(score).all(): return None
    Y=np.zeros((len(pool),len(pairs)))
    for k,(i,j) in enumerate(pairs): Y[i,k]=1
    rows=[]; lo=[]; hi=[]
    def add(row,low,high):
        rows.append(row);lo.append(low);hi.append(high)
    for j,(_,_,count) in enumerate(SLOTS):
        add(np.array([int(s==j) for _,s in pairs]),count,count)
    for i in range(len(pool)): add(Y[i],0,1)
    add(sal@Y,49600,50000)
    for t in set(team[pos=='RB']): add(Y[(team==t)&(pos=='RB')].sum(0),0,1)
    for i in np.flatnonzero(pos=='QB'):
        mates=(team==team[i])&np.isin(pos,['WR','TE'])
        add(Y[mates].sum(0)-Y[i],0,np.inf)
    for i in np.flatnonzero(np.isin(pos,['QB','RB','WR','TE'])):
        for d in np.flatnonzero((pos=='DST')&((team==opp[i])|(opp==team[i]))):
            if team[d]!=team[i]: add(Y[i]+Y[d],0,1)
    for prev in previous: add(Y[list(prev)].sum(0),0,8)
    result=milp(c=-(np.asarray(score,float)@Y),integrality=np.ones(len(pairs)),
                bounds=Bounds(0,1),constraints=LinearConstraint(np.vstack(rows),lo,hi),
                options={'time_limit':float(time_limit),'mip_rel_gap':.02})
    if result.x is None: return None
    indices=np.flatnonzero(Y@result.x>.5).tolist()
    lineup=[pool[i] for i in indices]
    if check_classic_lineup(lineup,salary_floor=49600,salary_cap=50000): return None
    if any(len(set(str(p['ID']) for p in lineup)&set(str(pool[i]['ID']) for i in prev))>8 for prev in previous): return None
    qb=next(p for p in lineup if p['Position']=='QB')
    if not any(p['Position'] in ('WR','TE') and canonical_team(p['TeamAbbrev'])==canonical_team(qb['TeamAbbrev']) for p in lineup): return None
    return indices

def supplemental_candidates(prepared,candidates,players,selection_draws,*,seed,n_worlds=12):
    ids=players.ID.astype(str).tolist()
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate scenario IDs')
    idx={ident:i for i,ident in enumerate(ids)}
    pool=[p for p in prepared if str(p.get('ID')) in idx and
          p.get('opportunity_verified') is True and player_opportunity_evidence(p)[0] and
          (p['Position']!='WR' or float(p['Salary'])>=4000) and
          (p['Position']!='RB' or float(p['Salary'])>=4500)]
    x=np.asarray(selection_draws)
    if x.ndim!=2 or x.shape[1]!=len(ids) or not np.isfinite(x).all(): raise ValueError('Invalid candidate simulation matrix')
    cols=[idx[str(p['ID'])] for p in pool]
    seen={tuple(sorted(str(p['ID']) for p in c['players'])) for c in candidates}
    combined=list(candidates); prev=[]; solved=0
    rng=np.random.default_rng(seed)
    worlds=rng.choice(len(x),size=min(n_worlds,len(x)),replace=False)
    scores=[x[:,cols].mean(0)]+[x[w,cols] for w in worlds]
    for score in scores:
        chosen=optimize_slot_lineup(pool,score,previous=prev)
        if chosen is None: continue
        prev.append(chosen);solved+=1
        lineup=[pool[i] for i in chosen]
        sig=tuple(sorted(str(p['ID']) for p in lineup))
        if sig in seen: continue
        seen.add(sig)
        combined.append(dict(candidate_origin="CLAUDE_MIP",players=lineup,projected_points=round(sum(float(p['proj']) for p in lineup),2),
                             salary=int(sum(float(p['Salary']) for p in lineup))))
    return combined,dict(enabled=True,method='Claude player-slot MILP adapted to Team AI2 coherent draws',
                         attempted=len(scores),feasible=solved,added_candidates=len(combined)-len(candidates),
                         original_candidates=len(candidates),final_portfolio_constraints_unchanged=True)
