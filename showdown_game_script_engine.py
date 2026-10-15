"""Experimental possession-driven Showdown simulator; research use only.
Not play-by-play, not historically calibrated. Keep baseline as production default.
"""
import numpy as np
import pandas as pd

def _shares(ids, pos, sal, rng, kind):
    if not ids: return np.array([])
    p=np.array([pos[i] for i in ids])
    mult=(np.where(p=="WR",1.,np.where(p=="TE",.72,np.where(p=="RB",.42,.05))) if kind=="target"
          else np.where(p=="RB",1.,np.where(p=="QB",.23,.04)))
    w=np.maximum(.1,(np.maximum(sal[ids],200)/6000)**1.6)*mult*rng.lognormal(0,.17,len(ids))
    return w/w.sum()

def simulate_game_scripts(players,teams,n_sims=10000,seed=315,return_audit=False):
    """Simulate correlated possessions, opportunities and DK points.

    Returns an (n_sims, n_players) matrix. Requires one confirmed QB per team.
    Experimental limitations: salary-based shares, approximate possessions and DST.
    """
    if len(teams)!=2 or int(n_sims)<1: raise ValueError("Two teams and positive simulation count required")
    players=players.reset_index(drop=True)
    n=len(players); n_sims=int(n_sims)
    rng=np.random.default_rng(seed)
    team=players["TeamAbbrev"].astype(str).tolist()
    pos=players["Position"].astype(str).str.upper().tolist()
    sal=pd.to_numeric(players["FLEX Salary"],errors="coerce").fillna(0).to_numpy(float)
    ids={t:[i for i in range(n) if team[i]==t] for t in teams}
    for t in teams:
        if sum(pos[i]=="QB" for i in ids[t])!=1: raise ValueError(f"{t}: select exactly one starting QB")
    qb={t:next(i for i in ids[t] if pos[i]=="QB") for t in teams}
    receivers={t:[i for i in ids[t] if pos[i] in ("WR","TE","RB")] for t in teams}
    carriers={t:[i for i in ids[t] if pos[i] in ("RB","QB","WR")] for t in teams}
    out=np.zeros((n_sims,n),dtype=np.float32); audits=[]
    for g in range(n_sims):
        pace=float(np.clip(rng.normal(1,.085),.8,1.22))
        eff={t:float(np.clip(rng.normal(1,.16),.60,1.45)) for t in teams}
        score={t:0 for t in teams}; patt={t:0 for t in teams}; ratt={t:0 for t in teams}
        sack={t:0 for t in teams}; turnover={t:0 for t in teams}; fg={t:0 for t in teams}; td={t:0 for t in teams}
        # pass yards, pass TD, INT, rush yards, rush TD, receptions, rec yards, rec TD
        stats=np.zeros((n,8),float)
        drives=int(np.clip(round(rng.normal(22*pace,2)),17,29)); first=int(rng.integers(0,2))
        for d in range(drives):
            t=teams[(first+d)%2]; opp=teams[(first+d+1)%2]
            margin=score[t]-score[opp]; late=d>drives*.66
            ppass=float(np.clip(.56+(.17 if margin < -7 and late else 0)-(.12 if margin>10 and late else 0),.34,.81))
            plays=int(np.clip(rng.poisson(5.4*pace)+1,2,13))
            pass_plays=int(rng.binomial(plays,ppass)); rush_plays=plays-pass_plays
            nsack=int(rng.binomial(pass_plays,.058)); attempts=pass_plays-nsack
            interceptions=int(rng.binomial(attempts,.024))
            completions=int(rng.binomial(max(0,attempts-interceptions),.65))
            py=float(rng.gamma(max(1,completions)*1.4,7.2*eff[t]/1.4)) if completions else 0.
            ry=float(rng.gamma(max(1,rush_plays)*1.8,4.4*eff[t]/1.8)) if rush_plays else 0.
            patt[t]+=attempts; ratt[t]+=rush_plays; sack[t]+=nsack
            stats[qb[t],0]+=py; stats[qb[t],2]+=interceptions
            drive_catches={}; drive_rush={}
            if completions and receivers[t]:
                rec_ids=receivers[t]; counts=rng.multinomial(completions,_shares(rec_ids,pos,sal,rng,"target"))
                yards=rng.multinomial(int(round(py)),counts/counts.sum())
                for j,i in enumerate(rec_ids):
                    stats[i,5]+=counts[j];stats[i,6]+=yards[j]
                    if counts[j]:drive_catches[i]=int(counts[j])
            if rush_plays and carriers[t]:
                run_ids=carriers[t]; counts=rng.multinomial(rush_plays,_shares(run_ids,pos,sal,rng,"rush"))
                yards=rng.multinomial(int(round(ry)),counts/counts.sum())
                for j,i in enumerate(run_ids):
                    stats[i,3]+=yards[j]
                    if counts[j]:drive_rush[i]=int(counts[j])
            intercepted=bool(interceptions and rng.random()<.8)
            if intercepted:turnover[t]+=1
            pscore=float(np.clip(.19+.009*(py+ry)+.07*(eff[t]-1),.04,.78))
            if not intercepted and rng.random()<pscore:
                is_td=rng.random()<float(np.clip(.32+(py+ry)/180,.30,.76))
                if is_td:
                    passing=bool(drive_catches and rng.random()<ppass)
                    if passing:
                        options=list(drive_catches); w=np.array([drive_catches[i] for i in options],float)
                        scorer=int(rng.choice(options,p=w/w.sum()))
                        stats[scorer,7]+=1;stats[qb[t],1]+=1
                    elif drive_rush:
                        options=list(drive_rush);w=np.array([drive_rush[i] for i in options],float)
                        scorer=int(rng.choice(options,p=w/w.sum()));stats[scorer,4]+=1
                    # A touchdown by an unlisted player remains a team TD only.
                    score[t]+=7;td[t]+=1
                else:fg[t]+=1;score[t]+=3
        for i in range(n):
            t=team[i];opp=teams[1] if t==teams[0] else teams[0]
            py,pt,inter,ry,rt,rec,recy,rect=stats[i]
            if pos[i]=="QB":
                val=.04*py+4*pt-inter+.1*ry+6*rt+3*(py>=300)+3*(ry>=100)
            elif pos[i] in ("RB","WR","TE"):
                val=.1*ry+6*rt+rec+.1*recy+6*rect+3*(ry>=100)+3*(recy>=100)
            elif pos[i]=="K":val=3*fg[t]+td[t]
            elif pos[i] in ("DST","D"):
                pa=score[opp]; bonus=10 if pa==0 else 7 if pa<=6 else 4 if pa<=13 else 1 if pa<=20 else 0 if pa<=27 else -1 if pa<=34 else -4
                val=bonus+2*turnover[opp]+sack[opp]
            else:val=0
            out[g,i]=float(val) if pos[i] in ("DST","D") else max(0,float(val))
        if return_audit:
            audits.append({"sim":g,"score_a":score[teams[0]],"score_b":score[teams[1]],
                           "pass_attempts_a":patt[teams[0]],"pass_attempts_b":patt[teams[1]],
                           "rush_attempts_a":ratt[teams[0]],"rush_attempts_b":ratt[teams[1]],
                           "td_a":td[teams[0]],"td_b":td[teams[1]]})
    return (out,pd.DataFrame(audits)) if return_audit else out
