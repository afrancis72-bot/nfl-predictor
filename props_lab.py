"""Opt-in football-stat prop simulator; does not alter existing DFS simulations.
Inputs are observed per-game rates, NOT DraftKings fantasy projections.
"""
import json
from urllib.request import Request, urlopen
from urllib.parse import urlencode
import numpy as np
import pandas as pd

MARKETS={'player_pass_yds':'passing_yards','player_rush_yds':'rushing_yards','player_reception_yds':'receiving_yards','player_receptions':'receptions','player_pass_completions':'completions','player_pass_attempts':'pass_attempts','player_rush_attempts':'carries'}
FIELDS=['player','team','game','position','games_played','pass_attempts_pg','pass_yards_pg','completions_pg','carries_pg','rushing_yards_pg','targets_pg','receptions_pg','receiving_yards_pg']

def standardize(frame):
    d=frame.copy(); rename={'Name':'player','TeamAbbrev':'team','Position':'position','Name + ID':'player','game':'game'}
    d=d.rename(columns={k:v for k,v in rename.items() if k in d.columns and v not in d.columns})
    for k in FIELDS:
        if k not in d: d[k]='' if k in ['player','team','game','position'] else np.nan
    d=d[FIELDS].copy()
    for k in FIELDS[4:]: d[k]=pd.to_numeric(d[k],errors='coerce')
    d['position']=d.position.astype(str).str.upper().str.strip()
    d['player']=d.player.astype(str).str.replace(r'\s*\(\d+\)$','',regex=True).str.strip()
    d['team']=d.team.astype(str).str.strip().str.upper()
    d=d[d.position.isin(['QB','RB','WR','TE']) & (d.player!='') & (d.player!='nan')].drop_duplicates(['player','team'])
    return d.reset_index(drop=True)

def simulate(frame, n=10000, seed=20261009):
    """Correlated team-volume Monte Carlo. Empirical per-game inputs required.
    Team targets are allocated jointly via multinomial; all sampled rates remain uncertain.
    Not an independently validated or fully play-by-play calibrated game simulator.
    """
    d=standardize(frame)
    if len(d)==0: raise ValueError('No valid QB/RB/WR/TE players.')
    numeric=FIELDS[4:]
    if not d[numeric[1:]].notna().any(axis=1).all():
        raise ValueError('Each player needs observed football-stat inputs; DK points alone are insufficient.')
    rng=np.random.default_rng(seed); n=int(n)
    if not 100<=n<=100000: raise ValueError('Simulation count must be 100–100,000')
    result={}; teams=d.team.unique()
    for team in teams:
        t=d[d.team==team]
        qbs=t[t.position=='QB']; qb=qbs.iloc[0] if len(qbs) else None
        team_pass=float(qb.pass_attempts_pg) if qb is not None and pd.notna(qb.pass_attempts_pg) and qb.pass_attempts_pg>0 else np.nan
        target_sum=t.targets_pg.fillna(0).clip(lower=0).sum()
        if not np.isfinite(team_pass): team_pass=max(1.,target_sum/0.85)
        team_rush=max(1.,t.carries_pg.fillna(0).clip(lower=0).sum())
        pace=rng.lognormal(-0.5*.13**2,.13,size=n)
        script=rng.normal(size=n)
        pass_n=np.maximum(0,rng.poisson(np.maximum(1,team_pass*pace*np.exp(.10*script))))
        rush_n=np.maximum(0,rng.poisson(np.maximum(1,team_rush*pace*np.exp(-.10*script))))
        # Targets share a team-wide attempt budget, including an unmodeled-targets bucket.
        rec=t[t.position.isin(['RB','WR','TE']) & (t.targets_pg.fillna(0)>0)]
        shares=rec.targets_pg.fillna(0).clip(lower=0).to_numpy(dtype=float)
        leftover=max(team_pass*.12,team_pass-shares.sum(),.1)
        probs=np.append(shares,leftover); probs=probs/probs.sum()
        targets=np.array([rng.multinomial(int(x),probs) for x in pass_n],dtype=np.int16)
        rushers=t[t.carries_pg.fillna(0)>0]
        carry_rates=rushers.carries_pg.fillna(0).clip(lower=0).to_numpy(dtype=float)
        carry_probs=np.append(carry_rates,max(team_rush*.08,.1)); carry_probs=carry_probs/carry_probs.sum()
        carries=np.array([rng.multinomial(int(x),carry_probs) for x in rush_n],dtype=np.int16)
        team_rec_yards=np.zeros(n); team_rec_comp=np.zeros(n)
        for j,(_,r) in enumerate(rec.iterrows()):
            key=(str(r.player),team); out=result.setdefault(key,{})
            tt=targets[:,j]; base_t=max(.01,float(r.targets_pg))
            rate=np.clip(float(r.receptions_pg)/base_t if pd.notna(r.receptions_pg) else .65,.15,.95)
            catch=rng.binomial(tt,rate)
            ypc=max(2.,float(r.receiving_yards_pg)/max(.1,float(r.receptions_pg))) if pd.notna(r.receiving_yards_pg) and pd.notna(r.receptions_pg) else 10.
            # Gamma total: per-catch positive yardage variance, plus mild game efficiency shock.
            eff=rng.lognormal(-.5*.17**2,.17,size=n)
            yds=rng.gamma(shape=np.maximum(catch,1)*2.0,scale=ypc/2.0,size=n)*eff
            yds=np.where(catch>0,yds,0.)
            out.update(targets=tt,receptions=catch,receiving_yards=yds)
            team_rec_yards+=yds; team_rec_comp+=catch
        for j,(_,r) in enumerate(rushers.iterrows()):
            key=(str(r.player),team); out=result.setdefault(key,{})
            cc=carries[:,j]
            ypc=float(r.rushing_yards_pg)/max(.1,float(r.carries_pg)) if pd.notna(r.rushing_yards_pg) and pd.notna(r.carries_pg) else 4.
            # Normal approximation permits negative net rushing yards, especially on few carries.
            yards=rng.normal(cc*ypc,np.sqrt(np.maximum(cc,1))*max(2.,abs(ypc)*1.1),size=n)
            out.update(carries=cc,rushing_yards=np.where(cc>0,yards,0.))
        if qb is not None:
            key=(str(qb.player),team); out=result.setdefault(key,{})
            # QB completions / yards reconciled to modeled receiver catches and yards,
            # plus unmodeled receivers' contribution.
            other_targets=targets[:,-1]
            other_catches=rng.binomial(other_targets,.63)
            other_yds=rng.gamma(np.maximum(other_catches,1)*2,4.,size=n)
            other_yds=np.where(other_catches>0,other_yds,0.)
            out.update(pass_attempts=pass_n,completions=team_rec_comp+other_catches,
                       passing_yards=team_rec_yards+other_yds)
    rows=[]
    for (player,team),stats in result.items():
        for metric,values in stats.items():
            v=np.asarray(values,dtype=float)
            rows.append({'player':player,'team':team,'stat':metric,'mean':round(float(v.mean()),2),
                         'median':round(float(np.median(v)),2),'p10':round(float(np.percentile(v,10)),2),
                         'p90':round(float(np.percentile(v,90)),2),'sd':round(float(v.std()),2),'position':str(d.loc[(d.player==player)&(d.team==team),'position'].iloc[0])})
    return pd.DataFrame(rows),result

def american_implied(odds):
    o=float(odds); return 100/(o+100) if o>0 else -o/(-o+100)

def american_profit(odds):
    o=float(odds); return o/100 if o>0 else 100/(-o)

def compare(summary,draws,odds):
    required={'player','market','line','side','price'}
    if not required.issubset(odds.columns): raise ValueError('Odds file requires: '+', '.join(sorted(required)))
    lookup={str(p).casefold():[(t,metrics) for (name,t),metrics in draws.items() if name.casefold()==str(p).casefold()] for p in odds.player.dropna().unique()}
    out=[]
    for _,r in odds.iterrows():
        metric=MARKETS.get(str(r.market),str(r.market)); matches=lookup.get(str(r.player).casefold(),[])
        if len(matches)!=1 or metric not in matches[0][1]: continue
        try: line=float(r.line); price=float(r.price)
        except (TypeError,ValueError): continue
        if not np.isfinite(line) or not np.isfinite(price) or price==0: continue
        side=str(r.side).lower().strip()
        if side not in ('over','under'): continue
        v=matches[0][1][metric]; p=float(np.mean(v>line if side=='over' else v<line)); push=float(np.mean(v==line))
        ev=p*american_profit(price)-(1-p-push)
        out.append({'player':r.player,'team':matches[0][0],'market':metric,'side':side,'line':line,'price':price,
                    'model_probability':round(p,4),'push_probability':round(push,4),
                    'market_implied':round(american_implied(price),4),'EV_per_$1':round(ev,4),
                    'bookmaker':r.get('bookmaker','')})
    return pd.DataFrame(out).sort_values('EV_per_$1',ascending=False) if out else pd.DataFrame()

def fetch_odds(api_key, regions='us', markets=None):
    """Fetch all currently listed NFL events and player props via The Odds API.
    Calls consume provider credits. Never logs or persists API keys.
    """
    if not api_key: raise ValueError('An Odds API key is required')
    markets=markets or list(MARKETS)
    root='https://api.the-odds-api.com/v4/sports/americanfootball_nfl'
    def get(path,params):
        url=root+path+'?'+urlencode(dict(params,apiKey=api_key))
        req=Request(url,headers={'User-Agent':'NFLPropsLab/1.0'})
        with urlopen(req,timeout=20) as res: return json.load(res)
    events=get('/events',{})
    rows=[]
    for e in events:
        data=get('/events/'+str(e['id'])+'/odds',{'regions':regions,'markets':','.join(markets),'oddsFormat':'american'})
        for book in data.get('bookmakers',[]):
            for m in book.get('markets',[]):
                for o in m.get('outcomes',[]):
                    if 'description' not in o or 'point' not in o: continue
                    rows.append({'player':o['description'],'market':m['key'],'side':o['name'],
                                 'line':o['point'],'price':o['price'],'bookmaker':book['key'],
                                 'event':e.get('home_team','')+' vs '+e.get('away_team','')})
    return pd.DataFrame(rows)
