"""Stat-driven NFL DFS simulator.

Football statistics are simulated first; DraftKings points are scored from those
outcomes.  Offensive players and the opposing DST share the same game scenario.
This is intentionally a transparent game-level model, not a play-by-play engine.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TEAM_ALIASES={"JAC":"JAX","WAS":"WSH","LA":"LAR"}

def _team(x):
    s=str(x).upper().strip(); return TEAM_ALIASES.get(s,s)

def _clean_name(x):
    return " ".join(str(x).lower().replace(".","").replace("'","").replace("-"," ").split())

def _opp(game, team):
    g=str(game).split()[0]
    if "@" not in g: return ""
    a,h=g.split("@",1); t=_team(team)
    return _team(h if _team(a)==t else a)

def _pa_points(points):
    p=np.asarray(points)
    return np.select([p==0,p<=6,p<=13,p<=20,p<=27,p<=34],[10,7,4,1,0,-1],default=-4).astype(float)

def _safe(v, default):
    try:
        z=float(v); return z if np.isfinite(z) else float(default)
    except Exception: return float(default)

def _multinomial_rows(rng, totals, probs):
    return np.asarray([rng.multinomial(int(max(0,n)), probs) for n in totals],dtype=np.int16)

def simulate_stat_driven_dfs(pool, rates, n_sims=10000, seed=20261009):
    """Return (sim_players, dk_points, stat_draws).

    rates is the pregame player baseline produced by props_data.make_rates().
    Sacks are explicit shared events: they reduce QB dropbacks available for pass
    attempts and score +1 for the opposing DST. Turnovers, TDs and points allowed
    are also shared between offense and DST.
    """
    x=pool[pool['optimizer_eligible']==True].copy().reset_index(drop=True)
    r=rates.copy()
    r['team']=r['team'].map(_team); r['_name']=r['player'].map(_clean_name)
    x['TeamAbbrev']=x['TeamAbbrev'].map(_team); x['_name']=x['Name'].map(_clean_name)
    statcols=[c for c in r.columns if c not in {'player','team','position','game','source','latest_observed_season','latest_observed_week'}]
    rm=r.drop_duplicates(['_name','team']).set_index(['_name','team'])
    ns=int(n_sims); rng=np.random.default_rng(seed)
    out=np.zeros((ns,len(x)),dtype=np.float32); stat_draws={}
    name_index={(str(row['_name']),str(row['TeamAbbrev']),str(row['Position'])):i for i,row in x.iterrows()}
    teams=sorted(x['TeamAbbrev'].dropna().astype(str).unique())
    games=sorted(x['game'].dropna().astype(str).unique())
    processed=set()

    for game in games:
        gx=x[x.game.astype(str)==game]; gteams=list(dict.fromkeys(gx.TeamAbbrev.astype(str)))
        if len(gteams)!=2: continue
        # Shared pace/script shock for both teams in the game.
        pace=rng.lognormal(-0.5*.10**2,.10,size=ns)
        script=rng.normal(0,1,size=ns)
        for ti,team in enumerate(gteams):
            opp=gteams[1-ti]; tx=gx[gx.TeamAbbrev==team]
            qbrows=tx[tx.Position.astype(str)=='QB']
            if qbrows.empty: continue
            # Highest-projected QB is treated as active starter for team-volume generation.
            qbrow=qbrows.sort_values('proj',ascending=False).iloc[0]; qname=str(qbrow['_name'])
            qr=rm.loc[(qname,team)] if (qname,team) in rm.index else None
            pass_pg=_safe(qr.get('pass_attempts_pg') if qr is not None else np.nan, 32.0)
            comp_pg=_safe(qr.get('completions_pg') if qr is not None else np.nan, pass_pg*.65)
            pass_yd_pg=_safe(qr.get('pass_yards_pg') if qr is not None else np.nan, 220.0)
            int_pg=_safe(qr.get('interceptions_pg') if qr is not None else np.nan, .65)
            sack_pg=_safe(qr.get('sacks_suffered_pg') if qr is not None else np.nan, 2.3)
            pass_td_pg=_safe(qr.get('passing_tds_pg') if qr is not None else np.nan, 1.45)

            skill=tx[tx.Position.astype(str).isin(['RB','WR','TE'])].copy()
            # Join observed opportunity; missing/depth players receive zero share unless no rates exist.
            rr=[]
            for _,p in skill.iterrows():
                key=(str(p['_name']),team); rr.append(rm.loc[key] if key in rm.index else None)
            target_rates=np.array([_safe(z.get('targets_pg') if z is not None else np.nan,0) for z in rr],float)
            carry_rates=np.array([_safe(z.get('carries_pg') if z is not None else np.nan,0) for z in rr],float)
            team_rush=max(12.0, carry_rates.sum()+_safe(qr.get('carries_pg') if qr is not None else np.nan,3.0))

            # Script: positive means this team trails -> more passing, less rushing.
            team_script=script if ti==0 else -script
            dropbacks=np.maximum(8,rng.poisson(np.maximum(8,(pass_pg+sack_pg)*pace*np.exp(.09*team_script))))
            # DST pass-rush strength is conservatively inferred from its DFS baseline until
            # a dedicated team sack-rate table is available; QB sack tendency supplies offense side.
            dstrow=x[(x.TeamAbbrev==opp)&(x.Position.astype(str)=='DST')]
            dst_mean=_safe(dstrow.iloc[0].get('proj') if len(dstrow) else np.nan,6.5)
            dst_sack_mult=float(np.clip(1.0+(dst_mean-6.5)*.035,.78,1.28))
            sack_rate=np.clip((sack_pg/max(pass_pg+sack_pg,1))*dst_sack_mult,.025,.14)
            sacks=rng.binomial(dropbacks,sack_rate)
            attempts=np.maximum(0,dropbacks-sacks)
            rushes=np.maximum(5,rng.poisson(np.maximum(5,team_rush*pace*np.exp(-.10*team_script))))

            # Target allocation from one team pass-attempt budget.
            leftover=max(pass_pg*.10, pass_pg-target_rates.sum(), .5)
            tprob=np.append(np.maximum(target_rates,0),leftover); tprob=tprob/tprob.sum()
            targ=_multinomial_rows(rng,attempts,tprob)
            # Carry allocation includes QB and an unmodeled bucket.
            qb_carry=_safe(qr.get('carries_pg') if qr is not None else np.nan,3.0)
            cbase=np.append(np.maximum(carry_rates,0),[max(qb_carry,0),max(team_rush*.06,.5)])
            cprob=cbase/cbase.sum(); carr=_multinomial_rows(rng,rushes,cprob)

            team_rec_yd=np.zeros(ns); team_comp=np.zeros(ns); player_rec={}; player_rush={}
            for j,(_,p) in enumerate(skill.iterrows()):
                z=rr[j]; tt=targ[:,j]
                tr=max(_safe(z.get('targets_pg') if z is not None else np.nan,0),.1)
                recpg=_safe(z.get('receptions_pg') if z is not None else np.nan,tr*.62)
                catch_rate=float(np.clip(recpg/tr,.18,.92)); recs=rng.binomial(tt,catch_rate)
                ypr=max(2.0,_safe(z.get('receiving_yards_pg') if z is not None else np.nan,recpg*9.5)/max(recpg,.1))
                recyd=np.where(recs>0,rng.gamma(np.maximum(recs,1)*2.2,ypr/2.2)*rng.lognormal(-.5*.14**2,.14,ns),0.)
                cc=carr[:,j]
                cpg=max(_safe(z.get('carries_pg') if z is not None else np.nan,0),.1)
                ypc=_safe(z.get('rushing_yards_pg') if z is not None else np.nan,cpg*4.1)/cpg
                rushyd=np.where(cc>0,rng.normal(cc*ypc,np.sqrt(np.maximum(cc,1))*max(1.8,abs(ypc)*.75),ns),0.)
                player_rec[j]=(tt,recs,recyd); player_rush[j]=(cc,rushyd)
                team_rec_yd+=recyd; team_comp+=recs

            # Unmodeled catches/yards reconcile QB production with team allocation.
            oth_t=targ[:,-1]; oth_c=rng.binomial(oth_t,.62); oth_y=np.where(oth_c>0,rng.gamma(np.maximum(oth_c,1)*2.0,4.5),0.)
            team_comp+=oth_c; team_rec_yd+=oth_y
            # Blend receiver-generated yards toward QB historical yards/attempt without breaking correlation.
            hist_ypa=pass_yd_pg/max(pass_pg,1); generated_ypa=team_rec_yd/np.maximum(attempts,1)
            scale=np.clip(hist_ypa/np.maximum(generated_ypa,.1),.72,1.35)
            team_rec_yd*=scale
            for j in player_rec: player_rec[j]=(player_rec[j][0],player_rec[j][1],player_rec[j][2]*scale)

            # Shared TD/turnover scoring. Implied points anchor scoring environment.
            implied=_safe(pd.to_numeric(tx.get('team_implied_points'),errors='coerce').dropna().median() if 'team_implied_points' in tx else np.nan,22.0)
            expected_td=np.clip((.55*(pass_td_pg+max(.3,team_rush*.035))+.45*(implied/7.0)),1.0,5.0)
            total_td=rng.poisson(expected_td,size=ns)
            pass_share=np.clip(pass_td_pg/max(pass_td_pg+max(.3,team_rush*.035),.1),.35,.78)
            pass_td=rng.binomial(total_td,pass_share); rush_td=total_td-pass_td
            int_rate=np.clip(int_pg/max(pass_pg,1),.005,.065); ints=rng.binomial(attempts,int_rate)

            # Allocate receiving and rushing TDs by opportunity shares.
            rec_td_alloc=np.zeros((ns,len(skill)),dtype=np.int8); rush_td_alloc=np.zeros((ns,len(skill)),dtype=np.int8)
            tp=np.maximum(target_rates,0)+.15; tp=tp/tp.sum() if tp.sum()>0 else np.repeat(1/len(skill),len(skill))
            cp=np.maximum(carry_rates,0)+.10; cp=cp/cp.sum() if cp.sum()>0 else np.repeat(1/len(skill),len(skill))
            for s in range(ns):
                if pass_td[s]>0: rec_td_alloc[s]=rng.multinomial(int(pass_td[s]),tp)
                if rush_td[s]>0: rush_td_alloc[s]=rng.multinomial(int(rush_td[s]),cp)

            # QB DK score.
            qidx=name_index.get((qname,team,'QB'))
            if qidx is not None:
                qb_carries=carr[:,-2]; qb_ypc=_safe(qr.get('rushing_yards_pg') if qr is not None else np.nan,15)/max(qb_carry,.5)
                qb_rushyd=np.where(qb_carries>0,rng.normal(qb_carries*qb_ypc,np.sqrt(np.maximum(qb_carries,1))*2.2,ns),0.)
                qb_rush_td=np.zeros(ns,dtype=int)  # conservative; team rush TDs allocated to listed skill rushers
                dk=.04*team_rec_yd+4*pass_td-ints+.1*qb_rushyd+6*qb_rush_td+3*(team_rec_yd>=300)+3*(qb_rushyd>=100)
                out[:,qidx]=dk.astype(np.float32)
                stat_draws[str(qbrow['Name'])]={'pass_attempts':attempts,'completions':team_comp,'passing_yards':team_rec_yd,'passing_tds':pass_td,'interceptions':ints,'sacks_taken':sacks,'rushing_yards':qb_rushyd}

            for j,(_,p) in enumerate(skill.iterrows()):
                idx=name_index.get((str(p['_name']),team,str(p['Position'])))
                if idx is None: continue
                tt,recs,recyd=player_rec[j]; cc,rushyd=player_rush[j]
                rtd=rec_td_alloc[:,j]; utd=rush_td_alloc[:,j]
                dk=recs+.1*recyd+6*rtd+.1*rushyd+6*utd+3*(recyd>=100)+3*(rushyd>=100)
                out[:,idx]=dk.astype(np.float32)
                stat_draws[str(p['Name'])]={'targets':tt,'receptions':recs,'receiving_yards':recyd,'receiving_tds':rtd,'carries':cc,'rushing_yards':rushyd,'rushing_tds':utd}

            # Opposing DST derives from this exact offensive scenario.
            didx_candidates=x.index[(x.TeamAbbrev==opp)&(x.Position.astype(str)=='DST')].tolist()
            if didx_candidates:
                didx=didx_candidates[0]
                # Fumble recoveries/defensive TDs are rare events tied partly to sacks/turnovers.
                fum=rng.binomial(np.maximum(sacks,0),.075)
                takeaways=ints+fum
                def_td=rng.binomial(np.minimum(takeaways,1),.075)
                saf=rng.binomial(1,.012,size=ns)
                # Field goals fill some of the gap between TD expectation and implied scoring.
                fg_mean=max(0.25,(implied-7*expected_td)/3.0)
                fgs=rng.poisson(fg_mean,size=ns)
                pts_allowed=7*total_td+3*fgs+2*saf
                dst_dk=sacks+2*ints+2*fum+6*def_td+2*saf+_pa_points(pts_allowed)
                out[:,didx]=dst_dk.astype(np.float32)
                stat_draws[str(x.loc[didx,'Name'])]={'sacks':sacks,'interceptions':ints,'fumble_recoveries':fum,'defensive_tds':def_td,'safeties':saf,'points_allowed':pts_allowed}
            processed.add(team)

    # Fail closed on players that could not be generated rather than silently inventing points.
    means=out.mean(axis=0)
    missing=(means==0)&(~x.Position.astype(str).eq('DST'))&(pd.to_numeric(x.proj,errors='coerce').fillna(0)>1)
    if missing.any():
        names=x.loc[missing,'Name'].astype(str).tolist()[:15]
        raise ValueError('Stat-driven engine could not match observed-stat baselines for: '+', '.join(names))
    x['simulation_engine']='Stat-driven football simulation'
    x['portfolio_relevant']=True
    x['sim_p99_sanity']=np.percentile(out,99,axis=0); x['sim_max_sanity']=out.max(axis=0)
    return x,out,stat_draws
