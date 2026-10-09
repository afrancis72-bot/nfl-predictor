"""Conservative, pregame-only opponent factors from nflverse schedules + weekly player stats.
No opponent guessed: missing schedule or insufficient evidence -> neutral factor 1.0.
"""
import numpy as np
import pandas as pd
from props_data import included_history

SCHEDULE_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
TEAM_ALIASES = {'JAX':'JAC','WSH':'WAS','LAR':'LA','LAC':'LAC','LV':'LV','OAK':'LV','SD':'LAC','STL':'LA'}

def normalize_team(x):
    v=str(x).upper().strip()
    return TEAM_ALIASES.get(v,v)

METRICS = {'pass_yards_pg':'passing_yards', 'rushing_yards_pg':'rushing_yards',
           'receiving_yards_pg':'receiving_yards', 'receptions_pg':'receptions',
           'completions_pg':'completions', 'targets_pg':'targets',
           'carries_pg':'carries', 'pass_attempts_pg':'attempts'}

def fetch_schedule(reader=None):
    return (reader or pd.read_csv)(SCHEDULE_URL)

def _schedule(schedules, season, week):
    s=schedules.copy()
    needed={'season','week','home_team','away_team'}
    if not needed.issubset(s.columns):
        raise ValueError('Schedule missing required fields: '+', '.join(sorted(needed-set(s.columns))))
    if 'game_type' in s.columns: s=s[s.game_type.astype(str).str.upper().eq('REG')]
    s['season']=pd.to_numeric(s.season,errors='coerce')
    s['week']=pd.to_numeric(s.week,errors='coerce')
    s=s[(s.season==int(season)) & s.week.between(1,int(week))]
    for c in ('home_team','away_team'): s[c]=s[c].map(normalize_team)
    s=s.drop_duplicates(['season','week','home_team','away_team'])
    if s.duplicated(['season','week','home_team']).any() or s.duplicated(['season','week','away_team']).any():
        raise ValueError('Schedule has duplicate team/week games')
    home=s[['season','week','home_team','away_team']].rename(columns={'home_team':'team','away_team':'opponent'})
    away=s[['season','week','away_team','home_team']].rename(columns={'away_team':'team','home_team':'opponent'})
    both=pd.concat([home,away],ignore_index=True)
    if both.duplicated(['season','week','team']).any(): raise ValueError('Schedule has conflicting opponents')
    return both

def matchup_factors(history,schedules,season,week, min_games=2, prior_strength=8, cap=.15):
    """Return team + position factors; defense concessions normalized by league/position.
    Shrinks 2-4 game evidence heavily toward 1.0 and caps at +/-15%.
    """
    h=included_history(history,season,week)
    schedule=_schedule(schedules,season,week)
    past=schedule[schedule.week<int(week)]
    current=schedule[schedule.week==int(week)]
    if current.empty: raise ValueError('Target-week schedule unavailable; opponent adjustments disabled')
    joined=h.merge(past,on=['season','week'],how='left',suffixes=('','_sched'))
    # Assign defense by actual team for each player-game, not by row order.
    joined['recent_team']=joined['recent_team'].map(normalize_team)
    joined['team']=joined['team'].map(normalize_team)
    joined['opponent']=joined['opponent'].map(normalize_team)
    joined=joined[joined.recent_team.eq(joined.team)].copy()
    if joined.empty: raise ValueError('No historical player-games matched schedule team/week')
    defenses=past[['season','week','opponent']].drop_duplicates().groupby('opponent').size()
    output=[]
    for _,row in current.iterrows():
        team=row.team; opp=row.opponent
        games=int(defenses.get(opp,0))
        for position in ('QB','RB','WR','TE'):
            group=joined[joined.position.astype(str).str.upper().eq(position)]
            against=group[group.opponent==opp]
            for rate,raw in METRICS.items():
                # Only adjust metrics where the stat is relevant to position.
                if position=='QB' and rate in ('receptions_pg','targets_pg','receiving_yards_pg','carries_pg','rushing_yards_pg'): continue
                if position!='QB' and rate in ('pass_attempts_pg','pass_yards_pg','completions_pg'): continue
                if position in ('WR','TE') and rate in ('carries_pg','rushing_yards_pg'): continue
                league=group.groupby(['season','week','opponent'])[raw].sum(min_count=1).dropna()
                opponent=against.groupby(['season','week','opponent'])[raw].sum(min_count=1).dropna()
                # Grouping across opponents and games is defense allowed by position per game.
                league_avg=float(league.mean()) if len(league) else np.nan
                opp_avg=float(opponent.mean()) if len(opponent) else np.nan
                if games<min_games or not np.isfinite(league_avg) or league_avg<=0 or not np.isfinite(opp_avg):
                    factor=1.; reason='insufficient opponent history'
                else:
                    observed=opp_avg/league_avg
                    reliability=games/(games+prior_strength)
                    factor=float(np.clip(1+reliability*(observed-1),1-cap,1+cap))
                    reason='opponent allowed vs league, shrunk to neutral'
                output.append({'team':team,'opponent':opp,'position':position,'stat_rate':rate,
                               'factor':round(factor,4),'opponent_games':games,
                               'opponent_allowed_pg':round(opp_avg,2) if np.isfinite(opp_avg) else np.nan,
                               'league_allowed_pg':round(league_avg,2) if np.isfinite(league_avg) else np.nan,
                               'reason':reason})
    return pd.DataFrame(output)

def apply_matchups(rates, factors, return_audit=False, apply_factors=True, role_filter=True):
    d=rates.copy()
    if 'position' not in d or 'team' not in d: raise ValueError('Player baselines require position and team')
    d['team']=d['team'].map(normalize_team)
    fct=factors.copy(); fct['team']=fct['team'].map(normalize_team); fct['opponent']=fct['opponent'].map(normalize_team)
    opponents=fct[['team','opponent']].drop_duplicates()
    d=d.merge(opponents,on='team',how='left',validate='many_to_one')
    excluded=d[d.opponent.isna()].copy()
    scheduled=d[d.opponent.notna()].copy()
    if scheduled.empty:
        raise ValueError('No player baselines matched the target-week schedule after team-code normalization.')
    # Current-week projectable-role screen. This is an opportunity filter, not an injury/active-roster claim.
    role_excluded=scheduled.iloc[0:0].copy()
    if role_filter:
        pos=scheduled['position'].astype(str).str.upper()
        pa=pd.to_numeric(scheduled.get('pass_attempts_pg',0),errors='coerce').fillna(0)
        ca=pd.to_numeric(scheduled.get('carries_pg',0),errors='coerce').fillna(0)
        tg=pd.to_numeric(scheduled.get('targets_pg',0),errors='coerce').fillna(0)
        role_ok=((pos=='QB')&(pa>=10)) | ((pos=='RB')&((ca+tg)>=3)) | (pos.isin(['WR','TE'])&(tg>=2))
        role_excluded=scheduled[~role_ok].copy()
        scheduled=scheduled[role_ok].copy()
    if apply_factors:
        for rate in METRICS:
            if rate not in scheduled: continue
            f=fct[fct.stat_rate==rate][['team','position','factor']].rename(columns={'factor':'_factor'})
            scheduled=scheduled.merge(f,on=['team','position'],how='left',validate='many_to_one')
            scheduled[rate]=pd.to_numeric(scheduled[rate],errors='coerce')*scheduled['_factor'].fillna(1.)
            scheduled=scheduled.drop(columns=['_factor'])
    all_excluded=pd.concat([excluded,role_excluded],ignore_index=True)
    audit={'matched_players':int(len(scheduled)),'excluded_players':int(len(all_excluded)),
           'matched_teams':sorted(scheduled.team.dropna().unique().tolist()),
           'excluded_teams':sorted(all_excluded.team.dropna().unique().tolist()),
           'bye_or_unmatched_players':int(len(excluded)),
           'low_role_players':int(len(role_excluded)),
           'excluded':all_excluded}
    return (scheduled.reset_index(drop=True),audit) if return_audit else scheduled.reset_index(drop=True)

def variance_diagnostics(summary, history, season, week):
    """Descriptive per-player historical SD vs simulated SD; no claim of calibration."""
    h=included_history(history,season,week)
    rows=[]
    for _,r in summary.iterrows():
        metric=str(r['stat'])
        src={'pass_attempts':'attempts','completions':'completions','passing_yards':'passing_yards',
             'carries':'carries','rushing_yards':'rushing_yards','targets':'targets',
             'receptions':'receptions','receiving_yards':'receiving_yards'}.get(metric)
        if src is None: continue
        obs=h[(h.player_name.astype(str)==str(r.player)) & (h.recent_team.astype(str)==str(r.team))][src].dropna()
        if len(obs)<3: continue
        actual_sd=float(obs.std(ddof=1)); sim_sd=float(r.sd)
        rows.append({'player':r.player,'team':r.team,'position':r.get('position',''),
                     'stat':metric,'observed_games':len(obs),
                     'observed_sd':round(actual_sd,2),'simulated_sd':round(sim_sd,2),
                     'sim_to_observed_sd':round(sim_sd/actual_sd,2) if actual_sd>0 else np.nan})
    return pd.DataFrame(rows)


def stabilize_variance(summary, draws, history, season, week, weight_observed=.25, min_games=3, cap_low=.75, cap_high=1.25):
    """Shrink simulated spread modestly toward observed current-season game-to-game SD.
    With only 3-4 games early in a season, observed SD receives 25% weight and the
    scale change is capped at +/-25%. This is variance stabilization, not proof of
    predictive calibration. Returns updated summary/draws plus an audit table.
    """
    h=included_history(history,season,week)
    srcmap={'pass_attempts':'attempts','completions':'completions','passing_yards':'passing_yards',
            'carries':'carries','rushing_yards':'rushing_yards','targets':'targets',
            'receptions':'receptions','receiving_yards':'receiving_yards'}
    out=summary.copy(); new_draws={k:{m:np.asarray(v,dtype=float).copy() for m,v in vals.items()} for k,vals in draws.items()}
    audits=[]
    for idx,r in out.iterrows():
        metric=str(r['stat']); src=srcmap.get(metric)
        if src is None: continue
        obs=h[(h.player_name.astype(str)==str(r.player)) & (h.recent_team.astype(str)==str(r.team))][src].dropna()
        if len(obs)<min_games: continue
        key=(str(r.player),str(r.team))
        if key not in new_draws or metric not in new_draws[key]: continue
        v=new_draws[key][metric]; sim_sd=float(np.std(v)); obs_sd=float(obs.std(ddof=1))
        if not np.isfinite(sim_sd) or sim_sd<=0 or not np.isfinite(obs_sd): continue
        blended=(1-weight_observed)*sim_sd+weight_observed*obs_sd
        scale=float(np.clip(blended/sim_sd,cap_low,cap_high))
        med=float(np.median(v)); nv=med+(v-med)*scale
        if metric in ('pass_attempts','completions','carries','targets','receptions'): nv=np.maximum(0,np.rint(nv))
        else: nv=np.maximum(0,nv)
        new_draws[key][metric]=nv
        out.at[idx,'mean']=round(float(np.mean(nv)),2); out.at[idx,'median']=round(float(np.median(nv)),2)
        out.at[idx,'p10']=round(float(np.percentile(nv,10)),2); out.at[idx,'p90']=round(float(np.percentile(nv,90)),2); out.at[idx,'sd']=round(float(np.std(nv)),2)
        audits.append({'player':r.player,'team':r.team,'position':r.get('position',''),'stat':metric,'observed_games':len(obs),
                       'raw_sim_sd':round(sim_sd,2),'observed_sd':round(obs_sd,2),'scale_applied':round(scale,3),'final_sd':round(float(np.std(nv)),2)})
    return out,new_draws,pd.DataFrame(audits)
