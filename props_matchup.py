"""Conservative, pregame-only opponent factors from nflverse schedules + weekly player stats.
No opponent guessed: missing schedule or insufficient evidence -> neutral factor 1.0.
"""
import numpy as np
import pandas as pd
from props_data import included_history

SCHEDULE_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
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
    for c in ('home_team','away_team'): s[c]=s[c].astype(str).str.upper().str.strip()
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
    joined=joined[joined.recent_team.astype(str).str.upper().eq(joined.team)].copy()
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

def apply_matchups(rates, factors):
    d=rates.copy()
    if 'position' not in d or 'team' not in d: raise ValueError('Player baselines require position and team')
    opponents=factors[['team','opponent']].drop_duplicates()
    d=d.merge(opponents,on='team',how='left',validate='many_to_one')
    if d.opponent.isna().any():
        raise ValueError('Some player teams have no target-week matchup (bye or team-code mismatch). Filter to scheduled players first.')
    for rate in METRICS:
        if rate not in d: continue
        f=factors[factors.stat_rate==rate][['team','position','factor']].rename(columns={'factor':'_factor'})
        d=d.merge(f,on=['team','position'],how='left',validate='many_to_one')
        d[rate]=pd.to_numeric(d[rate],errors='coerce')*d['_factor'].fillna(1.)
        d=d.drop(columns=['_factor'])
    return d

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
