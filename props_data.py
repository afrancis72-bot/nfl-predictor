"""Fail-closed nflverse player-game data preparation for NFL Props Lab.

Current-season regular-season player weeks ONLY. Historical backtesting may use
prior seasons, but prior-year games never silently enter the current projection.
"""
import numpy as np
import pandas as pd

BASE = 'https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv'
COUNT_COLS = {'pass_attempts_pg':'attempts','completions_pg':'completions',
              'pass_yards_pg':'passing_yards','carries_pg':'carries',
              'rushing_yards_pg':'rushing_yards','targets_pg':'targets',
              'receptions_pg':'receptions','receiving_yards_pg':'receiving_yards'}

def fetch_weekly(seasons, reader=None):
    reader = reader or pd.read_csv
    frames=[]
    for season in sorted(set(map(int,seasons))):
        df=reader(BASE.format(season=season)).copy()
        if 'season' not in df:
            raise ValueError(f'Source season column missing in {season} download')
        # NEVER overwrite the reported season with the requested year.
        if not pd.to_numeric(df['season'],errors='coerce').eq(season).all():
            raise ValueError(f'Source season mismatch in {season} download; refusing to project')
        frames.append(df)
    return pd.concat(frames,ignore_index=True)

def clean_weekly(raw):
    df=raw.copy()
    if 'team' in df:
        df['recent_team']=df['team'] if 'recent_team' not in df else df['recent_team'].fillna(df['team'])
    if 'player_name' not in df and 'player_display_name' in df:
        df['player_name']=df['player_display_name']
    required={'season','week','player_id','player_name','recent_team','position','season_type'}
    missing=required-set(df.columns)
    if missing:
        raise ValueError('Cannot validate weekly NFL records; missing: '+', '.join(sorted(missing)))
    df['season']=pd.to_numeric(df.season,errors='coerce')
    df['week']=pd.to_numeric(df.week,errors='coerce')
    df=df[df.season_type.astype(str).str.upper().str.strip().eq('REG')]
    df=df[df.position.astype(str).str.upper().isin(['QB','RB','WR','TE'])]
    df=df[df.week.between(1,18)&df.season.notna()]
    for c in ['player_id','player_name','recent_team']:
        df[c]=df[c].astype('string').str.strip()
    df=df.dropna(subset=['player_id','player_name','recent_team'])
    df=df[(df.player_id!='')&(df.player_name!='')&(df.recent_team!='')]
    df['recent_team']=df.recent_team.str.upper()
    for col in COUNT_COLS.values():
        if col not in df: raise ValueError(f'Missing required football statistic: {col}')
        df[col]=pd.to_numeric(df[col],errors='coerce')
    # One player-game is the atomic unit; duplicates can double-count history.
    key=['player_id','season','week']
    dup=df.duplicated(key,keep=False)
    if dup.any():
        example=df.loc[dup,key].head(3).to_dict('records')
        raise ValueError(f'Duplicate player-season-week records ({int(dup.sum())} rows), e.g. {example}')
    return df.sort_values(['season','week','player_id']).reset_index(drop=True)

def included_history(history,target_season,target_week):
    h=clean_weekly(history)
    season=int(target_season); week=int(target_week)
    if not 1<=week<=18: raise ValueError('Target week must be 1–18')
    h=h[(h.season==season)&(h.week<week)].copy()
    if h.empty: raise ValueError(f'No {season} regular-season weeks before Week {week}; no baseline generated')
    if h.week.max()>=week: raise ValueError('Future-week leakage detected')
    if h.groupby('player_id').size().max()>week-1:
        raise ValueError('Impossible games_played count; duplicate or future records detected')
    return h

def make_rates(history,target_season,target_week,window=8,prior_seasons=0):
    """Current-season-only pregame weighted averages; no retired-player carryover."""
    h=included_history(history,target_season,target_week)
    rows=[]
    for pid,g in h.groupby('player_id',sort=False):
        g=g.sort_values('week').tail(int(window))
        if len(g)<2: continue
        latest=g.iloc[-1]
        weights=np.power(.87,np.arange(len(g)-1,-1,-1))
        row={'player':str(latest.player_name),'player_id':str(pid),
             'team':str(latest.recent_team),'position':str(latest.position).upper(),
             'games_played':len(g),'game':'','source':'nflverse current-season REG weekly stats',
             'latest_observed_season':int(latest.season),'latest_observed_week':int(latest.week)}
        for dest,src in COUNT_COLS.items():
            vals=g[src].to_numpy(dtype=float); valid=np.isfinite(vals)
            row[dest]=float(np.average(vals[valid],weights=weights[valid])) if valid.any() else np.nan
        rows.append(row)
    rates=pd.DataFrame(rows)
    if rates.empty: raise ValueError('No players with two or more current-season games')
    if rates.games_played.max()>int(target_week)-1: raise ValueError('Baseline integrity failed: excess games')
    if rates.latest_observed_week.max()>=int(target_week): raise ValueError('Baseline integrity failed: future week')
    return rates.sort_values(['position','player']).reset_index(drop=True)

def validation_report(history,target_season,target_week):
    """Returns transparent source summary and every included player-week."""
    h=included_history(history,target_season,target_week)
    summary={'Season':int(target_season),'Target week':int(target_week),
             'Included weeks':', '.join(map(str,sorted(h.week.unique().astype(int)))),
             'Source player-games':len(h),'Unique players':h.player_id.nunique(),
             'Maximum player games':int(h.groupby('player_id').size().max()),
             'Duplicate player-weeks':int(h.duplicated(['player_id','season','week']).sum()),
             'Latest included week':int(h.week.max())}
    audit_cols=['player_id','player_name','recent_team','season','week','position']+list(COUNT_COLS.values())
    if 'opponent_team' in h: audit_cols.insert(5,'opponent_team')
    return summary,h[audit_cols].sort_values(['player_name','week']).reset_index(drop=True)

def walkforward(history,season,first_week=4,last_week=18,window=8):
    h=clean_weekly(history); out=[]
    for week in range(int(first_week),int(last_week)+1):
        actual=h[(h.season==int(season))&(h.week==week)]
        if actual.empty: continue
        try: rates=make_rates(h,season,week,window=window)
        except ValueError: continue
        merged=rates.merge(actual,on='player_id',suffixes=('_pred','_actual'))
        for _,r in merged.iterrows():
            for pred,obs in COUNT_COLS.items():
                p=r[pred]; y=r[obs]
                if pd.notna(p) and pd.notna(y):
                    out.append({'season':int(season),'week':week,'player':r['player'],
                                'team':r['recent_team'],'stat':obs,'prediction':float(p),
                                'actual':float(y),'error':float(p-y),'absolute_error':float(abs(p-y))})
    return pd.DataFrame(out)
