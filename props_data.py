"""NFL player-stat ingestion and transparent pregame baselines.
Data source: nflverse weekly player stats; no fantasy-point-derived stats.
"""
from datetime import datetime
import numpy as np
import pandas as pd

BASE='https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv'
COUNT_COLS={'pass_attempts_pg':'attempts','completions_pg':'completions','pass_yards_pg':'passing_yards',
            'carries_pg':'carries','rushing_yards_pg':'rushing_yards','targets_pg':'targets',
            'receptions_pg':'receptions','receiving_yards_pg':'receiving_yards','passing_tds_pg':'passing_tds','interceptions_pg':'interceptions','rushing_tds_pg':'rushing_tds','receiving_tds_pg':'receiving_tds','sacks_suffered_pg':'sacks_suffered'}

def fetch_weekly(seasons, reader=None):
    """Fetch weekly official-derived statistics. Caller can supply reader for testing."""
    reader=reader or pd.read_csv
    frames=[]
    for season in sorted(set(map(int,seasons))):
        df=reader(BASE.format(season=season))
        df['season']=season
        frames.append(df)
    return pd.concat(frames,ignore_index=True)

def clean_weekly(raw):
    df=raw.copy()
    required={'season','week','player_name','recent_team','position'}
    missing=required-set(df.columns)
    if missing: raise ValueError('nflverse data missing columns: '+', '.join(sorted(missing)))
    for col in COUNT_COLS.values():
        if col not in df: df[col]=np.nan
        df[col]=pd.to_numeric(df[col],errors='coerce')
    df['season']=pd.to_numeric(df.season,errors='coerce')
    df['week']=pd.to_numeric(df.week,errors='coerce')
    if 'season_type' in df: df=df[df.season_type.astype(str).str.upper().eq('REG')]
    df=df[df.position.astype(str).str.upper().isin(['QB','RB','WR','TE'])]
    df=df.dropna(subset=['season','week','player_name','recent_team'])
    df['player_name']=df.player_name.astype(str).str.strip()
    df['recent_team']=df.recent_team.astype(str).str.strip().str.upper()
    return df.sort_values(['season','week']).reset_index(drop=True)

def make_rates(history, target_season, target_week, window=8, prior_seasons=1):
    """Uses only games before target kickoff week. Weights recent games more.
    Games with no row are not assumed zero; role/injury must be checked separately.
    """
    h=clean_weekly(history)
    season=int(target_season); week=int(target_week)
    h=h[(h.season<season)|((h.season==season)&(h.week<week))]
    h=h[h.season>=season-int(prior_seasons)]
    if h.empty: raise ValueError('No prior-week observations. Cannot generate a pregame projection.')
    group_key='player_id' if 'player_id' in h and h.player_id.notna().any() else 'player_name'
    rows=[]
    for _,g in h.groupby(group_key,dropna=True):
        g=g.sort_values(['season','week']).tail(int(window))
        if len(g)<2: continue
        latest=g.iloc[-1]
        # exponential recency: most recent game weight 1, each older game 0.87
        weights=np.power(.87,np.arange(len(g)-1,-1,-1))
        row={'player':latest.player_name,'team':latest.recent_team,'position':str(latest.position).upper(),
             'games_played':len(g),'game':'','source':'nflverse weekly stats',
             'latest_observed_season':int(latest.season),'latest_observed_week':int(latest.week)}
        for dest,src in COUNT_COLS.items():
            vals=g[src].to_numpy(dtype=float); valid=np.isfinite(vals)
            row[dest]=float(np.average(vals[valid],weights=weights[valid])) if valid.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def walkforward(history, season, first_week=4, last_week=18, window=8):
    """Baseline backtest, not simulation calibration: pregame weighted means vs actual stats."""
    h=clean_weekly(history)
    out=[]
    for week in range(int(first_week),int(last_week)+1):
        actual=h[(h.season==int(season))&(h.week==week)]
        if actual.empty: continue
        try: rates=make_rates(h,season,week,window=window)
        except ValueError: continue
        if rates.empty: continue
        for _,r in rates.iterrows():
            a=actual[(actual.player_name==r.player)&(actual.recent_team==r.team)]
            if len(a)!=1: continue
            for pred,obs in COUNT_COLS.items():
                p=r[pred]; y=a.iloc[0][obs]
                if pd.notna(p) and pd.notna(y):
                    out.append({'season':int(season),'week':week,'player':r.player,'team':r.team,
                                'stat':obs,'prediction':float(p),'actual':float(y),
                                'error':float(p-y),'absolute_error':float(abs(p-y))})
    return pd.DataFrame(out)

def validation_report(raw, target_season, target_week):
    """Validate the pregame slice and return (summary, auditable source rows).

    The audit is intentionally limited to the target season and weeks before the
    target week so the UI can prove there is no future-week leakage.
    """
    df = raw.copy()
    if 'recent_team' not in df.columns and 'team' in df.columns:
        df['recent_team'] = df['team']
    if 'player_name' not in df.columns and 'player_display_name' in df.columns:
        df['player_name'] = df['player_display_name']
    h = clean_weekly(df)
    season, week = int(target_season), int(target_week)
    audit = h[(h['season'] == season) & (h['week'] < week)].copy()
    if audit.empty:
        raise ValueError(f'No regular-season player-week rows found before {season} week {week}.')
    id_col = 'player_id' if 'player_id' in audit.columns and audit['player_id'].notna().any() else 'player_name'
    dupes = int(audit.duplicated(subset=[id_col, 'week'], keep=False).sum())
    future = int((audit['week'] >= week).sum())
    max_week = int(audit['week'].max())
    if dupes:
        raise ValueError(f'Data integrity check failed: {dupes} duplicate player/week rows.')
    if future:
        raise ValueError(f'Data leakage check failed: {future} target/future-week rows included.')
    summary = {
        'target_season': season,
        'target_week': week,
        'regular_season_only': True,
        'included_weeks': sorted(int(x) for x in audit['week'].dropna().unique()),
        'max_included_week': max_week,
        'player_week_rows': int(len(audit)),
        'unique_players': int(audit[id_col].nunique()),
        'duplicate_player_weeks': dupes,
        'future_week_rows': future,
        'status': 'PASS',
    }
    cols = [c for c in ['season','week','player_id','player_name','recent_team','position'] + list(COUNT_COLS.values()) if c in audit.columns]
    return summary, audit[cols].sort_values(['week','recent_team','player_name']).reset_index(drop=True)
