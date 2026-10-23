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
    requested = sorted(set(map(int, seasons)))
    if not requested:
        raise ValueError("At least one historical season is required")
    frames = []
    for season in requested:
        if not 1999 <= season <= 2100:
            raise ValueError(f"Invalid NFL season: {season}")
        url = BASE.format(season=season)
        df = reader(url)
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(f"Empty or invalid nflverse player statistics for {season}: {url}")
        if "week" not in df.columns:
            raise ValueError(f"Missing week column in nflverse player statistics for {season}")
        # Never mutate the caller's cached reader DataFrame.
        df = df.copy()
        df["season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def clean_weekly(raw):
    df=raw.copy()

    # nflverse has used more than one public column name across releases.
    # Canonicalize those schema variants BEFORE validating the dataframe so
    # Props Lab and the DFS stat engine consume the same player/team fields.
    aliases = {
        'player_name': ['player_display_name', 'display_name', 'name'],
        'recent_team': ['team', 'team_abbr', 'team_abbrev'],
        'position': ['position_group', 'pos'],
    }
    for canonical, alternatives in aliases.items():
        if canonical not in df.columns:
            for alt in alternatives:
                if alt in df.columns:
                    df[canonical] = df[alt]
                    break

    # nflverse player_name can be abbreviated (e.g. 'J.Gibbs'), while
    # player_display_name is the full DK-compatible name. Prefer the latter
    # for identity matching, including when BOTH fields exist.
    if 'player_display_name' in df.columns:
        full = df['player_display_name'].astype('string').str.strip()
        good = full.notna() & full.ne('')
        if 'player_name' not in df.columns:
            df['player_name'] = full
        else:
            df.loc[good, 'player_name'] = full.loc[good]

    # Current nflverse uses passing_interceptions, not interceptions.
    if 'passing_interceptions' in df.columns:
        df['interceptions'] = df['passing_interceptions']

    required={'season','week','player_name','recent_team','position'}
    missing=required-set(df.columns)
    if missing:
        raise ValueError(
            'nflverse data missing columns after schema normalization: '
            + ', '.join(sorted(missing))
            + ' | available columns: '
            + ', '.join(map(str, df.columns))
        )
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

def included_history(history, target_season, target_week, prior_seasons=1):
    """Return the auditable pregame history slice used by matchup/variance tools."""
    h=clean_weekly(history)
    season=int(target_season); week=int(target_week)
    h=h[(h.season<season)|((h.season==season)&(h.week<week))]
    h=h[h.season>=season-int(prior_seasons)]
    return h.reset_index(drop=True)


def make_rates(history, target_season, target_week, window=8, prior_seasons=1):
    """Uses only games before target kickoff week. Weights recent games more.
    Games with no row are not assumed zero; role/injury must be checked separately.
    """
    h=clean_weekly(history)
    season=int(target_season); week=int(target_week)
    h=h[(h.season<season)|((h.season==season)&(h.week<week))]
    h=h[h.season>=season-int(prior_seasons)]
    if h.empty: raise ValueError('No prior-week observations. Cannot generate a pregame projection.')
    # Mixed feeds can have GSIS IDs for most players but null IDs for some.
    # Grouping directly by player_id silently discarded those players.
    # Use a stable ID where available and an explicit name/team fallback
    # for the remaining records, without merging unrelated same-name players.
    if 'player_id' in h.columns:
        ids = h['player_id'].astype('string').str.strip()
        valid_ids = ids.notna() & ids.ne('')
        h = h.copy()
        h['_history_identity'] = (
            'NAME:' + h['player_name'].astype(str).str.casefold()
            + '|TEAM:' + h['recent_team'].astype(str)
        )
        h.loc[valid_ids, '_history_identity'] = 'ID:' + ids.loc[valid_ids]
    else:
        h = h.copy()
        h['_history_identity'] = (
            'NAME:' + h['player_name'].astype(str).str.casefold()
            + '|TEAM:' + h['recent_team'].astype(str)
        )
    rows=[]
    for _,g in h.groupby('_history_identity',dropna=True):
        g=g.sort_values(['season','week']).tail(int(window))
        # One observed game is still usable for a new/low-history player.  Keeping it
        # is preferable to dropping the player entirely; downstream simulation retains
        # wider outcome variance for sparse roles.  Zero observations remain excluded.
        if len(g)<1: continue
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

def dk_classic_points(stats, position=None):
    """Partial DraftKings Classic offensive score with position-aware data.

    NFL weekly feeds may omit non-applicable stats entirely. Required stats
    are position-specific; missing applicable data must not become zero.
    Does not include fumbles lost or return touchdowns.
    """
    pos = str(position or stats.get('position', '')).upper()
    applicable = {
        'QB': ('passing_yards', 'passing_tds', 'interceptions',
               'rushing_yards', 'rushing_tds'),
        'RB': ('rushing_yards', 'rushing_tds', 'receiving_yards',
               'receptions', 'receiving_tds'),
        'WR': ('rushing_yards', 'rushing_tds', 'receiving_yards',
               'receptions', 'receiving_tds'),
        'TE': ('rushing_yards', 'rushing_tds', 'receiving_yards',
               'receptions', 'receiving_tds'),
    }
    if pos not in applicable:
        return np.nan
    fields = ('passing_yards', 'passing_tds', 'interceptions',
              'rushing_yards', 'rushing_tds', 'receiving_yards',
              'receptions', 'receiving_tds')
    values = {}
    for field in fields:
        if field not in applicable[pos]:
            values[field] = 0.
            continue
        try:
            value = float(stats.get(field, np.nan))
        except (TypeError, ValueError):
            return np.nan
        if not np.isfinite(value):
            return np.nan
        values[field] = value
    v = values
    return (0.04 * v['passing_yards'] + 4 * v['passing_tds']
            - v['interceptions'] + 0.1 * v['rushing_yards']
            + 6 * v['rushing_tds'] + 0.1 * v['receiving_yards']
            + v['receptions'] + 6 * v['receiving_tds']
            + (3 if v['passing_yards'] >= 300 else 0)
            + (3 if v['rushing_yards'] >= 100 else 0)
            + (3 if v['receiving_yards'] >= 100 else 0))


def walkforward_dk_offense(history, season, first_week=4, last_week=18, window=8):
    """Pregame partial-DK-point baseline vs subsequent observed offensive stats.

    Bonus thresholds are applied to mean-stat predictions as a naive baseline,
    not as expected-value probabilities. Missing score inputs are skipped.
    """
    h = clean_weekly(history)
    out = []
    for week in range(int(first_week), int(last_week) + 1):
        actual = h[(h.season == int(season)) & (h.week == week)]
        if actual.empty:
            continue
        try:
            rates = make_rates(h, season, week, window=window)
        except ValueError:
            continue
        for _, row in rates.iterrows():
            observed = actual[(actual.player_name == row.player)
                              & (actual.recent_team == row.team)]
            if len(observed) != 1:
                continue
            predicted_stats = {src: row[dest] for dest, src in COUNT_COLS.items()}
            pred = dk_classic_points(predicted_stats, row.position)
            truth = dk_classic_points(observed.iloc[0].to_dict(), row.position)
            if np.isfinite(pred) and np.isfinite(truth):
                out.append({'season': int(season), 'week': week,
                            'player': row.player, 'team': row.team,
                            'position': row.position,
                            'predicted_partial_dk': float(pred),
                            'actual_partial_dk': float(truth),
                            'error': float(pred - truth),
                            'absolute_error': float(abs(pred - truth))})
    return pd.DataFrame(out)


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
