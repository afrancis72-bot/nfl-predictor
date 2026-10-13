"""Post-slate diagnostic only. Never use actual scores to create pre-lock predictions."""
import pandas as pd
from winner_replay import parse_winner, key

def audit_simulation(winner, simulation):
    w=parse_winner(winner)
    s=simulation.copy()
    required={'Name','Mean','P95'}
    if not required.issubset(s.columns):
        raise ValueError('Simulation CSV must contain Name, Mean and P95.')
    s['_key']=s['Name'].map(key)
    if s['_key'].duplicated().any():
        raise ValueError('Duplicate player names in simulation; resolve aliases before auditing.')
    result=w.merge(s.drop(columns=['Name']),left_on=w['Name'].map(key),right_on='_key',how='left')
    result=result.drop(columns=['key_0','_key'],errors='ignore')
    result['Mean']=pd.to_numeric(result['Mean'],errors='coerce')
    result['P95']=pd.to_numeric(result['P95'],errors='coerce')
    if 'Actual' in result:
        result['Actual']=pd.to_numeric(result['Actual'],errors='coerce')
        result['Above model P95']=result['Actual']>result['P95']
        result['Actual minus mean']=result['Actual']-result['Mean']
    return result

def lineup_exposure(winner, lineups):
    w=parse_winner(winner)
    df=lineups.copy()
    if 'Players' not in df:
        raise ValueError('Lineup export requires a Players column.')
    lineups_as_sets=[{key(x) for x in str(v).split('|')} for v in df['Players'].fillna('')]
    rows=[]
    for name in w['Name']:
        hits=sum(key(name) in lineup for lineup in lineups_as_sets)
        rows.append({'Player':name,'Lineups':hits,'Exposure %':100*hits/len(df) if len(df) else 0})
    return pd.DataFrame(rows)
