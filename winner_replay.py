"""Pregame-only NFL Classic winner replay. Never infer pregame ownership from results."""
import re
import numpy as np
import pandas as pd

SLOTS=['QB','RB','RB','WR','WR','WR','TE','FLEX','DST']

def key(name):
    s=str(name).lower().strip()
    s=re.sub(r'\b(jr|sr|ii|iii|iv)\b','',s)
    return re.sub(r'[^a-z0-9]','',s)

def parse_winner(csv):
    """Accept Slot/Name long format, or a single-row QB/RB1/.../DST format."""
    df=pd.read_csv(csv) if not isinstance(csv,pd.DataFrame) else csv.copy()
    if {'Slot','Name'}.issubset(df.columns):
        out=df[['Slot','Name']].copy()
    elif {'Position','Name'}.issubset(df.columns):
        out=df[['Position','Name']].rename(columns={'Position':'Slot'})
    elif len(df)==1 and {'QB','RB1','RB2','WR1','WR2','WR3','TE','FLEX','DST'}.issubset(df.columns):
        out=pd.DataFrame({'Slot':SLOTS,'Name':[df.iloc[0][c] for c in ['QB','RB1','RB2','WR1','WR2','WR3','TE','FLEX','DST']]})
    else:
        raise ValueError('Upload Slot,Name rows (nine positions), or a one-row QB,RB1,RB2,WR1,WR2,WR3,TE,FLEX,DST CSV.')
    out['Slot']=out['Slot'].astype(str).str.upper().str.strip().replace({'DEF':'DST','D':'DST'})
    out['Name']=out['Name'].astype(str).str.strip()
    if sorted(out['Slot'].tolist())!=sorted(SLOTS):
        raise ValueError('Winner must have exactly QB, RB×2, WR×3, TE, FLEX, DST.')
    if out['Name'].map(key).duplicated().any():
        raise ValueError('Winner has duplicate player names.')
    return out

def replay(winner,pool,actual=None,reference_projection=None,quality_floor=.88,bank=None):
    """Audit winner against a *supplied pregame* slate; missing data remains unknown."""
    w=parse_winner(winner)
    p=pool.copy()
    if 'Name' not in p or 'Position' not in p: raise ValueError('Pregame pool needs Name and Position.')
    p['_key']=p['Name'].map(key)
    # Prevent silent ambiguous matches; use team-qualified sources to resolve aliases externally.
    duplicates=set(p.loc[p['_key'].duplicated(False),'_key'])
    rows=[]
    for _,r in w.iterrows():
        k=key(r['Name']); candidates=p[p['_key']==k]
        if k in duplicates or len(candidates)!=1:
            rows.append({'Slot':r['Slot'],'Winner player':r['Name'],'Match':'AMBIGUOUS' if k in duplicates else 'MISSING'})
            continue
        x=candidates.iloc[0]
        pos=str(x['Position']).upper()
        slot=r['Slot']
        legal=(slot==pos) or (slot=='FLEX' and pos in ('RB','WR','TE'))
        item={'Slot':slot,'Winner player':r['Name'],'Match':'MATCHED' if legal else 'WRONG POSITION',
              'Pregame name':x['Name'],'Position':pos}
        for col in ['Salary','proj','ceiling','p95_use','p99_use','ownership_pct','leverage_score','optimizer_eligible','injury_blocked','projection_source','weekly_projection_source','game','TeamAbbrev']:
            item[col]=x.get(col,np.nan)
        rows.append(item)
    detail=pd.DataFrame(rows)
    if actual is not None:
        a=pd.read_csv(actual) if not isinstance(actual,pd.DataFrame) else actual.copy()
        points_col=next((c for c in ['Actual','dk_points','DK Points','FPTS','Fantasy Points'] if c in a.columns),None)
        if 'Name' not in a or points_col is None: raise ValueError('Actual results need Name and Actual (or dk_points).')
        a=a.copy(); a['_key']=a['Name'].map(key)
        a=a.drop_duplicates('_key')
        actual_map=a.set_index('_key')[points_col].to_dict()
        detail['Actual DK']=detail['Winner player'].map(lambda n: actual_map.get(key(n),np.nan))
        detail['Actual DK']=pd.to_numeric(detail['Actual DK'],errors='coerce')
    matched=detail['Match'].eq('MATCHED').all()
    total_salary=pd.to_numeric(detail.get('Salary',pd.Series(dtype=float)),errors='coerce').sum(min_count=9)
    projected=pd.to_numeric(detail.get('proj',pd.Series(dtype=float)),errors='coerce').sum(min_count=9)
    actual_total=pd.to_numeric(detail.get('Actual DK',pd.Series(dtype=float)),errors='coerce').sum(min_count=9)
    eligible=detail.get('optimizer_eligible',pd.Series(dtype=bool)).eq(True)
    blocked=detail.get('injury_blocked',pd.Series(False,index=detail.index)).fillna(False).eq(True)
    reasons=[]
    if not matched: reasons.append('Some players are missing, ambiguous, or position-ineligible in the pregame pool')
    if pd.notna(total_salary) and total_salary>50000: reasons.append('Exceeds $50,000 salary cap')
    if matched and len(eligible)==9 and not eligible.all(): reasons.append('At least one player failed pregame optimizer eligibility')
    if blocked.any(): reasons.append('At least one player was injury-blocked in the supplied pool')
    if reference_projection is not None and pd.notna(projected) and projected<float(reference_projection)*quality_floor:
        reasons.append(f'Projected {projected:.2f} below quality floor {float(reference_projection)*quality_floor:.2f}')
    bank_status='NOT PROVIDED'
    if bank is not None:
        b=pd.read_csv(bank) if not isinstance(bank,pd.DataFrame) else bank.copy()
        target=frozenset(w['Name'].map(key))
        if 'Name' in b and 'lineup_id' in b:
            sets=[frozenset(g['Name'].map(key)) for _,g in b.groupby('lineup_id')]
        else:
            player_cols=[c for c in b.columns if re.match(r'^(QB|RB\d*|WR\d*|TE|FLEX|DST|D)$',str(c),re.I)]
            sets=[frozenset(key(x) for x in row if pd.notna(x)) for row in b[player_cols].itertuples(index=False,name=None)] if player_cols else []
        bank_status='PRESENT' if target in sets else 'ABSENT' if sets else 'UNREADABLE FORMAT'
    summary={'matched_players':int(detail['Match'].eq('MATCHED').sum()),'salary':total_salary,
             'projected_points':projected,'actual_points':actual_total,'salary_remaining':50000-total_salary if pd.notna(total_salary) else np.nan,
             'quality_floor_pass':(projected>=float(reference_projection)*quality_floor) if reference_projection is not None and pd.notna(projected) else None,
             'candidate_bank':bank_status,'rejection_reasons':reasons,
             'ownership_known':int(pd.to_numeric(detail.get('ownership_pct',pd.Series(dtype=float)),errors='coerce').notna().sum()),
             'ownership_warning':'Only use pre-lock sourced ownership. Historical/demo ownership or default 8% is not evidence of leverage.'}
    return detail,summary
