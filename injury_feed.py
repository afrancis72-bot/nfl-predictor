"""Live injury status discovery for DFS; unofficial ESPN public feed, fail closed."""
import datetime as dt
import json
import re
import urllib.request
import pandas as pd

FEED_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
TEAM_ALIAS={"JAC":"JAX","WSH":"WAS","OAK":"LV","SD":"LAC","STL":"LA","LAR":"LA","ARZ":"ARI"}
OUT={"OUT","INACTIVE","INJURED RESERVE","IR","RESERVE/INJURED","PUP","NFI","SUSPENDED","DOUBTFUL"}
# Doubtful is NOT confirmed absent; separate it below.
OUT.discard('DOUBTFUL')
QUESTIONABLE={"QUESTIONABLE","DOUBTFUL","GAME TIME DECISION","GTD"}

def norm_name(s):
    s=re.sub(r"[^a-z0-9 ]", " ",str(s).lower())
    p=s.split()
    while p and p[-1] in {'jr','sr','ii','iii','iv','v'}: p.pop()
    return ' '.join(p)

def norm_team(t):
    t=str(t or '').strip().upper()
    return TEAM_ALIAS.get(t,t)

def normalize_status(s):
    s=str(s or '').strip().upper().replace('_',' ').replace('-',' ')
    if s in OUT: return 'OUT'
    if s in QUESTIONABLE: return 'QUESTIONABLE'
    if s in {'ACTIVE','HEALTHY','AVAILABLE','PROBABLE'}: return 'ACTIVE'
    return 'UNKNOWN'

def parse_feed(payload, fetched_at=None):
    if not isinstance(payload,dict) or not isinstance(payload.get('injuries'),list):
        raise ValueError('Unexpected injury feed schema; automatic exclusions disabled')
    entries=[]
    for group in payload['injuries']:
        if not isinstance(group,dict): continue
        team=norm_team((group.get('team') or {}).get('abbreviation'))
        for item in group.get('injuries',[]):
            if not isinstance(item,dict): continue
            ath=item.get('athlete') or {}
            name=ath.get('fullName') or ath.get('displayName')
            if not name or not team: continue
            raw=item.get('status') or (item.get('type') or {}).get('name') or ''
            status=normalize_status(raw)
            entries.append({'team':team,'player':str(name),'status':status,'raw_status':str(raw),
                            'report_date':str(item.get('date') or ''),'source':'ESPN public injury feed',
                            'fetched_at':fetched_at or dt.datetime.now(dt.timezone.utc).isoformat()})
    return pd.DataFrame(entries,columns=['team','player','status','raw_status','report_date','source','fetched_at'])

def fetch_live(timeout=12):
    req=urllib.request.Request(FEED_URL,headers={'User-Agent':'Mozilla/5.0 (DFS injury status audit)','Accept':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as resp:
        data=json.load(resp)
    return parse_feed(data)

def match_to_pool(feed,pool):
    if feed is None or feed.empty: return pd.DataFrame(columns=['team','player','status','raw_status','source','fetched_at','matched_name','match_state'])
    idx={}
    for _,r in pool.iterrows():
        key=(norm_team(r.get('TeamAbbrev')),norm_name(r.get('Name')))
        idx.setdefault(key,[]).append(r)
    out=[]
    for _,r in feed.iterrows():
        key=(norm_team(r['team']),norm_name(r['player']))
        matches=idx.get(key,[])
        if len(matches)!=1: continue # Never auto-exclude ambiguous/non-slate players.
        p=matches[0]
        d=r.to_dict();d['matched_name']=str(p['Name']);d['match_state']='EXACT_TEAM_NAME'
        d['position']=str(p.get('Position',''))
        out.append(d)
    return pd.DataFrame(out)

def confirmed_out(matched):
    if matched is None or matched.empty:return []
    return [{'team':str(r['team']),'player':str(r['matched_name'])} for _,r in matched[matched.status.eq('OUT')].drop_duplicates(['team','matched_name']).iterrows()]
