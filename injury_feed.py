"""Official NFL public injury report reader. Fail-closed on structural ambiguity.
This is HTML ingestion, NOT the authenticated api.nfl.com API.
"""
import datetime as dt
import re
import urllib.request
import pandas as pd
from bs4 import BeautifulSoup

TEAM_ALIAS={"JAC":"JAX","WSH":"WAS","OAK":"LV","SD":"LAC","STL":"LA","LAR":"LA","ARZ":"ARI"}
TEAM_NAMES={
"ARIZONA CARDINALS":"ARI","CARDINALS":"ARI","ATLANTA FALCONS":"ATL","FALCONS":"ATL",
"BALTIMORE RAVENS":"BAL","RAVENS":"BAL","BUFFALO BILLS":"BUF","BILLS":"BUF",
"CAROLINA PANTHERS":"CAR","PANTHERS":"CAR","CHICAGO BEARS":"CHI","BEARS":"CHI",
"CINCINNATI BENGALS":"CIN","BENGALS":"CIN","CLEVELAND BROWNS":"CLE","BROWNS":"CLE",
"DALLAS COWBOYS":"DAL","COWBOYS":"DAL","DENVER BRONCOS":"DEN","BRONCOS":"DEN",
"DETROIT LIONS":"DET","LIONS":"DET","GREEN BAY PACKERS":"GB","PACKERS":"GB",
"HOUSTON TEXANS":"HOU","TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","COLTS":"IND",
"JACKSONVILLE JAGUARS":"JAX","JAGUARS":"JAX","KANSAS CITY CHIEFS":"KC","CHIEFS":"KC",
"LAS VEGAS RAIDERS":"LV","RAIDERS":"LV","LOS ANGELES CHARGERS":"LAC","CHARGERS":"LAC",
"LOS ANGELES RAMS":"LA","RAMS":"LA","MIAMI DOLPHINS":"MIA","DOLPHINS":"MIA",
"MINNESOTA VIKINGS":"MIN","VIKINGS":"MIN","NEW ENGLAND PATRIOTS":"NE","PATRIOTS":"NE",
"NEW ORLEANS SAINTS":"NO","SAINTS":"NO","NEW YORK GIANTS":"NYG","GIANTS":"NYG",
"NEW YORK JETS":"NYJ","JETS":"NYJ","PHILADELPHIA EAGLES":"PHI","EAGLES":"PHI",
"PITTSBURGH STEELERS":"PIT","STEELERS":"PIT","SAN FRANCISCO 49ERS":"SF","49ERS":"SF",
"SEATTLE SEAHAWKS":"SEA","SEAHAWKS":"SEA","TAMPA BAY BUCCANEERS":"TB","BUCCANEERS":"TB",
"TENNESSEE TITANS":"TEN","TITANS":"TEN","WASHINGTON COMMANDERS":"WAS","COMMANDERS":"WAS"}
COLUMNS=['team','player','status','raw_status','report_date','source','fetched_at']

def norm_name(s):
    p=re.sub(r"[^a-z0-9 ]"," ",str(s).lower()).split()
    while p and p[-1] in {'jr','sr','ii','iii','iv','v'}:p.pop()
    return ' '.join(p)
def norm_team(t):
    t=str(t or '').strip().upper()
    return TEAM_ALIAS.get(t,t)
def normalize_status(s):
    s=str(s or '').strip().upper().replace('_',' ').replace('-',' ')
    if s in {'OUT','INACTIVE','INJURED RESERVE','IR','RESERVE/INJURED','PUP','NFI','SUSPENDED'}:return 'OUT'
    if s in {'QUESTIONABLE','DOUBTFUL','GAME TIME DECISION','GTD'}:return 'QUESTIONABLE'
    if s in {'ACTIVE','HEALTHY','AVAILABLE','PROBABLE'}:return 'ACTIVE'
    return 'UNKNOWN'
def _team_label(s):
    s=' '.join(str(s).upper().split())
    if s in TEAM_NAMES:return TEAM_NAMES[s]
    # NFL page may prefix an abbreviation (e.g. 'CHI Bears').
    parts=s.split()
    if len(parts)>1 and norm_team(parts[0]) in set(TEAM_NAMES.values()):
        remainder=' '.join(parts[1:])
        if remainder in TEAM_NAMES and TEAM_NAMES[remainder]==norm_team(parts[0]):return norm_team(parts[0])
    return None

def parse_official_html(html,season,week,fetched_at=None):
    soup=BeautifulSoup(html,'html.parser')
    rows=[]
    # Parse only tables with the official NFL Player/Position/Injuries/Practice/Game Status schema.
    for table in soup.find_all('table'):
        head=table.find('thead') or table
        heads=[' '.join(x.stripped_strings).strip().lower() for x in head.find_all('th')]
        if not heads or not any('player' in h for h in heads) or not any('game status' in h for h in heads):continue
        pidx=next((i for i,h in enumerate(heads) if 'player' in h),None)
        sidx=next((i for i,h in enumerate(heads) if 'game status' in h),None)
        if pidx is None or sidx is None:continue
        # Only a nearby, unambiguous team heading may establish team identity.
        team=None
        for prev in table.find_all_previous(['h2','h3','h4','h5','h6'],limit=8):
            candidate=_team_label(' '.join(prev.stripped_strings))
            if candidate:team=candidate;break
        if team is None:continue
        for tr in table.find_all('tr'):
            cells=tr.find_all('td',recursive=False)
            if len(cells)<=max(pidx,sidx):continue
            player=' '.join(cells[pidx].stripped_strings).strip()
            raw=' '.join(cells[sidx].stripped_strings).strip()
            if not player or not re.search('[a-zA-Z]',player):continue
            rows.append({'team':team,'player':player,'status':normalize_status(raw),'raw_status':raw,
                         'report_date':f'{season} week {week}','source':'NFL.com official injury report',
                         'fetched_at':fetched_at or dt.datetime.now(dt.timezone.utc).isoformat()})
    df=pd.DataFrame(rows,columns=COLUMNS)
    if len(df)<10 or df.team.nunique()<3:
        raise ValueError(f'NFL HTML schema validation failed ({len(df)} rows, {df.team.nunique()} teams). No NFL exclusions applied.')
    # Conflicting rows for same identity are unsafe, never use either for automatic exclusion.
    bad=df.groupby(['team','player']).status.nunique()
    conflicts=set(bad[bad>1].index)
    if conflicts:df=df[~df.apply(lambda r:(r.team,r.player) in conflicts,axis=1)]
    return df.drop_duplicates(['team','player'])

def fetch_live(season=2026,week=5,timeout=12):
    url=f'https://www.nfl.com/injuries/league/{int(season)}/reg{int(week)}'
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Accept':'text/html'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        if response.status!=200:raise ValueError(f'NFL report HTTP {response.status}')
        html=response.read().decode('utf-8','replace')
    return parse_official_html(html,season,week)

def match_to_pool(feed,pool):
    if feed is None or feed.empty:return pd.DataFrame(columns=COLUMNS+['matched_name','match_state'])
    idx={}
    for _,r in pool.iterrows():
        key=(norm_team(r.get('TeamAbbrev')),norm_name(r.get('Name')))
        idx.setdefault(key,[]).append(r)
    out=[]
    for _,r in feed.iterrows():
        matches=idx.get((norm_team(r.team),norm_name(r.player)),[])
        if len(matches)!=1:continue
        p=matches[0];d=r.to_dict();d['matched_name']=str(p.Name);d['match_state']='EXACT_TEAM_NAME';d['position']=str(p.get('Position',''))
        out.append(d)
    return pd.DataFrame(out)

def confirmed_out(matched):
    if matched is None or matched.empty:return []
    return [{'team':str(r.team),'player':str(r.matched_name)} for _,r in matched[matched.status.eq('OUT')].drop_duplicates(['team','matched_name']).iterrows()]
