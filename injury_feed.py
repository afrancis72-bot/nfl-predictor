"""NFL public weekly news report ingestion. Never infer absence from practice status.
HTML from NFL.com is an unofficial integration: parsing failure is explicit.
"""
import datetime as dt
import re
import urllib.request
import pandas as pd
from bs4 import BeautifulSoup

TEAM_ALIAS={'JAC':'JAX','WSH':'WAS','OAK':'LV','SD':'LAC','STL':'LA','LAR':'LA','ARZ':'ARI'}
TEAMS={'CARDINALS':'ARI','FALCONS':'ATL','RAVENS':'BAL','BILLS':'BUF','PANTHERS':'CAR','BEARS':'CHI','BENGALS':'CIN','BROWNS':'CLE','COWBOYS':'DAL','BRONCOS':'DEN','LIONS':'DET','PACKERS':'GB','TEXANS':'HOU','COLTS':'IND','JAGUARS':'JAX','CHIEFS':'KC','RAIDERS':'LV','CHARGERS':'LAC','RAMS':'LA','DOLPHINS':'MIA','VIKINGS':'MIN','PATRIOTS':'NE','SAINTS':'NO','GIANTS':'NYG','JETS':'NYJ','EAGLES':'PHI','STEELERS':'PIT','49ERS':'SF','SEAHAWKS':'SEA','BUCCANEERS':'TB','TITANS':'TEN','COMMANDERS':'WAS'}
COLUMNS=['team','player','status','raw_status','report_date','source','fetched_at']
POSITIONS=r'QB|RB|FB|WR|TE|OL|OT|OG|G|C|T|DE|DT|DL|EDGE|LB|ILB|OLB|CB|DB|S|FS|SS|K|P|LS'

def norm_name(s):
    p=re.sub(r'[^a-z0-9 ]',' ',str(s).lower()).split()
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
def _heading_team(text):
    s=re.sub(r'[^A-Z0-9 ]',' ',str(text).upper()).strip()
    # Only use headings, not arbitrary prose; headings may include a city prefix.
    for name,abbr in TEAMS.items():
        if s==name or s.endswith(' '+name):return abbr
    return None

def parse_official_html(html,season,week,fetched_at=None):
    """Parse NFL.com's public weekly NEWS article, not its JS-rendered injury table."""
    soup=BeautifulSoup(html,'html.parser')
    # Restrict to main article where available; avoid menus and unrelated news.
    root=soup.find('article') or soup.find('main') or soup
    rows=[]; team=None
    now=fetched_at or dt.datetime.now(dt.timezone.utc).isoformat()
    for el in root.find_all(['h2','h3','h4','h5','h6','li','p'],recursive=True):
        if el.name.startswith('h'):
            team=_heading_team(el.get_text(' ',strip=True))
            continue
        if not team:continue
        content=el.get_text(' ',strip=True)
        m=re.match(r'^(OUT|QUESTIONABLE|DOUBTFUL)\s*:\s*(.+)$',content,re.I)
        if not m:continue
        status=normalize_status(m.group(1)); body=m.group(2)
        # One item can contain many comma-separated players with positions.
        chunks=re.split(r',\s*(?=(?:'+POSITIONS+r')\s+)',body,flags=re.I)
        for chunk in chunks:
            found=re.match(r'^(?:'+POSITIONS+r')\s+(.+?)(?:\s*\(|$)',chunk.strip(),re.I)
            if not found:continue
            player=found.group(1).strip().rstrip(' ,.;')
            if not player or len(player)>80:continue
            rows.append(dict(team=team,player=player,status=status,raw_status=m.group(1).upper(),report_date=f'{season} week {week}',source='NFL.com weekly injury news',fetched_at=now))
    df=pd.DataFrame(rows,columns=COLUMNS)
    if len(df)<10 or df.team.nunique()<5:
        raise ValueError(f'NFL weekly news schema validation failed ({len(df)} players, {df.team.nunique()} teams); no NFL exclusions applied')
    conflicts=df.groupby(['team','player']).status.nunique()
    bad=set(conflicts[conflicts>1].index)
    if bad:df=df[~df.apply(lambda r:(r.team,r.player) in bad,axis=1)]
    return df.drop_duplicates(['team','player'])

def fetch_live(season=2026,week=5,timeout=12):
    urls=[
        f'https://www.nfl.com/news/nfl-week-{int(week)}-injury-report-player-statuses-for-all-15-games',
        f'https://fantasy-www.nfl.com/news/nfl-week-{int(week)}-injury-report-player-statuses-for-all-15-games',
    ]
    errors=[]
    for url in urls:
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (compatible; InjuryAudit/1.0)','Accept':'text/html'})
            with urllib.request.urlopen(req,timeout=timeout) as r:
                if r.status!=200:raise ValueError(f'HTTP {r.status}')
                html=r.read().decode('utf-8','replace')
            return parse_official_html(html,season,week)
        except Exception as exc:errors.append(f'{url.split("/")[2]}: {str(exc)[:130]}')
    raise ValueError('Official weekly injury news unavailable: '+'; '.join(errors))

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
    return [{'team':norm_team(r.team),'player':str(r.matched_name)} for _,r in matched[matched.status=='OUT'].iterrows()]
