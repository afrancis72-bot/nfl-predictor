"""Automatic, fail-closed NFL injury intelligence from nflverse public release.

This is a structured injury/practice feed, NOT automated coach transcript analysis.
It never guesses opportunity redistribution. No feed => baseline unchanged and warning.
"""
from datetime import datetime, timezone
from io import BytesIO
import pandas as pd
import urllib.request

URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"
ALIASES = {"JAX":"JAC","WSH":"WAS","LAR":"LA","STL":"LA","OAK":"LV","SD":"LAC"}
EXCLUDE = {"OUT","IR","INACTIVE","SUSPENDED"}
VALID = {"OUT","IR","INACTIVE","SUSPENDED","DOUBTFUL","QUESTIONABLE","PROBABLE"}
def norm_team(s):
    s=str(s).strip().upper()
    return ALIASES.get(s,s)
def norm_name(s):
    return " ".join(str(s).lower().replace(".","").replace("'","").replace("-"," ").split())
def _get_csv(season,timeout=12):
    req=urllib.request.Request(URL.format(season=int(season)),headers={"User-Agent":"AI3Pro-NFL-Injury-Research/1.0"})
    with urllib.request.urlopen(req,timeout=timeout) as resp:
        if resp.status!=200: raise RuntimeError("Injury feed HTTP "+str(resp.status))
        data=resp.read(15_000_001)
    if len(data)>15_000_000: raise ValueError("Injury feed exceeds size limit")
    return pd.read_csv(BytesIO(data))
def get_injuries(season,week,fetcher=None):
    """Returns (current-week dataframe, metadata). No prior-week fallback."""
    url=URL.format(season=int(season))
    meta={"source":url,"checked_at":datetime.now(timezone.utc).isoformat(),
          "season":int(season),"week":int(week),"ok":False,"error":None}
    try:
        frame=(_get_csv if fetcher is None else fetcher)(int(season))
        required={"season","week","team","full_name","report_status","date_modified"}
        if not required.issubset(frame.columns):
            raise ValueError("Missing injury-feed columns: "+str(sorted(required-set(frame.columns))))
        x=frame.copy()
        x["season"]=pd.to_numeric(x["season"],errors="coerce")
        x["week"]=pd.to_numeric(x["week"],errors="coerce")
        x=x[(x["season"]==int(season)) & (x["week"]==int(week))].copy()
        if x.empty: raise ValueError("No reports for the requested season/week; do not reuse prior-week data")
        x["date_modified"]=pd.to_datetime(x["date_modified"],utc=True,errors="coerce")
        x=x.dropna(subset=["date_modified","full_name","team"])
        if x.empty: raise ValueError("No dated injury records")
        x["report_status"]=x["report_status"].fillna("").astype(str).str.upper().str.strip()
        x["practice_status"]=x.get("practice_status",pd.Series("",index=x.index)).fillna("").astype(str)
        x["team_key"]=x["team"].map(norm_team)
        x["name_key"]=x["full_name"].map(norm_name)
        x=x.sort_values("date_modified").drop_duplicates(["team_key","name_key"],keep="last")
        meta["ok"]=True
        meta["rows"]=len(x)
        meta["latest_report"]=x["date_modified"].max().isoformat()
        meta["stale"]=((pd.Timestamp.now(tz="UTC")-x["date_modified"].max()).total_seconds()>60*60*72)
        return x,meta
    except Exception as exc:
        meta["error"]=str(exc)
        return pd.DataFrame(),meta

def apply_injury_feed(pool,injuries,metadata):
    """Fail-closed: only verified structured OUT/IR/INACTIVE/SUSPENDED exclude.

    No questionable player automatically excluded, and no player score artificially boosted.
    """
    result=pool.copy()
    audit=[]
    if not metadata.get("ok") or metadata.get("stale") or injuries.empty:
        return result,pd.DataFrame(audit)
    lookup={(r.team_key,r.name_key):r for r in injuries.itertuples(index=False)}
    for idx,row in result.iterrows():
        key=(norm_team(row.get("TeamAbbrev","")),norm_name(row.get("Name","")))
        item=lookup.get(key)
        if item is None: continue
        status=item.report_status
        excluded=status in EXCLUDE
        if excluded:
            result.at[idx,"optimizer_eligible"]=False
            result.at[idx,"news_status"]=status
        audit.append({"Name":row.get("Name"),"TeamAbbrev":row.get("TeamAbbrev"),
                      "report_status":status or "NOT DESIGNATED",
                      "practice_status":item.practice_status,
                      "excluded":excluded,"source":metadata.get("source"),
                      "date_modified":str(item.date_modified)})
    return result,pd.DataFrame(audit)
