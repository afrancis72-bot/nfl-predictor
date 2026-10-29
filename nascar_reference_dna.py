import re, unicodedata
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import numpy as np, pandas as pd

BASE="https://nascar-reference.com"
UA="Mozilla/5.0 (compatible; NASCAR-Predictor/2.5)"

def _get(path,timeout=12):
    req=Request(BASE+path,headers={"User-Agent":UA,"Accept":"text/html,*/*"})
    with urlopen(req,timeout=timeout) as r:
        return r.read().decode("utf-8","replace")

def _slug(name):
    s=unicodedata.normalize("NFKD",str(name)).encode("ascii","ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+","-",s).strip("-")

def _tables(html):
    try:return pd.read_html(html)
    except:return []

def _normcol(c):
    if isinstance(c,tuple): c=" ".join(str(v) for v in c if str(v).lower()!="nan")
    c=unicodedata.normalize("NFKD",str(c)).encode("ascii","ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+"," ",c).strip()

def _flat(t):
    x=t.copy(); x.columns=[_normcol(c) for c in x.columns]; return x

def _find(cols,keys):
    keys=[_normcol(k) for k in keys]
    for k in keys:
        for c in cols:
            if c==k:return c
    for k in keys:
        for c in cols:
            if k in c:return c
    return None

def _num(v):
    return pd.to_numeric(pd.Series(v).astype(str).str.extract(r"([-+]?\d+(?:\.\d+)?)",expand=False),errors="coerce")

def _z(series, invert=False):
    x=pd.to_numeric(series,errors="coerce")
    if x.notna().sum()<2:return pd.Series(np.nan,index=x.index)
    sd=x.std(ddof=0)
    z=(x-x.mean())/(sd if sd and np.isfinite(sd) else 1)
    return -z if invert else z

def _pct_score(series,invert=False):
    x=pd.to_numeric(series,errors="coerce")
    if x.notna().sum()<2:return pd.Series(np.nan,index=x.index)
    r=x.rank(pct=True)
    if invert:r=1-r+1/len(x.dropna())
    return (r*100).clip(0,100)

def driver_page(name):
    return _get("/drivers/"+_slug(name))

def _page_text(html):
    # NASCAR Reference renders statistical sections as normal page text rather than <table> tags.
    return " | ".join(BeautifulSoup(html,"html.parser").stripped_strings)

def _between(txt,start,end=None):
    i=txt.lower().find(start.lower())
    if i<0:return ""
    i+=len(start)
    if end:
        j=txt.lower().find(end.lower(),i)
        if j>=0:return txt[i:j]
    return txt[i:]

def _metric_before_label(section,label):
    # With stripped_strings joined by pipes: "... | 10.45 | Avg Start | ..."
    m=re.search(rf"(?:^|\|)\s*([-+]?\d+(?:\.\d+)?)\s*\|\s*{re.escape(label)}(?:\s*\||$)",section,re.I)
    return float(m.group(1)) if m else np.nan

def parse_profile_text(html):
    txt=_page_text(html)
    # NASCAR Reference currently uses both headings across driver pages.
    season_heading="2026 Cup Season" if "2026 Cup Season" in txt else "2026 Season"
    season=_between(txt,season_heading,"Best Tracks")
    perf=_between(txt,"Performance Profile",season_heading)
    recent=_between(txt,"Recent Results","Year-by-Year Results")

    out={
        "season_races":_metric_before_label(season,"Races"),
        "season_wins":_metric_before_label(season,"Wins"),
        "season_top5":_metric_before_label(season,"Top 5"),
        "season_top10":_metric_before_label(season,"Top 10"),
        "season_avg_start":_metric_before_label(season,"Avg Start"),
        "season_avg_finish":_metric_before_label(season,"Avg Finish"),
        "last5_avg_finish":_metric_before_label(season,"Last 5 Avg"),
        "superspeedway_avg_finish":np.nan,
        "intermediate_avg_finish":np.nan,
        "short_track_avg_finish":np.nan,
        "road_course_avg_finish":np.nan,
        "superspeedway_elo":np.nan,
    }
    for label,key in [("Superspeedway","superspeedway_avg_finish"),
                      ("Intermediate","intermediate_avg_finish"),
                      ("Short Track","short_track_avg_finish"),
                      ("Road Course","road_course_avg_finish")]:
        out[key]=_metric_before_label(perf,label)

    # Recent Results: capture date, Cup, race name, start, finish, +/-.
    # Race name is non-greedy; finish accepts "1st" etc.
    pat=re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})\s*\|\s*Cup\s*\|\s*"
        r"(.+?)\s*\|\s*(\d+)\s*\|\s*(\d+)(?:st|nd|rd|th)?\s*\|\s*([+\-]?\d+|—|-)",
        re.I)
    races=[]
    for m in pat.finditer(recent):
        races.append({"date":f"{m.group(1)} {m.group(2)}","race":m.group(3).strip(),
                      "start":float(m.group(4)),"finish":float(m.group(5)),
                      "place_diff":float(m.group(4))-float(m.group(5))})
    out["recent_rows"]=races[:10]
    if races:
        rr=races[:10]
        out["recent_avg_start"]=float(np.mean([r["start"] for r in rr]))
        out["recent_avg_finish"]=float(np.mean([r["finish"] for r in rr]))
        out["recent_place_diff"]=float(np.mean([r["place_diff"] for r in rr]))
        out["recent_races"]=len(rr)
    else:
        out["recent_avg_start"]=out["season_avg_start"]
        out["recent_avg_finish"]=out["last5_avg_finish"] if pd.notna(out["last5_avg_finish"]) else out["season_avg_finish"]
        out["recent_place_diff"]=(out["recent_avg_start"]-out["recent_avg_finish"]
                                  if pd.notna(out["recent_avg_start"]) and pd.notna(out["recent_avg_finish"]) else np.nan)
        out["recent_races"]=0

    elo=_between(txt,"NR-Rating Elo Ratings","Performance Quality Score")
    # First Superspeedway occurrence after NR-Rating is the track-type Elo.
    m=re.search(r"(?:^|\|)\s*(\d{3,4})\s*\|\s*Superspeedway(?:\s*\||$)",elo,re.I)
    if m: out["superspeedway_elo"]=float(m.group(1))

    # Advanced Metrics are rendered as "Label | value | confidence..." in page text.
    adv=_between(txt,"Advanced Metrics","Career History")
    def metric_after_label(label):
        m=re.search(rf"{re.escape(label)}\s*\|\s*([-+]?\d+(?:\.\d+)?)",adv,re.I)
        return float(m.group(1)) if m else np.nan
    out["reference_reliability"]=metric_after_label("Reliability")
    out["pit_stop_quality"]=metric_after_label("Pit Stop Quality")
    out["tire_management"]=metric_after_label("Tire Management")
    out["late_race_performance"]=metric_after_label("Late-Race Performance")
    return out

def build_reference_dna(dk, selected_track):
    atl="Atlanta" in selected_track
    rows=[]
    for name in dk["Name"].astype(str):
        row={"Name":name}
        try:
            html=driver_page(name)
            q=parse_profile_text(html)
            row.update(q)
            row["comparable_track_avg_finish"]=q["superspeedway_avg_finish"] if atl else q["intermediate_avg_finish"]
            row["reference_ok"]=True
        except Exception:
            row.update({"recent_races":0,"recent_avg_finish":np.nan,"recent_avg_start":np.nan,
                        "recent_place_diff":np.nan,"season_avg_finish":np.nan,"last5_avg_finish":np.nan,
                        "comparable_track_avg_finish":np.nan,"superspeedway_elo":np.nan,"reference_ok":False})
        rows.append(row)
    x=pd.DataFrame(rows)

    # Current form: blend recent-10 (or last-5 fallback) and season finish.
    x["recent_finish_score"]=_pct_score(x["recent_avg_finish"],invert=True)
    x["season_finish_score"]=_pct_score(x["season_avg_finish"],invert=True)
    x["recent_form_score"]=x[["recent_finish_score","season_finish_score"]].mean(axis=1,skipna=True)
    x["place_diff_score"]=_pct_score(x["recent_place_diff"])
    x["comparable_track_score"]=_pct_score(x["comparable_track_avg_finish"],invert=True)
    x["track_type_elo_score"]=_pct_score(x["superspeedway_elo"]) if atl else np.nan

    # Prefer NASCAR Reference's own Reliability metric when present.
    # Fall back to finish consistency rather than inventing a value.
    source_rel=_pct_score(x["reference_reliability"]) if "reference_reliability" in x.columns else pd.Series(np.nan,index=x.index)
    fallback_rel=x[["season_finish_score","recent_finish_score"]].mean(axis=1,skipna=True)
    x["reliability_score"]=source_rel.combine_first(fallback_rel)

    comps=["recent_form_score","place_diff_score","comparable_track_score","track_type_elo_score","reliability_score"]
    x["reference_components_available"]=x[comps].notna().sum(axis=1)
    x["reference_coverage"]=x["reference_components_available"]/len(comps)
    return x

def blend_driver_dna(dk, ref):
    z=dk[["Name","Salary","AvgPointsPerGame"]].merge(ref,on="Name",how="left")
    z["dk_market_score"]=(0.55*_pct_score(z["AvgPointsPerGame"])+0.45*_pct_score(z["Salary"])).fillna(50)
    weights={"recent_form_score":.25,"place_diff_score":.10,"comparable_track_score":.20,
             "track_type_elo_score":.15,"reliability_score":.15,"dk_market_score":.15}
    vals=[]
    for _,r in z.iterrows():
        num=den=0.0
        for c,w in weights.items():
            v=r.get(c,np.nan)
            if pd.notna(v): num+=float(v)*w; den+=w
        vals.append(num/den if den else 50.0)
    z["driver_dna_score"]=vals
    z["data_coverage_pct"]=(15+85*z["reference_coverage"].fillna(0)).round(0)
    z["model_confidence"]=np.where(z["data_coverage_pct"]>=80,"HIGH",
                           np.where(z["data_coverage_pct"]>=55,"MEDIUM","LOW"))
    return z
