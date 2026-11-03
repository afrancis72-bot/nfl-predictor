"""Source-grounded historical-finish evidence in uploaded RunGood golf CSVs.

The standard RRG last-50 SG export contains "Recent -1S"..."Recent -12S"
and "Hist 2025"... event-history finish strings. These are *not* historical
strokes gained, and tour/event dates are not independently verified. Never
pretend they establish a calibrated probability of making the next cut.
"""
from __future__ import annotations

import re
import unicodedata
import numpy as np
import pandas as pd


def name_key(value):
    value=str(value or "").strip()
    if "," in value:
        last, first=value.split(",",1)
        if last.strip() and first.strip():
            value=first.strip()+" "+last.strip()
    value=unicodedata.normalize("NFKD",value).encode("ascii","ignore").decode().lower()
    return re.sub(r"[^a-z0-9]","",value)


def parse_finish(value):
    if pd.isna(value): return None
    s=str(value).strip().upper()
    if not s or s in {"NAN","NONE","-","--","L"}: return None
    if s in {"W","WIN","WINNER"}: return ("finish",1)
    if s in {"CUT","MC","MDF"}: return ("cut",None)
    if s in {"WD","W/D","DQ","DNF","DQF"}: return ("withdrawal",None)
    m=re.fullmatch(r"T?(\d{1,3})",s)
    if m and int(m.group(1))>0:
        return ("finish",int(m.group(1)))
    return None


def extract_finish_history(raw,dk,event_year=None):
    """Return DK-aligned review evidence and bounded experimental form signal.

    Last eight recent starts drive a small, optional form correction. A recorded
    withdrawal is never counted as a missed cut. Event-history columns are
    audit-only (event/venue comparability cannot be inferred from their names).
    """
    empty={"recent_matched":0,"historical_event_matched":0,"source":"not supplied",
           "recent_finish_columns":[],"event_history_columns":[]}
    if raw is None or raw.empty: return pd.DataFrame(),empty
    columns={str(c).strip():c for c in raw.columns}
    recent=[]
    for label,col in columns.items():
        match=re.fullmatch(r"Recent\s*-\s*(\d+)S",label,re.I)
        if match and 1<=int(match.group(1))<=8:
            recent.append((int(match.group(1)),col))
    recent.sort()
    event_history=[]
    for label,col in columns.items():
        match=re.fullmatch(r"Hist\s+(\d{4})",label,re.I)
        if match and (event_year is None or int(match.group(1))<int(event_year)):
            event_history.append((int(match.group(1)),col))
    event_history.sort(reverse=True)
    if not recent and not event_history: return pd.DataFrame(),empty
    name_col=next((columns[c] for c in columns
                   if re.sub(r"[^a-z0-9]","",c.casefold()) in
                   {"name","player","golfer","playername","golfername"}),None)
    if name_col is None:
        raise ValueError("RRG historical finishes require a golfer Name/Player field.")
    rows=[]
    for _,row in raw.iterrows():
        r={"_key":name_key(row[name_col])}
        recent_vals=[parse_finish(row[col]) for _,col in recent]
        started=[v for v in recent_vals if v is not None and v[0] in {"finish","cut"}]
        finishes=[v[1] for v in started if v[0]=="finish"]
        r["recent_starts"]=len(started)
        r["recent_cuts_made"]=len(finishes)
        r["recent_missed_cuts"]=len(started)-len(finishes)
        r["recent_top25"]=sum(n<=25 for n in finishes)
        r["recent_cut_rate"]=len(finishes)/len(started) if started else np.nan
        r["recent_top25_rate"]=r["recent_top25"]/len(started) if started else np.nan
        r["recent_withdrawals"]=sum(v is not None and v[0]=="withdrawal" for v in recent_vals)
        r["recent_finishes_observed"]=r["recent_starts"]+r["recent_withdrawals"]
        r["recent_last_result"]=str(row[recent[0][1]]) if recent else ""
        event_vals=[parse_finish(row[col]) for _,col in event_history]
        event_known=[v for v in event_vals if v is not None and v[0] in {"finish","cut"}]
        event_success=sum(v[0]=="finish" for v in event_known)
        r["historical_event_starts"]=len(event_known)
        r["historical_event_cuts_made"]=event_success
        r["historical_event_cut_rate"]=(event_success/len(event_known)
                                         if event_known else np.nan)
        # Provisional bounded tilt, NOT a historical cut-probability model:
        # eight raw RRG start results may mix tours/fields; do not overweight.
        n=r["recent_starts"]
        if n>=4:
            reliability=min(1.,n/8.)
            cut_delta=r["recent_cut_rate"]-0.65
            finish_delta=r["recent_top25_rate"]-0.30
            r["recent_form_adjustment_strokes"]=float(
                np.clip((-.18*cut_delta-.08*finish_delta)*reliability,-.12,.12))
            r["recent_form_evidence"]="recent finish history: bounded provisional adjustment (tour unverified)"
        else:
            r["recent_form_adjustment_strokes"]=0.
            r["recent_form_evidence"]="insufficient recent starts: no adjustment"
        rows.append(r)
    data=pd.DataFrame(rows)
    if data["_key"].eq("").any() or data["_key"].duplicated().any():
        raise ValueError("Historical finish source has missing or duplicate golfer names.")
    names=dk[["player"]].copy()
    names["_key"]=names["player"].map(name_key)
    combined=names.merge(data,on="_key",how="left",validate="one_to_one").drop(columns="_key")
    recent_matched=int(combined["recent_starts"].fillna(0).ge(4).sum())
    hist_matched=int(combined["historical_event_starts"].fillna(0).gt(0).sum())
    combined["recent_form_adjustment_strokes"]=combined[
        "recent_form_adjustment_strokes"].fillna(0.)
    combined["recent_form_evidence"]=combined[
        "recent_form_evidence"].fillna("no matched recent finish history")
    return combined,{"recent_matched":recent_matched,
                     "historical_event_matched":hist_matched,
                     "source":"RunGood recent finishes; no individually verified dates/tour level",
                     "recent_finish_columns":[str(n) for n,_ in recent],
                     "event_history_columns":[str(n) for n,_ in event_history]}
