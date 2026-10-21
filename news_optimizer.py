"""Auditable, conservative NFL news-to-projection adjustments.

Input news CSV: player,team,position,event_type,status,confidence,source_url,
published_at,notes,delta_pct,verified.
Only explicitly verified rows can alter projections. No guessed redistribution.
"""
import pandas as pd
import numpy as np

REQUIRED = {"player","team","event_type","status","source_url","published_at","verified","delta_pct"}
BLOCKED = {"OUT","IR","INACTIVE","SUSPENDED"}
STATUS = BLOCKED | {"QUESTIONABLE","DOUBTFUL","LIMITED","FULL","ACTIVE","UNKNOWN"}
MAX_ABS_DELTA = 0.25

def _key(value):
    return " ".join(str(value).casefold().replace(".","").replace("'","").split())

def validate_news(news):
    if news is None or news.empty:
        return pd.DataFrame(), ["No news rows supplied"]
    df = news.copy()
    df.columns = [str(c).strip().casefold() for c in df.columns]
    missing = REQUIRED - set(df.columns)
    if missing:
        return pd.DataFrame(), ["Missing columns: "+", ".join(sorted(missing))]
    df["verified"] = df["verified"].astype(str).str.strip().str.lower().isin(["true","1","yes"])
    df["delta_pct"] = pd.to_numeric(df["delta_pct"], errors="coerce")
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["status"] = df["status"].astype(str).str.upper().str.strip()
    df["player_key"] = df["player"].map(_key)
    df["team_key"] = df["team"].astype(str).str.upper().str.strip()
    df["usable"] = (df["verified"] & df["published_at"].notna() &
                    df["delta_pct"].notna() & df["delta_pct"].abs().le(MAX_ABS_DELTA) &
                    df["source_url"].astype(str).str.startswith(("https://","http://")) &
                    df["status"].isin(STATUS) & df["player_key"].ne(""))
    df["reason"] = np.where(df["usable"], "verified and bounded", "unverified, invalid or out of bounds")
    return df, []

def apply_news(pool, news, now=None):
    """Return (adjusted player pool, per-player audit, errors).

    Status exclusions only for VERIFIED individual player OUT/IR/INACTIVE/SUSPENDED.
    Other injuries and opponent news require explicit, independently reviewed
    player-level delta_pct; no inferred teammate/defender boosts.
    """
    df, errors = validate_news(news)
    result = pool.copy()
    if errors:
        return result, pd.DataFrame(), errors
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    df["usable"] &= df["published_at"].le(now)
    df = df[df["usable"]].sort_values("published_at")
    # Avoid double counting: latest verified event per player/team/event_type.
    df = df.drop_duplicates(["player_key","team_key","event_type"],keep="last")
    result["_news_key"] = result["Name"].map(_key)
    result["_news_team"] = result["TeamAbbrev"].astype(str).str.upper().str.strip()
    audits=[]
    for _,event in df.iterrows():
        match = (result["_news_key"].eq(event["player_key"]) &
                 result["_news_team"].eq(event["team_key"]))
        if not match.any():
            audits.append({"player":event["player"],"team":event["team"],"applied":False,
                           "reason":"not matched to current slate","source_url":event["source_url"]})
            continue
        idx=result.index[match]
        blocked = (event["status"] in BLOCKED and
                   str(event["event_type"]).strip().lower() in ("player_status","injury","inactive"))
        if blocked:
            result.loc[idx,"optimizer_eligible"] = False
            result.loc[idx,"news_status"] = event["status"]
        else:
            delta = float(event["delta_pct"])
            for col in ("proj","ceiling","p95_use","p99_use"):
                if col in result.columns:
                    result.loc[idx,col] = pd.to_numeric(result.loc[idx,col],errors="coerce") * (1+delta)
        audits.append({"player":event["player"],"team":event["team"],"applied":True,
                       "reason":"verified exclusion" if blocked else "reviewed projection adjustment",
                       "delta_pct":0 if blocked else float(event["delta_pct"]),
                       "source_url":event["source_url"],"published_at":str(event["published_at"])})
    result = result.drop(columns=["_news_key","_news_team"])
    return result, pd.DataFrame(audits), []
