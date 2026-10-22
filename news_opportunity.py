"""Evidence-based, bounded injury opportunity redistribution for NFL Classic.

Uses prior completed regular-season player-week data only. No data => no boosts.
Does not claim a full possession simulation; rescales projected fantasy distributions.
"""
import numpy as np
import pandas as pd
from auto_injury_feed import norm_team,norm_name

MAX_SHARE_BONUS = 0.22
MIN_TEAM_GAMES = 3
MIN_PLAYER_GAMES = 2

def _history(history,season,week):
    x=history.copy()
    aliases={"player_display_name":"player_name","team":"recent_team"}
    for src,dst in aliases.items():
        if dst not in x.columns and src in x.columns:x[dst]=x[src]
    required={"season","week","player_name","recent_team","position","targets","carries"}
    if not required.issubset(x.columns):
        raise ValueError("Usage history lacks: "+", ".join(sorted(required-set(x.columns))))
    x["season"]=pd.to_numeric(x["season"],errors="coerce")
    x["week"]=pd.to_numeric(x["week"],errors="coerce")
    x=x[(x["season"]<int(season))|((x["season"]==int(season))&(x["week"]<int(week)))].copy()
    if "season_type" in x.columns:
        x=x[x["season_type"].astype(str).str.upper().eq("REG")]
    x=x[x["season"].ge(int(season)-1)]
    x["team_key"]=x["recent_team"].map(norm_team)
    x["name_key"]=x["player_name"].map(norm_name)
    x["position"]=x["position"].astype(str).str.upper()
    for c in ("targets","carries"):
        x[c]=pd.to_numeric(x[c],errors="coerce").fillna(0).clip(lower=0)
    return x

def redistribute(pool,injuries,metadata,history,season,week):
    """Returns (pool, audit, status). Automatic boosts only with adequate prior evidence."""
    output=pool.copy()
    status={"ok":False,"reason":"","players_adjusted":0}
    if not metadata.get("ok") or metadata.get("stale") or injuries is None or injuries.empty:
        status["reason"]="Unavailable/stale injury feed: no redistribution"
        return output,pd.DataFrame(),status
    try:
        h=_history(history,season,week)
    except Exception as exc:
        status["reason"]=str(exc)
        return output,pd.DataFrame(),status
    if h.empty:
        status["reason"]="No eligible historical player-weeks"
        return output,pd.DataFrame(),status
    output["_team"]=output["TeamAbbrev"].map(norm_team)
    output["_name"]=output["Name"].map(norm_name)
    audit=[]
    # Out list restricted to confirmed feed statuses and current slate names.
    unavailable=set()
    for r in injuries.itertuples(index=False):
        if str(r.report_status).upper() in {"OUT","IR","INACTIVE","SUSPENDED"}:
            unavailable.add((str(r.team_key),str(r.name_key)))
    current=set(zip(output["_team"],output["_name"]))
    unavailable &= current
    # Only targets and carries, by position and team. Never reassign QB attempts.
    for team in sorted({t for t,_ in unavailable}):
        th=h[h.team_key.eq(team)]
        n_games=th[["season","week"]].drop_duplicates().shape[0]
        if n_games<MIN_TEAM_GAMES:continue
        for metric,positions in (("targets",{"WR","TE","RB"}),("carries",{"RB"})):
            sub=th[th.position.isin(positions)]
            if sub.empty:continue
            # Rates per team-game, then conservative recipient allocation by usage.
            rates=sub.groupby("name_key")[metric].sum()/n_games
            missing=sum(float(rates.get(n,0)) for t,n in unavailable if t==team)
            if missing<=0:continue
            recipients=output[(output["_team"]==team)&
                              output["Position"].astype(str).isin(positions)&
                              output["optimizer_eligible"].fillna(False).astype(bool)]
            recipients=recipients[~recipients["_name"].isin([n for t,n in unavailable if t==team])]
            if recipients.empty:continue
            available_rates={idx:float(rates.get(row["_name"],0)) for idx,row in recipients.iterrows()}
            # Require demonstrated opportunity, never allocate to zero-usage backups.
            available_rates={idx:v for idx,v in available_rates.items() if v>0 and
                             sub[sub.name_key.eq(output.at[idx,"_name"])][["season","week"]].drop_duplicates().shape[0]>=MIN_PLAYER_GAMES}
            total=sum(available_rates.values())
            if total<=0:continue
            for idx,rate in available_rates.items():
                # Partial recovery: 50% of vacated volume; rest absorbed by
                # personnel/game-plan changes. Clamp to protect against spurious feeds.
                gained=0.5*missing*rate/total
                pct=min(MAX_SHARE_BONUS,gained/max(rate,1.0))
                # A target contributes differently from a carry; modest projection
                # multiplier, combined metric effects capped below.
                boost=pct*(0.55 if metric=="targets" else 0.35)
                old=float(output.at[idx,"news_opportunity_boost"]) if "news_opportunity_boost" in output and pd.notna(output.at[idx,"news_opportunity_boost"]) else 0.0
                output.at[idx,"news_opportunity_boost"]=min(0.18,old+boost)
                audit.append({"Name":output.at[idx,"Name"],"TeamAbbrev":team,
                              "metric":metric,"baseline_per_game":round(rate,2),
                              "vacated_per_game":round(missing,2),"allocated_per_game":round(gained,2),
                              "incremental_boost":round(boost,4),"injury_source":metadata.get("source"),
                              "history_end":f"{season} week {int(week)-1}"})
    if "news_opportunity_boost" in output:
        factors=1+pd.to_numeric(output["news_opportunity_boost"],errors="coerce").fillna(0).clip(0,0.18)
        for col in ("proj","ceiling","p95_use","p99_use"):
            if col in output:
                output[col]=pd.to_numeric(output[col],errors="coerce")*factors
    output=output.drop(columns=["_team","_name"])
    status["ok"]=True
    status["players_adjusted"]=len(set((a["Name"],a["TeamAbbrev"]) for a in audit))
    status["reason"]="Historical usage redistribution applied" if audit else "No supported recipient adjustments"
    return output,pd.DataFrame(audit),status
