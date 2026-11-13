"""Conservative, auditable PGA TOUR season-stat reader for Team AI² Pro.

Uses the already-installed independent pga_tour_api client, which wraps the
public browser-facing StatDetails endpoint (unsupported by PGA TOUR).
Only current/future season slates may consume a live season stats snapshot.
A historical slate cannot use today's season standings (lookahead leakage).

Missing/ambiguous stats remain missing; values never fall back to rankings.
User-uploaded RunGood stats take precedence over these official-season figures.
"""
from __future__ import annotations

from datetime import date
import re
import pandas as pd
import numpy as np
from pga_input_integrity import align_to_dk

# Native IDs preserve meaningful leading zeroes.
# For each, use the value/average/percentage, not the leaderboard rank.
STAT_IDS={
    "sg_ott":"02567","sg_approach":"02568","sg_arg":"02569","sg_putting":"02564",
    "birdie_rate":"352","driving_distance":"101","driving_accuracy":"102",
    "gir_rate":"103","scrambling":"130","bogey_avoidance":"02414",
    "scoring_before_cut":"116",
    "par3_scoring":"142","par4_scoring":"143","par5_scoring":"144",
    "approach_175_200":"337","approach_200_225":"02358",
}
# Parsing percentage values should retain the site's 0-100 numeric scale.
STAT_VALUE_KEYS={
    "sg_ott":("avg","average"),
    "sg_approach":("avg","average"),
    "sg_arg":("avg","average"),
    "sg_putting":("avg","average"),
    "birdie_rate":("percentage","pct","percent","birdie_or_better_pct"),
    "driving_distance":("avg","average","yards","distance"),
    "driving_accuracy":("percentage","pct","percent"),
    "gir_rate":("percentage","pct","percent"),
    "scrambling":("percentage","pct","percent"),
    "bogey_avoidance":("percentage","pct","percent"),
    "scoring_before_cut":("avg","average"),
    "par3_scoring":("avg","average"),
    "par4_scoring":("avg","average"),
    "par5_scoring":("avg","average"),
    "approach_175_200":("avg","average","feet"),
    "approach_200_225":("avg","average","feet"),
}
META={"player_id","player_name","rank","rank_diff","rank_change_tendency","year","stat_id","country","country_flag"}
_NUMERIC=re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?$")

def _parse_numeric(x):
    """Accept plain numbers and percent, never feet/inches or rank fractions."""
    s=str(x).strip().replace(",","")
    if not _NUMERIC.fullmatch(s):return np.nan
    try:return float(s.removesuffix("%"))
    except (ValueError,TypeError):return np.nan

def normalize_stat_table(frame, stat_name):
    """Return (player, raw numeric stat), with validated original-stat value."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["player",stat_name]),"empty"
    if "player_name" not in frame.columns:
        return pd.DataFrame(columns=["player",stat_name]),"missing player_name"
    data=frame.copy()
    if "stat_id" in data.columns:
        data=data.loc[data["stat_id"].astype(str).eq(STAT_IDS[stat_name])].copy()
    if data.empty:return pd.DataFrame(columns=["player",stat_name]),"stat ID mismatch"
    names=list(data.columns)
    candidates=[]
    # One value column must correspond to the expected unit-bearing stat header.
    for want in STAT_VALUE_KEYS[stat_name]:
        matches=[c for c in names if str(c).casefold()==want and c not in META]
        if matches:
            candidates.extend(matches)
            break
    if not candidates:
        return pd.DataFrame(columns=["player",stat_name]),"expected value header unavailable: "+", ".join(map(str,names))
    if len(candidates)!=1:
        return pd.DataFrame(columns=["player",stat_name]),"ambiguous value columns"
    raw=data[candidates[0]].map(_parse_numeric)
    out=pd.DataFrame({"player":data["player_name"].astype(str).str.strip(),stat_name:raw})
    out=out[out.player.ne("")&out[stat_name].notna()].copy()
    if out.empty:return pd.DataFrame(columns=["player",stat_name]),"no valid numeric values"
    if out.player.duplicated().any():
        return pd.DataFrame(columns=["player",stat_name]),"duplicate player names in official rankings"
    return out.reset_index(drop=True),"accepted: "+candidates[0]

def retrieve_official_season_stats(players, tournament_start, *,
                                   provider=None, selected_stats=None,
                                   today=None):
    """Fetch selected PGA TOUR season stats and align exact player identity.

    Historical backtests must use saved pre-lock snapshots, never the LIVE
    season rankings. For a slate starting today, data might already include
    an in-progress event; users must verify the lock time.
    """
    lock=pd.Timestamp(tournament_start).date()
    now=pd.Timestamp(today).date() if today is not None else date.today()
    if lock<now:
        raise ValueError("Live PGA TOUR season stats cannot be used for a past slate: would leak post-lock results. Use dated pre-lock snapshots.")
    if provider is None:
        import pga_tour_api as provider
    chosen=tuple(selected_stats) if selected_stats is not None else tuple(STAT_IDS)
    unexpected=set(chosen)-set(STAT_IDS)
    if unexpected:raise ValueError(f"Unknown PGA TOUR stats {sorted(unexpected)}")
    field=players[["player"]].drop_duplicates("player").copy()
    if field.empty:raise ValueError("Empty DraftKings field")
    report=[];merged=field.copy()
    for stat in chosen:
        sid=STAT_IDS[stat]
        try:
            raw=provider.pga_stats(sid,year=int(lock.year),tour="R")
            clean,status=normalize_stat_table(raw,stat)
            if not clean.empty:
                aligned,match=align_to_dk(clean,field,f"PGA TOUR {stat}")
                matched=int(match["matched"])
                if not aligned.empty:
                    series=aligned.set_index("player")[stat]
                    merged[stat]=merged["player"].map(series)
                else:status="no current-field matches"
            else:matched=0
        except Exception as exc:
            matched=0;status=f"{type(exc).__name__}: {str(exc)[:125]}"
        report.append({"stat":stat,"stat_id":sid,"matched_golfers":matched,
                       "field_golfers":len(field),"status":status,
                       "source":"PGA TOUR website season StatDetails via unofficial client",
                       "season":int(lock.year),"retrieved_date":str(now),
                       "pre_lock_warning":bool(lock==now)})
    useful=[c for c in chosen if c in merged.columns]
    if useful:
        merged=merged[["player"]+useful]
    else:merged=pd.DataFrame(columns=["player"])
    return merged,pd.DataFrame(report)

def fill_missing_measured_stats(existing,official):
    """Merge official-season stats as fallback; preserve RunGood overrides."""
    merged=existing.copy()
    if official is None or official.empty or "player" not in official:
        return merged,{}
    source=official.set_index("player")
    audit={}
    for col in source.columns:
        if col=="player":continue
        incoming=merged["player"].map(source[col])
        old=pd.to_numeric(merged[col],errors="coerce") if col in merged else pd.Series(np.nan,index=merged.index)
        used=old.isna() & incoming.notna()
        merged[col]=old.combine_first(incoming)
        audit[col]={"official_filled":int(used.sum()),"previous_preserved":int(old.notna().sum()),
                    "still_missing":int(merged[col].isna().sum())}
    return merged,audit
