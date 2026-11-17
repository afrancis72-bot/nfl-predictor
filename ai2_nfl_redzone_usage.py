"""Research-only, prior-game player red-zone target and goal-line carry shares.

Real nflverse play-by-play GSIS player IDs are joined to *earlier* nflverse
weekly player IDs. This preserves full player names and prevents initials-based
mistaken identity matching. Each player's red-zone/goal-line opportunity share
is compared with his own overall team opportunity share and shrunk toward 1.

These factors NEVER create extra touchdowns, targets or rush attempts. An
optional simulation uses them only to allocate a previously simulated pass or
rushing touchdown to players with a real same-world catch or carry.

No live one-click use, no roster availability assertion, no profitable claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ai2_nfl_auto_baselines import _norm_name
from ai2_nfl_pbp_drives import engine_team
from props_data import clean_weekly

SOURCE = "nflverse-pregame-player-GSIS-redzone-target-goalline-carry"
FIELDS = {
    "season","week","season_type","game_id","posteam","yardline_100",
    "pass_attempt","rush_attempt","receiver_player_id","rusher_player_id",
    "two_point_attempt",
}
CHANNELS = ("receiving", "rushing")
PRIOR = {"receiving":12.0,"rushing":8.0}
MIN_TEAM_ZONE = {"receiving":10.0,"rushing":6.0}
FACTOR_RANGE = (0.45,2.20)


def _id(value):
    if pd.isna(value):
        return ""
    v = str(value).strip()
    return v if v and v.lower() not in ("nan","none","null") else ""


def _pregame_play_rows(pbp, season, week):
    if not isinstance(pbp,pd.DataFrame) or not FIELDS.issubset(pbp.columns):
        raise ValueError("Source lacks actual nflverse player-level play-by-play data")
    f=pbp.copy()
    f["season"]=pd.to_numeric(f["season"],errors="coerce")
    f["week"]=pd.to_numeric(f["week"],errors="coerce")
    f=f[(f["season"].ge(int(season)-1)) &
        ((f["season"].lt(int(season))) |
        (f["season"].eq(int(season)) & f["week"].lt(int(week)))) &
        f["season_type"].astype(str).eq("REG")].copy()
    f["team"]=f["posteam"].map(engine_team)
    f["yards_to_goal"]=pd.to_numeric(f["yardline_100"],errors="coerce")
    f=f[f["yards_to_goal"].between(1,99)].copy()
    for col in ("pass_attempt","rush_attempt","two_point_attempt"):
        f[col]=pd.to_numeric(f[col],errors="coerce").fillna(0)
    f=f[f["two_point_attempt"].eq(0) &
        (f["pass_attempt"].eq(1) | f["rush_attempt"].eq(1))].copy()
    if f.empty:
        raise ValueError("No past red-zone football observations")
    return f


def _player_labels(weekly, season, week):
    h=clean_weekly(weekly)
    if "player_id" not in h:
        raise ValueError("Pregame nflverse weekly GSIS player_id is required")
    h=h[(h["season"].ge(int(season)-1)) &
        ((h["season"].lt(int(season))) |
         (h["season"].eq(int(season)) & h["week"].lt(int(week))))].copy()
    if h.empty:
        raise ValueError("No pregame GSIS player identities")
    h["id"]=h["player_id"].map(_id)
    h=h[h["id"].ne("")]
    h["team"]=h["recent_team"].map(engine_team)
    h=h.sort_values(["season","week"])
    # Match only the most recent recorded PLAYER TEAM. A player transferred
    # since the observed PBP cannot inherit his prior club's scoring share.
    recent=h.drop_duplicates(["id"],keep="last")
    return {str(row["id"]):(str(row["team"]),_norm_name(row["player_name"]))
            for row in recent.to_dict("records")}


def pregame_redzone_profiles(pbp, weekly, season, week, *,
                             last_n_games=8):
    if not 3<=int(last_n_games)<=16:
        raise ValueError("Recent team games must be between three and sixteen")
    f=_pregame_play_rows(pbp,season,week)
    labels=_player_labels(weekly,season,week)
    result={}
    latest_season=latest_week=0
    for team,all_plays in f.groupby("team"):
        games=(all_plays[["season","week","game_id"]].drop_duplicates()
               .sort_values(["season","week","game_id"])
               .tail(last_n_games).copy())
        if len(games)<3:
            continue
        games["_game_weight"]=.87**np.arange(len(games)-1,-1,-1)
        subset=all_plays.merge(games,on=["season","week","game_id"],
                               how="inner",validate="many_to_one")
        if subset.empty:
            continue
        latest_season=max(latest_season,int(games.iloc[-1]["season"]))
        latest_week=(int(games.iloc[-1]["week"]) if
                     int(games.iloc[-1]["season"])==latest_season
                     else latest_week)
        for channel, event, player_field, zone_yards in (
            ("receiving","pass_attempt","receiver_player_id",20),
            ("rushing","rush_attempt","rusher_player_id",10),
        ):
            rows=subset[subset[event].eq(1)].copy()
            rows["id"]=rows[player_field].map(_id)
            rows=rows[rows["id"].ne("")].copy()
            if rows.empty:
                continue
            rows["zone"]=rows["yards_to_goal"].le(zone_yards)
            all_team=float(rows["_game_weight"].sum())
            rz_team=float(rows.loc[rows["zone"],"_game_weight"].sum())
            if all_team<=0 or rz_team<MIN_TEAM_ZONE[channel]:
                continue
            by_id=rows.groupby("id",sort=False)
            for pid,group in by_id:
                if labels.get(pid,(None,None))[0]!=team:
                    continue
                overall=float(group["_game_weight"].sum())
                inside=float(group.loc[group["zone"],"_game_weight"].sum())
                if overall<2.0:
                    continue
                baseline_share=overall/all_team
                rz_share=(inside+PRIOR[channel]*baseline_share)/(
                    rz_team+PRIOR[channel])
                factor=float(np.clip(
                    rz_share/max(baseline_share,1e-8),*FACTOR_RANGE))
                key=(str(team),str(labels[pid][1]),channel)
                if key in result:
                    raise ValueError("Duplicate GSIS/name in one opponent research role")
                result[key]={
                    "factor":round(factor,5),
                    "player_id":pid,
                    "channel":channel,
                    "observed_opportunities":round(overall,3),
                    "zone_opportunities":round(inside,3),
                    "team_zone_opportunities":round(rz_team,3),
                    "team_total_opportunities":round(all_team,3),
                    "team_games":len(games),
                    "source":SOURCE,
                }
    historic=f.sort_values(["season","week"]).iloc[-1]
    return {
        "source":SOURCE,
        "target_season":int(season),
        "target_week":int(week),
        "last_observed_season":int(historic["season"]),
        "last_observed_week":int(historic["week"]),
        "players":result,
        "team_context":"last eight prior observed team games; GSIS matched",
        "unmatched_prior_player_ids_not_imputed":True,
    }


def validated_redzone_lookup(profile, *, season, week, strength=0.):
    if not np.isfinite(strength) or not 0<=strength<=1:
        raise ValueError("Red-zone opportunity effect strength must be 0 to 1")
    if strength==0:
        return {}
    if not isinstance(profile,dict) or profile.get("source")!=SOURCE:
        raise ValueError("Red-zone roles must come from pregame nflverse evidence")
    if (profile.get("target_season")!=int(season) or
            profile.get("target_week")!=int(week)):
        raise ValueError("Wrong target week for red-zone player roles")
    yy=int(profile.get("last_observed_season",0))
    ww=int(profile.get("last_observed_week",99))
    if yy>int(season) or (yy==int(season) and ww>=int(week)):
        raise ValueError("Target-week red-zone observations are forbidden")
    players=profile.get("players")
    if not isinstance(players,dict):
        raise ValueError("Red-zone player factors must be source-matched roles")
    out={}
    for key,info in players.items():
        if not (isinstance(key,tuple) and len(key)==3 and
                key[2] in CHANNELS and isinstance(info,dict) and
                info.get("source")==SOURCE and info.get("channel")==key[2]):
            raise ValueError("Invalid red-zone GSIS player/role provenance")
        z=float(info.get("factor",np.nan))
        n=float(info.get("observed_opportunities",np.nan))
        sampled=float(info.get("team_zone_opportunities",np.nan))
        if (not np.isfinite([z,n,sampled]).all() or
                not FACTOR_RANGE[0]-1e-4<=z<=FACTOR_RANGE[1]+1e-4 or
                n<2-1e-4 or sampled<MIN_TEAM_ZONE[key[2]]-.001 or
                not _id(info.get("player_id"))):
            raise ValueError("Unsupported player goal-line/target sample")
        out[key]=float(1+strength*(z-1))
    return out


def factor_for(lookup,team,name,channel):
    return float(lookup.get((engine_team(team),_norm_name(name),channel),1.))
