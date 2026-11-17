"""NFL play-by-play -> audited, pregame-only possession outcome training data.

nflverse PBP includes a fixed-drive ID, offense/defense identities, field
position, scoring plays, and play counts.  This module deliberately reduces
actual completed PBP to ONE row per offensive possession, then freezes all
features BEFORE the target NFL week. No betting-market or target-week outcomes
can enter its parameters.

Download: nflverse/nflverse-data GitHub release pbp/play_by_play_YEAR.parquet
Requires pandas + pyarrow.  Data acquisition is for experimental/offline
validation only; never automatically downloads in the live app.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ai2_nfl_matchups import canonical_team

SOURCE = "nflverse-play-by-play-completed-prior-week-fixed-drive"
PBP_RELEASE = ("https://github.com/nflverse/nflverse-data/releases/download/"
               "pbp/play_by_play_{season}.parquet")
PBP_COLUMNS = (
    "game_id", "season", "week", "season_type", "play_id",
    "posteam", "defteam", "fixed_drive", "play_type",
    "pass_attempt", "rush_attempt", "sack",
    "pass_touchdown", "rush_touchdown", "interception",
    "fumble_lost", "field_goal_result",
    "yardline_100", "drive_time_of_possession",
)
OUTCOMES = ("pass_td", "rush_td", "field_goal", "turnover", "other")
MIN_GAME_DRIVES = 7


def load_historical_pbp(seasons, *, reader=None, cache_dir=None):
    """Load season files once; verify required columns, never fill missing PBP."""
    reader = reader or pd.read_parquet
    arrays = []
    for season in sorted(set(int(s) for s in seasons)):
        filename = f"play_by_play_{season}.parquet"
        path = (Path(cache_dir) / filename) if cache_dir else PBP_RELEASE.format(
            season=season)
        frame = reader(str(path), columns=list(PBP_COLUMNS))
        missing = set(PBP_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"nflverse {season} PBP missing: {sorted(missing)}")
        years = pd.to_numeric(frame["season"], errors="coerce")
        if years.notna().any() and not years.dropna().astype(int).eq(season).all():
            raise ValueError("PBP year mismatch; refusing possible data leakage")
        arrays.append(frame)
    if not arrays:
        raise ValueError("At least one historic PBP season is required")
    return pd.concat(arrays, ignore_index=True)


def _drive_seconds(raw, play_count):
    if pd.isna(raw):
        return float(np.clip(play_count * 27 + 35, 45, 480))
    parts = str(raw).split(":")
    try:
        if len(parts) == 2:
            seconds = float(parts[0])*60 + float(parts[1])
        elif len(parts) == 3:
            seconds = float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        else:
            seconds = float(raw)
    except (TypeError, ValueError):
        seconds = play_count*27 + 35
    return float(np.clip(seconds,45,480))


def pbp_to_drives(pbp):
    """Extract only OFFENSIVE possessions; reject ambiguous drive identity.

    This is a possessions proxy: we skip nonoffensive kickoffs and do not
    mistake a defensive return touchdown for an offensive passing/rushing TD.
    """
    if not isinstance(pbp,pd.DataFrame):
        raise ValueError("Play-by-play data must be a DataFrame")
    required = set(PBP_COLUMNS) - {"drive_time_of_possession"}
    if not required.issubset(pbp.columns):
        raise ValueError("Incomplete nflverse PBP drive schema: " +
                         str(sorted(required-set(pbp.columns))))
    f = pbp.copy()
    f = f[f["season_type"].astype(str).eq("REG")].copy()
    f = f[f[["game_id","fixed_drive","posteam","defteam"]].notna().all(axis=1)]
    f["offense"] = f["posteam"].map(canonical_team)
    f["defense"] = f["defteam"].map(canonical_team)
    f = f[f["offense"].ne(f["defense"])]
    f["play_id"] = pd.to_numeric(f["play_id"],errors="coerce")
    f = f.dropna(subset=["play_id"])
    f["season"] = pd.to_numeric(f["season"],errors="coerce")
    f["week"] = pd.to_numeric(f["week"],errors="coerce")
    f = f.dropna(subset=["season","week"])
    f = f.sort_values(["season","week","game_id","fixed_drive","play_id"],
                      kind="stable")
    for col in ("pass_attempt","rush_attempt","sack",
                "pass_touchdown","rush_touchdown","interception",
                "fumble_lost"):
        f[col]=pd.to_numeric(f[col],errors="coerce").fillna(0)
    rows = []
    for (game_id, drive), g in f.groupby(["game_id","fixed_drive"],sort=False):
        offenses = g["offense"].unique()
        defenses = g["defense"].unique()
        if len(offenses)!=1 or len(defenses)!=1:
            # A malformed fixed-drive grouping cannot train team scoring.
            continue
        actual_play = g["play_type"].astype(str).str.lower().isin(["pass","run"])
        # QB sacks are dropbacks but the administrative PBP play type may vary.
        offensive_play = (actual_play | g["sack"].gt(0) |
                          g["pass_attempt"].gt(0) | g["rush_attempt"].gt(0))
        play_count = int(offensive_play.sum())
        if not 1 <= play_count <= 24:
            continue
        pass_td = bool(g["pass_touchdown"].gt(0).any())
        rush_td = bool(g["rush_touchdown"].gt(0).any())
        made_fg = bool(g["field_goal_result"].astype(str).str.lower().eq(
            "made").any())
        lost = bool((g["interception"].gt(0) |
                     g["fumble_lost"].gt(0)).any())
        # For normal possessions terminal outcomes are mutually exclusive.
        # Leave out bad/ambiguous data instead of teaching multi-result drives.
        if sum((pass_td,rush_td,made_fg,lost)) > 1:
            continue
        if pass_td:
            outcome = "pass_td"
        elif rush_td:
            outcome = "rush_td"
        elif made_fg:
            outcome = "field_goal"
        elif lost:
            outcome = "turnover"
        else:
            outcome = "other"
        field = pd.to_numeric(g["yardline_100"],errors="coerce")
        valid_field=field.loc[offensive_play].dropna()
        start=float(valid_field.iloc[0]) if len(valid_field) else np.nan
        if not np.isfinite(start) or not 1 <= start <= 99:
            continue
        if "drive_time_of_possession" in g:
            t=g["drive_time_of_possession"].dropna()
            duration=_drive_seconds(t.iloc[0] if len(t) else np.nan,play_count)
        else:
            duration=_drive_seconds(np.nan,play_count)
        rows.append({
            "season":int(g.iloc[0]["season"]),
            "week":int(g.iloc[0]["week"]),
            "game_id":str(game_id),"drive":str(drive),
            "offense":str(offenses[0]),"defense":str(defenses[0]),
            "start_yardline_100":start,
            "plays":play_count,"duration_seconds":duration,
            "outcome":outcome,
            "turnover_interception":int(lost and bool(g["interception"].gt(0).any())),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No unambiguous offensive drives in supplied PBP")
    return out.sort_values(["season","week","game_id","drive"]).reset_index(drop=True)


def _smoothed_prob(frame, base, strength=42.):
    counts=Counter(frame["outcome"])
    a=np.asarray([float(counts[k]) for k in OUTCOMES])
    return ((a+strength*np.asarray(base))/(a.sum()+strength)).tolist()


def _recent(team_drives, team_games=8):
    if team_drives.empty:
        return team_drives
    games=(team_drives[["season","week","game_id"]]
           .drop_duplicates().sort_values(["season","week","game_id"])
           .tail(team_games))
    keys=set(map(tuple,games[["season","week","game_id"]].to_numpy()))
    return team_drives[[
        (row.season,row.week,row.game_id) in keys
        for row in team_drives.itertuples(index=False)
    ]]


def pregame_possession_profiles(drives, season, week, *,
                                min_team_drives=35):
    """Freeze pregame league, offense, defense and field-position distributions.

    The ONLY target-week representation is the requested date metadata.
    All rows used for estimates are strictly older, including prior seasons.
    """
    if not 10 <= min_team_drives <= 100:
        raise ValueError("Unsupported team-drive minimum")
    if not isinstance(drives,pd.DataFrame) or drives.empty:
        raise ValueError("Measured NFL possessions are required")
    required={"season","week","game_id","offense","defense","outcome",
              "plays","duration_seconds","start_yardline_100"}
    if not required.issubset(drives.columns):
        raise ValueError("Missing measured drive fields")
    historic=drives[
        (drives["season"].lt(int(season)))|
        (drives["season"].eq(int(season)) & drives["week"].lt(int(week)))
    ].copy()
    historic=historic[historic["season"].ge(int(season)-1)]
    if len(historic)<200:
        raise ValueError("Too few prior offensive possessions for game model")
    if not historic["outcome"].isin(OUTCOMES).all():
        raise ValueError("Unknown observed offensive drive result")
    games_per_team=historic.groupby(["season","week","game_id","offense"]).size()
    historic=historic.set_index(["season","week","game_id","offense"]).loc[
        games_per_team[games_per_team.ge(MIN_GAME_DRIVES)].index
    ].reset_index()
    if len(historic)<200:
        raise ValueError("Insufficient complete-team-game possession coverage")
    base=_smoothed_prob(historic,np.ones(len(OUTCOMES))/len(OUTCOMES),0.)
    by_team={}
    for team in sorted(set(historic["offense"])|set(historic["defense"])):
        off=_recent(historic[historic["offense"].eq(team)])
        deff=_recent(historic[historic["defense"].eq(team)])
        offense_prob=(
            _smoothed_prob(off,base) if len(off)>=min_team_drives else base)
        allowed_prob=(
            _smoothed_prob(deff,base) if len(deff)>=min_team_drives else base)
        by_team[team]={
            "offense":offense_prob,
            "allowed":allowed_prob,
            "offense_drives":int(len(off)),
            "defense_drives":int(len(deff)),
            "offense_games":int(off["game_id"].nunique()),
            "defense_games":int(deff["game_id"].nunique()),
            "recent_offensive_pass_fraction":None,
        }
    # Baseline outcome-based plays and clock are measured, not hardcoded odds.
    outcomes={}
    for cat in OUTCOMES:
        sample=historic[historic["outcome"].eq(cat)]
        if len(sample)<8:
            raise ValueError(f"Insufficient past {cat} drive examples")
        outcomes[cat]={
            "plays":sample["plays"].astype(int).to_list(),
            "seconds":sample["duration_seconds"].astype(int).to_list(),
        }
    starts=np.clip(
        historic["start_yardline_100"].to_numpy(float),1,99).astype(int)
    buckets=[(1,25),(26,50),(51,75),(76,99)]
    base_scoring=sum(historic["outcome"].isin(OUTCOMES[:3]))/len(historic)
    field_multiplier=[]
    for lo,hi in buckets:
        group=historic[historic["start_yardline_100"].between(lo,hi)]
        # Shrink sparse favorable-field samples heavily.
        scored=float(group["outcome"].isin(OUTCOMES[:3]).sum())
        shrunk=(scored+60*base_scoring)/(len(group)+60)
        field_multiplier.append(float(np.clip(
            shrunk/max(base_scoring,1e-5),.5,2.)))
    last=historic.sort_values(["season","week"]).iloc[-1]
    turnovers=historic[historic["outcome"].eq("turnover")]
    interception_rate=float(np.clip(
        (turnovers["turnover_interception"].sum()+12*.70)/
        (len(turnovers)+12), .1,.95))
    return {
        "source":SOURCE,
        "target_season":int(season),"target_week":int(week),
        "last_observed_season":int(last["season"]),
        "last_observed_week":int(last["week"]),
        "total_drives":int(len(historic)),
        "turnover_interception_rate":interception_rate,
        "league_probs":base,
        "teams":by_team,
        "outcomes":outcomes,
        "starting_field_positions":starts.tolist(),
        "field_bins":buckets,
        "field_scoring_multiplier":field_multiplier,
        "overall_mean_drives_per_team":float(
            historic.groupby(["game_id","offense"]).size().mean()),
    }


def check_profile_freshness(profile, season, week):
    if not isinstance(profile,dict) or profile.get("source") != SOURCE:
        raise ValueError("Unverified historical possession model")
    if (profile.get("target_season")!=int(season) or
            profile.get("target_week")!=int(week)):
        raise ValueError("Posession profile has wrong target season/week")
    oldyear=int(profile.get("last_observed_season",0))
    oldweek=int(profile.get("last_observed_week",99))
    if oldyear>int(season) or (oldyear==int(season) and oldweek>=int(week)):
        raise ValueError("Future week leaked into possession model")
    a=np.asarray(profile.get("league_probs"),float)
    if a.shape!=(len(OUTCOMES),) or not np.isfinite(a).all() or (
            a<0).any() or not np.isclose(a.sum(),1.):
        raise ValueError("Invalid PBP drive outcome distribution")
    if int(profile.get("total_drives",0))<200:
        raise ValueError("Too few observed historic offensive possessions")
    for team,z in profile.get("teams",{}).items():
        if not isinstance(z,dict):
            raise ValueError("Malformed possession team evidence")
        for key in ("offense","allowed"):
            v=np.asarray(z[key],float)
            if v.shape!=a.shape or not np.isfinite(v).all() or (
                    v<0).any() or not np.isclose(v.sum(),1.):
                raise ValueError("Invalid measured offense/defense probabilities")
    return True
