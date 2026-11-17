"""NFL official status evidence in the ONE unified Team AI² Pro Classic builder.

Only source bytes saved and independently re-parsed by the NFL.com capture
collector may alter a lineup. Scheduled GitHub Actions research artifacts are
NOT silently assumed available in a Streamlit process. At build time this
bridge polls official publisher links and checks its own persisted archives.

Source coverage is explicit. No verified source means NO added internal
teammates and no implied active-by-omission. Experimental full-universe
simulations are opt-in and do not overwrite user-approved portfolios.
"""
from __future__ import annotations

from datetime import datetime,timezone
from pathlib import Path

import pandas as pd

from ai2_nfl_inputs import parse_csv_payload
from ai2_nfl_matchups import canonical_team
from ai2_nfl_auto_baselines import _norm_name
from ai2_nfl_full_player_universe import (
    build_football_only_players, FOOTBALL_ONLY_COLUMNS,
)
from ai2_nfl_pregame_availability import (
    ACTIVE,OUT,validate_availability_snapshot,_utc,
)
from ai2_nfl_schedule_capture_guard import (
    GAMES_SOURCE,require_schedule_kickoff,
)
from ai2_nfl_verified_pregame_sources import combined_verified_game_snapshot
from scripts.auto_capture_nfl_status import collect_due_games


def _game(value):
    sides=str(value).split()[0].upper().split("@")
    if len(sides)!=2 or not all(sides):
        raise ValueError("DraftKings game must use AWAY@HOME")
    return "@".join(canonical_team(x) for x in sides)


def _slate_games(salaries):
    if not (isinstance(salaries,tuple) and len(salaries)>=2):
        raise ValueError("Build requires an actual DraftKings salary file")
    rows=parse_csv_payload(salaries[1])
    return sorted({_game(r.get("Game Info","")) for r in rows
                   if r.get("Game Info")})


def _archive_dirs(root):
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and (p/"manifest.json").is_file())


def _read_one(path,*,as_of_utc):
    # Full original bytes and status rows must be independently reparsed
    # before using even a single official game-day claim.
    from ai2_nfl_official_inactives_capture import load_verified_capture
    from ai2_nfl_official_named_active import load_verified_named_active_capture
    import json
    meta=json.loads((path/"manifest.json").read_text())
    category=meta.get("archive_type")
    if category=="NFL_OFFICIAL_INACTIVES_PREGAME_CAPTURE":
        snap,_=load_verified_capture(path,as_of_utc=as_of_utc)
    elif category=="NFL_OFFICIAL_NAMED_ACTIVE_NEWS_CAPTURE":
        snap,_=load_verified_named_active_capture(path,as_of_utc=as_of_utc)
    else:
        raise ValueError("Unknown or unverifiable official game-day archive type")
    return snap,meta,category


def prepare_unified_nfl_availability(salaries,*,season,week,
                                     data_dir,now_utc=None,
                                     schedule=None,collector=collect_due_games):
    """One prebuild source pass shared by all lineup seeds for a single slate.

    Existing OFFICIAL game status reports from the old pipeline remain intact;
    these additional sources are never equated with a complete active list.
    """
    as_of=_utc(now_utc or datetime.now(timezone.utc).isoformat(),"now_utc")
    slate_games=_slate_games(salaries)
    base=Path(data_dir)/"nfl_official_captures"
    report={
        "mode":"EXPERIMENTAL_VERIFIED_PREGAME_AVAILABILITY",
        "as_of_utc":as_of.isoformat(),
        "season":int(season),"week":int(week),
        "slate_games":slate_games,
        "attempted_automatic_source_collection":False,
        "collector_result":"NOT_RUN",
        "source_coverage":[],"source_urls":[],
        "verified_active_players":0,"verified_out_players":0,
        "football_only_players_added":0,
        "errors":[],
        "game_snapshot_count":0,
        "complete_game_day_roster_verified":False,
        "active_by_inactive_list_omission":False,
        "predictive_accuracy_validated":False,
        "does_not_change_approved_lineups":True,
    }
    if schedule is None:
        try:
            schedule=pd.read_csv(GAMES_SOURCE)
        except (OSError,ValueError,TimeoutError) as exc:
            report["errors"].append("Published schedule unavailable: "+str(exc)[:230])
            return {"snapshots_by_game":{},"report":report}
    try:
        collection=collector(schedule=schedule,now_utc=as_of.isoformat(),
                             out_dir=base)
        report["attempted_automatic_source_collection"]=True
        report["collector_result"]=collection.get("result","UNAVAILABLE")
        report["errors"].extend(
            str(x.get("error",x))[:210] for x in collection.get("errors",[]))
    except (OSError,ValueError,TimeoutError,KeyError) as exc:
        report["collector_result"]="SOURCE_COLLECTION_FAILED"
        report["errors"].append(str(exc)[:230])
    candidates={}
    for path in _archive_dirs(base):
        try:
            snap,meta,kind=_read_one(path,as_of_utc=as_of.isoformat())
            if (int(snap["season"])!=int(season) or
                int(snap["week"])!=int(week) or
                _game(snap["game"]) not in slate_games):
                continue
            expected=require_schedule_kickoff(
                schedule,season=season,week=week,game=snap["game"],
                kickoff_utc=snap["kickoff_utc"])
            # Require a source-specific schedule archive generated in the
            # same successful game-day HTTP collection transaction.
            check=path/"schedule_crosscheck.json"
            if not check.is_file():
                raise ValueError("Missing independent scheduled-kickoff evidence")
            import json
            actual=json.loads(check.read_text())
            if (_game(actual["game"])!=_game(snap["game"]) or
                actual.get("source")!=GAMES_SOURCE or
                _utc(actual["independently_verified_kickoff_utc"],
                     "archive kickoff")!=_utc(expected,"scheduled kickoff")):
                raise ValueError("Archived published kickoff evidence disagrees")
            key=(_game(snap["game"]),kind)
            captured=_utc(snap["captured_at_utc"],"captured_at_utc")
            if key not in candidates or captured>candidates[key][0]:
                candidates[key]=(captured,path,snap,meta)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            report["errors"].append("Archive rejected: "+str(exc)[:230])
    snapshots={}
    for game in slate_games:
        choices=[item[1] for (g,kind),item in candidates.items() if g==game]
        if not choices:
            report["source_coverage"].append({
                "game":game,"status":"NO_VERIFIED_GAME_DAY_ARTICLE",
                "known_active":0,"known_out":0,
                "internal_football_teammates":"NOT_ENABLED_WITHOUT_POSITIVE_EVIDENCE"})
            continue
        try:
            combined,provenance=combined_verified_game_snapshot(
                choices,season=season,week=week,game=game,
                as_of_utc=as_of.isoformat())
            indexed=validate_availability_snapshot(
                combined,season=season,week=week,game=game,
                as_of_utc=as_of.isoformat())
        except (ValueError,OSError,KeyError,TypeError) as exc:
            report["errors"].append(f"{game}: rejected conflicting article claims: {exc}")
            continue
        active=sum(x["status"]==ACTIVE for x in indexed.values())
        out=sum(x["status"]==OUT for x in indexed.values())
        snapshots[game]=combined
        report["source_urls"].extend(x["url"] for x in provenance["official_sources"])
        report["verified_active_players"]+=active
        report["verified_out_players"]+=out
        report["source_coverage"].append({
            "game":game,"status":"ARCHIVED_OFFICIAL_SOURCE_CONTENT_REVERIFIED",
            "known_active":active,"known_out":out,
            "internal_football_teammates":"REQUIRES_PRIOR_ROLE_AND_EXPLICIT_ACTIVE"})
    report["source_urls"]=sorted(set(report["source_urls"]))
    report["game_snapshot_count"]=len(snapshots)
    return {"snapshots_by_game":snapshots,"report":report}


def apply_status_sources_to_pool(pool,rates,*,context,season,week):
    """Apply OFFICIAL OUT constraints, then internalize only positive actives.

    Only team+player exact normalized matches are used for priced players and
    prospective on-field teammates; unknowns never get promoted to active.
    """
    if not isinstance(pool,pd.DataFrame) or not isinstance(rates,pd.DataFrame):
        raise ValueError("Live NFL pool/rates must be explicit observed DataFrames")
    snapshots=context.get("snapshots_by_game",{})
    report=dict(context.get("report",{}))
    x=pool.copy()
    raw_keys={}
    known_out=[]
    as_of=report.get("as_of_utc")
    for game,snap in snapshots.items():
        indexed=validate_availability_snapshot(
            snap,season=season,week=week,game=game,as_of_utc=as_of)
        for (team,name),record in indexed.items():
            if record["status"]==OUT:
                raw_keys[(team,name)]=record
                known_out.append({"team":team,"player":record["player"]})
    if "optimizer_eligible" not in x:
        raise ValueError("Current NFL model pool missing optimizer eligibility")
    for i,row in x.iterrows():
        identity=(canonical_team(row["TeamAbbrev"]),_norm_name(row["Name"]))
        if identity in raw_keys:
            x.at[i,"optimizer_eligible"]=False
    internal=[]
    for game,snap in snapshots.items():
        subset=x[x["game"].astype(str).map(_game).eq(game)].copy()
        if subset.empty:
            continue
        # Use the precise matchup string in the live salary pool, which
        # can differ from nflverse's WSH/WAS alias. The validator checks
        # both through canonical_team.
        salary_game=str(subset["game"].iloc[0])
        row=build_football_only_players(
            subset,rates,season=season,week=week,
            known_out=known_out,require_verified_active=True,
            availability_snapshot=snap,as_of_utc=as_of)
        if not row.empty:
            row["game"]=salary_game
            internal.append(row)
    extras=(pd.concat(internal,ignore_index=True) if internal
            else pd.DataFrame(columns=FOOTBALL_ONLY_COLUMNS))
    report["football_only_players_added"]=len(extras)
    report["applied_verified_out_to_priced_players"]=sum(
        (canonical_team(row["TeamAbbrev"]),_norm_name(row["Name"])) in raw_keys
        for _,row in pool[pool["optimizer_eligible"]==True].iterrows())
    report["effective_scorer"]=(
        "OFFICIAL_GATED_ADDITIONAL_PLAYER_UNIVERSE"
        if len(extras) else "EXISTING_PLAYER_UNIVERSE_NO_NEW_ACTIVE_PLAYERS")
    return x,extras,known_out,report
