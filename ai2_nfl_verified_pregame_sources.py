"""Combine independently archived official OUT + explicit ACTIVE source captures.

No inference from absence; explicit contradictions fail closed. A positive
claim is supported only if source bytes can be re-parsed and the claim appears
in the official article. Historical postgame captures cannot be retroactively
reinterpreted as pregame because each archive's capture clock is validated.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ai2_nfl_pregame_availability import (
    validate_availability_snapshot,_utc,
)
from ai2_nfl_official_inactives_capture import load_verified_capture
from ai2_nfl_official_named_active import load_verified_named_active_capture


def combined_verified_game_snapshot(archives,*,season,week,game,as_of_utc):
    if not isinstance(archives,(list,tuple)) or not archives:
        raise ValueError("Provide one or more archived official pregame source directories")
    all_sources=[]
    combined=[]
    keys=set()
    captured=[]
    kickoffs=[]
    for root in archives:
        path=Path(root)
        manifest_path=path/"manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Missing archived official source manifest")
        import json
        manifest=json.loads(manifest_path.read_text())
        kind=manifest.get("archive_type")
        if kind=="NFL_OFFICIAL_INACTIVES_PREGAME_CAPTURE":
            snapshot,manifest=load_verified_capture(path,as_of_utc=as_of_utc)
        elif kind=="NFL_OFFICIAL_NAMED_ACTIVE_NEWS_CAPTURE":
            snapshot,manifest=load_verified_named_active_capture(
                path,as_of_utc=as_of_utc)
        else:
            raise ValueError("Unrecognized official NFL archive source type")
        if (int(snapshot["season"])!=int(season) or
            int(snapshot["week"])!=int(week) or
            str(snapshot["game"])!=str(game)):
            raise ValueError("Official source archive belongs to a different matchup")
        indexed=validate_availability_snapshot(
            snapshot,season=season,week=week,game=game,
            as_of_utc=as_of_utc)
        for identity,record in indexed.items():
            if identity in keys:
                raise ValueError("Conflicting or duplicate official source claims for one game player")
            keys.add(identity)
        combined+=snapshot["records"]
        captured.append(_utc(snapshot["captured_at_utc"],"captured_at_utc"))
        kickoffs.append(_utc(snapshot["kickoff_utc"],"kickoff_utc"))
        all_sources.append({
            "kind":kind,"url":manifest["source_url"],
            "source_html_sha256":manifest["source_html_sha256"],
            "captured_at_utc":snapshot["captured_at_utc"],
            "status_claims":len(snapshot["records"]),
        })
    if len(set(kickoffs))!=1:
        raise ValueError("Official report archives disagree about matchup kickoff")
    latest=max(captured)
    out={
        "season":int(season),"week":int(week),"game":str(game),
        "kickoff_utc":kickoffs[0].isoformat(),
        "captured_at_utc":latest.isoformat(),
        "records":combined,
    }
    validate_availability_snapshot(
        out,season=season,week=week,game=game,as_of_utc=as_of_utc)
    return out,{
        "archive_count":len(archives),
        "all_source_content_independently_reparsed":True,
        "positive_actives_from_inactive_omission":False,
        "combined_source_claim_count":len(combined),
        "official_sources":all_sources,
        "production_approved":False,
    }
