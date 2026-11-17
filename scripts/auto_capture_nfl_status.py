"""Unattended *research* pregame official-news polling/capture.

- No DraftKings optimizer writes, no gameplay changes, no betting actions.
- Derive season/week/matchup/kickoff from published schedule, never user input.
- Discover official NFL.com article links from publisher pages.
- Fetch each discovered article, reparse the same raw HTML per game.
- Persist independently confirmed explicit OUT and named ACTIVE claims before
  kickoff with original source bytes and hashes; UNKNOWN stays UNKNOWN.
- A missing/late/malformed source results in diagnostics, not invented roles.
- Post-kickoff windows never record retroactive 'pregame' snapshots.

This is research-only. GitHub Actions schedule is a separate read-only launcher.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import hashlib

import pandas as pd

from ai2_nfl_official_article_discovery import discover_official_links
from ai2_nfl_official_inactives_capture import (
    fetch_official_html,parse_official_inactive_article,make_pregame_snapshot,
    save_new_capture,
)
from ai2_nfl_official_named_active import (
    parse_named_active_news,make_named_active_snapshot,save_named_active_capture,
)
from ai2_nfl_schedule_capture_guard import (
    GAMES_SOURCE,verified_match_kickoff,
)
from ai2_nfl_pregame_availability import _utc


def due_games(schedule,*,now,lookahead_minutes=160):
    now=_utc(now,"now")
    if not 15<=lookahead_minutes<=240:
        raise ValueError("Pregame polling window must be bounded 15–240 min")
    if not isinstance(schedule,pd.DataFrame):
        raise ValueError("Published football schedule is mandatory")
    needed={"season","week","game_type","away_team","home_team","gameday","gametime"}
    if not needed.issubset(schedule.columns):
        raise ValueError("Incomplete published schedule; no game-day inference")
    years={now.year,now.year-1}
    frame=schedule[(schedule["game_type"]=="REG") &
                   pd.to_numeric(schedule["season"],errors="coerce").isin(years)].copy()
    frame=frame[frame["gameday"].astype(str).isin(
        {(now+timedelta(days=offset)).date().isoformat()
         for offset in (-1,0,1)})]
    due=[]
    for row in frame.to_dict("records"):
        game=f'{row["away_team"]}@{row["home_team"]}'
        try:
            kickoff=verified_match_kickoff(schedule,
                season=int(row["season"]),week=int(row["week"]),game=game)
            dt=_utc(kickoff,"published kickoff")
        except (ValueError,TypeError):
            # Independent schedule could not establish a unique real kickoff:
            # NEVER infer from stale user input or force a capture.
            continue
        lead=(dt-now).total_seconds()/60
        if 0<lead<=lookahead_minutes:
            due.append({
                "season":int(row["season"]),"week":int(row["week"]),
                "game":game,"kickoff_utc":dt.isoformat(),
                "lead_minutes":round(lead,2),
            })
    due.sort(key=lambda r:(r["kickoff_utc"],r["game"]))
    if len(due)!=len({(r["season"],r["week"],r["game"]) for r in due}):
        raise ValueError("Duplicate scheduled game identity")
    return due


def _archive_label(game,category,source_sha,utc):
    now=_utc(utc,"utc")
    safe=game.replace("@","_")
    return f'{category}_{safe}_{now.strftime("%Y%m%dT%H%M%SZ")}_{source_sha[:12]}'


def collect_due_games(*,schedule,now_utc,out_dir,
                      discovery=discover_official_links,
                      get_html=fetch_official_html):
    """Capture source evidence for each imminent NFL matchup.

    Source URL discovery alone NEVER admits any football-only player.
    """
    now=_utc(now_utc,"now_utc")
    due=due_games(schedule,now=now.isoformat())
    report={
        "report_type":"UNATTENDED_RESEARCH_SOURCE_COLLECTION",
        "checked_at_utc":now.isoformat(),
        "schedule_source":GAMES_SOURCE,
        "games_due":due,"candidate_articles":[],
        "source_archives":[],"errors":[],
        "positive_active_by_omission":False,
        "official_article_content_required":True,
        "production_app_modified":False,
        "betting_action_taken":False,
    }
    if not due:
        report["result"]="NO_IMMINENT_GAMES"
        return report
    weeks=sorted({g["week"] for g in due})
    sources={}
    for week in weeks:
        discovered=discovery(week=week)
        sources[week]=discovered["articles"]
        report["candidate_articles"]+=[
            {"week":week,**item} for item in discovered["articles"]]
        report["errors"]+=discovered.get("errors",[])
    root=Path(out_dir)
    cache={}
    seen=set()
    for game in due:
        for item in sources[game["week"]]:
            url=item["url"];kind=item["kind"]
            key=(game["season"],game["week"],game["game"],url)
            if key in seen:
                continue
            seen.add(key)
            try:
                if url not in cache:
                    cache[url]=get_html(url)
                raw=cache[url]
                if kind=="OFFICIAL_INACTIVES":
                    parsed=parse_official_inactive_article(
                        raw,url,season=game["season"],week=game["week"],
                        game=game["game"])
                    snap=make_pregame_snapshot(parsed,
                        kickoff_utc=game["kickoff_utc"],
                        captured_at_utc=now.isoformat())
                    label=_archive_label(game["game"],"out",
                                         parsed["source_html_sha256"],
                                         now.isoformat())
                    saved=save_new_capture(raw,parsed,snap,root/label)
                elif kind=="EXPLICIT_INJURY_UPDATES":
                    parsed=parse_named_active_news(
                        raw,url,season=game["season"],week=game["week"],
                        game=game["game"])
                    snap=make_named_active_snapshot(parsed,
                        kickoff_utc=game["kickoff_utc"],
                        captured_at_utc=now.isoformat())
                    label=_archive_label(game["game"],"named_active",
                                         parsed["source_html_sha256"],
                                         now.isoformat())
                    saved=save_named_active_capture(raw,parsed,snap,root/label)
                else:
                    raise ValueError("Unexpected discovery URL category")
                # Archive the schedule cross-check alongside the raw source
                # so future model audits can verify the actual game/kickoff.
                (Path(saved)/"schedule_crosscheck.json").write_text(json.dumps({
                    "source":GAMES_SOURCE,
                    "season":game["season"],"week":game["week"],
                    "game":game["game"],
                    "independently_verified_kickoff_utc":game["kickoff_utc"],
                    "captured_at_utc":now.isoformat(),
                    "claimed_future_kickoff_override_used":False,
                },indent=2,sort_keys=True))
                report["source_archives"].append({
                    "game":game["game"],"season":game["season"],
                    "week":game["week"],"kind":kind,
                    "url":url,"archive_dir":saved,
                    "source_html_sha256":parsed["source_html_sha256"],
                    "source_rows_explicit":len(snap["records"]),
                    "captured_before_kickoff":True,
                })
            except (ValueError,OSError,FileExistsError) as e:
                # A candidate news story can cover OTHER games from the same
                # week. Its failure for this matchup is not an active status.
                report["errors"].append({
                    "game":game["game"],"url":url,"kind":kind,
                    "error":str(e)[:250]})
    report["result"]=(
        "CAPTURED_OFFICIAL_PREGAME_EVIDENCE"
        if report["source_archives"] else
        "NO_MATCHING_VALID_PREGAME_OFFICIAL_ARTICLE")
    return report


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-base",default="nfl_official_captures")
    ap.add_argument("--report",default="nfl_official_capture_run.json")
    options=ap.parse_args(argv)
    now=datetime.now(timezone.utc).isoformat()
    schedule=pd.read_csv(GAMES_SOURCE)
    report=collect_due_games(
        schedule=schedule,now_utc=now,out_dir=options.out_base)
    Path(options.report).write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps(report,sort_keys=True))
    print("Research official-source collection only; production optimizer unchanged")


if __name__=="__main__":
    main()
