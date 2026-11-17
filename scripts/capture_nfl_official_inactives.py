"""CLI entry point for read-only official NFL inactive article captures.

Normal mode:
  python scripts/capture_nfl_official_inactives.py --url ... --season 2026 \
    --week 11 --game BUF@MIA --kickoff-utc 2026-11-22T18:00:00Z \
    --out-base nfl_official_capture/

A successful pre-kickoff run stores raw HTML + source digest + individual
verified OUT status entries in an append-only directory, to be retained as
a GitHub Actions artifact.

Parser-smoke mode may read an ALREADY FINISHED historical article, but can
never emit a pregame availability snapshot; it exists solely to test parsing.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime,timezone
from pathlib import Path

from ai2_nfl_official_inactives_capture import (
    fetch_official_html,parse_official_inactive_article,
    make_pregame_snapshot,save_new_capture,verify_snapshot_against_raw_html,
)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--url",required=True,help="Exact official NFL.com /news/ inactive article")
    p.add_argument("--season",required=True,type=int)
    p.add_argument("--week",required=True,type=int)
    p.add_argument("--game",required=True,help="Canonical away@home NFL team abbreviations")
    p.add_argument("--kickoff-utc",help="Verified scheduled kickoff, required for pregame capture")
    p.add_argument("--out-base",default="nfl_official_captures")
    p.add_argument("--parser-smoke-only",action="store_true",
                   help="May inspect past articles; never emits pregame statuses")
    args=p.parse_args(argv)
    raw=fetch_official_html(args.url)
    captured=datetime.now(timezone.utc).isoformat()
    parsed=parse_official_inactive_article(
        raw,args.url,season=args.season,week=args.week,game=args.game)
    digest=parsed["source_html_sha256"]
    if args.parser_smoke_only:
        # Logging only. Do not archive status snapshots or claim pregame capture.
        print(json.dumps({
            "mode":"PARSER_SMOKE_ONLY_NOT_PREGAME",
            "source_url":args.url,
            "captured_at_utc":captured,
            "source_html_sha256":digest,
            "article_published_at_utc":parsed["published_at_utc"],
            "game":args.game,"season":args.season,"week":args.week,
            "inactive_counts":{
                team:len(rows) for team,rows in parsed["inactive_by_team"].items()
            },
            "pregame_snapshot_created":False,
            "production_approved":False,
        },indent=2,sort_keys=True))
        return
    if not args.kickoff_utc:
        p.error("--kickoff-utc is required for pregame capture")
    snap=make_pregame_snapshot(
        parsed,kickoff_utc=args.kickoff_utc,captured_at_utc=captured)
    verify_snapshot_against_raw_html(raw,parsed,snap)
    label=f"{args.season}_wk{args.week}_{args.game.replace('@','_')}_{digest[:12]}_{captured[:19].replace(':','')}"
    dest=Path(args.out_base)/label
    path=save_new_capture(raw,parsed,snap,dest)
    print(json.dumps({
        "mode":"PREGAME_OUT_ONLY_OFFICIAL_SOURCE_CAPTURE",
        "archive_dir":path,
        "source":args.url,
        "captured_at_utc":captured,
        "n_explicit_out":len(snap["records"]),
        "n_positive_active":0,
        "source_html_sha256":digest,
        "eligible_to_autopromote_predictive_model":False,
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
