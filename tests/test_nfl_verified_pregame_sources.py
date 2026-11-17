"""Merged archive gate tests: strict-source semantics, no active-by-omission."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ai2_nfl_verified_pregame_sources import combined_verified_game_snapshot
from ai2_nfl_official_inactives_capture import (
    parse_official_inactive_article, make_pregame_snapshot,
)
from ai2_nfl_official_named_active import (
    parse_named_active_news,make_named_active_snapshot,
)
from test_nfl_official_inactives_capture import html as out_html,URL as OUT_URL
from test_nfl_official_named_active import fixture as active_html,URL as ACTIVE_URL

GAME="KC@ATL"
CAPTURE="2026-11-15T16:50:00Z"
KICKOFF="2026-11-15T18:00:00Z"


def store(root,raw,parsed,snap,kind):
    root.mkdir()
    (root/"official_source.html").write_bytes(raw)
    name=("inactive_player_snapshot.json" if kind=="out"
          else "active_player_snapshot.json")
    (root/name).write_text(json.dumps(snap))
    (root/"verified_parser.json").write_text(json.dumps(parsed))
    manifest={
        "archive_type":("NFL_OFFICIAL_INACTIVES_PREGAME_CAPTURE"
                        if kind=="out" else
                        "NFL_OFFICIAL_NAMED_ACTIVE_NEWS_CAPTURE"),
        "http_capture_was_performed_by_this_program":True,
        "source_reparsed_and_statuses_matched":True,
        "production_approved":False,
        "source_url":parsed["source_url"],
        "source_html_sha256":hashlib.sha256(raw).hexdigest(),
        "capture_utc":snap["captured_at_utc"],"kickoff_utc":snap["kickoff_utc"],
    }
    (root/"manifest.json").write_text(json.dumps(manifest))


def named_fixture():
    return (active_html()
            .replace(b"/teams/carolina-panthers/",b"/teams/kansas-city-chiefs/")
            .replace(b"Carolina Panthers 5-5",b"Kansas City Chiefs 5-5"))


class MergedProofTests(unittest.TestCase):
    def test_two_verified_sources_merge_and_duplicated_sources_reject(self):
        raw1=out_html()
        p1=parse_official_inactive_article(
            raw1,OUT_URL,season=2026,week=10,game=GAME)
        snap1=make_pregame_snapshot(
            p1,kickoff_utc=KICKOFF,captured_at_utc=CAPTURE)
        raw2=named_fixture()
        p2=parse_named_active_news(
            raw2,ACTIVE_URL,season=2026,week=10,game=GAME)
        snap2=make_named_active_snapshot(
            p2,kickoff_utc=KICKOFF,captured_at_utc=CAPTURE)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            a=root/"out";b=root/"active"
            store(a,raw1,p1,snap1,"out")
            store(b,raw2,p2,snap2,"active")
            merged,audit=combined_verified_game_snapshot(
                [a,b],season=2026,week=10,game=GAME,as_of_utc=CAPTURE)
            self.assertEqual(len(merged["records"]),13)
            self.assertEqual(len(audit["official_sources"]),2)
            self.assertFalse(audit["positive_actives_from_inactive_omission"])
            self.assertFalse(audit["production_approved"])
            with self.assertRaisesRegex(ValueError,"duplicate"):
                combined_verified_game_snapshot(
                    [a,a],season=2026,week=10,game=GAME,
                    as_of_utc=CAPTURE)
            with self.assertRaisesRegex(ValueError,"Future source"):
                combined_verified_game_snapshot(
                    [a,b],season=2026,week=10,game=GAME,
                    as_of_utc="2026-11-15T16:00:00Z")


if __name__=="__main__":
    unittest.main()
