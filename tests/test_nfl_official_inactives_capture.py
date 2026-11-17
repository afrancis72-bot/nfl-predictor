"""NFL.com HTML extraction tests: sources below are synthetic fixtures.

These tests do not authenticate the real NFL article or fabricate pregame
snapshots. The official HTML capture code must reject post-kickoff collection,
wrong season, unrelated editorial text and changed archived evidence.
"""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from ai2_nfl_official_inactives_capture import (
    fetch_official_html,parse_official_inactive_article,make_pregame_snapshot,
    verify_snapshot_against_raw_html,save_new_capture,load_verified_capture,
)

URL="https://www.nfl.com/news/nfl-week-10-inactives-synthetic-test"
KICKOFF="2026-11-15T18:00:00Z"
CAPTURE="2026-11-15T16:59:00Z"


def html(*,week=10,year=2026,with_second=True,
         published="2026-11-15T16:32:00Z",order="right",
         include_editorial=False):
    editorial=("<aside><h3>Chiefs</h3><ul><li>QB NOT INACTIVE</li></ul></aside>"
               if include_editorial else "")
    second=(
        "<h3>FALCONS</h3><ul><li>WR Example Falcons</li><li>QB Example Starter</li>"
        "<li>DL Example DL</li><li>LB Example LB</li><li>C Example C</li></ul>"
        if with_second else "")
    if order=="wrong":
        second="<h3>BILLS</h3><ul><li>WR Some Bill</li></ul>"+second
    return (f"""<!doctype html><html><head>
        <title>NFL Week {week} inactives: 2026 synthetic newsroom sample</title>
        <meta property="article:published_time" content="{published}">
        </head><body>
        <nav><h3>CHIEFS</h3><ul><li>RB Editorial Only</li></ul></nav>
        {editorial}
        <article><h1>NFL Week {week} inactives: synthetic testing only</h1>
        <h2>EARLY GAMES</h2>
        <h3>CHIEFS</h3><ul>
            <li>RB Kenneth Example</li><li>QB Backup Example (emergency third QB)</li>
            <li>WR Recipient Example</li><li>TE Example Tightend</li>
            <li>CB Example Defender</li><li>DL Example Lineman</li>
        </ul>
        {second}
        <h2>Related Content</h2><p>WR Editorial Name is active</p>
        </article>
        </body></html>""").encode("utf-8")


class FakeResponse:
    status_code=200
    url=URL
    headers={"Content-Type":"text/html; charset=utf-8"}
    content=html()


class FakeClient:
    def get(self,url,**kwargs):
        return FakeResponse()


class OfficialSourceCaptureTests(unittest.TestCase):
    def parse(self,data=None,**kwargs):
        return parse_official_inactive_article(
            data if data is not None else html(),URL,season=2026,
            week=10,game="KC@ATL",**kwargs)

    def test_exposes_only_two_exact_teams_and_actual_bullet_positions(self):
        d=self.parse(data=html(include_editorial=False))
        self.assertEqual(list(d["inactive_by_team"]),["KC","ATL"])
        self.assertEqual(len(d["inactive_by_team"]["KC"]),6)
        self.assertEqual(len(d["inactive_by_team"]["ATL"]),5)
        self.assertEqual(d["inactive_by_team"]["KC"][0]["player"],"Kenneth Example")
        self.assertNotIn("Editorial Only",str(d))
        self.assertNotIn("Editorial Name",str(d))
        self.assertEqual(d["inactive_by_team"]["KC"][1]["note"],
                         "emergency third QB")
        self.assertTrue(d["verified_out_only"])
        self.assertFalse(d["verified_active_from_omission"])

    def test_full_snapshot_retains_hash_for_each_out_not_active(self):
        d=self.parse()
        snap=make_pregame_snapshot(
            d,kickoff_utc=KICKOFF,captured_at_utc=CAPTURE)
        self.assertEqual(len(snap["records"]),11)
        self.assertTrue(all(row["status"]=="VERIFIED_OUT"
                            for row in snap["records"]))
        self.assertTrue(all(row["source_html_sha256"]==d["source_html_sha256"]
                            for row in snap["records"]))
        self.assertTrue(verify_snapshot_against_raw_html(html(),d,snap))
        self.assertFalse(any(row["status"]=="VERIFIED_ACTIVE"
                             for row in snap["records"]))

    def test_rejects_archive_tampering(self):
        d=self.parse()
        snap=make_pregame_snapshot(d,kickoff_utc=KICKOFF,captured_at_utc=CAPTURE)
        mutated=deepcopy(snap)
        mutated["records"][0]["player"]="Fake Addition"
        with self.assertRaisesRegex(ValueError,"statuses/timestamps altered"):
            verify_snapshot_against_raw_html(html(),d,mutated)
        with self.assertRaisesRegex(ValueError,"differs from source"):
            verify_snapshot_against_raw_html(
                html().replace(b"Kenneth Example",b"Other Example"),d,snap)

    def test_rejects_post_kickoff_backfill_and_publication_after_capture(self):
        d=self.parse()
        with self.assertRaisesRegex(ValueError,"Past-kickoff"):
            make_pregame_snapshot(d,kickoff_utc=KICKOFF,
                                  captured_at_utc="2026-11-15T18:01:00Z")
        with self.assertRaisesRegex(ValueError,"published after collection"):
            make_pregame_snapshot(d,kickoff_utc=KICKOFF,
                                  captured_at_utc="2026-11-15T16:00:00Z")

    def test_rejects_bad_headline_wrong_year_bad_game_and_missing_team(self):
        for item in (html(week=9),html(year=2025,published="2025-11-15T16:32:00Z"),
                     html(with_second=False),html(order="wrong")):
            with self.subTest(data=item[:100]),self.assertRaises(ValueError):
                self.parse(data=item)
        with self.assertRaises(ValueError):
            parse_official_inactive_article(html(),URL,season=2026,
                                             week=10,game="KC@MIA")

    def test_rejects_unsupported_html_host_and_non_html_http_status(self):
        for url in ("http://www.nfl.com/news/article",
                    "https://nfl.com.attacker.invalid/news/evil",
                    "https://www.nfl.com/players/some-player"):
            with self.subTest(url=url),self.assertRaisesRegex(ValueError,"Only exact NFL"):
                fetch_official_html(url,client=FakeClient())
        data=fetch_official_html(URL,client=FakeClient())
        self.assertEqual(data,html())
        class BadClient:
            def get(self,url,**kwargs):
                r=FakeResponse()
                r.status_code=403
                return r
        with self.assertRaisesRegex(ValueError,"HTTP 403"):
            fetch_official_html(URL,client=BadClient())
        class RedirectClient:
            def get(self,url,**kwargs):
                r=FakeResponse()
                r.status_code=302
                return r
        with self.assertRaisesRegex(ValueError,"HTTP 302"):
            fetch_official_html(URL,client=RedirectClient())

    def test_no_fake_fixture_archiving_as_real_source(self):
        d=self.parse()
        snap=make_pregame_snapshot(d,kickoff_utc=KICKOFF,captured_at_utc=CAPTURE)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/"not_a_real_source"
            with self.assertRaisesRegex(ValueError,"Synthetic HTML"):
                save_new_capture(html(),d,snap,dest,real_network_fetch=False)
            self.assertFalse(dest.exists())

    def test_archived_pregame_source_reparses_and_rejects_altered_or_future_data(self):
        import hashlib
        import tempfile
        from pathlib import Path
        source=html()
        d=self.parse(data=source)
        snap=make_pregame_snapshot(d,kickoff_utc=KICKOFF,captured_at_utc=CAPTURE)
        manifest={
            "archive_type":"NFL_OFFICIAL_INACTIVES_PREGAME_CAPTURE",
            "http_capture_was_performed_by_this_program":True,
            "source_reparsed_and_statuses_matched":True,
            "source_html_sha256":hashlib.sha256(source).hexdigest(),
            "source_url":URL,
            "capture_utc":snap["captured_at_utc"],
            "kickoff_utc":snap["kickoff_utc"],
        }
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/"official_source.html").write_bytes(source)
            (root/"inactive_player_snapshot.json").write_text(json.dumps(snap))
            (root/"verified_parser.json").write_text(json.dumps(d))
            (root/"manifest.json").write_text(json.dumps(manifest))
            restored,_=load_verified_capture(root,as_of_utc=CAPTURE)
            self.assertEqual(restored,snap)
            with self.assertRaisesRegex(ValueError,"later pregame capture"):
                load_verified_capture(root,as_of_utc="2026-11-15T16:00:00Z")
            corrupted=deepcopy(snap);corrupted["records"][0]["player"]="Injected Name"
            (root/"inactive_player_snapshot.json").write_text(json.dumps(corrupted))
            with self.assertRaisesRegex(ValueError,"statuses/timestamps altered"):
                load_verified_capture(root,as_of_utc=CAPTURE)
            (root/"inactive_player_snapshot.json").write_text(json.dumps(snap))
            (root/"official_source.html").write_bytes(source.replace(b"Kenneth Example",b"Someone Else"))
            with self.assertRaisesRegex(ValueError,"SHA-256 has changed"):
                load_verified_capture(root,as_of_utc=CAPTURE)

    def test_publication_time_requires_machine_readable_utc(self):
        source=html().replace(
            b'<meta property="article:published_time" content="2026-11-15T16:32:00Z">',
            b'')
        with self.assertRaisesRegex(ValueError,"machine-readable"):
            self.parse(data=source)
        source2=html(published="2026-11-15T16:32:00")
        with self.assertRaisesRegex(ValueError,"machine-readable"):
            self.parse(data=source2)

    def test_duplicates_outside_article_do_not_pollute_roster(self):
        raw=html(include_editorial=True)
        d=self.parse(data=raw)
        self.assertEqual(len(d["inactive_by_team"]["KC"]),6)
        self.assertFalse("NOT INACTIVE" in json.dumps(d))


if __name__=="__main__":
    unittest.main()
