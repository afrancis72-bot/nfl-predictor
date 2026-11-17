"""Conservative NFL game-day ACTIVE claim extraction tests, synthetic HTML only."""
import copy
import unittest

from ai2_nfl_official_named_active import (
    parse_named_active_news,make_named_active_snapshot,verify_named_active_raw,
)
from ai2_nfl_pregame_availability import validate_availability_snapshot

URL="https://www.nfl.com/news/nfl-week-10-injuries-to-monitor-for-sunday-games"
PUBLISHED="2026-11-15T15:00:00Z"


def fixture(*,real=True,publication=PUBLISHED,include_unrelated=True):
    return f"""<html><head><title>NFL Week 10 injuries to monitor for Sunday's games</title>
<meta property="article:published_time" content="{publication}"></head>
<body><nav><a href="/teams/atlanta-falcons/">Atlanta Falcons</a></nav>
<main><h1>NFL Week 10 injuries to monitor for Sunday's games</h1>
<a href="/teams/atlanta-falcons/">Atlanta Falcons 3-6</a>
<ul>
<li>WR <a href="/players/drake-london/">Drake London</a> (ankle; questionable)
 is expected to play on Sunday against Carolina. London is officially active.</li>
<li>RB <a href="/players/example-rb/">Example RB</a> (knee)
 is expected to play against Carolina.</li>
<li>TE <a href="/players/example-te/">Example TE</a> (ankle)
 might be active on Sunday.</li>
</ul>
<a href="/teams/carolina-panthers/">Carolina Panthers 5-5</a>
<ul><li>WR <a href="/players/example-player/">Example Player</a> (hamstring)
 is active for Sunday's game against Atlanta.</li>
<li>WR <a href="/players/example-scratched/">Example Scratched</a> is inactive.</li>
</ul><h2>Related Content</h2>
<li>WR <a href="/players/editorial-player/">Editorial Player</a> is active.</li>
</main></body></html>""".encode("utf-8")


class NamedActiveClaimTests(unittest.TestCase):
    def parse(self,raw=None):
        return parse_named_active_news(
            raw or fixture(),URL,season=2026,week=10,game="ATL@CAR")

    def test_only_explicit_active_named_rows_inside_team_cards(self):
        result=self.parse()
        self.assertEqual(set(result["active_by_team"]),{"ATL","CAR"})
        self.assertEqual(result["active_by_team"]["ATL"][0]["player"],"Drake London")
        self.assertEqual(result["active_by_team"]["CAR"][0]["player"],"Example Player")
        self.assertEqual(sum(map(len,result["active_by_team"].values())),2)
        self.assertNotIn("Example RB",str(result))
        self.assertNotIn("Example TE",str(result))
        self.assertNotIn("Editorial Player",str(result))
        self.assertFalse(result["inferred_from_no_inactive_listing"])

    def test_pregame_status_snapshot_provenance_and_hash(self):
        result=self.parse()
        snap=make_named_active_snapshot(
            result,kickoff_utc="2026-11-15T18:00:00Z",
            captured_at_utc="2026-11-15T16:40:00Z")
        self.assertEqual(len(snap["records"]),2)
        self.assertTrue(all(r["source_type"]=="NFL_OFFICIAL_EXPLICIT_ACTIVE_NEWS"
                            for r in snap["records"]))
        self.assertTrue(all(r["status"]=="VERIFIED_ACTIVE" for r in snap["records"]))
        self.assertTrue(verify_named_active_raw(fixture(),result,snap))
        validate_availability_snapshot(
            snap,season=2026,week=10,game="ATL@CAR")

    def test_cannot_make_history_pregame_after_kickoff(self):
        parsed=self.parse()
        with self.assertRaisesRegex(ValueError,"Past-kickoff"):
            make_named_active_snapshot(parsed,kickoff_utc="2026-11-15T18:00:00Z",
                captured_at_utc="2026-11-15T21:00:00Z")

    def test_archive_detects_modifications(self):
        parsed=self.parse()
        snap=make_named_active_snapshot(parsed,kickoff_utc="2026-11-15T18:00:00Z",
             captured_at_utc="2026-11-15T16:40:00Z")
        forged=copy.deepcopy(snap)
        forged["records"][0]["player"]="Somebody Else"
        with self.assertRaisesRegex(ValueError,"was altered"):
            verify_named_active_raw(fixture(),parsed,forged)
        with self.assertRaisesRegex(ValueError,"differs from archived"):
            verify_named_active_raw(fixture().replace(b"Drake London",b"Fake Player"),
                                    parsed,snap)

    def test_wrong_week_year_game_or_no_explicit_present_active_fails(self):
        with self.assertRaisesRegex(ValueError,"Not a week-specific"):
            parse_named_active_news(fixture(),URL,season=2026,week=11,game="ATL@CAR")
        with self.assertRaisesRegex(ValueError,"Wrong NFL season"):
            parse_named_active_news(fixture(),URL,season=2025,week=10,game="ATL@CAR")
        with self.assertRaisesRegex(ValueError,"no unambiguous"):
            parse_named_active_news(fixture(),URL,season=2026,week=10,game="SF@LAC")
        weaker=fixture().replace(b"is officially active",b"is likely to play").replace(
            b"is active for Sunday's game",b"is expected to play for Sunday's game")
        with self.assertRaisesRegex(ValueError,"no unambiguous"):
            self.parse(weaker)

    def test_publication_time_and_name_anchors_are_required(self):
        without=fixture().replace(
            b'<meta property="article:published_time" content="2026-11-15T15:00:00Z">',
            b'')
        with self.assertRaisesRegex(ValueError,"machine-readable"):
            self.parse(without)
        altered=fixture().replace(b'/players/drake-london/',b'/non-player/')
        changed=self.parse(altered)
        self.assertNotIn("ATL",changed["active_by_team"])

if __name__=="__main__":
    unittest.main()
