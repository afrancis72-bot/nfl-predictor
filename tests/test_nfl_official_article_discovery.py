"""Bounded official NFL.com news-article discovery tests."""
import unittest
from ai2_nfl_official_article_discovery import (
    extract_candidate_links, discover_official_links, fetch_official_hub_html,
)

SOURCE=b"""<html><body>
<a href="/news/nfl-week-12-inactives-players-ruled-out-for-sunday-s-13-games">Inactives</a>
<a href="https://www.nfl.com/news/nfl-week-12-injuries-to-monitor-for-sunday-games">
Injuries</a>
<a href="https://www.nfl.com/news/nfl-week-11-inactives-players-ruled-out">Older</a>
<a href="https://nfl.com.attacker.example/news/nfl-week-12-inactives">Fake</a>
<a href="/news/nfl-week-12-inactives-players-ruled-out-for-sunday-s-13-games?tracking=yes">
Duplicate</a>
<a href="/news/week-12-injuries-sunday-pregame-2026-nfl-season">Earlier</a>
</body></html>"""


class DiscoveryTests(unittest.TestCase):
    def test_filters_expected_week_exact_nfl_article_links(self):
        found=extract_candidate_links(SOURCE,"https://www.nfl.com/news",week=12)
        self.assertEqual([x["kind"] for x in found],
                         ["OFFICIAL_INACTIVES","EXPLICIT_INJURY_UPDATES",
                          "EXPLICIT_INJURY_UPDATES"])
        self.assertEqual(len(found),3)
        self.assertNotIn("attacker",str(found))
        self.assertNotIn("week-11",str(found))

    def test_requires_exact_official_hub(self):
        for url in ("https://nfl.com.attacker.example/news",
                    "http://www.nfl.com/news","https://www.nfl.com/billing"):
            with self.subTest(url=url),self.assertRaises(ValueError):
                fetch_official_hub_html(url)
        with self.assertRaisesRegex(ValueError,"requires exact official"):
            extract_candidate_links(SOURCE,"https://attacker.example/news",week=12)

    def test_partial_hubs_surface_errors_not_invent_source(self):
        def fake_fetch(url):
            if url.endswith("inactives/"):
                raise ValueError("No game-day report yet")
            return SOURCE
        result=discover_official_links(
            week=12,hubs=("https://www.nfl.com/news",
                          "https://www.nfl.com/inactives/"),fetcher=fake_fetch)
        self.assertEqual(result["candidate_count"],3)
        self.assertEqual(len(result["errors"]),1)
        self.assertFalse(result["content_and_status_verified"])

    def test_empty_week_not_an_active_player_signal(self):
        result=discover_official_links(
            week=13,hubs=("https://www.nfl.com/news",),
            fetcher=lambda _: SOURCE)
        self.assertEqual(result["articles"],[])
        self.assertFalse(result["content_and_status_verified"])

    def test_rejects_bad_week(self):
        with self.assertRaisesRegex(ValueError,"1–22"):
            discover_official_links(week=99,hubs=())

if __name__=="__main__":
    unittest.main()
