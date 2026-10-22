"""Offline pagination regression tests: no external API requests."""
import unittest
from ai2_market_pages import fetch_complete_props

class PaginationTests(unittest.TestCase):
    def test_two_pages_and_completeness(self):
        calls = []
        def transport(url, headers):
            calls.append(url)
            if len(calls) == 1:
                return [{"event_id": "a"}], {"X-Result-Has-More": "true", "X-Next-Offset": "1"}
            return [{"event_id": "b"}], {"X-Result-Has-More": "false"}
        rows = fetch_complete_props("americanfootball_nfl", ["player_rush_yds"],
                                    "test-key", transport=transport)
        self.assertEqual(len(rows), 2)
        self.assertIn("offset=1", calls[1])
        self.assertNotIn("test-key", calls[0])

    def test_empty_page_with_more_results_fails(self):
        with self.assertRaises(ValueError):
            fetch_complete_props("americanfootball_nfl", ["player_rush_yds"],
                                 "test-key", transport=lambda u, h: ([], {
                                     "X-Result-Has-More": "true", "X-Next-Offset": "1"}))

    def test_oversized_page_fails(self):
        with self.assertRaises(ValueError):
            fetch_complete_props("americanfootball_nfl", ["player_rush_yds"],
                                 "test-key", page_size=1,
                                 transport=lambda u, h: ([{}, {}], {
                                     "X-Result-Has-More": "false"}))

    def test_missing_completeness_header_fails(self):
        with self.assertRaises(ValueError):
            fetch_complete_props("americanfootball_nfl", ["player_rush_yds"],
                                 "test-key", transport=lambda u, h: ([], {}))

    def test_incomplete_page_budget_fails(self):
        with self.assertRaises(ValueError):
            fetch_complete_props("americanfootball_nfl", ["player_rush_yds"],
                                 "test-key", max_pages=1,
                                 transport=lambda u, h: ([{}], {"X-Result-Has-More": "true",
                                                                  "X-Next-Offset": "1"}))

if __name__ == "__main__":
    unittest.main()
