"""Tests for post-simulation market isolation and review-only findings."""
import unittest
from ai2_market_challenge import freeze_simulation, compare_frozen_to_props

class MarketChallengeTests(unittest.TestCase):
    def setUp(self):
        self.frozen = freeze_simulation("run-1", [
            {"player_id": "p1", "market_key": "player_rush_yds",
             "samples": [80, 85, 90, 95, 100]}], "2026-10-22T18:00:00Z")
        self.prop = {"player_id": "p1", "market_key": "player_rush_yds",
                     "period": "FULL", "line": 65.5, "over_price": -110,
                     "under_price": -110, "bookmaker": "test",
                     "last_update": "2026-10-22T17:59:00Z",
                     "fetched_at": "2026-10-22T18:01:00Z"}

    def test_large_gap_flagged_without_mutation(self):
        original = self.frozen
        findings = compare_frozen_to_props(self.frozen, [self.prop])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["status"], "REVIEW_ONLY")
        self.assertAlmostEqual(findings[0]["probability_gap"], .5)
        self.assertEqual(original, self.frozen)

    def test_retrieval_must_follow_freeze(self):
        prop = {**self.prop, "fetched_at": "2026-10-22T17:59:30Z"}
        self.assertEqual(compare_frozen_to_props(self.frozen, [prop]), [])

    def test_stale_quote_rejected(self):
        prop = {**self.prop, "last_update": "2026-10-22T17:00:00Z"}
        self.assertEqual(compare_frozen_to_props(self.frozen, [prop]), [])

    def test_unverified_player_id_not_name_matched(self):
        prop = {**self.prop, "player_id": "other", "player_name": "Similar Name"}
        self.assertEqual(compare_frozen_to_props(self.frozen, [prop]), [])

    def test_partial_game_market_rejected(self):
        self.assertEqual(compare_frozen_to_props(self.frozen, [{**self.prop, "period": "1H"}]), [])

    def test_freeze_rejects_bad_samples(self):
        with self.assertRaises(ValueError):
            freeze_simulation("bad", [{"player_id": "p", "market_key": "player_rush_yds",
                                       "samples": [float("nan")]}], "2026-10-22T18:00:00Z")

if __name__ == "__main__":
    unittest.main()
