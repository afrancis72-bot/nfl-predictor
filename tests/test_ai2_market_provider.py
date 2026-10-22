"""Offline tests for market provider: no real API requests or credentials."""
import unittest
from datetime import datetime, timedelta, timezone
from ai2_market_challenge import freeze_simulation
from ai2_market_provider import retrieve_market_props

class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.frozen = freeze_simulation("r", [
            {"player_id": "dk1", "market_key": "player_rush_yds",
             "samples": [40, 50, 60]}], "2026-10-22T18:00:00Z")
        self.board = [{"event_id": "game1", "player_name": "Test Player",
                       "market_key": "player_rush_yds", "period": "FULL",
                       "line": 50.5, "over_price": -110, "under_price": -110,
                       "bookmaker": "draftkings",
                       "last_update": "2026-10-22T18:01:00Z"}]

    def test_explicit_identity_mapping_only(self):
        calls = []
        def fetch(url, headers):
            calls.append((url, headers))
            return self.board
        self.assertEqual(retrieve_market_props(self.frozen, sport="NFL", api_key="test-secret",
                                               identity_map={}, fetcher=fetch), [])
        rows = retrieve_market_props(self.frozen, sport="NFL", api_key="test-secret",
                                     identity_map={("game1", "Test Player"): "dk1"}, fetcher=fetch)
        self.assertEqual(rows[0]["player_id"], "dk1")
        self.assertEqual(rows[0]["period"], "FULL")
        self.assertEqual(calls[0][1]["X-API-Key"], "test-secret")
        self.assertNotIn("test-secret", calls[0][0])

    def test_mismatched_frozen_player_excluded(self):
        rows = retrieve_market_props(self.frozen, sport="NFL", api_key="secret",
                                     identity_map={("game1", "Test Player"): "other"},
                                     fetcher=lambda u, h: self.board)
        self.assertEqual(rows, [])

    def test_no_network_without_freeze(self):
        with self.assertRaises(TypeError):
            retrieve_market_props(None, sport="NFL", api_key="secret", identity_map={},
                                  fetcher=lambda u, h: self.fail("network called"))

    def test_truncated_board_rejected(self):
        with self.assertRaises(ValueError):
            retrieve_market_props(self.frozen, sport="NFL", api_key="secret",
                                  identity_map={}, max_rows=1,
                                  fetcher=lambda u, h: self.board)

    def test_future_freeze_rejected(self):
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        frozen = freeze_simulation("future", [
            {"player_id": "dk1", "market_key": "player_rush_yds",
             "samples": [50]}], future)
        with self.assertRaises(ValueError):
            retrieve_market_props(frozen, sport="NFL", api_key="secret",
                                  identity_map={}, fetcher=lambda u, h: [])

if __name__ == "__main__":
    unittest.main()
