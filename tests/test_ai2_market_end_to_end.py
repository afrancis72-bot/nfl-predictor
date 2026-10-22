"""Check that the post-simulation audit flow preserves a frozen baseline."""
import unittest
from ai2_market_challenge import freeze_simulation, compare_frozen_to_props

class MarketEndToEndTests(unittest.TestCase):
    def test_simulation_frozen_before_odds_comparison(self):
        frozen = freeze_simulation(
            "audit-001",
            [{"player_id": "player-1", "market_key": "player_rush_yds",
              "samples": [82, 90, 97, 104]}],
            "2026-10-22T18:00:00Z",
        )
        original_digest = frozen.digest
        prop = {
            "player_id": "player-1",
            "market_key": "player_rush_yds",
            "period": "FULL",
            "line": 70.5,
            "over_price": -110,
            "under_price": -110,
            "last_update": "2026-10-22T17:59:00Z",
            "fetched_at": "2026-10-22T18:01:00Z",
        }
        findings = compare_frozen_to_props(frozen, [prop])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["status"], "REVIEW_ONLY")
        self.assertEqual(frozen.digest, original_digest)
        self.assertEqual(findings[0]["simulation_digest"], original_digest)

if __name__ == "__main__":
    unittest.main()
