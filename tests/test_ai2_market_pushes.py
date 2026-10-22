"""Additional correctness checks for push-aware market comparisons."""
import unittest
from ai2_market_challenge import freeze_simulation, compare_frozen_to_props

class PushAwareMarketTests(unittest.TestCase):
    def test_pushes_excluded_from_two_way_no_vig_comparison(self):
        frozen = freeze_simulation("push", [{"player_id": "p", "market_key": "player_receptions",
                                           "samples": [3, 4, 4, 4, 5]}], "2026-10-22T18:00:00Z")
        prop = {"player_id": "p", "market_key": "player_receptions", "period": "FULL",
                "line": 4, "over_price": -110, "under_price": -110,
                "last_update": "2026-10-22T18:00:00Z",
                "fetched_at": "2026-10-22T18:01:00Z"}
        # Unconditional P(over)=.2, but among settled bets P(over)=.5.
        self.assertEqual(compare_frozen_to_props(frozen, [prop]), [])

    def test_all_pushes_do_not_produce_spurious_edge(self):
        frozen = freeze_simulation("push", [{"player_id": "p", "market_key": "player_receptions",
                                           "samples": [4, 4, 4]}], "2026-10-22T18:00:00Z")
        prop = {"player_id": "p", "market_key": "player_receptions", "period": "FULL",
                "line": 4, "over_price": -110, "under_price": -110,
                "last_update": "2026-10-22T18:00:00Z",
                "fetched_at": "2026-10-22T18:01:00Z"}
        self.assertEqual(compare_frozen_to_props(frozen, [prop]), [])

if __name__ == "__main__":
    unittest.main()
