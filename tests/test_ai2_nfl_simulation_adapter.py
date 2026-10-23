"""Fail-closed NFL simulation adapter tests (no real-data validation claim)."""
import unittest
from unittest.mock import patch
import pandas as pd
from ai2_nfl_simulation_adapter import run_verified_stat_simulation


def pool():
    return pd.DataFrame([dict(ID="101", Name="Example QB", Position="QB",
                              TeamAbbrev="AAA", Salary=6500, game="AAA@BBB",
                              proj=20.0, optimizer_eligible=True,
                              weekly_projection_valid=True, opportunity_verified=True),\n                         dict(ID="102", Name="Example RB", Position="RB",\n                              TeamAbbrev="BBB", Salary=5100, game="AAA@BBB",\n                              proj=11.0, optimizer_eligible=True,\n                              weekly_projection_valid=True, opportunity_verified=True)])


def rates():
    return pd.DataFrame([dict(player="Example QB", team="AAA", position="QB",
                              games_played=6, pass_attempts_pg=32,
                              carries_pg=3, targets_pg=0, source="NFLVERSE_OBSERVED"),\n                         dict(player="Example RB", team="BBB", position="RB",\n                              games_played=6, pass_attempts_pg=0, carries_pg=12,\n                              targets_pg=3, source="NFLVERSE_OBSERVED")])


class AdapterTests(unittest.TestCase):
    def test_verified_inputs_reach_engine(self):
        with patch("ai2_nfl_simulation_adapter.simulate_stat_driven_dfs",
                   return_value=("players", "points", "stats")) as engine:
            result = run_verified_stat_simulation(pool(), rates(), n_sims=10000)
        self.assertEqual(result, ("players", "points", "stats"))
        engine.assert_called_once()
        self.assertIs(engine.call_args.kwargs["allow_role_proxies"], False)

    def test_unverified_opportunity_blocks(self):
        p = pool()
        p.loc[0, "opportunity_verified"] = False
        with self.assertRaisesRegex(ValueError, "verified"):
            run_verified_stat_simulation(p, rates())

    def test_role_proxy_blocks(self):
        r = rates()
        r.loc[0, "source"] = "ROLE_PROXY_NO_HISTORY"
        with self.assertRaisesRegex(ValueError, "role-proxy"):
            run_verified_stat_simulation(pool(), r)

    def test_missing_provenance_blocks(self):
        with self.assertRaisesRegex(ValueError, "provenance"):
            run_verified_stat_simulation(pool(), rates().drop(columns="source"))

    def test_incomplete_game_blocks(self):
        p = pool().iloc[:1].copy()
        with self.assertRaisesRegex(ValueError, "Incomplete matchup"):
            run_verified_stat_simulation(p, rates())

    def test_missing_matchup_blocks(self):
        p = pool()
        p.loc[0, "game"] = ""
        with self.assertRaisesRegex(ValueError, "matchup"):
            run_verified_stat_simulation(p, rates())


if __name__ == "__main__":
    unittest.main()
