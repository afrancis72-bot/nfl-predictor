"""Fail-closed NFL simulation adapter tests (no real-data validation claim)."""
import unittest
from unittest.mock import patch
import pandas as pd
from ai2_nfl_simulation_adapter import run_verified_stat_simulation


def pool():
    return pd.DataFrame([dict(ID="101", Name="Example QB", Position="QB",
                              TeamAbbrev="AAA", Salary=6500, game="AAA@BBB",
                              proj=20.0, optimizer_eligible=True,
                              weekly_projection_valid=True, opportunity_verified=True),
                         dict(ID="102", Name="Example RB", Position="RB",
                              TeamAbbrev="BBB", Salary=5100, game="AAA@BBB",
                              proj=11.0, optimizer_eligible=True,
                              weekly_projection_valid=True, opportunity_verified=True)])


def rates():
    return pd.DataFrame([dict(player="Example QB", team="AAA", position="QB",
                              games_played=6, pass_attempts_pg=32,
                              carries_pg=3, targets_pg=0, source="NFLVERSE_OBSERVED"),
                         dict(player="Example RB", team="BBB", position="RB",
                              games_played=6, pass_attempts_pg=0, carries_pg=12,
                              targets_pg=3, source="NFLVERSE_OBSERVED")])


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

    def test_missing_draftkings_id_blocks(self):
        p = pool()
        p.loc[0, "ID"] = ""
        with self.assertRaisesRegex(ValueError, "nonempty DraftKings ID"):
            run_verified_stat_simulation(p, rates())

    def test_fractional_simulation_count_blocks(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            run_verified_stat_simulation(pool(), rates(), n_sims=10000.5)

    def test_non_numeric_historical_rate_blocks(self):
        r = rates()
        r.loc[0, "pass_attempts_pg"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Invalid historical opportunity"):
            run_verified_stat_simulation(pool(), r)

    def test_inapplicable_opportunity_nulls_are_allowed(self):
        r = rates()
        r.loc[1, "pass_attempts_pg"] = float("nan")
        with patch("ai2_nfl_simulation_adapter.simulate_stat_driven_dfs",
                   return_value=("players", "points", "stats")) as engine:
            run_verified_stat_simulation(pool(), r)
        engine.assert_called_once()

    def test_missing_applicable_targets_blocks(self):
        r = rates()
        r.loc[1, "targets_pg"] = float("nan")
        with self.assertRaisesRegex(ValueError, "targets_pg"):
            run_verified_stat_simulation(pool(), r)

    def test_nonfinite_projection_blocks(self):
        p = pool()
        p.loc[0, "proj"] = float("nan")
        with self.assertRaisesRegex(ValueError, "positive finite independent projection"):
            run_verified_stat_simulation(p, rates())

    def test_invalid_salary_blocks(self):
        p = pool()
        p.loc[0, "Salary"] = 0
        with self.assertRaisesRegex(ValueError, "positive finite DraftKings salary"):
            run_verified_stat_simulation(p, rates())

    def test_unsupported_position_blocks(self):
        p = pool()
        p.loc[0, "Position"] = "K"
        with self.assertRaisesRegex(ValueError, "Unsupported NFL roster position"):
            run_verified_stat_simulation(p, rates())

    def test_missing_historical_games_blocks(self):
        r = rates()
        r.loc[0, "games_played"] = 0
        with self.assertRaisesRegex(ValueError, "games_played"):
            run_verified_stat_simulation(pool(), r)

    def test_blank_source_blocks(self):
        r = rates()
        r.loc[0, "source"] = ""
        with self.assertRaisesRegex(ValueError, "nonempty"):
            run_verified_stat_simulation(pool(), r)

    def test_missing_matchup_blocks(self):
        p = pool()
        p.loc[0, "game"] = ""
        with self.assertRaisesRegex(ValueError, "matchup"):
            run_verified_stat_simulation(p, rates())


if __name__ == "__main__":
    unittest.main()
