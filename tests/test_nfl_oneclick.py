import unittest
from unittest.mock import patch
from ai2_nfl_oneclick import build_oneclick_portfolio, _slate_week

class OneClickTests(unittest.TestCase):
    def setUp(self):
        csv = ("ID,Name,Position,Salary,Game Info,TeamAbbrev\\n"
               "1,Example,QB,6500,DAL@ARI 11/01/2026 01:00PM ET,DAL\\n")
        self.saved = ("salary.csv", csv.replace("\\n", "\n").encode(),
                      "2026-10-27T22:00:00+00:00")

    def test_week_from_dk_slate(self):
        self.assertEqual(_slate_week(self.saved), (2026, 8))

    @patch("ai2_nfl_oneclick.create_observed_model_pool")
    @patch("ai2_nfl_oneclick.collect_automatic_baselines")
    def test_auto_research_without_projection_upload(self, research, pool):
        research.return_value = {"blockers": [], "injury_meta": {"source": "test"}}
        pool.side_effect = ValueError("Not enough observed players")
        with self.assertRaisesRegex(ValueError, "Not enough observed players"):
            build_oneclick_portfolio(self.saved, None, seed=42)
        research.assert_called_once_with(self.saved, 2026, 8)
        pool.assert_called_once()


    @patch("ai2_nfl_oneclick.export_classic_candidates", return_value="QB,RB,...\\n")
    @patch("ai2_nfl_oneclick.select_simulated_portfolio")
    @patch("ai2_nfl_oneclick.build_classic_candidates")
    @patch("ai2_nfl_oneclick.run_verified_stat_simulation")
    @patch("ai2_nfl_oneclick.create_observed_model_pool")
    @patch("ai2_nfl_oneclick.collect_automatic_baselines")
    def test_build_20_uses_own_stats_and_one_injury_pass(
            self, research, model_pool, simulation, candidates, selection, exporter):
        import numpy as np
        import pandas as pd
        research.return_value = {"blockers": [], "injury_meta": {"source": "official"}}
        players = pd.DataFrame([
            {"ID": str(i), "Name": f"P{i}", "Position": "WR", "Salary": 5000,
             "proj": 10.0, "projection_source": "observed", "TeamAbbrev":"DAL"}
            for i in range(9)])
        raw = pd.DataFrame([{"player": "P0"}])
        outs = [{"player": "Unavailable", "team": "DAL"}]
        model_pool.return_value = (players, raw, outs, {"complete_games":1})
        simulation.return_value = (players, np.full((10000,9), 11.0),
                                   {"P1": {"targets": np.full(10000, 6),
                                            "carries": np.zeros(10000)}})
        candidates.return_value = [{"players": players.to_dict("records"), "salary":50000} for _ in range(20)]
        selection.return_value = [
            {"candidate": {"players": players.to_dict("records"),"salary":50000},
             "contest_tier":"$3" if i < 5 else "$0.50",
             "simulated_mean":99.0,"simulated_p90":105.0,
             "simulated_p99":120.0,"p90_per_1000":2.1}
            for i in range(20)]
        output = build_oneclick_portfolio(self.saved, seed=117)
        self.assertEqual(len(output["summary"]), 20)
        self.assertEqual(output["seed"], 117)
        self.assertEqual(output["csv"], "QB,RB,...\\n")
        self.assertEqual(simulation.call_args.kwargs["out_players"], outs)
        self.assertIs(simulation.call_args.args[1], raw)
        self.assertEqual(simulation.call_args.kwargs["seed"],117)
        self.assertEqual(candidates.call_args.kwargs["seed"],117)
        modeled = {row["Name"]:row for row in candidates.call_args.args[0]}
        self.assertEqual(modeled["P1"]["projected_targets"],6.0)
        self.assertEqual(modeled["P1"]["projected_touches"],6.0)

    def test_seed_bounds(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            build_oneclick_portfolio(self.saved, None, seed=-1)

    def test_seed_string_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            build_oneclick_portfolio(self.saved, None, seed="42")

if __name__ == "__main__":
    unittest.main()
