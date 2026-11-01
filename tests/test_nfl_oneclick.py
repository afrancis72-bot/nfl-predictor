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
        research.return_value = {
            "blockers": [], "injury_meta": {"source": "https://www.nfl.com/injuries/",
                                           "checked_utc": "2026-10-28T22:00:00Z",
                                           "practice_rows": []},
            "players": __import__("pandas").DataFrame(
                [{"Name": "Breece Hall", "Team": "NYJ",
                  "current_status": "Q", "DK Status": "Q"}]),
            "questionable_count": 1,
            "excluded_count": 0
        }
        players = pd.DataFrame([
            {"ID": str(i), "Name": f"P{i}", "Position": "WR", "Salary": 5000,
             "proj": 10.0, "projection_source": "observed", "TeamAbbrev":"DAL"}
            for i in range(9)])
        # A prior week's changed opinion must not become an invisible ban.
        players.loc[0, "Name"] = "Malik Willis"
        players.loc[2, "Name"] = "Adonai Mitchell"
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
        # One-click orchestration test uses a stub scoring ledger. Full
        # scoring component math is tested in dedicated reconciliation tests.
        verified = [
            {"Player":p["Name"],"Team":p["TeamAbbrev"],
             "Position":p["Position"],"DraftKings ID":str(p["ID"]),
             "Audit status":"Verified: per-simulation DK points match components",
             "Rebuilt mean DK":11.0}
            for p in players.to_dict("records")
        ]
        with patch("ai2_nfl_oneclick.summarize_simulated_players",
                   return_value=verified):
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
        self.assertIn("Malik Willis", modeled)
        self.assertIn("Adonai Mitchell", modeled)
        self.assertEqual(output["audit"]["user_vetoed"], [])
        self.assertEqual(output["pregame_report"]["official_injury_url"],
                         "https://www.nfl.com/injuries/")
        self.assertEqual(output["pregame_report"]["injuries"][0]["Player"],
                         "Breece Hall")



    @patch("ai2_nfl_oneclick.run_verified_stat_simulation")
    @patch("ai2_nfl_oneclick.create_observed_model_pool")
    @patch("ai2_nfl_oneclick.collect_automatic_baselines")
    def test_missing_player_stat_details_block_lineup_generation(
            self, research, pool, simulation):
        import numpy as np
        import pandas as pd
        research.return_value = {
            "blockers":[],
            "injury_meta":{"source":"https://www.nfl.com/injuries/"}}
        players=pd.DataFrame([{
            "ID":"100","Name":"Sample Receiver","Position":"WR",
            "Salary":6200,"TeamAbbrev":"DAL","proj":13.0,
            "projection_source":"nflverse weekly stats"}])
        pool.return_value=(players,pd.DataFrame([{"player":"Sample Receiver"}]),
                           [],{"complete_games":1})
        # Old failure: fantasy points present, no football stats.
        simulation.return_value=(players,np.full((10000,1),11.),
                                 {"__MODEL_VERSION__":"broken"})
        with self.assertRaisesRegex(ValueError,
                                    "scoring components could not be verified"):
            build_oneclick_portfolio(self.saved,seed=10)

    def test_seed_bounds(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            build_oneclick_portfolio(self.saved, None, seed=-1)

    def test_seed_string_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            build_oneclick_portfolio(self.saved, None, seed="42")

if __name__ == "__main__":
    unittest.main()
