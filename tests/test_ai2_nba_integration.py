"""Offline NBA predictor tests: no fabricated player or market data."""
import unittest
from datetime import date
from unittest.mock import patch
import numpy as np
import pandas as pd

from nba_engine.data_sources import norm_team, season_key, normalize_logs, scoreboard, discover_game_log_files
from nba_engine.predictor import (build_projections, simulate_stat_lines,
                                  summarize_simulated_projections, build_prefilled_prop_worksheet,
                                  prepare_bet_lines, evaluate_bets, parlay_summary)

class NBAEngineTests(unittest.TestCase):
    @staticmethod
    def history():
        rows=[]
        for dt in pd.date_range("2025-11-01", "2026-10-27", freq="7D"):
            for pid,name,team,opp,points in [
                ("1","Player One","LAL","GSW",27),
                ("2","Player Two","GSW","LAL",19),
                ("3","Player Three","BOS","NYK",22),
                ("4","Player Four","NYK","BOS",18),
            ]:
                rows.append({"Date":dt.strftime("%Y-%m-%d"),"Player":name,"ID":pid,
                             "Team":team,"Opp":opp,"Minutes":34,
                             "Points":points+(int(dt.day)%7),"Rebounds":7,"Assists":5,
                             "3PM":3,"Steals":1,"Blocks":1,"Turnovers":2})
        return normalize_logs(pd.DataFrame(rows))

    def test_schedule_json_with_realistic_fields(self):
        body={"events":[{"id":"401","date":"2026-10-30T23:00Z","status":{"type":{"state":"pre","shortDetail":"7:00 PM"}},
                "competitions":[{"competitors":[
                  {"homeAway":"away","team":{"abbreviation":"GS"}},
                  {"homeAway":"home","team":{"abbreviation":"LAL"}}]}]}]}
        with patch("nba_engine.data_sources._get_json",return_value=body):
            df=scoreboard(date(2026,10,30))
        self.assertEqual(df["matchup"].iloc[0],"GSW @ LAL")
        self.assertEqual(df["state"].iloc[0],"pre")

    def test_discover_dynamic_snapshot_name(self):
        rows=[{"name":"NBA_players_game_log_2025-26_as_of_2026-10-27.csv"},
              {"name":"NBA_players_game_log_2025-26_as_of_2026-10-29.csv"},
              {"name":"NBA_players_game_log_2026-27_as_of_2026-10-29.csv"}]
        with patch("nba_engine.data_sources._get_json",return_value=rows):
            found=discover_game_log_files(date(2026,10,29))
        self.assertEqual(len(found),2)
        self.assertEqual(found[0]["as_of"],"2026-10-29")

    def test_multiple_matchups_have_correct_opponent(self):
        history=self.history()
        proj=build_projections(history,{"LAL":"GSW","GSW":"LAL","BOS":"NYK","NYK":"BOS"},date(2026,10,29),min_minutes=12)
        self.assertEqual(len(proj),4)
        assert (proj.loc[proj["team"]=="BOS","opponent"]=="NYK").all()
        assert (proj.loc[proj["team"]=="LAL","opponent"]=="GSW").all()
        self.assertTrue((proj["current_season_games"]>0).all())
        self.assertTrue((proj["minutes"]<=48).all())

    def test_injuries_and_minutes_overrides_do_not_modify_history(self):
        history=self.history()
        before=history.copy(deep=True)
        p=build_projections(history,{"LAL":"GSW","GSW":"LAL"},date(2026,10,29),
                            excluded=["2"],minute_overrides={"1":40})
        self.assertEqual(len(p),1)
        self.assertAlmostEqual(float(p["minutes"].iloc[0]),40)
        pd.testing.assert_frame_equal(history,before)

    def test_future_game_does_not_leak(self):
        history=self.history()
        expected=build_projections(history,{"LAL":"GSW","GSW":"LAL"},date(2026,10,29))
        extra=history.iloc[[0]].copy()
        extra["date"]=pd.Timestamp("2026-10-29")
        extra["pts"]=900
        added=pd.concat([history,extra],ignore_index=True)
        actual=build_projections(added,{"LAL":"GSW","GSW":"LAL"},date(2026,10,29))
        pd.testing.assert_frame_equal(expected,actual)

    def test_simulation_seed_reproducibility_and_fibonacci_variation(self):
        history = self.history()
        projections = build_projections(history, {"LAL":"GSW","GSW":"LAL"}, date(2026,10,29))
        first = simulate_stat_lines(projections, history, trials=2000, seed=13)
        repeated = simulate_stat_lines(projections, history, trials=2000, seed=13)
        next_fib = simulate_stat_lines(projections, history, trials=2000, seed=21)
        self.assertEqual(set(first), set(repeated))
        for market in first:
            np.testing.assert_array_equal(first[market], repeated[market])
        self.assertTrue(any(not np.array_equal(first[market],next_fib[market]) for market in first))
        # Using a different seed must NEVER change model means.
        rerun_projections = build_projections(history, {"LAL":"GSW","GSW":"LAL"}, date(2026,10,29))
        pd.testing.assert_frame_equal(projections,rerun_projections)

    def test_seed_changes_market_hit_probabilities_but_not_projected_means(self):
        history = self.history()
        p = build_projections(history, {"LAL":"GSW","GSW":"LAL"}, date(2026,10,29))
        pid = str(p["player_id"].iloc[0])
        markets = pd.DataFrame([
            {"player_id":pid, "market":"Points", "line":29.5,"side":"Over","odds":-110},
            {"player_id":pid, "market":"Points", "line":31.5,"side":"Under","odds":-110},
            {"player_id":pid, "market":"PRA", "line":40.5,"side":"Over","odds":110},
        ])
        lines = prepare_bet_lines(markets, p)
        a, _ = evaluate_bets(lines,simulate_stat_lines(p,history,trials=10000,seed=42))
        b, _ = evaluate_bets(lines,simulate_stat_lines(p,history,trials=10000,seed=34))
        self.assertTrue(np.any(np.abs(a["win_probability"]-b["win_probability"]) > 1e-8))
        self.assertTrue(np.array_equal(a["line"].to_numpy(),b["line"].to_numpy()))
        self.assertTrue(np.array_equal(a["odds"].to_numpy(),b["odds"].to_numpy()))

    def test_joint_stats_and_parlay_probabilities(self):
        history=self.history()
        p=build_projections(history,{"LAL":"GSW","GSW":"LAL"},date(2026,10,29))
        draws=simulate_stat_lines(p,history,trials=2000,seed=10)
        pid=str(p["player_id"].iloc[0])
        self.assertTrue(np.array_equal(draws[(pid,"PRA")],
                         draws[(pid,"Points")]+draws[(pid,"Rebounds")]+draws[(pid,"Assists")]))
        lines=pd.DataFrame([
          {"player_id":pid,"market":"Points","line":25.5,"side":"Over","odds":-110},
          {"player_id":pid,"market":"Assists","line":4.5,"side":"Under","odds":105},
        ])
        bets=prepare_bet_lines(lines,p)
        report,status=evaluate_bets(bets,draws)
        ticket=parlay_summary(status,[0,1])
        self.assertAlmostEqual(ticket["all_win"]+ticket["any_push_no_loss"]+ticket["any_loss"],1,places=4)
        self.assertFalse(ticket["has_push"])
        self.assertTrue(((report["win_probability"]>=0)&(report["win_probability"]<=1)).all())


    def test_simulated_projections_change_with_seed_while_baselines_remain_fixed(self):
        history = self.history()
        baseline = build_projections(history, {"LAL":"GSW","GSW":"LAL"}, date(2026,10,29))
        a = simulate_stat_lines(baseline, history, trials=10000, seed=42)
        b = simulate_stat_lines(baseline, history, trials=10000, seed=34)
        repeated = simulate_stat_lines(baseline, history, trials=10000, seed=42)
        first = summarize_simulated_projections(baseline, a)
        second = summarize_simulated_projections(baseline, b)
        replay = summarize_simulated_projections(baseline, repeated)
        pd.testing.assert_frame_equal(first, replay)
        np.testing.assert_array_equal(
            first["baseline_projection"].to_numpy(), second["baseline_projection"].to_numpy()
        )
        self.assertTrue(np.any(np.abs(
            first["simulation_mean"].to_numpy() - second["simulation_mean"].to_numpy()
        ) > 1e-6))
        self.assertTrue((first["simulation_p10"] <= first["simulation_median"]).all())
        self.assertTrue((first["simulation_median"] <= first["simulation_p90"]).all())
        self.assertTrue((first["monte_carlo_se"] >= 0).all())
        self.assertEqual(len(first), len(baseline)*5)
        pid = str(baseline["player_id"].iloc[0])
        mean = first.loc[(first["player_id"]==pid)&(first["market"]=="PRA"), "simulation_mean"].iloc[0]
        self.assertAlmostEqual(float(mean), float(np.mean(a[(pid, "PRA")])), places=8)

    def test_prefilled_worksheet_includes_actual_simulated_p10_p90_not_book_lines(self):
        history=self.history()
        p=build_projections(history,{"LAL":"GSW","GSW":"LAL"},date(2026,10,29))
        samples=simulate_stat_lines(p,history,trials=2000,seed=55)
        summary=summarize_simulated_projections(p,samples,
                                                 markets=["Points","PRA","Assists"])
        ids=[str(p["player_id"].iloc[1]),str(p["player_id"].iloc[0])]
        worksheet=build_prefilled_prop_worksheet(
            p,summary,ids,["PRA","Points"])
        self.assertEqual(len(worksheet),4)
        self.assertEqual(worksheet["player_id"].tolist(),[ids[0],ids[0],ids[1],ids[1]])
        self.assertEqual(worksheet["market"].tolist(),["PRA","Points","PRA","Points"])
        self.assertTrue(worksheet["line"].isna().all())
        self.assertTrue(worksheet["odds"].isna().all())
        self.assertIn("sim_p10",worksheet)
        self.assertIn("sim_p90",worksheet)
        for r in worksheet.itertuples(index=False):
            arr=samples[(r.player_id,r.market)]
            self.assertAlmostEqual(r.sim_p10,round(float(np.quantile(arr,.10)),2))
            self.assertAlmostEqual(r.sim_p90,round(float(np.quantile(arr,.90)),2))
            self.assertAlmostEqual(r.sim_mean,round(float(np.mean(arr)),2))
            model_row=p.loc[p["player_id"]==r.player_id].iloc[0]
            self.assertAlmostEqual(r.model_projection,round(float(model_row[r.market]),2))
            self.assertLessEqual(r.sim_p10,r.sim_p90)

    def test_prefilled_ranges_reject_missing_or_wrong_market_but_allow_no_selections(self):
        history=self.history()
        p=build_projections(history,{"LAL":"GSW","GSW":"LAL"},date(2026,10,29))
        samples=simulate_stat_lines(p,history,trials=1000,seed=42)
        summary=summarize_simulated_projections(p,samples,markets=["Points"])
        empty=build_prefilled_prop_worksheet(p,summary,[],["PRA"])
        self.assertEqual(len(empty),0)
        self.assertEqual(list(empty.columns)[-3:],["line","side","odds"])
        with self.assertRaises(ValueError):
            build_prefilled_prop_worksheet(p,summary,["unlisted"],["Points"])
        with self.assertRaises(ValueError):
            build_prefilled_prop_worksheet(p,summary,[str(p["player_id"].iloc[0])],["PRA"])
        with self.assertRaises(ValueError):
            build_prefilled_prop_worksheet(p,summary,[str(p["player_id"].iloc[0])],["NotAMarket"])

    def test_integer_line_pushes_are_not_counted_as_losses(self):
        history=self.history()
        p=build_projections(history,{"LAL":"GSW","GSW":"LAL"},date(2026,10,29))
        draws=simulate_stat_lines(p,history,trials=2000,seed=3)
        pid=str(p["player_id"].iloc[0])
        bets=prepare_bet_lines(pd.DataFrame([
            {"player_id":pid,"market":"Rebounds","line":7,"side":"Over","odds":-110},
            {"player_id":pid,"market":"Assists","line":5,"side":"Over","odds":-110},
        ]),p)
        report,status=evaluate_bets(bets,draws)
        self.assertTrue((report["push_probability"]>0).any())
        self.assertTrue(parlay_summary(status,[0,1])["has_push"])

if __name__=="__main__":
    unittest.main()
