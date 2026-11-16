"""No lookahead and independent rush/receive/pass defense evidence tests."""
import unittest
import pandas as pd
from ai2_nfl_defense_components import (
    METRICS, defense_component_profiles, component_pregame_grading_rows)


def history():
    players = [("Arm", "A", "QB"), ("Back", "B", "RB"),
               ("Wide", "C", "WR"), ("End", "D", "TE")]
    rows = []
    for week in range(1, 5):
        for name, team, position in players:
            r = dict(
                season=2025, week=week, player_id=name, player_name=name,
                position=position, recent_team=team, season_type="REG",
                opponent_team=("CHI" if week < 4 else "DAL"),
                attempts=30 if position == "QB" else 0,
                passing_yards=240 if position == "QB" else 0,
                carries=20 if position == "RB" else 0,
                rushing_yards=80 if position == "RB" else 0,
                targets=4 if position == "RB" else 6 if position == "WR"
                        else 5 if position == "TE" else 0,
                receiving_yards=20 if position == "RB" else 60 if position == "WR"
                                else 35 if position == "TE" else 0)
            if week == 4:
                if position == "QB":
                    r["passing_yards"] = 300
                if position == "RB":
                    r["rushing_yards"] = 40
                    r["receiving_yards"] = 72
                if position == "WR":
                    r["receiving_yards"] = 100
                if position == "TE":
                    r["receiving_yards"] = 12
            rows.append(r)
    return pd.DataFrame(rows)


class ComponentDefenseTests(unittest.TestCase):
    def test_runner_and_receiver_are_separate_for_same_running_back(self):
        p = defense_component_profiles(
            history(), 2025, 5, min_observations=1,
            min_opportunities=2, shrink_opportunities=5)
        effects = {
            (x.position, x.channel): x.shrunk_efficiency_pct
            for x in p.itertuples()}
        self.assertGreater(effects[("QB", "pass_yards_per_attempt")], 0)
        self.assertLess(effects[("RB", "rush_yards_per_carry")], 0)
        self.assertGreater(effects[("RB", "receive_yards_per_target")], 0)
        self.assertGreater(effects[("WR", "receive_yards_per_target")], 0)
        self.assertLess(effects[("TE", "receive_yards_per_target")], 0)
        self.assertEqual(len(effects), len(METRICS))

    def test_future_week_stats_cannot_change_pregame_profiles(self):
        h = history()
        initial = defense_component_profiles(
            h, 2025, 4, min_observations=1, min_opportunities=2)
        h.loc[h.week.eq(4), "receiving_yards"] = 2000
        later = pd.concat([h, h[h.week.eq(4)].assign(week=5)],
                          ignore_index=True)
        after = defense_component_profiles(
            later, 2025, 4, min_observations=1, min_opportunities=2)
        pd.testing.assert_frame_equal(initial, after)
        self.assertFalse((initial.defense == "DAL").any())

    def test_target_week_results_do_not_change_player_prior(self):
        h = history()
        scored = component_pregame_grading_rows(h, 2025, 4)
        qb = scored[scored.player.eq("Arm")].iloc[0]
        self.assertAlmostEqual(qb["prior_yards_per_opp"], 8.)
        self.assertAlmostEqual(qb["actual_yards"], 300.)
        h.loc[(h.week.eq(4)) & h.player_name.eq("Arm"), "passing_yards"] = 500
        scored2 = component_pregame_grading_rows(h, 2025, 4)
        self.assertAlmostEqual(
            scored2[scored2.player.eq("Arm")].iloc[0]["prior_yards_per_opp"], 8.)

    def test_refuses_unsupported_opponent_and_duplicate_games(self):
        h = history()
        with self.assertRaisesRegex(ValueError, "opponent_team"):
            defense_component_profiles(h.drop(columns="opponent_team"), 2025, 5)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            defense_component_profiles(
                pd.concat([h, h.iloc[[0]]]), 2025, 5)
        h.loc[h.week.eq(4), "opponent_team"] = float("nan")
        p = defense_component_profiles(
            h, 2025, 5, min_observations=1,
            min_opportunities=2)
        self.assertFalse((p.defense == "NAN").any())


if __name__ == "__main__":
    unittest.main()
