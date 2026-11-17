"""Experimental opponent efficiency must change REAL football stats, not add DK points."""
import unittest

import numpy as np
import pandas as pd

from ai2_nfl_defense_components import SOURCE
from ai2_nfl_sim_component_factors import validated_component_lookup
from dfs_stats_engine import simulate_stat_driven_dfs
from test_ai2_nfl_team_pace_scoring import sim_fixture


class MatchupConditionedSimulationTests(unittest.TestCase):
    def profiles(self, season=2026, week=9):
        return pd.DataFrame([
            dict(defense="BBB", position=p, channel=channel,
                 observations=15, opportunities=80,
                 shrunk_efficiency_pct=delta, source=SOURCE,
                 target_season=season, target_week=week)
            for p, channel, delta in (
                ("QB", "pass_yards_per_attempt", -10.0),
                ("RB", "rush_yards_per_carry", -9.0),
                ("RB", "receive_yards_per_target", 8.0),
                ("WR", "receive_yards_per_target", -12.0),
                ("TE", "receive_yards_per_target", 10.0),
            )
        ])

    def simulate(self, strength, profiles=None, seed=31):
        pool, rates = sim_fixture()
        return simulate_stat_driven_dfs(
            pool, rates, n_sims=270, seed=seed, allow_role_proxies=False,
            component_profiles=profiles, component_strength=strength,
            component_target_season=2026, component_target_week=9)

    def test_no_change_to_original_scoring_at_zero_strength(self):
        _, base, observed = self.simulate(0.)
        _, same, details = self.simulate(0., self.profiles())
        np.testing.assert_array_equal(base, same)
        self.assertNotIn("__OPPONENT_COMPONENT_EFFICIENCY__|AAA", details)

    def test_qb_defense_affects_actual_yards_and_preserves_receiver_accounting(self):
        _, baseline_scores, baseline = self.simulate(0.)
        _, scores, adjusted = self.simulate(.5, self.profiles())
        self.assertFalse(np.array_equal(baseline_scores, scores))
        factor = adjusted["__OPPONENT_COMPONENT_EFFICIENCY__|AAA"]["pass_ypa_factor"]
        self.assertAlmostEqual(factor, .95)
        qb_yards = adjusted["QAAA"]["passing_yards"]
        np.testing.assert_allclose(qb_yards,
            baseline["QAAA"]["passing_yards"] * .95, atol=.002)
        for team in ("AAA", "BBB"):
            qb = adjusted[f"Q{team}"]["passing_yards"]
            complete = sum(adjusted[f"{pos}{team}"]["receiving_yards"]
                           for pos in ("R", "W", "T"))
            complete += adjusted[f"__OTHER_RECEIVER__|{team}"]["receiving_yards"]
            np.testing.assert_allclose(qb, complete, atol=.002)
            for player in ("R", "W", "T"):
                name = player+team
                for column in ("targets", "receptions", "receiving_tds",
                               "carries", "rushing_tds"):
                    np.testing.assert_array_equal(
                        baseline[name][column], adjusted[name][column])
            np.testing.assert_array_equal(
                baseline[f"__TEAM_SCORING__|{team}"]["simulated_offensive_tds"],
                adjusted[f"__TEAM_SCORING__|{team}"]["simulated_offensive_tds"])
            np.testing.assert_array_equal(
                baseline[f"__TEAM_PLAYS__|{team}"]["simulated_plays"],
                adjusted[f"__TEAM_PLAYS__|{team}"]["simulated_plays"])
        np.testing.assert_allclose(
            adjusted["RAAA"]["rushing_yards"],
            baseline["RAAA"]["rushing_yards"] * .955, atol=.002)
        # Only AAA offense faces BBB; BBB is neutral against AAA.
        np.testing.assert_allclose(adjusted["QBBB"]["passing_yards"],
                                   baseline["QBBB"]["passing_yards"], atol=.002)

    def test_profile_requires_source_date_sample_and_channel(self):
        for bad in ("source", "target_week", "observations",
                    "shrunk_efficiency_pct", "channel"):
            sample=self.profiles()
            if bad=="source":
                sample.loc[0,bad]="made up"
            elif bad=="target_week":
                sample.loc[0,bad]=10
            elif bad=="observations":
                sample.loc[0,bad]=1
            elif bad=="shrunk_efficiency_pct":
                sample.loc[0,bad]=99
            else:
                sample.loc[0,bad]="free_fantasy_points"
            with self.subTest(field=bad), self.assertRaises(ValueError):
                validated_component_lookup(
                    sample, season=2026, week=9, strength=.5)

    def test_forbidden_double_application(self):
        pool, rates = sim_fixture()
        with self.assertRaisesRegex(ValueError, "cannot be stacked"):
            simulate_stat_driven_dfs(pool, rates, n_sims=40,
                defense_weight=.5, component_strength=.5,
                component_profiles=self.profiles(),
                component_target_season=2026, component_target_week=9)

    def test_missing_opponent_signal_remains_neutral(self):
        _, base, _ = self.simulate(0.)
        _, neutral, _ = self.simulate(.5, self.profiles(season=2026,week=9)
            .query("defense == 'AAA'"))
        np.testing.assert_array_equal(base, neutral)


if __name__ == "__main__":
    unittest.main()
