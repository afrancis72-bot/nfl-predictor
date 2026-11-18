"""Separate, independent scenario-by-scenario checks of football accounting."""
import unittest
from copy import deepcopy

import numpy as np

from ai2_nfl_game_conservation import audit_team_game_conservation
from dfs_stats_engine import simulate_stat_driven_dfs
from test_ai2_nfl_team_pace_scoring import sim_fixture


class NFLGameConservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool,cls.rates=sim_fixture()
        _,cls.scores,cls.stats=simulate_stat_driven_dfs(
            cls.pool,cls.rates,n_sims=300,seed=2401,allow_role_proxies=False)

    def test_both_teams_conserve_each_scenario_and_ledger_is_explicit(self):
        for team in ("AAA","BBB"):
            a=audit_team_game_conservation(self.stats,self.pool,team)
            self.assertEqual(a["worlds_verified"],300)
            self.assertEqual(self.stats[f"__TEAM_CONSERVATION__|{team}"],a)
            self.assertTrue(a["conserved_pass_attempt_slots"])
            self.assertFalse(a["unmodeled_receiver_targets_separately_identified"])
            self.assertFalse(a["no_target_passes_separately_identified"])
            self.assertGreater(a["mean_other_or_no_target_attempt_slots"],0)
            self.assertIn("no-target",a["audit_warning"])

    def test_fails_closed_on_missing_attempt_slot(self):
        corrupt=deepcopy(self.stats)
        corrupt["__OTHER_RECEIVER__|AAA"]["targets"][0]+=1
        with self.assertRaisesRegex(ValueError,"pass attempt slots"):
            audit_team_game_conservation(corrupt,self.pool,"AAA")

    def test_fails_closed_on_touchdown_without_catch(self):
        corrupt=deepcopy(self.stats)
        other=corrupt["__OTHER_RECEIVER__|BBB"]
        other["receiving_tds"][0]=other["receptions"][0]+1
        with self.assertRaisesRegex(ValueError,"passing TDs|receiver exceeds"):
            audit_team_game_conservation(corrupt,self.pool,"BBB")

    def test_fails_closed_on_carry_outside_team_budget(self):
        corrupt=deepcopy(self.stats)
        corrupt["__OTHER_RUSHER__|AAA"]["carries"][0]+=1
        with self.assertRaisesRegex(ValueError,"rush attempts"):
            audit_team_game_conservation(corrupt,self.pool,"AAA")

    def test_fails_closed_when_scoring_not_shared(self):
        corrupt=deepcopy(self.stats)
        corrupt["__TEAM_SCORING__|AAA"]["simulated_offensive_points"][0]+=7
        with self.assertRaisesRegex(ValueError,"offensive points"):
            audit_team_game_conservation(corrupt,self.pool,"AAA")

    def test_research_result_not_mistaken_for_prediction_validation(self):
        report=self.stats["__TEAM_CONSERVATION__|AAA"]
        self.assertTrue(report["research_only"])
        self.assertFalse(report["predictive_accuracy_validated"])
        self.assertTrue(np.isfinite(self.scores).all())


if __name__=="__main__":
    unittest.main()
