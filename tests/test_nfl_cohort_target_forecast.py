"""Fail-closed WR teammate co-observation forecasting (no future results)."""
import unittest

import numpy as np
import pandas as pd

from ai2_nfl_cohort_target_forecast import conditional_wr_targets
from props_data import make_rates


def scenario():
    rows=[]
    for week in range(1,10):
        if week<=8:
            rows.append(dict(season=2026,week=week,season_type="REG",
                             player_name="Receiver A",recent_team="AAA",
                             position="WR",targets=6 if week%2==0 else 12))
            if week%2==0:
                rows.append(dict(season=2026,week=week,season_type="REG",
                                 player_name="Receiver B",recent_team="AAA",
                                 position="WR",targets=10))
            rows.append(dict(season=2026,week=week,season_type="REG",
                             player_name="Receiver C",recent_team="AAA",
                             position="WR",targets=4))
        else:
            # HINDSIGHT only, never usable for a week-9 forecast.
            rows.append(dict(season=2026,week=week,season_type="REG",
                             player_name="Receiver A",recent_team="AAA",
                             position="WR",targets=95))
    history=pd.DataFrame(rows)
    rates=make_rates(history,2026,9)
    roster=pd.DataFrame([
        dict(Name=name,TeamAbbrev="AAA",Position="WR")
        for name in ("Receiver A","Receiver B","Receiver C")
    ])
    return history,rates,roster


class CohortForecastTests(unittest.TestCase):
    def setUp(self):
        self.history,self.rates,self.roster=scenario()

    def test_requires_explicit_research_only_execution(self):
        with self.assertRaisesRegex(ValueError,"retrospective research only"):
            conditional_wr_targets(self.history,self.rates,self.roster,
                                   season=2026,week=9,strength=.5)

    def test_same_team_coobservation_changes_real_player_target_rate(self):
        revised,ledger=conditional_wr_targets(
            self.history,self.rates,self.roster,
            season=2026,week=9,strength=.5,research_proxy=True)
        a=self.rates[self.rates.player.eq("Receiver A")].iloc[0]
        b=revised[revised.player.eq("Receiver A")].iloc[0]
        self.assertLess(b.targets_pg,a.targets_pg)
        self.assertGreaterEqual(b.targets_pg,.8*a.targets_pg)
        self.assertLessEqual(b.targets_pg,1.2*a.targets_pg)
        change=[x for x in ledger if x["player"]=="Receiver A"][0]
        self.assertEqual(change["co_observed_games"],4)
        self.assertEqual(change["not_co_observed_games"],4)
        self.assertFalse(change["verified_game_day_active"])
        self.assertTrue(change["anchor_is_pregame_pool_proxy_not_certified_active"])
        self.assertEqual(change["target_week"],9)
        np.testing.assert_allclose(
            self.rates["carries_pg"].fillna(0),
            revised["carries_pg"].fillna(0))
        self.assertEqual(set(self.rates.player),set(revised.player))

    def test_future_boxscore_has_no_influence(self):
        a,la=conditional_wr_targets(
            self.history,self.rates,self.roster,season=2026,week=9,
            strength=.5,research_proxy=True)
        truncated=self.history[self.history.week.lt(9)].copy()
        b,lb=conditional_wr_targets(
            truncated,self.rates,self.roster,season=2026,week=9,
            strength=.5,research_proxy=True)
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(la,lb)

    def test_missing_teammate_contrast_keeps_rates_unchanged(self):
        absent=self.roster[self.roster.Name.ne("Receiver B")]
        # A and C are both in every prior team week; neither has an
        # eligible non-co-observed contrast, so no uplift is asserted.
        revised,ledger=conditional_wr_targets(
            self.history,self.rates,absent,season=2026,week=9,
            strength=1.,research_proxy=True)
        self.assertFalse(ledger)
        pd.testing.assert_frame_equal(self.rates,revised)

    def test_duplicate_same_week_identity_fails_closed(self):
        dup=pd.concat([self.history,self.history.iloc[[0]]],ignore_index=True)
        with self.assertRaisesRegex(ValueError,"Duplicate historical"):
            conditional_wr_targets(dup,self.rates,self.roster,
                                   season=2026,week=9,strength=.5,
                                   research_proxy=True)

    def test_neutral_strength_does_not_modify_scoring_data(self):
        unchanged,ledger=conditional_wr_targets(
            self.history,self.rates,self.roster,season=2026,week=9,
            strength=0.,research_proxy=False)
        self.assertFalse(ledger)
        pd.testing.assert_frame_equal(self.rates,unchanged)


if __name__=="__main__":
    unittest.main()
