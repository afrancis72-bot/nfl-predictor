"""Paired stat diagnostics cannot quietly select different historical players."""
import unittest

import pandas as pd

from scripts.audit_nfl_opportunity_forecasts import (
    compare_stat_rows, paired_stat_summary, _strict_matched_player_games,
)


def make():
    return pd.DataFrame([
        dict(season=2026,week=7,game="AAA@BBB",player="A RB",
             team="AAA",position="RB",stat="carries",category="opportunity",
             model="neutral",actual=15.,prior_baseline=10.,predicted_mean=12.,
             predicted_p10=5.,predicted_p90=24.,n_sims=200),
        dict(season=2026,week=7,game="AAA@BBB",player="B WR",
             team="BBB",position="WR",stat="targets",category="opportunity",
             model="neutral",actual=6.,prior_baseline=11.,predicted_mean=10.,
             predicted_p10=4.,predicted_p90=15.,n_sims=200),
    ])


class OpportunityComparisonTests(unittest.TestCase):
    def test_paired_scores_calculate_lower_true_error(self):
        first=make()
        second=first.copy()
        second["model"]="possession"
        second["predicted_mean"]=[15.,7.]
        merged=compare_stat_rows(first,second)
        x=paired_stat_summary(merged)
        self.assertAlmostEqual(x["neutral_mae"],3.5)
        self.assertAlmostEqual(x["possession_mae"],.5)
        self.assertEqual(x["n"],2)

    def test_missing_player_or_stat_does_not_improve_score(self):
        with self.assertRaisesRegex(ValueError,"stat pools differ"):
            compare_stat_rows(make(),make().iloc[:1])

    def test_changed_actual_does_not_pass(self):
        second=make()
        second.loc[0,"actual"]=99
        with self.assertRaisesRegex(ValueError,"different football truth"):
            compare_stat_rows(make(),second)

    def test_duplicate_player_stat_not_silently_counted(self):
        copied=pd.concat([make(),make().iloc[:1]],ignore_index=True)
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            compare_stat_rows(copied,make())

    def test_fantasy_score_cohort_and_stat_cohort_match(self):
        stats=make()
        scores=stats[["season","week","game","player","team","position"]]
        _strict_matched_player_games(scores,stats)
        with self.assertRaisesRegex(ValueError,"Missing player-level"):
            _strict_matched_player_games(scores,
                                         stats.iloc[:1])


if __name__=="__main__":
    unittest.main()
