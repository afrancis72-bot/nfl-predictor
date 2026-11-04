"""Unmodeled real teammates keep DFS-projection workload conservative."""
import unittest
import pandas as pd
from dfs_stats_engine import _unmodeled_skill_opportunities


class OtherTeammateRoleTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame([
            {"player":"Starter Back","team":"NYG","position":"RB",
             "carries_pg":19.0,"targets_pg":4.0,
             "latest_observed_season":2026,"latest_observed_week":7},
            {"player":"Backup Back","team":"NYG","position":"RB",
             "carries_pg":8.0,"targets_pg":2.0,
             "latest_observed_season":2026,"latest_observed_week":7},
            {"player":"Third Receiver","team":"NYG","position":"WR",
             "carries_pg":0.0,"targets_pg":4.1,
             "latest_observed_season":2026,"latest_observed_week":7},
            {"player":"Stale Runner","team":"NYG","position":"RB",
             "carries_pg":30.0,"targets_pg":6.0,
             "latest_observed_season":2026,"latest_observed_week":1},
            {"player":"Injured RB","team":"NYG","position":"RB",
             "carries_pg":13.0,"targets_pg":3.0,
             "latest_observed_season":2026,"latest_observed_week":7},
            {"player":"Opponent","team":"DAL","position":"RB",
             "carries_pg":20.0,"targets_pg":5.0,
             "latest_observed_season":2026,"latest_observed_week":7},
        ])

    def test_recent_nonrosterable_teammates_not_transferred_to_starter(self):
        carry,target,names=_unmodeled_skill_opportunities(
            self.sample(),"NYG",["Starter Back"],{("injured rb","NYG")})
        self.assertAlmostEqual(carry,8.0)
        self.assertAlmostEqual(target,6.1)
        self.assertEqual({v["player"] for v in names},{"Backup Back","Third Receiver"})

    def test_active_backup_is_not_other(self):
        carry,target,names=_unmodeled_skill_opportunities(
            self.sample(),"NYG",
            ["Starter Back","Backup Back","Third Receiver"],{("injured rb","NYG")})
        self.assertEqual((carry,target,names),(0.0,0.0,[]))


if __name__=="__main__":
    unittest.main()
