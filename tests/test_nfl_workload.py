import unittest
import pandas as pd
from ai2_nfl_workload import redistribute_opportunities

class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.rates = pd.DataFrame([
            {"player":"Starter","team":"NYJ","position":"RB","carries_pg":18.0,"targets_pg":4.0},
            {"player":"Backup","team":"NYJ","position":"RB","carries_pg":6.0,"targets_pg":1.0},
            {"player":"Other","team":"NYJ","position":"RB","carries_pg":2.0,"targets_pg":0.0},
            {"player":"Receiver","team":"NYJ","position":"WR","carries_pg":0.0,"targets_pg":7.0},
            {"player":"Elsewhere","team":"BUF","position":"RB","carries_pg":8.0,"targets_pg":2.0},
        ])
    def test_confirmed_out_raises_teammate_workload_with_cap(self):
        adjusted, audit = redistribute_opportunities(self.rates, {("Starter","NYJ"):"OUT"})
        backup = adjusted.loc[adjusted.player.eq("Backup")].iloc[0]
        self.assertGreater(backup.carries_pg, 6.0)
        self.assertLessEqual(backup.carries_pg, 9.6 + 1e-8)
        self.assertEqual(float(adjusted.loc[adjusted.player.eq("Elsewhere"),"carries_pg"].iloc[0]), 8.0)
        self.assertGreater(len(audit), 0)
    def test_questionable_does_not_redistribute(self):
        adjusted, audit = redistribute_opportunities(self.rates, {("Starter","NYJ"):"QUESTIONABLE"})
        self.assertEqual(len(audit), 0)
        self.assertEqual(float(adjusted.iloc[1].carries_pg), 6.0)
    def test_no_evidence_no_fabricated_role(self):
        r = self.rates.copy()
        r.loc[1,"carries_pg"] = 0.0
        adjusted, audit = redistribute_opportunities(r, {("Starter","NYJ"):"OUT"})
        self.assertEqual(float(adjusted.iloc[1].carries_pg), 0.0)

if __name__ == "__main__":
    unittest.main()
