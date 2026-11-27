import unittest
import numpy as np
import pandas as pd
from ai2_nfl_draft_matchup import review_strength, paired_projection_rows

class DraftMatchupTests(unittest.TestCase):
    def test_strength(self):
        self.assertEqual(review_strength(.25,True),.25)
        self.assertEqual(review_strength(0,False),0)
        with self.assertRaises(ValueError): review_strength(.25,False)
        with self.assertRaises(ValueError): review_strength(1,True)
    def test_pair_by_id(self):
        p=pd.DataFrame([dict(ID="a",Name="A",TeamAbbrev="X",Position="RB"),dict(ID="b",Name="B",TeamAbbrev="Y",Position="WR")])
        draws=np.tile([12.,9.],(10000,1))
        neutral=np.tile([8.,10.],(10000,1))
        rows=paired_projection_rows(p,draws,p.iloc[::-1],neutral,seed=47,season=2026,week=12)
        self.assertEqual([r["delta_mean"] for r in rows],[2.,1.])
        with self.assertRaises(ValueError): paired_projection_rows(p,draws,p.iloc[:1],neutral,seed=47,season=2026,week=12)
