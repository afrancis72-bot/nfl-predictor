import unittest
import pandas as pd
import numpy as np
from snap_level_research import simulate_snaps, summarize_player_props

def sample_players():
    rows=[]
    for team in ("SEA","DEN"):
        for pos,name,salary in [("QB","QB",10000),("RB","RB",7000),("WR","WR1",9000),("WR","WR2",4500),("TE","TE",4000),("K","K",5000),("DST","DST",4200)]:
            rows.append({"Name":team+" "+name,"TeamAbbrev":team,"Position":pos,"FLEX Salary":salary})
    return pd.DataFrame(rows)

class SnapResearchTests(unittest.TestCase):
    def test_reproducible_shapes_and_statistics(self):
        p=sample_players()
        a,x=simulate_snaps(p,["SEA","DEN"],n_games=12,seed=7)
        b,y=simulate_snaps(p,["SEA","DEN"],n_games=12,seed=7)
        self.assertEqual(a.shape,(12,len(p)))
        self.assertTrue(np.array_equal(a,b))
        self.assertTrue((x["targets"]>=x["receptions"]).all())
        self.assertTrue((x["pass_att"]>=x["pass_cmp"]).all())
        self.assertTrue(np.isfinite(a).all())
        self.assertFalse(summarize_player_props(x).empty)
    def test_requires_confirmed_qb(self):
        p=sample_players()
        p=p[p["Name"]!="DEN QB"]
        with self.assertRaises(ValueError):
            simulate_snaps(p,["SEA","DEN"],n_games=2)

if __name__=="__main__":
    unittest.main()
