import unittest
import numpy as np
import pandas as pd
from ai2_nfl_sim_portfolio import select_simulated_portfolio

def player(name, pos, salary, team):
    return {"ID":name,"Name":name,"Position":pos,"Salary":salary,
            "TeamAbbrev":team,"opponent":"XYZ","opportunity_verified":True,"projected_targets":5,"projected_touches":12}

def roster(qb):
    return [player(qb,"QB",6500,"A"),player("r1","RB",7500,"B"),
            player("r2","RB",7200,"C"),player("w1","WR",7500,"D"),
            player("w2","WR",6000,"E"),player("cheap","WR",4500,"F"),
            player("t1","TE",4700,"G"),player("t2","TE",2900,"H"),
            player("def","DST",3100,"I")]

class ExposureTests(unittest.TestCase):
    def evaluate(self, count, premium):
        groups=[roster("q"+str(n)) for n in range(count)]
        all_players={p["ID"]:p for group in groups for p in group}
        frame=pd.DataFrame(list(all_players.values()))
        draws=np.full((120,len(frame)),12.)
        candidate=[{"players":group,"salary":49900} for group in groups]
        return select_simulated_portfolio(candidate,frame,draws,count=count,
            max_exposure=1.,upside_share=0.,premium_count=premium)
    def test_three_value_uses_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate(3,0)
    def test_two_premium_value_uses_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate(2,2)
    def test_single_lineup_retains_correlated_p90(self):
        result=self.evaluate(1,1)
        self.assertEqual(len(result),1)
        self.assertAlmostEqual(result[0]["p90_per_1000"],108./49.9,places=3)

if __name__=="__main__":
    unittest.main()
