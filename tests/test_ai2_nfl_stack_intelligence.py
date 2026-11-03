"""Same-world QB and receiver stack co-ceiling test."""
import unittest
import numpy as np
from ai2_nfl_stack_intelligence import score_stack_lift


class StackLiftRegression(unittest.TestCase):
    def test_positive_correlation_gets_higher_lift(self):
        x=np.linspace(0,99,100)
        roster=[{"ID":"qb","Position":"QB","TeamAbbrev":"DET"},
                {"ID":"wr","Position":"WR","TeamAbbrev":"DET"}]
        correlated=np.stack([x,x],axis=1)
        decorrelated=np.stack([x,np.roll(x,37)],axis=1)
        index={"qb":0,"wr":1}
        good=score_stack_lift(roster,index,correlated)
        other=score_stack_lift(roster,index,decorrelated)
        self.assertGreater(good,other)
        self.assertGreater(good,.8)

    def test_no_receiver_no_bonus(self):
        roster=[{"ID":"qb","Position":"QB","TeamAbbrev":"DET"},
                {"ID":"wr","Position":"WR","TeamAbbrev":"SEA"}]
        self.assertEqual(score_stack_lift(roster,{"qb":0,"wr":1},
                                        np.ones((100,2))),0.)


if __name__=="__main__":
    unittest.main()
