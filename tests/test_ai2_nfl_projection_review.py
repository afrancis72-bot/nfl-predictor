"""Review-only projection inflation flags use the SAME post-simulation player audit."""
import unittest
import pandas as pd
from ai2_nfl_projection_review import projection_review_flags


class ProjectionReviewTests(unittest.TestCase):
    def test_flags_high_mean_puka_bijan_but_not_good_midrange(self):
        pool = pd.DataFrame([
            {"ID":"1","Name":"Puka Nacua","proj":20.2},
            {"ID":"2","Name":"Bijan Robinson","proj":19.0},
            {"ID":"3","Name":"Ordinary Player","proj":13.0}
        ])
        audits=[
            {"Player":"Puka Nacua","Team":"LAR","Position":"WR","DraftKings ID":"1",
             "Sim mean DK":33.27,"Audit status":"Verified: per-simulation DK points match components"},
            {"Player":"Bijan Robinson","Team":"ATL","Position":"RB","DraftKings ID":"2",
             "Sim mean DK":29.44,"Audit status":"Verified: per-simulation DK points match components"},
            {"Player":"Ordinary Player","Team":"ATL","Position":"WR","DraftKings ID":"3",
             "Sim mean DK":16.0,"Audit status":"Verified: per-simulation DK points match components"}
        ]
        flags=projection_review_flags(audits,pool)
        self.assertEqual({r["Player"] for r in flags},{"Puka Nacua","Bijan Robinson"})
        self.assertEqual(pool.loc[0,"proj"],20.2)
    def test_does_not_certify_unverifiable_simulation(self):
        pool=pd.DataFrame([{"ID":"1","Name":"Bad","proj":3.}])
        self.assertEqual(projection_review_flags(
            [{"Player":"Bad","DraftKings ID":"1","Sim mean DK":41.,
              "Audit status":"Simulation missing component details"}],pool),[])


if __name__=="__main__":
    unittest.main()
