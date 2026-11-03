"""Predicted football props must translate to correlated DK distributions."""
import unittest
import numpy as np
import pandas as pd
from ai2_nfl_prop_intelligence import player_prop_decisions


class StatFirstPropRegression(unittest.TestCase):
    def test_model_prop_projection_and_market_polarity(self):
        n=7000
        # Model 6 rushing yards per carry; ten carries each simulation
        rushing=np.arange(n,dtype=float)%140
        points=0.1*rushing
        players=pd.DataFrame([{"ID":"rb","Name":"Sample Runner","Position":"RB",
                                "TeamAbbrev":"LAR","Salary":5100}])
        stats={"Sample Runner":{
            "carries":np.full(n,10.0),
            "rushing_yards":rushing,
            "targets":np.full(n,2.0),
            "receptions":np.full(n,2.0),
            "receiving_yards":np.zeros(n),
            "rushing_tds":np.zeros(n),
            "receiving_tds":np.zeros(n)}}
        result=player_prop_decisions(
            players,stats,points.reshape(-1,1),
            pregame_role={"rb":{"Role gate":"PASS",
                                "Role rationale":"consistent prior-game rushes"}},
            market_lines={("rb","rushing_yards"):{"line":55.5,"over_american_odds":-110}})
        self.assertEqual(len(result),1)
        row=result[0]
        self.assertEqual(row["Historical role validation"],"PASS")
        self.assertEqual(row["Rushing yards projection"],69.5)
        self.assertEqual(row["Rushing yards market line"],55.5)
        self.assertGreater(row["Rushing yards model over %"],50)
        self.assertEqual(row["Model mean DK"],round(float(points.mean()),2))
        self.assertIn("estimated unit EV",row)

    def test_no_market_line_means_no_real_over_claim(self):
        n=7000
        players=pd.DataFrame([{"ID":"dst","Name":"Defense","Position":"DST",
                                "TeamAbbrev":"ARI","Salary":3400}])
        out=player_prop_decisions(players,{},np.full((n,1),7.0))
        self.assertEqual(out[0]["Market comparison"],"NO VERIFIED SPORTSBOOK LINE")
        self.assertEqual(out[0]["Model mean DK"],7.0)
        self.assertEqual(out[0]["Stat evidence"],
                         "Unavailable / ambiguous - no stat-specific confidence")

    def test_heldout_worlds_do_not_change_prop_projection(self):
        n=7000
        players=pd.DataFrame([{"ID":"qb","Name":"Sample QB",
                                "Position":"QB","TeamAbbrev":"DAL",
                                "Salary":6200}])
        base=np.full(10000,230.0)
        base[7000:]=5000.0
        data={"Sample QB":{"passing_yards":base,
                             "passing_tds":np.ones(10000),
                             "interceptions":np.zeros(10000),
                             "rushing_yards":np.zeros(10000),
                             "rushing_tds":np.zeros(10000)}}
        out=player_prop_decisions(players,data,np.full((n,1),13.2))
        self.assertEqual(out[0]["Passing yards projection"],230.0)
        self.assertEqual(out[0]["Model mean DK"],13.2)

    def test_invalid_market_line_fails_closed(self):
        n=100
        players=pd.DataFrame([{"ID":"qb","Name":"QB",
                                "Position":"QB","TeamAbbrev":"DET","Salary":6000}])
        data={"QB":{"passing_yards":np.full(n,240.)}}
        with self.assertRaisesRegex(ValueError,"Invalid supplied market prop line"):
            player_prop_decisions(
                players,data,np.ones((n,1)),market_lines={
                    ("qb","passing_yards"):{"line":-1}})


if __name__=="__main__":
    unittest.main()
