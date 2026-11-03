"""PGA joint-world optimization and portfolio audit, using synthetic scenarios.

Synthetic worlds are designed to identify objective errors, not to infer actual
tournament cut probabilities or claim a profitable betting strategy.
"""
import unittest
import numpy as np
import pandas as pd

from ai2_pga_portfolio_scenarios import (
    audit_portfolio_worlds, choose_joint_scenario_portfolio,
)
from portfolio_optimizer import PortfolioSettings, optimize_portfolio


class PGAScenarioPortfolioTests(unittest.TestCase):
    @staticmethod
    def crafted_worlds():
        names=[f"Golfer {j:02d}" for j in range(18)]
        draws=np.full((100,18),50.,dtype=np.float32)
        draws[:,0:6]=80.    # A: strong dependable scorer
        draws[:,6:12]=78.   # B: complementary 6/6 outcomes
        draws[90:,12:18]=150.  # C: rare exceptional upside, usually weak
        cuts=np.zeros((100,18),dtype=bool)
        cuts[:55,0:6]=True
        cuts[45:,6:12]=True
        cuts[:55,12:18]=True
        candidates=[
            (0.,48000,frozenset(names[:6])),
            (0.,48000,frozenset(names[6:12])),
            (0.,48000,frozenset(names[12:18])),
        ]
        return names,draws,cuts,candidates

    def test_portfolio_union_cut_is_not_sum_of_lineup_probabilities(self):
        names,draws,cuts,ranked=self.crafted_worlds()
        portfolio=pd.DataFrame([{"lineup":i+1,"player":p} for i,r in
                                enumerate(ranked[:2]) for p in sorted(r[2])])
        lines,overall=audit_portfolio_worlds(
            portfolio,pd.DataFrame({"player":names}),draws,cuts,
            strategy="Cut Equity")
        self.assertEqual(list(lines.sim_lineup_6of6_pct),[55.,55.])
        self.assertAlmostEqual(overall["sim_portfolio_at_least_one_6of6_pct"],100.)
        self.assertAlmostEqual(overall["sim_portfolio_expected_6of6_count"],1.1)
        self.assertEqual(overall["lineup_count"],2)
        self.assertAlmostEqual(overall["sim_portfolio_best_score_mean"],480.)
        self.assertEqual(overall["sim_portfolio_best_score_p99"],480.)

    def test_cut_coverage_and_gpp_tail_choose_different_legal_entries(self):
        names,draws,cuts,ranked=self.crafted_worlds()
        kwargs=dict(ranked=ranked,players=names,draws=draws,cuts=cuts,
                    lineup_count=2,max_appearances=1,min_unique=3,
                    roster_size=6,seed=43)
        cut=choose_joint_scenario_portfolio(strategy="Cut Equity",**kwargs)
        gpp=choose_joint_scenario_portfolio(strategy="GPP Ceiling",**kwargs)
        self.assertEqual(set(cut),{ranked[0][2],ranked[1][2]})
        self.assertIn(ranked[2][2],set(gpp))
        self.assertNotEqual(set(cut),set(gpp))
        def audit(picks):
            frame=pd.DataFrame([{"lineup":i+1,"player":p} for i,r in
                                enumerate(picks) for p in sorted(r)])
            return audit_portfolio_worlds(
                frame,pd.DataFrame({"player":names}),draws,cuts)[1]
        cu,gu=audit(cut),audit(gpp)
        self.assertGreater(cu["sim_portfolio_at_least_one_6of6_pct"],
                           gu["sim_portfolio_at_least_one_6of6_pct"])
        self.assertGreater(gu["sim_portfolio_best_score_p99"],
                           cu["sim_portfolio_best_score_p99"])

    def test_reject_misaligned_or_non_boolean_cut_worlds(self):
        names,draws,cuts,ranked=self.crafted_worlds()
        portfolio=pd.DataFrame([{"lineup":1,"player":p} for p in names[:6]])
        frame=pd.DataFrame({"player":names})
        with self.assertRaises(ValueError):
            audit_portfolio_worlds(portfolio,frame,draws,cuts[:,1:])
        with self.assertRaises(ValueError):
            audit_portfolio_worlds(portfolio,frame,draws,cuts.astype(int))
        with self.assertRaises(ValueError):
            audit_portfolio_worlds(portfolio,frame.assign(player=names[:17]+[names[0]]),draws,cuts)

    def test_realistic_eleven_lineup_portfolio_keeps_hard_guards(self):
        rng=np.random.default_rng(41)
        n=34
        players=[f"PGA Golfer {i:02d}" for i in range(n)]
        raw=rng.normal(loc=75,scale=23,size=(200,n)).astype(np.float32)
        cuts=(rng.random((200,n))<np.linspace(.55,.87,n)[None,:])
        mu=raw.mean(axis=0)
        p75=np.quantile(raw,.75,axis=0)
        p90=np.quantile(raw,.90,axis=0)
        salary=np.array([7600+(i%9)*150 for i in range(n)])
        predictions=pd.DataFrame({
            "player":players,
            "salary":salary,
            "dk_points_proxy":mu,
            "dk_value_per_1000":mu/(salary/1000),
            "win_pct":np.linspace(.05,.01,n),
            "top10_pct":np.linspace(.3,.1,n),
            "make_cut_pct":cuts.mean(axis=0),
            "v11_mean_dk":mu,
            "v11_p75_dk":p75,
            "v11_p90_dk":p90,
        })
        settings=PortfolioSettings(
            lineup_count=11,max_exposure=.35,min_unique=3,
            min_player_salary=3000,salary_floor=45000,salary_cap=50000,
            candidate_pool_size=n,candidate_samples=12000,
            target_candidates=1100,strategy="Cut Equity",seed=17)
        original=predictions.copy(deep=True)
        lineup,summary,expo=optimize_portfolio(
            predictions,settings,scenario_scores=raw,scenario_cuts=cuts)
        self.assertEqual(len(summary),11)
        self.assertEqual(len(lineup),66)
        self.assertLessEqual(expo.lineups.max(),3)
        self.assertTrue(summary.salary.between(45000,50000).all())
        sets=[set(g.player) for _,g in lineup.groupby("lineup")]
        self.assertTrue(all(len(x&y)<=3 for i,x in enumerate(sets)
                            for y in sets[i+1:]))
        self.assertGreaterEqual(len(lineup.player.unique()),22)
        _,summary2=audit_portfolio_worlds(lineup,predictions,raw,cuts,
                                          strategy="Cut Equity")
        self.assertTrue(0<=summary2["sim_portfolio_at_least_one_6of6_pct"]<=100)
        pd.testing.assert_frame_equal(predictions,original)
        self.assertTrue(pd.DataFrame([summary2]).to_csv(index=False))


if __name__=="__main__":
    unittest.main()
