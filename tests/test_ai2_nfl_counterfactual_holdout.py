"""No favored QB, no fabricated stack, and no evaluation-data leakage.

All fixtures synthetic. Real NFL calibration requires frozen pregame predictions
and completed games, and is NOT established by these unit/regression checks.
"""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from ai2_nfl_qb_conditional import anchored_qb_stack
from ai2_nfl_qb_counterfactual import quarterback_counterfactuals
from ai2_nfl_qb_audit import qb_and_partners
from ai2_nfl_lineup import check_classic_lineup
from ai2_nfl_sim_portfolio import select_simulated_portfolio
from ai2_nfl_projection_accountability import uncertain_role_exposures


def player(ident,pos,team,salary,proj=12):
    return {"ID":ident,"Name":ident,"Position":pos,"TeamAbbrev":team,
            "opponent":"OPP","Salary":salary,"proj":proj,
            "opportunity_verified":True}


class NFLAccountabilityTests(unittest.TestCase):
    def test_solver_finds_actual_legal_genuine_qb_stacks_at_exact_salary(self):
        pools={"QB":[player("QB_A","QB","A",6500,20),
                     player("QB_B","QB","B",6500,20)],
               "RB":[player(f"RB{i}","RB",f"RBTEAM{i}",6100,15) for i in range(3)],
               "WR":[player("WR_A","WR","A",6300,15),
                     player("WR_B","WR","B",6300,15),
                     player("WR_C","WR","C",6300,15),
                     player("WR_A2","WR","A",4400,9),
                     player("WR_B2","WR","B",4400,9)],
               "TE":[player("TE1","TE","T",4700,11)],
               "DST":[player("DST1","DST","DEF",3300,8)]}
        for qb in pools["QB"]:
            one=anchored_qb_stack(pools,qb,min_receivers=1)
            two=anchored_qb_stack(pools,qb,min_receivers=2)
            for lineup in (one,two):
                self.assertIsNotNone(lineup)
                self.assertEqual(check_classic_lineup(lineup,salary_floor=49600),[])
                self.assertEqual(sum(int(p["Salary"]) for p in lineup),50000)
                self.assertEqual(qb_and_partners(lineup)[0]["ID"],qb["ID"])
            self.assertEqual(len(qb_and_partners(two)[1]),2)
        pools["WR"]=[p for p in pools["WR"] if p["TeamAbbrev"]!="A"]
        self.assertIsNone(anchored_qb_stack(pools,pools["QB"][0]))

    def test_counterfactual_is_a_real_lineup_not_sum_of_individual_percentiles(self):
        def roster(qbid,team):
            return [player(qbid,"QB",team,6000),
                    player("WR"+team,"WR",team,5500),
                    player("RB1","RB","R1",6000),player("RB2","RB","R2",6000),
                    player("WR3","WR","X",5500),player("WR4","WR","Y",5500),
                    player("TE1","TE","Z",4500),player("WR5","WR","Q",5000),
                    player("DST1","DST","D",3000)]
        a={"candidate":{"players":roster("QB_A","A"),"salary":47000},
           "simulated_mean":135,"simulated_p90":171,"simulated_p99":202}
        b={"candidate":{"players":roster("QB_B","B"),"salary":47000},
           "simulated_mean":150,"simulated_p90":180,"simulated_p99":215}
        names=sorted({p["ID"] for r in (a,b) for p in r["candidate"]["players"]})
        vals=np.full((500,len(names)),10.,dtype=float)
        lookup={x:i for i,x in enumerate(names)}
        vals[:,lookup["QB_A"]]=31
        vals[:,lookup["QB_B"]]=12
        details,summary=quarterback_counterfactuals(
            [a,b],[b],names,vals)
        self.assertEqual(details["QB_A"]["Counterfactual stacked candidates"],1)
        self.assertEqual(details["QB_A"]["Counterfactual status"],
                         "Not selected; compare documented lineup values")
        self.assertGreater(details["QB_A"]["Best legal stack holdout mean"],
                           details["QB_B"]["Best legal stack holdout mean"])
        self.assertEqual(summary["holdout_worlds"],500)

    def test_selection_uses_train_only_and_holdout_cannot_reverse_the_pick(self):
        positions=["QB","RB","RB","WR","WR","WR","TE","WR","DST"]
        roster1=[player(str(i),pos,"A" if pos in ("QB","WR") else f"X{i}",5500)
                 for i,pos in enumerate(positions)]
        roster2=[p.copy() for p in roster1]
        roster2[0]=player("otherQB","QB","B",5500)
        roster2[3]=player("otherWR","WR","B",5500)
        all_players={str(p["ID"]):p for row in (roster1,roster2) for p in row}
        names=list(all_players)
        idx={x:i for i,x in enumerate(names)}
        training=np.full((300,len(names)),10.)
        heldout=np.full((200,len(names)),10.)
        training[:,idx["0"]]=25
        training[:,idx["otherQB"]]=10
        heldout[:,idx["0"]]=1
        heldout[:,idx["otherQB"]]=40
        notes={}
        candidates=[{"players":roster1,"salary":49500},
                    {"players":roster2,"salary":49500}]
        with patch("ai2_nfl_sim_portfolio.check_classic_lineup",return_value=[]):
            selected=select_simulated_portfolio(
                candidates,pd.DataFrame({"ID":names}),training,
                count=1,max_exposure=1.,premium_count=0,
                require_opportunity=False,validation_draws=heldout,
                diagnostics=notes)
        self.assertEqual(selected[0]["candidate"]["players"][0]["ID"],"0")
        self.assertEqual(notes["independent_scenario_audit"]["holdout_worlds"],200)
        self.assertGreater(
            notes["qb_counterfactuals"]["otherQB"]["Best legal stack holdout mean"],
            notes["qb_counterfactuals"]["0"]["Best legal stack holdout mean"])

    def test_exposes_uncertain_role_by_actual_portfolio_exposure(self):
        roster=[player("rb","RB","R1",6500),player("qb","QB","A",6500)]
        ranked=[{"candidate":{"players":roster}} for _ in range(5)]
        flags=[{"Player":"rb","Team":"R1","Concern":"Carries significantly above observed role",
                "Increase %":120.,"Prior player-games":8}]
        result=uncertain_role_exposures(flags,ranked)
        self.assertEqual(result[0]["Lineups using flagged player"],5)
        self.assertEqual(result[0]["Portfolio exposure %"],100.)
        self.assertTrue(result[0]["Review priority"].startswith("HIGH"))
        self.assertEqual(uncertain_role_exposures([],ranked),[])


if __name__=="__main__":
    unittest.main()
