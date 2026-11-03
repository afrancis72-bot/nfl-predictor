"""Live unified NFL QB/stack safeguards with synthetic slate fixtures."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from ai2_nfl_candidate_builder import build_classic_candidates
from ai2_nfl_sim_portfolio import select_simulated_portfolio
from ai2_nfl_qb_audit import qb_and_partners, qb_selection_report
from ai2_nfl_lineup import check_classic_lineup


def fixture():
    rows=[]
    for pos,count,salary in (("QB",3,6500),("RB",8,6500),
                             ("WR",12,5500),("TE",4,5000),("DST",4,3500)):
        for i in range(count):
            # Every QB has at least one eligible receiving teammate.
            team=("QBTEAM"+str(i) if pos=="QB" else
                  "QBTEAM"+str(i) if pos=="WR" and i<3 else
                  "DIVERSE"+pos+str(i))
            rows.append({"ID":f"{pos}{i}","Name":f"{pos}{i}",
                         "Position":pos,"TeamAbbrev":team,
                         "opponent":"OTHER"+str(i),"Salary":salary,
                         "proj":12+i*.4,"opportunity_verified":True})
    return rows


class NFLStacksRegression(unittest.TestCase):
    def test_each_eligible_qb_keeps_legal_receiver_stack_in_candidate_shortlist(self):
        rows=fixture()
        # All requested lineups are exactly $50k with a WR flex.
        candidates=build_classic_candidates(
            rows,count=24,attempts=2500,seed=19,
            salary_floor=49600,max_exposure=.35)
        self.assertGreaterEqual(len(candidates),3)
        stacks={}
        for c in candidates:
            self.assertEqual(check_classic_lineup(c["players"],salary_floor=49600),[])
            qb,partners=qb_and_partners(c["players"])
            if partners:
                stacks[str(qb["ID"])]=True
        self.assertEqual(set(stacks),{"QB0","QB1","QB2"})

    def test_no_forced_invalid_stack_and_qb_reason_in_audit(self):
        rows=fixture()
        candidates=build_classic_candidates(
            rows,count=12,attempts=2500,seed=19,
            salary_floor=49600,max_exposure=.35)
        audit=qb_selection_report(rows,candidates,[],
                                  diagnostics={"scored_qbs_after_efficiency":{},
                                               "stacked_qbs_after_efficiency":{}})
        self.assertEqual(len(audit),3)
        self.assertTrue(all(r["Stacked QB candidates"]>0 for r in audit))
        self.assertTrue(all("Stacked candidates after efficiency" in r for r in audit))

    def test_qb_stack_rescue_keeps_evidence_backed_near_floor(self):
        rows=fixture()
        qb0=next(r for r in rows if r["ID"]=="QB0")
        qb1=next(r for r in rows if r["ID"]=="QB1")
        rb=[r for r in rows if r["Position"]=="RB"][:2]
        wr=[r for r in rows if r["Position"]=="WR"]
        te=next(r for r in rows if r["Position"]=="TE")
        dst=next(r for r in rows if r["Position"]=="DST")
        one=[qb0]+rb+[wr[0],wr[3],wr[4]]+[te,wr[5],dst]
        two=[qb1]+rb+[wr[1],wr[3],wr[4]]+[te,wr[5],dst]
        candidates=[{"players":one,"salary":50000},
                    {"players":two,"salary":50000}]
        players=pd.DataFrame({"ID":[x["ID"] for x in rows]})
        # First QB slightly better overall, other QB has near-market ceiling.
        raw=np.ones((200,len(rows)),dtype=np.float32)*10
        ix={str(p["ID"]):i for i,p in enumerate(rows)}
        raw[:,ix["QB0"]]=20
        raw[:,ix["QB1"]]=19
        diag={}
        with patch("ai2_nfl_sim_portfolio.check_classic_lineup",return_value=[]):
            result=select_simulated_portfolio(
                candidates,players,raw,count=1,max_exposure=1,
                premium_count=0,require_opportunity=False,
                p90_efficiency_retention=.995,diagnostics=diag)
        self.assertEqual(len(result),1)
        self.assertGreaterEqual(diag["scored_qbs_after_efficiency"].get("QB1",0),1)
        self.assertGreaterEqual(diag["stacked_qbs_after_efficiency"].get("QB1",0),1)
        self.assertEqual(diag["final_qb_max_allowed"],1)
        self.assertEqual(diag["final_stack_count"],1)
        self.assertLessEqual(max(diag["final_qb_counts"].values()),1)


if __name__=="__main__":
    unittest.main()
