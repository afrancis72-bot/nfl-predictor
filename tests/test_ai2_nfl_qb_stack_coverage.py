"""Regress QB receiver stacking on the real unified NFL pipeline.

The fixtures are *synthetic*: they prove selection code properties, not that
Goff, Stafford or Prescott should be used in any particular future slate.
"""
from collections import Counter
import unittest
import numpy as np
import pandas as pd

from ai2_nfl_candidate_builder import build_classic_candidates
from ai2_nfl_lineup import check_classic_lineup
from ai2_nfl_qb_audit import qb_and_partners, qb_selection_report
from ai2_nfl_sim_portfolio import select_simulated_portfolio


def player(ident,pos,team,salary,proj=12):
    return {"ID":str(ident),"Name":str(ident),"Position":pos,
            "TeamAbbrev":team,"opponent":"OPP",
            "Salary":salary,"proj":float(proj),
            "projection_source":"observed synthetic fixture",
            "opportunity_verified":True}


def build_fixture():
    players=[]
    for team,proj in (("A",20),("B",17),("C",15)):
        players.append(player("QB"+team,"QB",team,7100,proj))
    players.extend(player("RB"+str(i),"RB","R"+str(i),6500,13)
                   for i in range(10))
    for team in ("A","B","C","OTHER"):
        players.extend(player("WR"+team+str(i),"WR",team,5500,
                              18 if team=="A" else 15 if team=="B" else 12)
                       for i in range(4))
    players.extend(player("TE"+str(i),"TE","A" if i==0 else "C" if i==1
                          else "OTHER",4500,10) for i in range(4))
    players.extend(player("DST"+str(i),"DST","D"+str(i),3000,6)
                   for i in range(4))
    return players


class QBStackLiveTests(unittest.TestCase):
    def test_guided_candidates_retain_legal_stacks_for_all_qbs(self):
        data=build_fixture()
        out=build_classic_candidates(data,count=60,attempts=4000,
                                     seed=43,max_exposure=.35,
                                     salary_floor=49600)
        self.assertEqual(len(out),60)
        qbs=Counter()
        stacked=Counter()
        for item in out:
            lineup=item["players"]
            self.assertFalse(check_classic_lineup(lineup,salary_floor=49600))
            qb,partners=qb_and_partners(lineup)
            qbs[str(qb["ID"])]+=1
            stacked[str(qb["ID"])]+=bool(partners)
        self.assertEqual(set(qbs),{"QBA","QBB","QBC"})
        self.assertTrue(all(stacked[k]>=1 for k in qbs))
        counts=Counter(p["ID"] for entry in out for p in entry["players"])
        self.assertLessEqual(max(counts.values()),21)

    def test_qb_selection_report_is_evidence_not_hindsight(self):
        rows=build_fixture()
        candidates=build_classic_candidates(rows,count=30,attempts=4000,
                                             seed=11,max_exposure=.35)
        selected=[]
        for entry in candidates[:4]:
            selected.append({"candidate":entry,"contest_tier":"$0.50",
                             "portfolio_role":"balanced"})
        report=qb_selection_report(
            rows,candidates,selected,
            diagnostics={"scored_qbs_before_efficiency":{"QBA":10,"QBB":5},
                         "scored_qbs_after_efficiency":{"QBA":4,"QBB":0},
                         "qb_best_simulated_p99":{"QBB":204.51}},
            team_environment={"B":{"expected_pass_attempts":35.25,
                                   "source":"prior-game nflverse sample"}})
        by_id={row["QB ID"]:row for row in report}
        self.assertEqual(len(by_id),3)
        self.assertGreater(by_id["QBB"]["Stacked QB candidates"],0)
        self.assertEqual(by_id["QBB"]["After efficiency filter"],0)
        self.assertIn("P90 salary-efficiency filter",
                      by_id["QBB"]["QB selection status"])
        self.assertEqual(by_id["QBB"]["Prior-game pass attempts expected"],35.25)

    def _roster(self,qb_id, qb_team):
        return [
            player(qb_id,"QB",qb_team,7000),
            player("RB1","RB","R1",6500),
            player("RB2","RB","R2",6500),
            player("WA","WR","A",5600),
            player("WX","WR","X",5500),
            player("WY","WR","Y",5500),
            player("TE","TE","T",4500),
            player("WZ","WR","Z",5500),
            player("DST","DST","D",3000)
        ]

    def _run_two(self,qb_bonus):
        stacked=self._roster("QBA","A")
        standalone=self._roster("QBB","B")
        self.assertEqual(check_classic_lineup(stacked,salary_floor=49600),[])
        self.assertEqual(check_classic_lineup(standalone,salary_floor=49600),[])
        all_players={str(p["ID"]):p for roster in (stacked,standalone)
                     for p in roster}
        frame=pd.DataFrame(list(all_players.values()))
        matrix=np.tile(np.linspace(10,20,200)[:,None],(1,len(frame)))
        qbi={name:i for i,name in enumerate(frame.ID)}
        matrix[:,qbi["QBB"]]+=qb_bonus
        notes={}
        out=select_simulated_portfolio([
            {"players":stacked,"salary":49600},
            {"players":standalone,"salary":49600}
            ],frame,matrix,count=1,max_exposure=1,premium_count=1,
            upside_share=1,require_opportunity=True,diagnostics=notes)
        return out[0],notes

    def test_comparable_upside_prefers_real_qb_wr_stack(self):
        item,notes=self._run_two(0)
        qb,receivers=qb_and_partners(item["candidate"]["players"])
        self.assertEqual(qb["ID"],"QBA")
        self.assertEqual([p["ID"] for p in receivers],["WA"])
        self.assertEqual(item["selection_explanation"]["qb_receiving_teammates"],["WA"])
        self.assertEqual(notes["scored_qbs_before_efficiency"]["QBB"],1)
        self.assertEqual(notes["scored_qbs_after_efficiency"]["QBA"],1)

    def test_materially_better_standalone_qb_still_wins(self):
        item,notes=self._run_two(14)
        qb,receivers=qb_and_partners(item["candidate"]["players"])
        self.assertEqual(qb["ID"],"QBB")
        self.assertFalse(receivers)
        self.assertIn("standalone QB ceiling",item["selection_explanation"]["qb_stack_note"])

    def test_rb_teammate_is_not_direct_qb_receiver_stack(self):
        roster=self._roster("QB0","A")
        roster[3]["TeamAbbrev"]="X"
        roster[1]["TeamAbbrev"]="A"
        _,partners=qb_and_partners(roster)
        self.assertEqual(partners,[])


if __name__=="__main__":
    unittest.main()
