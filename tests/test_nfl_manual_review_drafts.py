"""NFL human-review draft functionality must not masquerade as Bentley release."""
import unittest
from unittest.mock import patch
import json

from ai2_nfl_review import (snapshot_draft,load_draft,load_approved,
                           snapshot_approved,review_tables)
from ai2_nfl_oneclick import build_oneclick_portfolio
from ai2_nfl_release_gate import nfl_classic_release_status


def sample_ranked():
    slots=[("QB",7300),("RB",6900),("RB",6500),
           ("WR",6200),("WR",5800),("WR",5300),
           ("TE",4200),("WR",4600),("DST",2800)]
    players=[
        {"ID":str(i+1),"Name":f"Sample {position} {i}",
         "Position":position,
         "TeamAbbrev":("AAA" if i in (0,8) or i%2 else "BBB"),
         "Salary":salary,"proj":10.,
         "opponent":("BBB" if i in (0,8) or i%2 else "AAA")}
        for i,(position,salary) in enumerate(slots)]
    salary=sum(p["Salary"] for p in players)
    return [{"candidate":{"players":players,"salary":salary},
             "contest_tier":"$0.50",
             "simulated_mean":115.3,"simulated_p90":148.8,
             "simulated_p99":162.1}]


class DraftSeparationTests(unittest.TestCase):
    def setUp(self):
        self.ranked=sample_ranked()
        self.salaries=(
            "salary.csv",
            b"ID,Name,Position,Salary,Game Info,TeamAbbrev\n"
            b"1,Example,QB,6500,DAL@ARI 11/01/2026 01:00PM ET,DAL\n",
            "2026-10-28T22:00:00Z")

    def test_can_save_drafts_without_claiming_certification(self):
        packet=snapshot_draft(self.ranked,"fake-slate-hash",47)
        parsed=load_draft(packet)
        self.assertEqual(parsed["format"],"ai2-nfl-UNVALIDATED-draft-v1")
        self.assertIs(parsed["bentley_certified"],False)
        self.assertIs(parsed["user_review_required"],True)
        self.assertIs(parsed["auto_submit_to_draftkings"],False)
        self.assertEqual(parsed["simulation_seed"],47)
        self.assertEqual(len(parsed["lineups"]),1)
        self.assertFalse(nfl_classic_release_status()["ready_for_entries"])
        with self.assertRaisesRegex(ValueError,"Unsupported approved"):
            load_approved(packet)

    def test_protected_approved_snapshot_is_not_reinterpreted_as_draft(self):
        approved=snapshot_approved(self.ranked,"hash",47)
        load_approved(approved)
        with self.assertRaisesRegex(ValueError,"Not a saved human-review"):
            load_draft(approved)

    @patch("ai2_nfl_oneclick.create_observed_model_pool")
    @patch("ai2_nfl_oneclick.collect_automatic_baselines")
    def test_missing_official_feed_blocks_certified_but_not_manual_draft(
            self,baseline,player_pool):
        baseline.return_value={
            "blockers":["Official NFL injury feed unavailable: temporary outage"],
            "pregame_role_evidence":{"one":{"Role gate":"PASS"}},
        }
        player_pool.side_effect=ValueError("Reached observed player pool")
        with self.assertRaisesRegex(ValueError,"Official game-status data"):
            build_oneclick_portfolio(self.salaries,seed=47)
        player_pool.assert_not_called()
        with self.assertRaisesRegex(ValueError,"Reached observed player pool"):
            build_oneclick_portfolio(
                self.salaries,seed=47,draft_manual_review=True)
        player_pool.assert_called_once()

    def test_player_role_and_salary_review_still_enforced(self):
        summary,_,_=review_tables(self.ranked)
        self.assertEqual(summary["Salary"].iloc[0],sum(
            p["Salary"] for p in self.ranked[0]["candidate"]["players"]))
        broken=json.loads(json.dumps(self.ranked))
        broken[0]["candidate"]["salary"]+=250
        with self.assertRaisesRegex(ValueError,"Salary mismatch"):
            snapshot_draft(broken,"fake-slate",42)

    def test_no_false_release_status(self):
        report=nfl_classic_release_status()
        self.assertEqual(report["checks_passed"],0)
        self.assertEqual(report["checks_total"],9)
        self.assertTrue(report["must_not_build_lineups"])


if __name__=="__main__":
    unittest.main()
