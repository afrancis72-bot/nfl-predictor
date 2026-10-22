import unittest
import pandas as pd
from receiver_matchup_exposure import receiver_injury_exposure

class ReceiverExposureTests(unittest.TestCase):
    def setUp(self):
        self.receivers=pd.DataFrame([
            dict(player_id="wr1",offense_team="BUF",opponent_team="NYJ",receiver_rank="WR1",alignment="slot_pass",routes=120,matchup_routes=18),
            dict(player_id="wr2",offense_team="BUF",opponent_team="NYJ",receiver_rank="WR2",alignment="slot_pass",routes=110,matchup_routes=72),
            dict(player_id="wr3",offense_team="BUF",opponent_team="NYJ",receiver_rank="WR3",alignment="slot_pass",routes=80,matchup_routes=48)])
        self.vuln=pd.DataFrame([dict(team="NYJ",weakness="slot_pass",factor=1.08,supported=True)])
    def test_wr2_wr3_can_benefit_more_than_wr1(self):
        x=receiver_injury_exposure(self.receivers,self.vuln).set_index("player_id")
        self.assertGreater(x.at["wr2","efficiency_factor"],x.at["wr1","efficiency_factor"])
        self.assertGreater(x.at["wr3","efficiency_factor"],x.at["wr1","efficiency_factor"])
    def test_rank_does_not_override_actual_exposure(self):
        r=self.receivers.copy()
        r.loc[r.player_id=="wr1","matchup_routes"]=95
        x=receiver_injury_exposure(r,self.vuln).set_index("player_id")
        self.assertGreater(x.at["wr1","efficiency_factor"],x.at["wr2","efficiency_factor"])
    def test_missing_matchup_evidence_neutral(self):
        r=self.receivers.copy()
        r["matchup_routes"]=0
        x=receiver_injury_exposure(r,self.vuln)
        self.assertTrue((x.efficiency_factor==1.).all())
        self.assertFalse(x.supported.any())
    def test_opponent_must_match(self):
        r=self.receivers.copy()
        r["opponent_team"]="MIA"
        x=receiver_injury_exposure(r,self.vuln)
        self.assertTrue((x.efficiency_factor==1.).all())
if __name__=="__main__":
    unittest.main()
