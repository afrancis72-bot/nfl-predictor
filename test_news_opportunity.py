import unittest
import pandas as pd
from news_opportunity import redistribute

class OpportunityTests(unittest.TestCase):
    def setUp(self):
        self.pool=pd.DataFrame([
            {"Name":"Absent Wideout","TeamAbbrev":"NYJ","Position":"WR","optimizer_eligible":False,"proj":14.0},
            {"Name":"Active Wideout","TeamAbbrev":"NYJ","Position":"WR","optimizer_eligible":True,"proj":12.0},
            {"Name":"Other Wideout","TeamAbbrev":"NYJ","Position":"WR","optimizer_eligible":True,"proj":9.0}])
        self.injuries=pd.DataFrame([{"team_key":"NYJ","name_key":"absent wideout","report_status":"OUT"}])
        self.meta={"ok":True,"stale":False,"source":"unit-test"}
        rows=[]
        for w in range(1,5):
            for name,t in [("Absent Wideout",8),("Active Wideout",6),("Other Wideout",3)]:
                rows.append({"season":2026,"week":w,"player_name":name,"recent_team":"NYJ",
                             "position":"WR","targets":t,"carries":0,"season_type":"REG"})
        self.history=pd.DataFrame(rows)
    def test_injury_boosts_active_not_absent(self):
        out,audit,status=redistribute(self.pool,self.injuries,self.meta,self.history,2026,7)
        self.assertTrue(status["ok"])
        self.assertEqual(float(out.iloc[0]["proj"]),14)
        self.assertGreater(float(out.iloc[1]["proj"]),12)
        self.assertGreater(float(out.iloc[2]["proj"]),9)
        self.assertLessEqual(float(out.iloc[1]["proj"]),12*1.18)
        self.assertEqual(status["players_adjusted"],2)
    def test_no_future_data_leak(self):
        h=self.history.copy()
        h["week"]=8
        out,audit,status=redistribute(self.pool,self.injuries,self.meta,h,2026,7)
        self.assertEqual(float(out.iloc[1]["proj"]),12)
    def test_feed_failure_keeps_baseline(self):
        out,audit,status=redistribute(self.pool,self.injuries,{"ok":False},self.history,2026,7)
        self.assertFalse(status["ok"])
        self.assertTrue(audit.empty)
        self.assertEqual(float(out.iloc[1]["proj"]),12)
if __name__=="__main__":
    unittest.main()
