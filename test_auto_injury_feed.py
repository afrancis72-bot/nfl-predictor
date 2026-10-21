import unittest
import pandas as pd
from auto_injury_feed import get_injuries,apply_injury_feed

class FeedTests(unittest.TestCase):
    def sample(self,season):
        return pd.DataFrame([
          dict(season=2026,week=7,team="LA",full_name="Example Receiver",
               report_status="Out",practice_status="DNP",date_modified="2026-10-21T15:00:00Z"),
          dict(season=2026,week=6,team="LA",full_name="Other Receiver",
               report_status="Out",practice_status="DNP",date_modified="2026-10-15T15:00:00Z"),
          dict(season=2026,week=7,team="NYJ",full_name="Questionable Back",
               report_status="Questionable",practice_status="Limited",date_modified="2026-10-21T15:00:00Z")])
    def test_week_filter_and_status(self):
        reports,meta=get_injuries(2026,7,fetcher=self.sample)
        self.assertEqual(len(reports),2)
        meta["stale"]=False
        pool=pd.DataFrame([dict(Name="Example Receiver",TeamAbbrev="LAR",optimizer_eligible=True,proj=15.0),
                           dict(Name="Questionable Back",TeamAbbrev="NYJ",optimizer_eligible=True,proj=13.0)])
        changed,audit=apply_injury_feed(pool,reports,meta)
        self.assertFalse(bool(changed.iloc[0].optimizer_eligible))
        self.assertTrue(bool(changed.iloc[1].optimizer_eligible))
        self.assertEqual(changed.iloc[1]["proj"],13.0)
        self.assertEqual(len(audit),2)
    def test_missing_week_fails_closed(self):
        reports,meta=get_injuries(2026,8,fetcher=self.sample)
        self.assertFalse(meta["ok"])
        self.assertTrue(reports.empty)
    def test_stale_fails_closed(self):
        reports,meta=get_injuries(2026,7,fetcher=self.sample)
        meta["stale"]=True
        pool=pd.DataFrame([dict(Name="Example Receiver",TeamAbbrev="LA",optimizer_eligible=True)])
        changed,audit=apply_injury_feed(pool,reports,meta)
        self.assertTrue(bool(changed.iloc[0].optimizer_eligible))
        self.assertTrue(audit.empty)
if __name__=="__main__":
    unittest.main()
