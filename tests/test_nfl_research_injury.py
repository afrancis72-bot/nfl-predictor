import unittest
from unittest.mock import patch
import pandas as pd
from ai2_nfl_auto_intelligence import collect_automatic_baselines

class ResearchTest(unittest.TestCase):
    @patch("ai2_nfl_auto_intelligence.make_rates")
    @patch("ai2_nfl_auto_intelligence.build_historical_baselines")
    def test_injury_status(self, mocked, make_rates):
        make_rates.return_value = pd.DataFrame([{"player":"A","team":"T","position":"RB","carries_pg":1.0,"targets_pg":1.0}])
        mocked.return_value = pd.DataFrame([
            {"ID":"1","Name":"A","DK Status":"","match_status":"matched"},
            {"ID":"2","Name":"B","DK Status":"","match_status":"matched"},
            {"ID":"3","Name":"C","DK Status":"Q","match_status":"matched"}
        ])
        saved = ("dk.csv", b"ID,Name\n1,A\n2,B\n3,C\n", "now")
        result = collect_automatic_baselines(saved, 2026, 8,
            history=pd.DataFrame({"x":[1]}),
            injury_fetcher=lambda: ({"b":"OUT"}, {"source":"test"}))
        self.assertEqual(result["candidate_count"],1)
        self.assertFalse(result["release_ready"])
