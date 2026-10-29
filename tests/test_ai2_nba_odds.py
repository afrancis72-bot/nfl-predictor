"""Check actual quoted odds ingestion and independent NBA model matching."""
import unittest
from datetime import date
from unittest.mock import patch
import numpy as np
import pandas as pd
from nba_engine.odds_feed import (
    BOOKMAKERS, player_id_for_name, parse_event_props, load_current_props, normalize_csv_lines,
)

class NBAMarketImportTests(unittest.TestCase):
    def setUp(self):
        self.proj = pd.DataFrame([
            {"player_id":"1629029","player":"Luka Dončić","team":"LAL","opponent":"GSW","Points":33.0},
            {"player_id":"201939","player":"Stephen Curry","team":"GSW","opponent":"LAL","Points":26.2},
            {"player_id":"1630058","player":"Player Unknown","team":"BOS","opponent":"NYK","Points":12.0},
        ])

    def sample_event(self):
        return {
            "id":"evt-nba-1",
            "home_team":"Golden State Warriors",
            "away_team":"Los Angeles Lakers",
            "commence_time":"2026-10-31T02:00:00Z",
            "bookmakers":[{
                "key":"hardrockbet_fl","title":"Hard Rock Bet FL","last_update":"2026-10-30T20:10:00Z",
                "markets":[{"key":"player_points","outcomes":[
                    {"description":"Luka Doncic","name":"Over","point":32.5,"price":-110},
                    {"description":"Luka Doncic","name":"Under","point":32.5,"price":-110},
                    {"description":"Stephen Curry","name":"Over","point":26.5,"price":+100},
                    {"description":"Someone Not On Slate","name":"Over","point":20.5,"price":-115},
                ]}]
            }]
        }

    def test_hard_rock_available_as_a_bookmaker(self):
        self.assertEqual(BOOKMAKERS["Hard Rock Bet Florida"],"hardrockbet_fl")

    def test_normalize_accents_and_identify_unique_players(self):
        self.assertEqual(player_id_for_name("Luka Doncic",self.proj),"1629029")
        self.assertIsNone(player_id_for_name("Nikola Jokic",self.proj))

    def test_parse_odds_separately_from_projections(self):
        rows,unmatched = parse_event_props(self.sample_event(),self.proj,"hardrockbet_fl")
        self.assertEqual(len(rows),3)
        self.assertIn("Someone Not On Slate",unmatched)
        self.assertEqual(rows[0]["line"],32.5)
        self.assertEqual(rows[0]["player_id"],"1629029")
        self.assertNotIn("projection",rows[0])

    def test_api_event_only_selected_day_and_team(self):
        data=self.sample_event()
        calls=[]
        def mock_get(url,key,**args):
            calls.append(url)
            if url.endswith("/events"):
                return ([{"id":"evt-nba-1","home_team":data["home_team"],
                    "away_team":data["away_team"],"commence_time":data["commence_time"]},
                    {"id":"evt-next-day","home_team":data["home_team"],
                    "away_team":data["away_team"],"commence_time":"2026-11-01T02:00:00Z"}],{})
            return data,{"x-requests-remaining":"75"}
        with patch("nba_engine.odds_feed._api_get",side_effect=mock_get):
            rows,meta=load_current_props(self.proj,{"LAL":"GSW","GSW":"LAL"},
                "sample_api_key",markets=["Points"],target_day=date(2026,10,30))
        self.assertEqual(meta["event_count"],1)
        self.assertEqual(len(calls),2)
        self.assertEqual(len(rows),3)

    def test_import_name_only_csv_without_dummy_lines(self):
        uploaded=pd.DataFrame([
            {"Player Name":"Luka Doncic","Team":"LAL","Market":"player_points",
             "Point":32.5,"Outcome":"Over","Price":-110},
            {"Player Name":"Stephen Curry","Team":"GSW","Market":"Points",
             "Point":26.5,"Outcome":"Under","Price":115},
            {"Player Name":"Stephen Curry","Team":"GSW","Market":"Points",
             "Point":np.nan,"Outcome":"Under","Price":115},
        ])
        frame,meta=normalize_csv_lines(uploaded,self.proj)
        self.assertEqual(len(frame),2)
        self.assertEqual(frame.iloc[0]["market"],"Points")
        self.assertEqual(frame.iloc[0]["player_id"],"1629029")
        self.assertEqual(meta["missing_lines"],1)

    def test_do_not_accept_id_name_mismatch(self):
        raw=pd.DataFrame([{"player_id":"1629029","player":"Stephen Curry",
                           "market":"Points","line":30.5,"side":"Over","odds":-110}])
        rows,diag=normalize_csv_lines(raw,self.proj)
        self.assertTrue(rows.empty)
        self.assertEqual(len(diag["unmatched"]),1)

if __name__=="__main__":
    unittest.main()
