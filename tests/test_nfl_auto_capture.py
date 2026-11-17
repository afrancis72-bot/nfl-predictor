"""Unattended source acquisition tests with ONLY local synthetic fixtures."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from scripts.auto_capture_nfl_status import due_games,collect_due_games
from test_nfl_official_inactives_capture import html as inactive_html,URL as INACTIVE_URL
from test_nfl_official_named_active import fixture as named_html,URL as ACTIVE_URL
from ai2_nfl_official_inactives_capture import load_verified_capture
from ai2_nfl_official_named_active import load_verified_named_active_capture


def schedule():
    return pd.DataFrame([
        {"season":2026,"week":10,"game_type":"REG",
         "away_team":"KC","home_team":"ATL",
         "gameday":"2026-11-15","gametime":"13:00"},
        {"season":2026,"week":10,"game_type":"REG",
         "away_team":"ATL","home_team":"CAR",
         "gameday":"2026-11-15","gametime":"13:00"},
        {"season":2026,"week":12,"game_type":"REG",
         "away_team":"NE","home_team":"NYJ",
         "gameday":"2026-11-29","gametime":"13:00"},
    ])


def candidate(kind,url):
    return lambda *,week:{"articles":[{"kind":kind,"url":url}],
                          "errors":[]}


class AutoNFLResearchCaptureTests(unittest.TestCase):
    NOW="2026-11-15T16:50:00Z"

    def test_schedule_driven_imminent_matchups_and_no_postgame_capture(self):
        due=due_games(schedule(),now=self.NOW)
        self.assertEqual(len(due),2)
        self.assertEqual(set(d["game"] for d in due),{"KC@ATL","ATL@CAR"})
        self.assertTrue(all(d["week"]==10 for d in due))
        self.assertEqual(due_games(schedule(),now="2026-11-15T19:00:00Z"),[])
        self.assertEqual(due_games(schedule(),now="2026-11-13T16:50:00Z"),[])

    def test_capture_real_data_flow_for_explicit_out_from_mock_html(self):
        subset=schedule().iloc[[0]].copy()
        with tempfile.TemporaryDirectory() as temp:
            result=collect_due_games(schedule=subset,now_utc=self.NOW,
                out_dir=Path(temp)/"archives",
                discovery=candidate("OFFICIAL_INACTIVES",INACTIVE_URL),
                capture_clock=lambda:self.NOW,
                get_html=lambda url:inactive_html())
            self.assertEqual(result["result"],"CAPTURED_OFFICIAL_PREGAME_EVIDENCE")
            self.assertEqual(len(result["source_archives"]),1)
            path=result["source_archives"][0]["archive_dir"]
            snap,manifest=load_verified_capture(path,as_of_utc=self.NOW)
            self.assertEqual(len(snap["records"]),11)
            self.assertTrue(all(r["status"]=="VERIFIED_OUT" for r in snap["records"]))
            self.assertTrue(result["positive_active_by_omission"] is False)
            self.assertFalse(result["production_app_modified"])

    def test_capture_explicit_active_from_mock_news_not_expected_to_play(self):
        subset=schedule().iloc[[1]].copy()
        with tempfile.TemporaryDirectory() as temp:
            result=collect_due_games(schedule=subset,now_utc=self.NOW,
                out_dir=Path(temp)/"archives",
                discovery=candidate("EXPLICIT_INJURY_UPDATES",ACTIVE_URL),
                capture_clock=lambda:self.NOW,
                get_html=lambda url:named_html())
            self.assertEqual(len(result["source_archives"]),1)
            snap,manifest=load_verified_named_active_capture(
                result["source_archives"][0]["archive_dir"],as_of_utc=self.NOW)
            self.assertEqual(len(snap["records"]),2)
            self.assertTrue(all(r["status"]=="VERIFIED_ACTIVE" for r in snap["records"]))
            self.assertFalse(result["production_app_modified"])

    def test_report_unavailable_without_invented_status_or_fake_capture(self):
        subset=schedule().iloc[[0]].copy()
        with tempfile.TemporaryDirectory() as temp:
            result=collect_due_games(schedule=subset,now_utc=self.NOW,
                out_dir=Path(temp)/"archives",
                discovery=lambda *,week:{"articles":[],"errors":[]})
            self.assertEqual(result["source_archives"],[])
            self.assertEqual(result["result"],"NO_MATCHING_VALID_PREGAME_OFFICIAL_ARTICLE")
            self.assertFalse((Path(temp)/"archives").exists())

    def test_bad_article_does_not_produce_game_day_status(self):
        subset=schedule().iloc[[0]].copy()
        with tempfile.TemporaryDirectory() as temp:
            result=collect_due_games(schedule=subset,now_utc=self.NOW,
                out_dir=Path(temp)/"archives",
                discovery=candidate("OFFICIAL_INACTIVES",INACTIVE_URL),
                capture_clock=lambda:self.NOW,
                get_html=lambda url:named_html())
            self.assertEqual(result["source_archives"],[])
            self.assertEqual(result["result"],"NO_MATCHING_VALID_PREGAME_OFFICIAL_ARTICLE")
            self.assertEqual(len(result["errors"]),1)

if __name__=="__main__":
    unittest.main()
