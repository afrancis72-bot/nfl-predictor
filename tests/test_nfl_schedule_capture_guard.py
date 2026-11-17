"""Published nflverse schedule kickoff evidence guard, using offline mock CSV."""
import unittest
import pandas as pd

from ai2_nfl_schedule_capture_guard import (
    verified_match_kickoff,require_schedule_kickoff,fetch_and_verify_kickoff,
)


def games():
    return pd.DataFrame([
        {"season":2026,"week":10,"game_type":"REG","away_team":"KC",
         "home_team":"ATL","gameday":"2026-11-15","gametime":"13:00"},
        {"season":2026,"week":10,"game_type":"REG","away_team":"WSH",
         "home_team":"MIA","gameday":"2026-11-15","gametime":"09:30"},
        {"season":2026,"week":2,"game_type":"REG","away_team":"KC",
         "home_team":"ATL","gameday":"2026-09-13","gametime":"13:00"},
    ])


class ScheduledCaptureGuardTests(unittest.TestCase):
    def test_eastern_kickoff_conversion_makes_november_sunday_utc(self):
        actual=verified_match_kickoff(games(),season=2026,week=10,
                                      game="KC@ATL")
        self.assertEqual(actual,"2026-11-15T18:00:00+00:00")

    def test_uses_away_home_alias_and_early_morning_et(self):
        self.assertEqual(
            verified_match_kickoff(games(),season=2026,week=10,game="WAS@MIA"),
            "2026-11-15T14:30:00+00:00")

    def test_validates_expected_kickoff_from_independent_schedule(self):
        result=require_schedule_kickoff(games(),season=2026,week=10,
                                        game="KC@ATL",kickoff_utc="2026-11-15T18:00:00Z")
        self.assertEqual(result,"2026-11-15T18:00:00+00:00")

    def test_rejects_fake_later_kickoff_that_would_enable_postgame_backfill(self):
        with self.assertRaisesRegex(ValueError,"does not match"):
            require_schedule_kickoff(games(),season=2026,week=10,
                                    game="KC@ATL",kickoff_utc="2026-11-22T18:00:00Z")

    def test_rejects_wrong_game_week_or_duplicate_rows(self):
        with self.assertRaisesRegex(ValueError,"exactly one"):
            verified_match_kickoff(games(),season=2026,week=9,game="KC@ATL")
        duplicate=pd.concat([games(),games().iloc[[0]]],ignore_index=True)
        with self.assertRaisesRegex(ValueError,"exactly one"):
            verified_match_kickoff(duplicate,season=2026,week=10,game="KC@ATL")

    def test_cli_integration_reader_gets_stable_public_schedule_source(self):
        urlnames=[]
        def fake_reader(url):
            urlnames.append(url)
            return games()
        result=fetch_and_verify_kickoff(
            season=2026,week=10,game="KC@ATL",
            kickoff_utc="2026-11-15T18:00:00Z",reader=fake_reader)
        self.assertEqual(result,"2026-11-15T18:00:00+00:00")
        self.assertEqual(len(urlnames),1)
        self.assertIn("nflverse",urlnames[0])

if __name__=="__main__":
    unittest.main()
