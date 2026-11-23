"""NBA regulation minutes, team points, joint markets, injury source regression.

Synthetic games are only offline fixtures, never claimed real sports data.
"""
from datetime import date,datetime
from zoneinfo import ZoneInfo
from io import BytesIO
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd

from nba_engine.predictor import build_projections
from nba_engine.game_conservation import (
    normalize_team_minutes,observed_complete_team_games,
    calibrate_team_scoring_baselines,simulate_conserved_nba_games)
from nba_engine.injury_report import (
    normalize_person, parse_official_injury_pdf,
    injury_exclusions, match_excluded_ids)


WHEN=date(2026,11,20)
TEAMS={"IND":"CLE","CLE":"IND"}


def complete_games():
    rows=[]
    # Every synthetic team game covers exactly ten players x 24 regulation
    # minutes; historical scores are approximately 110 and 120, not 160/167.
    for day in range(1,17):
        d=pd.Timestamp(f"2026-11-{day:02d}")
        for team,opp,total in [("IND","CLE",111),("CLE","IND",121)]:
            for i in range(10):
                pts=total//10+(1 if i<total%10 else 0)
                rows.append({
                    "date":d,"team":team,"opponent":opp,
                    "player_id":f"{team}_{i}",
                    "player":("Tyrese Haliburton" if team=="IND" and i==0
                              else "Evan Mobley" if team=="CLE" and i==0
                              else f"Example {team} {i}"),
                    "minutes":24.,"pts":float(pts),
                    "reb":float(3+i%4),"ast":float(2+i%3),
                    "threes":float(i%4),"stl":1.,
                    "blk":float(i%2),"tov":1.,
                })
    return pd.DataFrame(rows)


class NBAConservationTests(unittest.TestCase):
    def setUp(self):
        self.logs=complete_games()
        self.proj=build_projections(
            self.logs,TEAMS,WHEN,min_minutes=8)

    def test_rate_limited_index_can_find_actual_public_raw_logs(self):
        import requests
        from nba_engine.data_sources import discover_game_log_files
        response=requests.Response()
        response.status_code=403
        error=requests.HTTPError("Rate limit",response=response)
        class Head:
            status_code=200
        with patch("nba_engine.data_sources._get_json",side_effect=error),patch(
                "nba_engine.data_sources.requests.head",
                return_value=Head()) as heads:
            chosen=discover_game_log_files(WHEN)
        self.assertEqual(len(chosen),3)
        self.assertTrue(heads.called)
        self.assertTrue(all(row["as_of"]=="2026-11-20" for row in chosen))

    def test_prior_team_games_are_real_roster_scores_not_player_average_sums(self):
        g=observed_complete_team_games(self.logs,WHEN)
        self.assertEqual(set(g["team"]),set(TEAMS))
        self.assertTrue((g["minutes"]==240).all())
        self.assertEqual(set(g.loc[g.team=="IND","pts"]),{111})
        self.assertEqual(set(g.loc[g.team=="CLE","pts"]),{121})

    def test_overbooked_rotation_normalized_to_240_minutes(self):
        before=self.proj.copy()
        for key in ("minutes",):
            before[key]=36.
        before["minutes_sd"]=4.
        rotated,audit=normalize_team_minutes(before)
        self.assertTrue((audit["projected_minutes_before"]==360).all())
        self.assertTrue((audit["projected_minutes_after"]==240).all())
        for team,g in rotated.groupby("team"):
            self.assertAlmostEqual(float(g.minutes.sum()),240.)
            self.assertAlmostEqual(float(g.pts.sum()),
                float(before.loc[before.team==team,"pts"].sum())*240/360)

    def test_points_plus_bench_exactly_conserve_historical_team_score(self):
        rotated,audit=normalize_team_minutes(self.proj)
        draws,teams=simulate_conserved_nba_games(
            rotated,self.logs,when=WHEN,trials=10000,seed=47)
        self.assertEqual(len(teams),2)
        for team,group in rotated.groupby("team"):
            total=np.sum(np.stack([
                draws[(str(pid),"Points")]
                for pid in group.player_id.astype(str)]),axis=0)
            self.assertGreaterEqual(total.min(),65)
            self.assertLessEqual(total.max(),180)
            self.assertTrue(np.all(total>0))
            self.assertLess(abs(total.mean()-(111 if team=="IND" else 121)),5)
            reported=teams.loc[teams.team==team].iloc[0]
            self.assertAlmostEqual(total.mean(),reported.mean_team_points,delta=.05)
            self.assertEqual(reported.mean_unmodeled_bench_points,0.)
            self.assertTrue(reported.team_score_conservation_passed)
        self.assertLess(float(teams.mean_team_points.sum()),270.)
        self.assertTrue((teams.market_total_used_to_fit==False).all())

    def test_independent_player_baselines_fit_realistic_team_total(self):
        p,_=normalize_team_minutes(self.proj)
        p.loc[p.team=="IND","pts"]*=1.6
        original=float(p.loc[p.team=="IND","pts"].sum())
        for market,parts in __import__(
                "nba_engine.predictor",fromlist=["MARKETS"]).MARKETS.items():
            p[market]=p[list(parts)].sum(axis=1)
        adjusted,audit=calibrate_team_scoring_baselines(p,self.logs,WHEN)
        self.assertGreater(original,100)
        self.assertTrue((audit.constrained_player_baseline_points<=
                         audit.expected_team_points_from_complete_history+0.02).all())
        self.assertFalse(audit.uses_sportsbook_total.any())
        self.assertNotEqual(original,
            float(adjusted.loc[adjusted.team=="IND","pts"].sum()))
        for team,g in adjusted.groupby("team"):
            self.assertLess(float(g.Points.sum()),150)
            self.assertAlmostEqual(
                float(g.Points.sum()),
                float(audit.loc[audit.team==team,
                    "constrained_player_baseline_points"].iloc[0]),places=2)

    def test_same_seed_repeats_and_combo_points_reconciles(self):
        p,_=normalize_team_minutes(self.proj)
        a,_=simulate_conserved_nba_games(
            p,self.logs,when=WHEN,trials=1000,seed=7)
        b,_=simulate_conserved_nba_games(
            p,self.logs,when=WHEN,trials=1000,seed=7)
        for k in a:
            np.testing.assert_array_equal(a[k],b[k])
        pid=p.player_id.iloc[0]
        np.testing.assert_array_equal(
            a[(pid,"PRA")],
            a[(pid,"Points")]+a[(pid,"Rebounds")]+a[(pid,"Assists")])

    def test_missing_rotation_does_not_fake_a_game(self):
        short=self.proj.loc[
            ~self.proj.player_id.isin(["IND_6","IND_7","IND_8","IND_9",
                                       "CLE_6","CLE_7","CLE_8","CLE_9"])]
        with self.assertRaisesRegex(ValueError,"fewer than seven"):
            normalize_team_minutes(short)
        oversized=self.proj.copy()
        oversized["minutes"]*=2
        with self.assertRaisesRegex(ValueError,"normalize rotation"):
            simulate_conserved_nba_games(
                oversized,self.logs,when=WHEN,trials=1000,seed=42)

    def test_absent_injured_players_cannot_get_old_points(self):
        # Fixture team injury list strictly matched on BOTH team + full name.
        statuses=pd.DataFrame([
            {"game":"IND@CLE","team":"IND","player":"Tyrese Haliburton",
             "player_key":normalize_person("Tyrese Haliburton"),"status":"Out"},
            {"game":"IND@CLE","team":"CLE","player":"Evan Mobley",
             "player_key":normalize_person("Evan Mobley"),"status":"Out"},
            {"game":"IND@CLE","team":"IND","player":"Example IND 1",
             "player_key":normalize_person("Example IND 1"),"status":"Questionable"},
        ])
        unavailable,uncertain=injury_exclusions(statuses,TEAMS)
        ids,unmatched=match_excluded_ids(self.logs,unavailable,TEAMS,WHEN)
        self.assertEqual(set(ids),{"IND_0","CLE_0"})
        self.assertEqual(unmatched,[])
        self.assertEqual(uncertain[("IND",normalize_person("Example IND 1"))],
                         "Questionable")
        fresh=build_projections(self.logs,TEAMS,WHEN,
                                excluded=ids,min_minutes=8)
        self.assertNotIn("Tyrese Haliburton",set(fresh.player))
        self.assertNotIn("Evan Mobley",set(fresh.player))
        p,_=normalize_team_minutes(fresh)
        draws,report=simulate_conserved_nba_games(
            p,self.logs,when=WHEN,trials=1000,seed=42)
        self.assertNotIn(("IND_0","Points"),draws)
        self.assertNotIn(("CLE_0","Points"),draws)
        self.assertLess(float(report.mean_team_points.sum()),270.)

    def test_partial_report_skips_only_unsubmitted_team_game(self):
        from nba_engine.injury_report import fetch_current_official_injuries
        class Page:
            def extract_text(self,extraction_mode="layout"):
                # The official pypdf visual extraction is reverse row order.
                lines=[
                    "11/23/2026 07:30 (ET) IND@CLE Indiana Pacers"+
                    "      Haliburton, Tyrese     Out",
                    "                 Cleveland Cavaliers"+
                    "      Mobley, Evan           Out",
                    "10:00 (ET) UTA@LAL Utah Jazz     NOT   YET SUBMITTED",
                    "                 Los Angeles Lakers"+
                    "      Reaves, Austin       Questionable",
                ]
                return "\\n".join(reversed(lines))
        class Reader:
            def __init__(self,_):
                self.pages=[Page()]
        games={"IND@CLE","UTA@LAL"}
        with patch("nba_engine.injury_report.PdfReader",Reader):
            table=parse_official_injury_pdf(
                b"%PDF-1.4 fake",target_games=games,
                document_date=date(2026,11,23),allow_partial=True)
        self.assertEqual(table.attrs["submitted_games"],["IND@CLE"])
        self.assertEqual(table.attrs["pending_games"],["UTA@LAL"])
        self.assertEqual(table.attrs["unsubmitted_teams"],
                         [("UTA@LAL","UTA")])
        self.assertEqual(set(table.game),{"IND@CLE"})
        self.assertNotIn("Austin Reaves",set(table.player))
        with patch("nba_engine.injury_report.PdfReader",Reader):
            with self.assertRaisesRegex(ValueError,"not been submitted"):
                parse_official_injury_pdf(
                    b"%PDF-1.4 fake",target_games=games,
                    document_date=date(2026,11,23))
        class Response:
            status_code=200
            content=b"%PDF-1.4 fake"
        with patch("nba_engine.injury_report.PdfReader",Reader):
            rows,meta=fetch_current_official_injuries(
                date(2026,11,23),target_games=games,
                now=datetime(2026,11,23,14,47,
                             tzinfo=ZoneInfo("America/New_York")),
                get=lambda *args,**kwargs:Response())
        self.assertEqual(meta["games_reported"],["IND@CLE"])
        self.assertEqual(meta["games_pending"],["UTA@LAL"])
        self.assertEqual(meta["status"],"PARTIAL_NBA_REPORT")
        self.assertTrue((rows.game=="IND@CLE").all())

    def test_per_game_runner_keeps_complete_game_when_another_missing_data(self):
        from nba_engine.slate_runner import run_verified_slate
        status=pd.DataFrame([
            {"game":"IND@CLE","team":"IND","player":"Tyrese Haliburton",
             "player_key":normalize_person("Tyrese Haliburton"),
             "status":"Out"},
            {"game":"IND@CLE","team":"CLE","player":"Evan Mobley",
             "player_key":normalize_person("Evan Mobley"),
             "status":"Out"},
        ])
        opponents={"IND":"CLE","CLE":"IND","BOS":"NYK","NYK":"BOS"}
        a=run_verified_slate(
            self.logs,opponents,status,{"IND@CLE","NYK@BOS"},
            when=WHEN,trials=1000,seed=89)
        self.assertEqual(a["games_completed"],["IND@CLE"])
        self.assertIn("NYK@BOS",a["games_failed"])
        self.assertEqual(set(a["teams_map"]),{"IND","CLE"})
        self.assertTrue(
            (a["game_audit"]["team_score_conservation_passed"]).all())
        self.assertEqual(len(a["projections"]["team"].unique()),2)
        self.assertNotIn("Tyrese Haliburton",set(a["projections"]["player"]))
        self.assertNotIn("Evan Mobley",set(a["projections"]["player"]))
        replay=run_verified_slate(
            self.logs,opponents,status,{"IND@CLE"},
            when=WHEN,trials=1000,seed=89)
        for key in a["samples"]:
            np.testing.assert_array_equal(a["samples"][key],replay["samples"][key])
        with self.assertRaisesRegex(ValueError,"None of the selected"):
            run_verified_slate(
                self.logs,opponents,status,{"NYK@BOS"},
                when=WHEN,trials=1000,seed=89)

    def test_fake_official_pdf_layout_carries_team_across_page(self):
        class Page:
            def __init__(self,lines):self.lines=lines
            def extract_text(self,extraction_mode="layout"):
                return "\n".join(reversed(self.lines))
        class Reader:
            def __init__(self,_):self.pages=[
                Page([
                    "11/20/2026     07:30 (ET)    IND@CLE     Indiana Pacers"+
                    "       Furphy, Johnny         Out",
                ]),
                Page([
                    "                                    Haliburton, Tyrese    Out   Hamstring",
                    "                  Cleveland Cavaliers    Mobley, Evan    Out   Toe",
                    "                                    Nembhard, Andrew     Questionable    Hip",
                ]),
            ]
        with patch("nba_engine.injury_report.PdfReader",Reader):
            table=parse_official_injury_pdf(
                b"%PDF-1.4 placeholder",target_games={"IND@CLE"},
                document_date=WHEN)
        out,unsure=injury_exclusions(table,TEAMS)
        self.assertIn(("IND",normalize_person("Tyrese Haliburton")),out)
        self.assertIn(("CLE",normalize_person("Evan Mobley")),out)
        self.assertEqual(unsure[("CLE",normalize_person("Andrew Nembhard"))],
                         "Questionable")


if __name__=="__main__":
    unittest.main()
