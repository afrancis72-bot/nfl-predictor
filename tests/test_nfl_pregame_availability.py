"""Strict evidence/selection gate tests. Example source links are SYNTHETIC.

An HTTPS link + a claimed published timestamp are not proof that real data
were independently fetched. These are source-contract/provenance tests;
separate verified source collection is needed for a deployable integration.
"""
import copy
import unittest

import numpy as np
import pandas as pd

from ai2_nfl_pregame_availability import (
    validate_availability_snapshot,classify_game_players,
    ACTIVE, OUT, UNKNOWN,
)
from ai2_nfl_full_player_universe import (
    build_football_only_players,validate_football_only_players,
)
from dfs_stats_engine import simulate_stat_driven_dfs
from test_nfl_full_player_universe import fixture

SYNTHETIC_URL="https://www.nfl.com/news/synthetic-test-not-a-real-availability-report"


def evidence(player,status,source,*,team="AAA",
             published="2026-10-18T15:42:00Z"):
    return dict(
        player=player,team=team,status=status,source_type=source,
        source_url=SYNTHETIC_URL,published_at_utc=published,
    )


def snapshot(*records):
    return dict(
        season=2026,week=7,game="AAA@BBB",
        kickoff_utc="2026-10-18T17:00:00Z",
        captured_at_utc="2026-10-18T16:15:00Z",
        records=list(records),
    )


class PregameAvailabilityResearchTests(unittest.TestCase):
    def setUp(self):
        self.pool,self.rates=fixture()

    def extras(self,snap=None,**more):
        return build_football_only_players(
            self.pool,self.rates,season=2026,week=7,
            require_verified_active=True,availability_snapshot=snap,
            **more)

    def test_no_snapshot_keeps_unverified_teammates_out_of_football_world(self):
        extras=self.extras()
        self.assertEqual(len(extras),0)
        self.assertEqual(extras.columns.tolist(),
                         build_football_only_players(self.pool,self.rates,
                                  season=2026,week=7).columns.tolist())

    def test_positive_active_row_is_required_and_status_remains_unverified_content(self):
        snap=snapshot(evidence("Missing Salary WR",ACTIVE,"NFL_GAME_DAY_ACTIVE_ROSTER"))
        extra=self.extras(snap)
        self.assertEqual(extra.Name.tolist(),["Missing Salary WR"])
        self.assertEqual(extra.optimizer_eligible.tolist(),[False])
        self.assertTrue(extra.role_evidence_status.iloc[0].endswith(
                "UNVERIFIED_CONTENT"))
        self.assertEqual(extra.availability_source_url.iloc[0],SYNTHETIC_URL)
        validate_football_only_players(extra,self.pool,self.rates,
                   availability_snapshot=snap,season=2026,week=7)

    def test_unlisted_and_questionable_players_remain_outside_simulation(self):
        snap=snapshot(evidence("Missing Salary WR",UNKNOWN,
                               "NFL_OFFICIAL_GAME_STATUS_REPORT"))
        self.assertTrue(self.extras(snap).empty)
        desc=classify_game_players(self.pool,self.rates,snap,
                         season=2026,week=7,game="AAA@BBB")
        self.assertIn(("AAA","missing salary wr"),desc["uncertain"])
        self.assertIn(("AAA","minor team wr"),desc["uncertain"])
        self.assertEqual(desc["verified_active"],[])

    def test_out_status_does_not_resurrect_or_infer_from_missing_inactive_list(self):
        snap=snapshot(evidence("Missing Salary WR",OUT,"NFL_OFFICIAL_INACTIVES"),
                      evidence("Minor Team WR",ACTIVE,
                               "NFL_GAME_DAY_ACTIVE_ROSTER"))
        extra=self.extras(snap)
        self.assertEqual(extra.Name.tolist(),["Minor Team WR"])
        desc=classify_game_players(self.pool,self.rates,snap,
                         season=2026,week=7,game="AAA@BBB")
        self.assertIn(("AAA","missing salary wr"),desc["verified_out"])

    def test_selected_player_out_fails_the_lineup_evidence_gate(self):
        snap=snapshot(evidence("WAAA",OUT,"NFL_OFFICIAL_INACTIVES"))
        with self.assertRaisesRegex(ValueError,"Confirmed-OUT"):
            self.extras(snap)

    def test_out_from_final_game_status_report_allowed_but_doubtful_is_not_out(self):
        snap=snapshot(evidence("Missing Salary WR",OUT,
                               "NFL_OFFICIAL_GAME_STATUS_REPORT"))
        self.assertTrue(self.extras(snap).empty)
        with self.assertRaisesRegex(ValueError,"Source cannot prove"):
            validate_availability_snapshot(snapshot(
                evidence("Missing Salary WR",ACTIVE,"NFL_OFFICIAL_INACTIVES")),
                season=2026,week=7,game="AAA@BBB")
        with self.assertRaisesRegex(ValueError,"Source cannot prove"):
            validate_availability_snapshot(snapshot(
                evidence("Missing Salary WR",ACTIVE,"NFL_OFFICIAL_GAME_STATUS_REPORT")),
                season=2026,week=7,game="AAA@BBB")

    def test_rejects_future_asof_and_postgame_backfill(self):
        good=snapshot(evidence("Missing Salary WR",ACTIVE,
                               "NFL_GAME_DAY_ACTIVE_ROSTER"))
        with self.assertRaisesRegex(ValueError,"Future availability"):
            self.extras(good,as_of_utc="2026-10-18T16:00:00Z")
        late=copy.deepcopy(good);late["captured_at_utc"]="2026-10-18T18:00:00Z"
        with self.assertRaisesRegex(ValueError,"Post-kickoff"):
            self.extras(late)
        post=copy.deepcopy(good)
        post["records"][0]["published_at_utc"]="2026-10-18T17:01:00Z"
        with self.assertRaisesRegex(ValueError,"publication after"):
            self.extras(post)
        naive=copy.deepcopy(good);naive["captured_at_utc"]="2026-10-18T16:15:00"
        with self.assertRaisesRegex(ValueError,"explicitly UTC"):
            self.extras(naive)

    def test_rejects_wrong_team_game_duplicate_contradictory_or_fake_source(self):
        base=evidence("Missing Salary WR",ACTIVE,"NFL_GAME_DAY_ACTIVE_ROSTER")
        with self.assertRaisesRegex(ValueError,"game identity"):
            self.extras({**snapshot(base),"game":"AAA@CCC"})
        with self.assertRaisesRegex(ValueError,"Duplicate/conflicting"):
            self.extras(snapshot(base,evidence("Missing Salary WR",OUT,
                                               "NFL_OFFICIAL_INACTIVES")))
        with self.assertRaisesRegex(ValueError,"team identity"):
            self.extras(snapshot(evidence("Missing Salary WR",ACTIVE,
                                           "NFL_GAME_DAY_ACTIVE_ROSTER",team="CCC")))
        forged={**base,"source_url":"https://nfl.com.attacker.example/active"}
        with self.assertRaisesRegex(ValueError,"official NFL HTTPS"):
            self.extras(snapshot(forged))
        HTTP={**base,"source_url":"http://www.nfl.com/news/active"}
        with self.assertRaisesRegex(ValueError,"official NFL HTTPS"):
            self.extras(snapshot(HTTP))

    def test_uncertified_direct_injection_or_metadata_modification_rejected(self):
        snap=snapshot(evidence("Missing Salary WR",ACTIVE,
                               "NFL_GAME_DAY_ACTIVE_ROSTER"))
        ghost=self.extras(snap)
        with self.assertRaisesRegex(ValueError,"Unverified official ACTIVE"):
            validate_football_only_players(ghost,self.pool,self.rates)
        modified=ghost.copy()
        modified.loc[0,"availability_source_url"]="https://www.nfl.com/news/other"
        with self.assertRaisesRegex(ValueError,"Altered player availability"):
            validate_football_only_players(modified,self.pool,self.rates,
                    availability_snapshot=snap,season=2026,week=7)

    def simulate(self,extra=None,snap=None):
        return simulate_stat_driven_dfs(
            self.pool,self.rates,n_sims=340,seed=21117,
            allow_role_proxies=False,wr_role_team_rng_isolated=True,
            football_only_players=extra,
            football_universe_target_season=2026 if extra is not None else None,
            football_universe_target_week=7 if extra is not None else None,
            football_availability_snapshot=snap,
        )

    def test_empty_evidence_is_identical_to_original_model(self):
        pool0,points0,stats0=self.simulate()
        pool1,points1,stats1=self.simulate(extra=self.extras())
        self.assertEqual(pool0.ID.tolist(),pool1.ID.tolist())
        self.assertEqual(pool0.columns.tolist(),pool1.columns.tolist())
        np.testing.assert_array_equal(points0,points1)
        self.assertEqual(stats1["__FOOTBALL_PLAYER_UNIVERSE__"][
                            "historical_football_only_players"],0)

    def test_active_adds_only_sourced_player_without_changing_team_attempts_or_dk_rows(self):
        snap=snapshot(evidence("Missing Salary WR",ACTIVE,
                               "NFL_GAME_DAY_ACTIVE_ROSTER"),
                      evidence("Minor Team WR",UNKNOWN,
                               "NFL_OFFICIAL_GAME_STATUS_REPORT"))
        extra=self.extras(snap)
        p0,v0,z0=self.simulate()
        p1,v1,z1=self.simulate(extra,snap=snap)
        self.assertEqual(p0.ID.tolist(),p1.ID.tolist())
        self.assertEqual(p0.columns.tolist(),p1.columns.tolist())
        self.assertEqual(v0.shape,v1.shape)
        self.assertGreater(z1["Missing Salary WR"]["targets"].mean(),0)
        self.assertNotIn("Minor Team WR",z1)
        self.assertEqual(z1["__FOOTBALL_PLAYER_UNIVERSE__"][
                          "football_only_players_exported_to_optimizer"],0)
        for team in ("AAA","BBB"):
            root="__TEAM_PLAYS__|"+team
            for metric in ("simulated_pass_attempts","simulated_plays",
                           "simulated_dropbacks","simulated_sacks"):
                np.testing.assert_array_equal(z0[root][metric],z1[root][metric])
        self.assertTrue(z1["__FOOTBALL_PLAYER_UNIVERSE__"][
                        "source_asserted_game_day_active"])
        self.assertFalse(z1["__FOOTBALL_PLAYER_UNIVERSE__"][
                        "source_content_independently_fetched"])

    def test_documented_explicit_gate_prevents_accidental_unverified_mode(self):
        snap=snapshot(evidence("Missing Salary WR",ACTIVE,
                               "NFL_GAME_DAY_ACTIVE_ROSTER"))
        with self.assertRaisesRegex(ValueError,"cannot be ignored"):
            build_football_only_players(self.pool,self.rates,season=2026,week=7,
                                        availability_snapshot=snap)
        # A claimed official status without independent source verification
        # is not license to declare the production model ready.
        self.assertEqual(len(self.extras(snap)),1)

if __name__=="__main__":
    unittest.main()
