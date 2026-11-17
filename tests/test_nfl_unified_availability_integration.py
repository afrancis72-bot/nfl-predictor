"""Single unified NFL Classic source bridge: same builder, honest source ledger.

All HTML/schedules here are synthetic fixtures; no source content is asserted
to be genuine public data. True non-test captures require original NFL.com
HTTPS responses obtained before actual independently checked kickoff.
"""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from ai2_nfl_unified_availability import (
    prepare_unified_nfl_availability, apply_status_sources_to_pool,
)
from scripts.auto_capture_nfl_status import collect_due_games
from test_nfl_official_inactives_capture import html as inactive_html, URL as OUT_URL
from test_nfl_official_named_active import fixture as active_html, URL as ACTIVE_URL

NOW="2026-11-15T16:50:00Z"
SALARIES=(
    "synthetic_DKSalaries.csv",
    b"Position,Name,ID,Salary,Game Info,TeamAbbrev,AvgPointsPerGame\n"
    b"WR,Kenneth Example,1,6200,KC@ATL 11/15/2026 01:00PM ET,KC,9.3\n"
    b"QB,Example Quarterback,2,5500,KC@ATL 11/15/2026 01:00PM ET,KC,15.0\n"
    b"WR,Drake London,3,7900,KC@ATL 11/15/2026 01:00PM ET,ATL,18.2\n",
)


def schedule():
    return pd.DataFrame([{
        "season":2026,"week":10,"game_type":"REG",
        "away_team":"KC","home_team":"ATL",
        "gameday":"2026-11-15","gametime":"13:00",
    }])


def named_fixture():
    return active_html().replace(
        b"/teams/carolina-panthers/",b"/teams/kansas-city-chiefs/"
    ).replace(b"Carolina Panthers 5-5",b"Kansas City Chiefs 5-5")


def fake_collector_with(kinds):
    def collection(*,schedule,now_utc,out_dir):
        articles=[
            {"kind":kind,"url":(OUT_URL if kind=="OFFICIAL_INACTIVES" else ACTIVE_URL)}
            for kind in kinds]
        return collect_due_games(
            schedule=schedule,now_utc=now_utc,out_dir=out_dir,
            discovery=lambda *,week:{"articles":articles,"errors":[]},
            get_html=lambda url:(inactive_html() if url==OUT_URL else named_fixture()),
            capture_clock=lambda:now_utc,
        )
    return collection


def player_pool():
    return pd.DataFrame([
        {"Name":"Kenneth Example","TeamAbbrev":"KC","Position":"WR",
         "game":"KC@ATL","ID":"1","Salary":6200,"proj":10.0,"optimizer_eligible":True},
        {"Name":"Example Quarterback","TeamAbbrev":"KC","Position":"QB",
         "game":"KC@ATL","ID":"2","Salary":5500,"proj":15.0,"optimizer_eligible":True},
        {"Name":"Drake London","TeamAbbrev":"ATL","Position":"WR",
         "game":"KC@ATL","ID":"3","Salary":7900,"proj":19.0,"optimizer_eligible":True},
    ])


def prior_rates():
    return pd.DataFrame([
        {"player":"Example Player","team":"KC","position":"WR",
         "games_played":5,"targets_pg":4.0,"carries_pg":0.0,
         "latest_observed_season":2026,"latest_observed_week":9},
        {"player":"Drake London","team":"ATL","position":"WR",
         "games_played":5,"targets_pg":9.0,"carries_pg":0.0,
         "latest_observed_season":2026,"latest_observed_week":9},
        {"player":"Kenneth Example","team":"KC","position":"WR",
         "games_played":5,"targets_pg":2.0,"carries_pg":0.0,
         "latest_observed_season":2026,"latest_observed_week":9},
    ])


class UnifiedNFLAvailabilityIntegrationTests(unittest.TestCase):
    def test_no_verified_source_yields_no_added_active_players(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=prepare_unified_nfl_availability(
                SALARIES,season=2026,week=10,data_dir=tmp,now_utc=NOW,
                schedule=schedule(),collector=fake_collector_with([]))
            self.assertEqual(ctx["snapshots_by_game"],{})
            self.assertEqual(ctx["report"]["game_snapshot_count"],0)
            x,extra,outs,ledger=apply_status_sources_to_pool(
                player_pool(),prior_rates(),context=ctx,season=2026,week=10)
            self.assertTrue(extra.empty)
            self.assertEqual(outs,[])
            self.assertEqual(x.optimizer_eligible.tolist(),
                             player_pool().optimizer_eligible.tolist())
            self.assertEqual(ledger["effective_scorer"],
                             "EXISTING_PLAYER_UNIVERSE_NO_NEW_ACTIVE_PLAYERS")

    def test_verified_inactive_article_excludes_priced_player_without_implying_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=prepare_unified_nfl_availability(
                SALARIES,season=2026,week=10,data_dir=tmp,now_utc=NOW,
                schedule=schedule(),
                collector=fake_collector_with(["OFFICIAL_INACTIVES"]))
            self.assertEqual(ctx["report"]["game_snapshot_count"],1)
            self.assertEqual(ctx["report"]["verified_active_players"],0)
            self.assertEqual(ctx["report"]["verified_out_players"],11)
            x,extra,outs,ledger=apply_status_sources_to_pool(
                player_pool(),prior_rates(),context=ctx,season=2026,week=10)
            self.assertFalse(bool(x.loc[x.Name=="Kenneth Example",
                                          "optimizer_eligible"].iloc[0]))
            self.assertTrue(extra.empty)
            self.assertEqual(ledger["football_only_players_added"],0)
            self.assertEqual(len(outs),11)
            self.assertEqual(len(ledger["source_urls"]),1)

    def test_both_types_source_verified_and_internal_nonselectable_teammate(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=prepare_unified_nfl_availability(
                SALARIES,season=2026,week=10,data_dir=tmp,now_utc=NOW,
                schedule=schedule(),
                collector=fake_collector_with([
                    "OFFICIAL_INACTIVES","EXPLICIT_INJURY_UPDATES"]))
            self.assertEqual(ctx["report"]["verified_active_players"],2)
            self.assertEqual(ctx["report"]["verified_out_players"],11)
            self.assertEqual(len(ctx["report"]["source_urls"]),2)
            x,extra,outs,ledger=apply_status_sources_to_pool(
                player_pool(),prior_rates(),context=ctx,season=2026,week=10)
            self.assertEqual(extra.Name.tolist(),["Example Player"])
            self.assertEqual(extra.optimizer_eligible.tolist(),[False])
            self.assertEqual(extra.Salary.tolist(),[0])
            self.assertEqual(extra.game.tolist(),["KC@ATL"])
            self.assertEqual(ledger["football_only_players_added"],1)
            self.assertEqual(ledger["applied_verified_out_to_priced_players"],1)
            # No independent prospective MAE/P90 claim is permitted.
            self.assertFalse(ledger["predictive_accuracy_validated"])
            self.assertFalse(ledger["complete_game_day_roster_verified"])

    def test_future_or_bad_kickoff_archives_cannot_enter_a_game(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=prepare_unified_nfl_availability(
                SALARIES,season=2026,week=10,data_dir=tmp,
                now_utc="2026-11-15T12:00:00Z",
                schedule=schedule(),
                collector=fake_collector_with(["OFFICIAL_INACTIVES"]))
            self.assertEqual(ctx["report"]["game_snapshot_count"],0)
            self.assertEqual(ctx["report"]["verified_out_players"],0)

    def test_multiple_classic_games_validate_independent_status_snapshots(self):
        # Exact same multi-game interface the unified Classic optimizer uses,
        # not two unrelated research single-game apps.
        from ai2_nfl_full_player_universe import (
            build_football_only_players,validate_football_only_players)
        from test_nfl_full_player_universe import fixture
        from test_nfl_pregame_availability import snapshot,evidence,ACTIVE
        p1,r1=fixture()
        p2=p1.copy()
        r2=r1.copy()
        p2["TeamAbbrev"]=p2["TeamAbbrev"].replace({"AAA":"CCC","BBB":"DDD"})
        p2["game"]="CCC@DDD"
        p2["ID"]="B_"+p2["ID"].astype(str)
        p2["Name"]="Second "+p2["Name"].astype(str)
        r2["team"]=r2["team"].replace({"AAA":"CCC","BBB":"DDD"})
        r2["player"]="Second "+r2["player"].astype(str)
        p=pd.concat([p1,p2],ignore_index=True)
        rates=pd.concat([r1,r2],ignore_index=True)
        status1=snapshot(evidence("Missing Salary WR",ACTIVE,"NFL_GAME_DAY_ACTIVE_ROSTER"))
        status2=snapshot(evidence("Second Missing Salary WR",ACTIVE,
                                  "NFL_GAME_DAY_ACTIVE_ROSTER",team="CCC"))
        status2["game"]="CCC@DDD"
        sources={"AAA@BBB":status1,"CCC@DDD":status2}
        extras=pd.concat([
            build_football_only_players(
                p1,rates,season=2026,week=7,
                require_verified_active=True,availability_snapshot=status1),
            build_football_only_players(
                p2,rates,season=2026,week=7,
                require_verified_active=True,availability_snapshot=status2),
        ],ignore_index=True)
        self.assertEqual(len(extras),2)
        passed=validate_football_only_players(
            extras,p,rates,availability_snapshot=sources,
            season=2026,week=7)
        self.assertEqual(len(passed),2)
        self.assertFalse(passed.optimizer_eligible.any())
        with self.assertRaisesRegex(ValueError,"game not on the DK slate"):
            validate_football_only_players(
                extras,p,rates,availability_snapshot={
                    **sources,"OTHER@EAA":status1},
                season=2026,week=7)

    def test_final_unified_sim_adapter_really_consumes_verified_football_teammate(self):
        import numpy as np
        from test_nfl_full_player_universe import fixture
        from test_nfl_pregame_availability import snapshot,evidence,ACTIVE
        from ai2_nfl_simulation_adapter import run_verified_stat_simulation
        pool,rates=fixture()
        snap=snapshot(
            evidence("Missing Salary WR",ACTIVE,"NFL_GAME_DAY_ACTIVE_ROSTER"))
        ctx={"snapshots_by_game":{"AAA@BBB":snap},
             "report":{"as_of_utc":snap["captured_at_utc"],
                       "game_snapshot_count":1,"verified_active_players":1,
                       "verified_out_players":0,"source_urls":[],
                       "predictive_accuracy_validated":False}}
        x,extras,outs,ledger=apply_status_sources_to_pool(
            pool,rates,context=ctx,season=2026,week=7)
        self.assertEqual(len(extras),1)
        ctrl_p,ctrl_scores,ctrl_stats=run_verified_stat_simulation(
            pool,rates,n_sims=450,seed=20261117,
            injury_presence={})
        exp_p,exp_scores,exp_stats=run_verified_stat_simulation(
            x,rates,n_sims=450,seed=20261117,
            injury_presence={},
            football_only_players=extras,
            football_universe_target_season=2026,
            football_universe_target_week=7,
            football_availability_snapshots=ctx["snapshots_by_game"],
            football_availability_as_of_utc=snap["captured_at_utc"])
        self.assertEqual(exp_p.ID.tolist(),ctrl_p.ID.tolist())
        self.assertEqual(exp_p.columns.tolist(),ctrl_p.columns.tolist())
        self.assertEqual(exp_scores.shape,ctrl_scores.shape)
        self.assertFalse(np.array_equal(exp_scores,ctrl_scores))
        self.assertGreater(np.asarray(exp_stats["Missing Salary WR"]["targets"]).mean(),0)
        self.assertEqual(exp_stats["__FOOTBALL_PLAYER_UNIVERSE__"][
                         "historical_football_only_players"],1)
        self.assertEqual(exp_stats["__FOOTBALL_PLAYER_UNIVERSE__"][
                         "football_only_players_exported_to_optimizer"],0)
        empty_ctx={"snapshots_by_game":{},
                   "report":{"as_of_utc":snap["captured_at_utc"],
                             "predictive_accuracy_validated":False}}
        unchanged,empty,_,_=apply_status_sources_to_pool(
            pool,rates,context=empty_ctx,season=2026,week=7)
        same_p,same_scores,_=run_verified_stat_simulation(
            unchanged,rates,n_sims=450,seed=20261117,
            injury_presence={},
            football_only_players=empty,
            football_universe_target_season=2026,
            football_universe_target_week=7,
            football_availability_snapshots={},
            football_availability_as_of_utc=snap["captured_at_utc"])
        self.assertEqual(same_p.ID.tolist(),ctrl_p.ID.tolist())
        np.testing.assert_array_equal(same_scores,ctrl_scores)

    def test_mode_signature_stays_same_oneclick_builder(self):
        from inspect import signature
        from ai2_nfl_oneclick import build_oneclick_portfolio
        params=signature(build_oneclick_portfolio).parameters
        self.assertIn("simulation_mode",params)
        self.assertIn("availability_context",params)
        self.assertEqual(params["simulation_mode"].default,"existing")
        with self.assertRaisesRegex(ValueError,"prebuild check"):
            build_oneclick_portfolio(SALARIES,seed=43,count=20,
                                     simulation_mode="availability_aware")

if __name__=="__main__":
    unittest.main()
