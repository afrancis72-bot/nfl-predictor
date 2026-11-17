"""NFL drive-level research engine: no future leaks and actual football conservation."""
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from ai2_nfl_pbp_drives import (
    SOURCE, PBP_COLUMNS, pbp_to_drives,
    pregame_possession_profiles, check_profile_freshness,
)
from ai2_nfl_possession_engine import simulate_possession_worlds
from dfs_stats_engine import simulate_stat_driven_dfs
from test_ai2_nfl_team_pace_scoring import sim_fixture


def synthetic_pbp():
    """Two complete games per week, drive winners vary by team and field."""
    rows = []
    for wk in range(1, 11):
        for gameidx, sides in enumerate((("AAA","BBB"),("CCC","DDD"))):
            game=f"2025_{wk:02d}_G{gameidx}"
            for drive in range(1, 25):
                off=sides[(drive-1) % 2]
                deff=sides[drive % 2]
                # BBB concedes more pass TDs than AAA; alternate all outcomes.
                if drive % 8 in (0,3):
                    outcome="pass_td" if off in ("AAA","CCC") else "other"
                elif drive % 8 == 1:
                    outcome="rush_td"
                elif drive % 8 in (2,6):
                    outcome="field_goal"
                elif drive % 8 == 4:
                    outcome="turnover"
                else:
                    outcome="other"
                for play in range(4):
                    final=(play==3)
                    fields=dict(
                        game_id=game, season=2025, week=wk,
                        season_type="REG",
                        play_id=drive*10+play,
                        posteam=off, defteam=deff,
                        fixed_drive=drive,
                        play_type="pass" if play in (0,1,3) else "run",
                        pass_attempt=int(play in (0,1,3)),
                        rush_attempt=int(play==2),
                        sack=0,
                        pass_touchdown=int(final and outcome=="pass_td"),
                        rush_touchdown=int(final and outcome=="rush_td"),
                        interception=int(final and outcome=="turnover"),
                        fumble_lost=0,
                        field_goal_result=(
                            "made" if final and outcome=="field_goal" else ""),
                        yardline_100=float(78-(drive%4)*15),
                        drive_time_of_possession="02:24",
                    )
                    rows.append(fields)
    return pd.DataFrame(rows, columns=PBP_COLUMNS)


def fixture():
    drives=pbp_to_drives(synthetic_pbp())
    profile=pregame_possession_profiles(drives,2026,9)
    return drives,profile


class PosessionHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.drives,cls.profile=fixture()

    def test_drive_extractor_uses_one_row_per_actual_possession(self):
        self.assertEqual(len(self.drives),480)
        self.assertTrue(self.drives["plays"].eq(4).all())
        self.assertSetEqual(set(self.drives["outcome"]),
                            {"pass_td","rush_td","field_goal","turnover","other"})
        self.assertEqual(self.profile["source"],SOURCE)

    def test_future_season_or_week_cannot_look_back_into_target(self):
        base=self.profile
        new=self.drives.copy()
        new["season"]=2026
        new["week"]=9
        new["outcome"]="pass_td"
        appended=pd.concat([self.drives,new],ignore_index=True)
        after=pregame_possession_profiles(appended,2026,9)
        self.assertEqual(base,after)
        with self.assertRaisesRegex(ValueError,"wrong target|Future week"):
            check_profile_freshness(self.profile,2025,4)

    def test_provenance_and_bad_probabilities_fail_closed(self):
        for mutation in ("source","last_observed_week","league_probs"):
            bad=deepcopy(self.profile)
            if mutation=="source":
                bad["source"]="gambling projections"
            elif mutation=="last_observed_week":
                bad["last_observed_season"]=2026
                bad["last_observed_week"]=9
            else:
                bad["league_probs"]=[.9,.8,0.,0.,0.]
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                check_profile_freshness(bad,2026,9)

    def test_shared_clock_scoring_and_team_play_conservation(self):
        worlds=simulate_possession_worlds(
            "AAA@BBB",self.profile,season=2026,week=9,
            n_sims=125,seed=44)
        for team in ("AAA","BBB"):
            w=worlds[team]
            self.assertTrue(np.all(w["plays"] >= w["dropbacks"]))
            self.assertTrue(np.all(w["drives"] > 0))
            self.assertTrue(np.all(w["plays"] > 0))
            np.testing.assert_array_equal(
                w["points"],7*(w["pass_tds"]+w["rush_tds"])+3*w["field_goals"])
            np.testing.assert_array_equal(
                w["drives"],
                sum(w["outcomes"][key] for key in w["outcomes"]))
        # Two offenses share 60 minutes, not two independent game clocks.
        self.assertTrue(np.all(worlds["AAA"]["drives"]+
                               worlds["BBB"]["drives"] <=36))
        again=simulate_possession_worlds(
            "AAA@BBB",self.profile,season=2026,week=9,
            n_sims=125,seed=44)
        np.testing.assert_array_equal(
            worlds["AAA"]["points"],again["AAA"]["points"])

    def test_opponent_strength_changes_drives_before_fantasy_points(self):
        profile=deepcopy(self.profile)
        altered=deepcopy(self.profile)
        # Structural sensitivity test only, not an invented empirical finding.
        altered["teams"]["BBB"]["allowed"]=[.35,.20,.15,.10,.20]
        old=simulate_possession_worlds(
            "AAA@BBB",profile,season=2026,week=9,
            n_sims=400,seed=33,strength=1.)
        new=simulate_possession_worlds(
            "AAA@BBB",altered,season=2026,week=9,
            n_sims=400,seed=33,strength=1.)
        self.assertFalse(np.array_equal(
            old["AAA"]["points"],new["AAA"]["points"]))

    def test_stat_scoring_obeys_possession_worlds_with_real_player_opportunity(self):
        pool,rates=sim_fixture()
        _,scores,stats=simulate_stat_driven_dfs(
            pool,rates,seed=70,n_sims=150,allow_role_proxies=False,
            possession_profiles=self.profile,
            possession_strength=1.,
            possession_target_season=2026,possession_target_week=9)
        self.assertEqual(scores.shape,(150,len(pool)))
        self.assertIn("__POSSESSION_WORLD__|AAA@BBB",stats)
        for team in ("AAA","BBB"):
            w=stats["__POSSESSION_TEAM__|"+team]
            p=stats["__TEAM_PLAYS__|"+team]
            scoring=stats["__TEAM_SCORING__|"+team]
            np.testing.assert_array_equal(
                w["simulated_plays"],p["simulated_plays"])
            np.testing.assert_array_equal(
                w["simulated_dropbacks"],p["simulated_dropbacks"])
            np.testing.assert_array_equal(
                w["simulated_pass_tds"],stats["Q"+team]["passing_tds"])
            np.testing.assert_array_equal(
                w["simulated_rush_tds"],
                scoring["simulated_offensive_tds"]-
                stats["Q"+team]["passing_tds"])
            np.testing.assert_array_equal(
                w["simulated_field_goals"],scoring["simulated_field_goals"])
            np.testing.assert_allclose(
                stats["Q"+team]["passing_yards"],
                sum(stats[prefix+team]["receiving_yards"] for prefix in
                    ("R","W","T"))+
                stats["__OTHER_RECEIVER__|"+team]["receiving_yards"],
                atol=.001)

    def test_inactive_possession_flag_changes_nothing(self):
        pool,rates=sim_fixture()
        _,baseline,stats=simulate_stat_driven_dfs(
            pool,rates,seed=70,n_sims=110,allow_role_proxies=False)
        _,zero,untouched=simulate_stat_driven_dfs(
            pool,rates,seed=70,n_sims=110,allow_role_proxies=False,
            possession_profiles=self.profile,possession_strength=0.)
        np.testing.assert_array_equal(baseline,zero)
        self.assertNotIn("__POSSESSION_WORLD__|AAA@BBB",untouched)


if __name__=="__main__":
    unittest.main()
