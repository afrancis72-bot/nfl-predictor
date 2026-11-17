"""Synthetic, no-network checks for pregame player red-zone/goal-line evidence."""
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from ai2_nfl_redzone_usage import (
    SOURCE, pregame_redzone_profiles, validated_redzone_lookup, factor_for,
)
from dfs_stats_engine import simulate_stat_driven_dfs
from test_ai2_nfl_team_pace_scoring import sim_fixture


def inputs():
    plays=[]
    weekly=[]
    for team in ("AAA","BBB"):
        for player,pos,gsis in (
            ("Q"+team,"QB",team+"-QB"),
            ("R"+team,"RB",team+"-RB"),
            ("W"+team,"WR",team+"-WR"),
            ("T"+team,"TE",team+"-TE"),
        ):
            for week in range(1,9):
                weekly.append(dict(season=2025,week=week,season_type="REG",
                    player_name=player,player_display_name=player,
                    player_id=gsis,recent_team=team,position=pos))
        for week in range(1,9):
            game=f"2025_{week:02d}_{team}"
            for play in range(1,31):
                is_pass=play<=20
                is_goal=play<=9 if is_pass else play<=24
                if is_pass:
                    receiver=team+("-WR" if play<=9 or play>=18 else "-TE")
                    rusher=np.nan
                else:
                    receiver=np.nan
                    rusher=team+("-QB" if play in (21,22,23) else "-RB")
                plays.append(dict(
                    season=2025,week=week,season_type="REG",
                    game_id=game,posteam=team,
                    yardline_100=(8 if is_goal else 50),
                    pass_attempt=int(is_pass),rush_attempt=int(not is_pass),
                    two_point_attempt=0,
                    receiver_player_id=receiver,rusher_player_id=rusher))
    return pd.DataFrame(plays),pd.DataFrame(weekly)


class RedZoneUsageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pbp,cls.weekly=inputs()
        cls.profiles=pregame_redzone_profiles(cls.pbp,cls.weekly,2026,9)

    def test_source_and_player_role_factors(self):
        p=self.profiles
        self.assertEqual(p["source"],SOURCE)
        factors=validated_redzone_lookup(p,season=2026,week=9,strength=1.)
        self.assertGreater(factor_for(factors,"AAA","WAAA","receiving"),1.)
        self.assertGreater(factor_for(factors,"AAA","QAAA","rushing"),1.)
        self.assertLess(factor_for(factors,"AAA","RAAA","rushing"),1.)
        self.assertEqual(factor_for(factors,"AAA","TAAA","rushing"),1.)
        self.assertEqual(factor_for(factors,"BBB","unmatched","receiving"),1.)

    def test_no_future_week_data_can_change_pregame_roles(self):
        bad=self.pbp.copy()
        bad["season"]=2026
        bad["week"]=9
        bad["receiver_player_id"]="BBB-TE"
        bad["rusher_player_id"]="BBB-RB"
        changed=pregame_redzone_profiles(
            pd.concat([self.pbp,bad],ignore_index=True),
            self.weekly,2026,9)
        self.assertEqual(self.profiles,changed)

    def test_unknown_player_id_stays_unattributed(self):
        altered=self.pbp.copy()
        altered.loc[
            altered["rusher_player_id"].notna(),"rusher_player_id"
        ]="NONEXISTENT"
        p=pregame_redzone_profiles(altered,self.weekly,2026,9)
        lookup=validated_redzone_lookup(p,season=2026,week=9,strength=.5)
        self.assertEqual(factor_for(lookup,"AAA","QAAA","rushing"),1.)

    def test_bad_provenance_and_strength_rejected(self):
        cases=[("source","invented"),
               ("target_week",10),
               ("last_observed_season",2027)]
        for field,value in cases:
            p=deepcopy(self.profiles)
            p[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validated_redzone_lookup(p,season=2026,week=9,strength=1.)
        with self.assertRaises(ValueError):
            validated_redzone_lookup(self.profiles,season=2026,week=9,
                                     strength=1.01)

    def test_zero_strength_preserves_original_engine_exactly(self):
        pool,rates=sim_fixture()
        _,neutral,_=simulate_stat_driven_dfs(
            pool,rates,n_sims=240,seed=101,allow_role_proxies=False)
        _,unchanged,stats=simulate_stat_driven_dfs(
            pool,rates,n_sims=240,seed=101,allow_role_proxies=False,
            redzone_profiles=self.profiles,redzone_strength=0.)
        np.testing.assert_array_equal(neutral,unchanged)
        self.assertNotIn("__REDZONE_OPPORTUNITY__|AAA",stats)

    def test_only_touchdown_recipients_change_not_team_scoring(self):
        pool,rates=sim_fixture()
        _,neutral,base=simulate_stat_driven_dfs(
            pool,rates,n_sims=360,seed=53,allow_role_proxies=False)
        _,adjusted,exp=simulate_stat_driven_dfs(
            pool,rates,n_sims=360,seed=53,allow_role_proxies=False,
            redzone_profiles=self.profiles,redzone_strength=1.,
            redzone_target_season=2026,redzone_target_week=9)
        self.assertFalse(np.array_equal(neutral,adjusted))
        self.assertGreater(
            exp["__REDZONE_OPPORTUNITY__|AAA"]["qb_goal_line_factor"],1.)
        for team in ("AAA","BBB"):
            for key in ("simulated_offensive_tds","simulated_field_goals",
                        "simulated_offensive_points"):
                np.testing.assert_array_equal(
                    base["__TEAM_SCORING__|"+team][key],
                    exp["__TEAM_SCORING__|"+team][key])
            for name in ("Q","R","W","T"):
                p=name+team
                for field in ("carries","rushing_yards","receptions",
                              "targets","receiving_yards"):
                    if field in base[p]:
                        np.testing.assert_array_equal(
                            base[p][field],exp[p][field])
            np.testing.assert_array_equal(
                exp["Q"+team]["passing_tds"],
                base["Q"+team]["passing_tds"])

    def test_cannot_stack_with_old_td_conversion_method(self):
        pool,rates=sim_fixture()
        with self.assertRaisesRegex(ValueError,"cannot stack"):
            simulate_stat_driven_dfs(
                pool,rates,n_sims=100,allow_role_proxies=False,
                redzone_profiles=self.profiles,redzone_strength=.5,
                redzone_target_season=2026,redzone_target_week=9,
                receiving_td_weight=.2)


if __name__=="__main__":
    unittest.main()
