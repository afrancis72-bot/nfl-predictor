"""Role-specific defensive adjustments must change actual stat rates only
from prior-week player/defense evidence, never from the target-game result.
"""
import unittest
from pathlib import Path
import pandas as pd
from ai2_nfl_role_defense_engine import (
    SOURCE, role_channel_profiles, role_channel_observations,
    validated_role_lookup, apply_role_defense_to_rates,
)


def history():
    rows=[]
    players=[
        ("Top Wideout","WR",10,8,94),
        ("Second Wideout","WR",6,4,56),
        ("Third Wideout","WR",2,1,18),
        ("Starting Quarterback","QB",34,0,0),
        ("Lead Runner","RB",15,2,15),
        ("Second Runner","RB",8,1,8),
        ("Starting Tight End","TE",5,3,43),
        ("Backup Tight End","TE",3,2,22),
    ]
    for week in range(1,12):
        for name,pos,volume,catch,yd in players:
            targets=volume if pos in ("WR","TE") else catch if pos=="RB" else 0
            # Completed MIA defensive games: WR3 has more volume compared
            # with his own earlier pregame average, not with an elite WR1.
            if name=="Third Wideout" and week>=4 and week<=10:
                targets=5;catch=3;yd=46
            if name=="Third Wideout" and week==11:
                targets=30;catch=24;yd=333  # MUST NOT LEAK into W11 forecast
            rows.append(dict(
                season=2025,week=week,season_type="REG",
                player_display_name=name,player_id=name,
                recent_team="KC",position=pos,
                opponent_team=("MIA" if 4<=week<=10 else "NYJ"),
                passing_yards=254 if pos=="QB" else 0,
                passing_tds=2 if pos=="QB" else 0,
                passing_interceptions=0,
                attempts=volume if pos=="QB" else 0,
                carries=volume if pos=="RB" else 0,
                rushing_yards=(volume*4 if pos=="RB" else 0),
                rushing_tds=0,targets=targets,
                receptions=catch,receiving_yards=yd,receiving_tds=0))
    return pd.DataFrame(rows)


def current_slate_roles(status=""):
    return [
        {"Player":"Third Wideout","Team":"KC","Opponent":"MIA",
         "Position":"WR","Role":"WR3","Slate status":status,
         "Player history games":10,
         "Role evidence":"Pregame within-team usage rank; not official lineup or coverage"}
    ]


def rate_frame():
    return pd.DataFrame([{
        "player":"Third Wideout","team":"KC","position":"WR",
        "games_played":8,"source":"nflverse pregame observed weekly stats",
        "targets_pg":3.0,"receptions_pg":1.9,"receiving_yards_pg":27.,
        "receiving_tds_pg":.1,"carries_pg":0.,
        "rushing_yards_pg":0.,"rushing_tds_pg":0.,
    }])


class RoleDefenseStatEngineTests(unittest.TestCase):
    def test_previous_defensive_wr3_volume_adjustment_is_positive(self):
        p,obs=role_channel_profiles(history(),2025,11)
        wr3=p[(p.defense=="MIA")&(p.role=="WR3")&
              (p.channel=="targets")]
        self.assertEqual(len(wr3),1)
        self.assertGreater(int(wr3.iloc[0]["observations"]),4)
        self.assertGreater(float(wr3.iloc[0]["shrunk_relative_pct"]),0)
        self.assertEqual(wr3.iloc[0]["source"],SOURCE)
        self.assertEqual(int(wr3.iloc[0]["target_week"]),11)
        self.assertTrue((obs.week<11).all())
        self.assertIn("QB1",set(p.role))
        self.assertIn("RB1",set(p.role))
        self.assertIn("TE1",set(p.role))

    def test_target_game_stats_never_change_pregame_factor(self):
        a=history()
        x,_=role_channel_profiles(a,2025,11)
        b=a.copy()
        b.loc[(b.week==11)&(b.player_display_name=="Third Wideout"),
              ["targets","receptions","receiving_yards"]]=[99,99,999]
        y,_=role_channel_profiles(b,2025,11)
        pd.testing.assert_frame_equal(
            x.sort_values(["defense","role","channel"]).reset_index(drop=True),
            y.sort_values(["defense","role","channel"]).reset_index(drop=True))

    def test_role_factor_changes_real_rates_and_maintains_conversion(self):
        p,_=role_channel_profiles(history(),2025,11)
        base=rate_frame()
        changed,ledger=apply_role_defense_to_rates(
            base,current_slate_roles(),p,season=2025,week=11,strength=1.)
        self.assertEqual(len(ledger),1)
        self.assertEqual(ledger[0]["pregame_role"],"WR3")
        self.assertIn("targets",ledger[0]["factor_by_stat_channel"])
        self.assertGreater(float(changed.iloc[0].targets_pg),
                           float(base.iloc[0].targets_pg))
        self.assertGreater(float(changed.iloc[0].receptions_pg),0.)
        self.assertNotIn("dk_fantasy_point_bonus",changed.columns)
        # Audit proof must expose actual pregame stat inputs that changed.
        changes=ledger[0]["stat_rate_changes"]
        self.assertIn("targets_pg",changes)
        for field,record in changes.items():
            self.assertAlmostEqual(record["before"],float(base.iloc[0][field]),places=5)
            self.assertAlmostEqual(record["after"],float(changed.iloc[0][field]),places=5)
            self.assertAlmostEqual(record["delta"],
                                   record["after"]-record["before"],places=5)
        self.assertNotIn("passing_tds_pg",changes)
        self.assertEqual(float(base.iloc[0].targets_pg),3.0)

    def test_out_and_questionable_are_not_promoted_as_active_wr3(self):
        p,_=role_channel_profiles(history(),2025,11)
        for s in ("OUT","IR","QUESTIONABLE","Q","GTD"):
            z,ledger=apply_role_defense_to_rates(
                rate_frame(),current_slate_roles(s),p,
                season=2025,week=11,strength=1.)
            self.assertFalse(ledger)
            self.assertEqual(float(z.iloc[0].targets_pg),3.)

    def test_neutral_strength_no_evidence_no_mutation(self):
        base=rate_frame()
        z,ledger=apply_role_defense_to_rates(
            base,None,None,season=2025,week=11,strength=0.)
        self.assertEqual(len(ledger),0)
        pd.testing.assert_frame_equal(z,base)

    def test_forged_wrong_week_and_low_sample_rejected(self):
        p,_=role_channel_profiles(history(),2025,11)
        with self.assertRaisesRegex(ValueError,"wrong-week"):
            validated_role_lookup(p,season=2025,week=12,strength=.5)
        forged=p.copy()
        forged["source"]="unsourced reddit matchup power ranking"
        with self.assertRaisesRegex(ValueError,"Unverified"):
            validated_role_lookup(forged,season=2025,week=11,strength=.5)
        bogus=p.copy()
        bogus.loc[bogus.index[0],"observations"]=1
        with self.assertRaisesRegex(ValueError,"Insufficient"):
            validated_role_lookup(bogus,season=2025,week=11,strength=.5)

    def test_no_wrong_teams_or_role_guesses(self):
        p,_=role_channel_profiles(history(),2025,11)
        fake=current_slate_roles()
        fake[0]["Team"]="MIA"
        z,ledger=apply_role_defense_to_rates(
            rate_frame(),fake,p,season=2025,week=11,strength=1.)
        self.assertEqual(len(ledger),0)
        fake=current_slate_roles()
        fake[0]["Role evidence"]="I assume WR3 because $4100 salary"
        z,ledger=apply_role_defense_to_rates(
            rate_frame(),fake,p,season=2025,week=11,strength=1.)
        self.assertEqual(len(ledger),0)

    def test_unified_scoring_path_wired_but_bentley_release_remains_locked(self):
        root=Path(__file__).resolve().parents[1]
        one=(root/"ai2_nfl_oneclick.py").read_text()
        adapter=(root/"ai2_nfl_simulation_adapter.py").read_text()
        ui=(root/"unified_app.py").read_text()
        self.assertIn('role_defense_strength = 0.5',one)
        self.assertIn('role_defense_profiles=research.get("opponent_role_stat_profiles")',one)
        self.assertIn('role_rates,role_ledger=apply_role_defense_to_rates',adapter)
        self.assertIn('active, role_rates, n_sims=',adapter)
        self.assertIn('stat_draws["__ROLE_DEFENSE_STAT_CHANNELS__"]',adapter)
        self.assertIn('disabled=not release_ready',ui)
        self.assertIn('assert_nfl_classic_ready_for_entries()',ui)
        self.assertIn('independent_simulator_accuracy_validated":False',adapter)


if __name__=="__main__":
    unittest.main()
