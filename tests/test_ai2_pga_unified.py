"""Offline PGA unified integration checks; synthetic test fixtures are not predictions."""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch

from round_parlay import (simulate_groups, build_six_leg_tickets,
                          parse_groupings_text, _api_groupings)

from ai2_pga_oneclick import (
    parse_dk, normalize_otis, market_baseline, run_tournament,
    dk_roster_ids, normalize_groupings_upload,
    evaluate_groups, normalize_independent_stats,
    normalize_rrg_course_fit, apply_rrg_course_fit, group_ratings_source,
    bookmaker_filtered_tickets,
)

class TestPGAOneclick(unittest.TestCase):
    def slate(self, n=24):
        return pd.DataFrame({
            "Name":[f"Golfer {i:02d}" for i in range(n)],
            "ID":[str(50000+i) for i in range(n)],
            "Salary":[7800+(i%8)*200 for i in range(n)],
            "Status":["Active"]*n,
        })
    def test_current_dk_field_and_real_id_parsing(self):
        d=parse_dk(self.slate())
        self.assertEqual(len(d),24)
        self.assertIn("dk_id",d)
        bad=self.slate()
        bad.loc[0,"Name"]=bad.loc[1,"Name"]
        with self.assertRaises(ValueError):
            parse_dk(bad)
        bad=self.slate()
        bad["Salary"]=bad["Salary"].astype(object)
        bad.loc[1,"Salary"]="?"
        with self.assertRaises(ValueError):
            parse_dk(bad)

    def test_otis_field_match_and_rank_not_used_for_skill(self):
        dk=parse_dk(self.slate())
        otis=pd.DataFrame({
            "Player":dk["player"],"True Skill":np.linspace(30,80,len(dk)),
            "Course Fit":np.linspace(45,75,len(dk)),
            "Form":np.linspace(48,68,len(dk)),
            "Rank":range(1,len(dk)+1),"Model":range(1,len(dk)+1),
        })
        normalized,match=normalize_otis(otis,dk)
        self.assertEqual(match,len(dk))
        self.assertIn("otis_model_audit",normalized)
        self.assertNotIn("otis_model",normalized)
        with self.assertRaises(ValueError):
            normalize_otis(otis.head(4),dk)

    def test_independent_measured_strokes_gained_is_primary(self):
        dk=parse_dk(self.slate())
        source=pd.DataFrame({
            "Player":dk["player"],
            "SG: Total":np.linspace(-1.1,1.5,len(dk)),
            "SG: APP":np.linspace(-.8,1.1,len(dk)),
            "SG: OTT":np.linspace(-.5,.65,len(dk)),
            "SG: PUTT":np.linspace(.25,-.2,len(dk)),
            "Measured Rounds":[48]*len(dk),
        })
        normalized,covered=normalize_independent_stats(source,dk)
        self.assertEqual(covered,len(dk))
        self.assertIn("sg_approach",normalized)
        result=run_tournament(self.slate(),tournament="Observed SG Acceptance",
            course_par=72,no_cut=True,sims=1000,seed=55,lineup_count=2,
            max_exposure=.5,salary_floor=46000,min_unique=2,
            independent_raw=source)
        self.assertEqual(result["metadata"]["independent_sg_match"],len(dk))
        self.assertIn("Independent observed",result["metadata"]["source_mode"])
        self.assertEqual(len(result["lineups"]),12)

    def test_rrg_course_fit_categories_are_independently_weighted_and_audited(self):
        dk=parse_dk(self.slate())
        n=len(dk)
        # RRG Player Fit CSV examples: independently computed OTT/APP/ARG/PUTT/SCORING
        # course correlation effects, and their own preweighted FIT composite.
        rrg=pd.DataFrame({
            "Player":dk["player"][::-1].to_numpy(),
            "OTT":np.linspace(-.5,.5,n),
            "APP":np.linspace(-1.,1.,n),
            "ARG":np.linspace(.2,-.2,n),
            "PUTT":np.linspace(-.4,.4,n),
            "SCORING":np.linspace(-.7,.7,n),
            "FIT":np.linspace(40.,90.,n),
            "Rank":list(range(1,n+1)),
        })
        matched,coverage=normalize_rrg_course_fit(rrg,dk)
        self.assertEqual(coverage["matched"],n)
        self.assertEqual(coverage["components"],n)
        self.assertIn("rrg_fit_rank_audit",matched)
        base=market_baseline(dk)
        tilt=apply_rrg_course_fit(base,matched)
        self.assertGreater(tilt["rrg_fit_adjustment_strokes"].abs().max(),.01)
        self.assertLessEqual(tilt["rrg_fit_adjustment_strokes"].abs().max(),.22)
        self.assertTrue((tilt["rrg_fit_evidence"]==
                         "Our weighting of RRG skill-fit categories").all())
        self.assertTrue(np.allclose(tilt["expected_round"]-base["expected_round"],
                            tilt["rrg_fit_adjustment_strokes"]))
        self.assertEqual(tilt["rrg_fit_rank_audit"].nunique(),n)
        # A radically changed RRG FIT value should not change our independent
        # component-weighted evaluation when the component columns are present.
        changed=rrg.copy()
        changed["FIT"]=10000.-changed["FIT"]
        matched2,_=normalize_rrg_course_fit(changed,dk)
        tilt2=apply_rrg_course_fit(base,matched2)
        np.testing.assert_allclose(tilt["rrg_fit_adjustment_strokes"],
                                   tilt2["rrg_fit_adjustment_strokes"])
        full=run_tournament(self.slate(),tournament="RRG Course Fit Acceptance",
            course_par=72,no_cut=True,sims=1000,seed=42,lineup_count=2,
            max_exposure=.5,salary_floor=46000,min_unique=2,
            course_fit_raw=rrg)
        self.assertEqual(full["metadata"]["rrg_fit_match"],n)
        self.assertEqual(full["metadata"]["rrg_fit_components"],n)
        self.assertIn("rrg_fit_adjustment_strokes",full["projections"])
        self.assertEqual(len(full["summary"]),2)
        ratings,source=group_ratings_source(full)
        self.assertIn("RRG",source)
        self.assertGreater(float(ratings["otis_course_fit"].std()),0.)

    def test_rrg_fit_only_fallback_and_bad_course_export(self):
        dk=parse_dk(self.slate())
        n=len(dk)
        base=market_baseline(dk)
        fit_only=pd.DataFrame({
            "Golfer":dk["player"],"FIT":np.linspace(-1.,1.,n),
            "Course Fit Rank":list(range(1,n+1)),
        })
        normalized,meta=normalize_rrg_course_fit(fit_only,dk)
        self.assertEqual(meta["overall_only"],n)
        adjusted=apply_rrg_course_fit(base,normalized)
        self.assertTrue((adjusted["rrg_fit_evidence"]==
                         "RRG overall FIT only (reduced weight)").all())
        one_category=pd.DataFrame({"Player":dk["player"],
                                  "APP":np.linspace(-1.,1.,n)})
        normalized2,_=normalize_rrg_course_fit(one_category,dk)
        category_impact=apply_rrg_course_fit(base,normalized2)
        self.assertGreater(category_impact["rrg_fit_adjustment_strokes"].abs().max(),
                           adjusted["rrg_fit_adjustment_strokes"].abs().max())
        with self.assertRaises(ValueError):
            normalize_rrg_course_fit(fit_only[["Golfer","Course Fit Rank"]],dk)
        with self.assertRaises(ValueError):
            normalize_rrg_course_fit(fit_only.head(4),dk)
        with self.assertRaises(ValueError):
            normalize_rrg_course_fit(
                pd.concat([fit_only,fit_only.iloc[:1]],ignore_index=True),dk)

    def test_one_button_sim_projections_align_seed_and_legal_dk_six(self):
        d=self.slate()
        args=dict(tournament="Synthetic Acceptance Tournament",course_par=72,
                no_cut=False,cut_size=12,sims=1000,
                lineup_count=2,max_exposure=.5,salary_floor=46000,
                min_unique=2,strategy="Balanced / Single Entry")
        a=run_tournament(d,seed=13,**args)
        b=run_tournament(d,seed=13,**args)
        c=run_tournament(d,seed=34,**args)
        self.assertEqual(len(a["summary"]),2)
        self.assertEqual(len(a["lineups"]),12)
        self.assertTrue((a["summary"]["salary"]<=50000).all())
        self.assertTrue((a["summary"]["salary"]>=46000).all())
        self.assertTrue((a["exposure"]["exposure_pct"]<=50+1e-5).all())
        np.testing.assert_array_equal(a["raw_draws"],b["raw_draws"])
        self.assertFalse(np.array_equal(a["raw_draws"],c["raw_draws"]))
        np.testing.assert_allclose(a["raw_draws"].mean(0),
                                   a["projections"]["v11_mean_dk"].to_numpy(),
                                   atol=.02)
        self.assertEqual(
            dk_roster_ids(a["lineups"],a["projections"]).decode("utf-8-sig").splitlines()[0],
            "G,G,G,G,G,G"
        )
        self.assertEqual(a["metadata"]["otis_match"],0)

    def test_mixed_2_ball_3_ball_group_csv_and_invalid_rosters(self):
        names=[f"Golfer {i:02d}" for i in range(20)]
        source=pd.DataFrame([
            {"tee_time":"9:00","golfer1":names[0],"golfer2":names[1],
             "golfer3":np.nan},
            {"tee_time":"9:10","golfer1":names[2],"golfer2":names[3],
             "golfer3":names[4]},
            {"tee_time":"9:20","golfer1":names[5],"golfer2":names[5],
             "golfer3":""},
            {"tee_time":"9:30","golfer1":names[6],"golfer2":"","golfer3":""},
        ])
        g=normalize_groupings_upload(source,round_no=3)
        self.assertEqual(g["group_size"].tolist(),[2,3])
        self.assertEqual(g["player3"].tolist()[0],"")
        self.assertEqual(g["round"].tolist(),[3,3])
        self.assertEqual(len(g),2)

    def test_official_tee_time_feed_accepts_true_two_ball_and_three_ball(self):
        mock_rounds={"rounds":[
            {"roundInt":3,"groups":[
                {"groupNumber":12,"teeTime":1793466000000,
                 "players":[{"displayName":"Pair One"},{"displayName":"Pair Two"}]},
                {"groupNumber":13,"teeTime":1793466000000,
                 "players":[{"displayName":"Trio One"},{"displayName":"Trio Two"},
                            {"displayName":"Trio Three"}]},
                {"groupNumber":14,"teeTime":1793466000000,
                 "players":[{"displayName":"Solo"}]},
                {"groupNumber":15,"teeTime":1793466000000,
                 "players":[{"displayName":"Four One"},{"displayName":"Four Two"},
                            {"displayName":"Four Three"},{"displayName":"Four Four"}]},
            ]},
        ]}
        with patch("round_parlay._pga_graphql",return_value={"teeTimesCompressedV2":{"payload":"fake"}}):
            with patch("round_parlay._decompress_payload",return_value=mock_rounds):
                groups,tid,note=_api_groupings("Event",2026,3,tournament_id="R2026999")
        self.assertEqual(tid,"R2026999")
        self.assertEqual(groups["group_size"].tolist(),[2,3])
        self.assertEqual(groups["player3"].iloc[0],"")

    def test_weekend_head_to_head_ties_are_three_way_losses_or_two_way_pushes(self):
        names=[f"Golfer {i:02d}" for i in range(24)]
        groups=normalize_groupings_upload(pd.DataFrame([
            {"player1":names[0],"player2":names[1],"tee_time":"9:00"},
            {"player1":names[2],"player2":names[3],"tee_time":"9:10"},
            {"player1":names[4],"player2":names[5],"tee_time":"9:20"},
            {"player1":names[6],"player2":names[7],"tee_time":"9:30"},
            {"player1":names[8],"player2":names[9],"tee_time":"9:40"},
            {"player1":names[10],"player2":names[11],"tee_time":"9:50"},
        ]),3)
        ratings=pd.DataFrame({"key":[x.lower() for x in names],
                              "round_rating":[50.]*len(names)})
        # Two-player, three-outcome (Tie selectable, golfer bet loses):
        threeway,_=simulate_groups(groups,ratings,n_sims=5000,seed=42,
                                   two_ball_tie_rule="loss")
        golfers=threeway[threeway.status.eq("OK")]
        ties=threeway[threeway.status.eq("TIE_OUTCOME")]
        self.assertEqual(len(golfers),12)
        self.assertEqual(len(ties),6)
        self.assertTrue((golfers["push_pct"]==0).all())
        self.assertTrue((golfers["tie_outcome_pct"]>0).all())
        for group, g in threeway.groupby("group"):
            picks=g[g.status.eq("OK")]
            tie=g[g.status.eq("TIE_OUTCOME")].iloc[0]
            self.assertAlmostEqual(float(picks["win_pct"].sum()+tie.win_pct),100,places=6)
            self.assertAlmostEqual(float(picks.iloc[0].loss_pct),
                100-float(picks.iloc[0].win_pct),places=6)
        # Two-way: tie pushes either golfer, rather than losing the bet.
        twoway,_=simulate_groups(groups,ratings,n_sims=5000,seed=42,
                                 two_ball_tie_rule="push")
        golfer2=twoway[twoway.status.eq("OK")]
        self.assertFalse(twoway.status.eq("TIE_OUTCOME").any())
        self.assertTrue((golfer2["push_pct"]>0).all())
        self.assertAlmostEqual(float(golfer2.iloc[0].push_pct),
                               float(golfer2.iloc[0].tie_outcome_pct))
        self.assertAlmostEqual(float(golfer2.iloc[0].win_pct+
                           golfer2.iloc[0].loss_pct+golfer2.iloc[0].push_pct),100)
        # Weekend six-leg parlay must be possible from six real two-golfer pairs,
        # and tie-win is displayed separately, never auto-forced as a parlay leg.
        summary,legs=build_six_leg_tickets(threeway,ticket_count=1)
        self.assertEqual(len(legs),6)
        self.assertTrue((legs["matchup_type"]=="2-ball").all())
        self.assertEqual(int(summary["two_ball_legs"].iloc[0]),6)
        self.assertFalse(legs["player"].eq("Tie").any())
        self.assertTrue((legs["push_pct"]==0).all())
        self.assertAlmostEqual(float(summary["strict_6_of_6_pct"].iloc[0]),
            float(summary["all_legs_non_loss_pct"].iloc[0]),places=6)

    def test_mixed_six_leg_parlay_from_3_pairs_and_3_triples(self):
        names=[f"Golfer {i:02d}" for i in range(24)]
        groups=[]
        for i in range(3):
            groups.append({"player1":names[2*i],"player2":names[2*i+1],
                           "player3":"","tee_time":"9:00"})
        for i in range(3):
            groups.append({"player1":names[6+3*i],"player2":names[7+3*i],
                           "player3":names[8+3*i],"tee_time":"9:10"})
        groupings=normalize_groupings_upload(pd.DataFrame(groups),3)
        ratings=pd.DataFrame({"key":[x.lower() for x in names],
                              "round_rating":[50.]*len(names)})
        probs,_=simulate_groups(groupings,ratings,n_sims=5000,seed=55)
        summary,legs=build_six_leg_tickets(probs,ticket_count=1)
        self.assertEqual(len(legs),6)
        self.assertEqual(int(summary["two_ball_legs"].iloc[0]),3)
        self.assertEqual(set(legs["matchup_type"]),{"2-ball","3-ball"})
        self.assertEqual(int(legs["group"].nunique()),6)
        self.assertEqual(len(legs["player"].unique()),6)

    def test_only_bookmaker_listed_tee_groups_can_appear_in_six_leg_shortlist(self):
        names=[f"Golfer {i:02d}" for i in range(30)]
        raw=pd.DataFrame([
            {"player1":names[i*2],"player2":names[i*2+1],
             "tee_time":f"9:{i:02d}"}
            for i in range(10)
        ])
        groups=normalize_groupings_upload(raw,round_no=2)
        ratings=pd.DataFrame({"key":[n.lower() for n in names],
                              "round_rating":[50.]*len(names)})
        simulated,_=simulate_groups(groups,ratings,n_sims=1000,seed=42)
        # No sportsbook market list: no actionable six-leg ticket.
        p,s,l=bookmaker_filtered_tickets(simulated,groups,[],ticket_count=1)
        self.assertTrue(p.empty)
        self.assertTrue(s.empty)
        self.assertTrue(l.empty)
        # Five confirmed groups cannot form a six-leg ticket.
        p,s,l=bookmaker_filtered_tickets(simulated,groups,[1,2,3,4,5],ticket_count=1)
        self.assertEqual(p["group"].nunique(),5)
        self.assertTrue(s.empty)
        self.assertTrue(l.empty)
        # A restricted six-market list must NEVER include unlisted groups 3/8/9/10.
        ids=[1,2,4,5,6,7]
        p,s,l=bookmaker_filtered_tickets(simulated,groups,ids,ticket_count=1)
        self.assertEqual(set(p["group"]),set(ids))
        self.assertEqual(len(l),6)
        self.assertEqual(set(l["group"]),set(ids))
        self.assertNotIn(3,set(l["group"]))
        self.assertNotIn(8,set(l["group"]))
        with self.assertRaises(ValueError):
            bookmaker_filtered_tickets(simulated,groups,[1,99],ticket_count=1)

    def test_only_five_valid_matchups_never_force_six_leg_ticket(self):
        players=[f"Golfer {i:02d}" for i in range(10)]
        groups=normalize_groupings_upload(pd.DataFrame(
            [{"player1":players[2*i],"player2":players[2*i+1]}
             for i in range(5)]),round_no=3)
        ratings=pd.DataFrame({"key":[x.lower() for x in players],
                              "round_rating":[50.]*10})
        probs,_=simulate_groups(groups,ratings,n_sims=1000)
        summary,legs=build_six_leg_tickets(probs)
        self.assertTrue(summary.empty)
        self.assertTrue(legs.empty)

    def test_threeball_research_clearly_market_only(self):
        d=self.slate()
        model=run_tournament(d,tournament="Synthetic Tournament",course_par=72,
                no_cut=True,sims=1000,seed=21,lineup_count=2,
                max_exposure=.5,salary_floor=46000,min_unique=2)
        p=model["projections"]["player"].tolist()
        groups=pd.DataFrame([{"player1":p[i*3],"player2":p[i*3+1],
                               "player3":p[i*3+2],"tee_time":"09:00 AM"}
                             for i in range(6)])
        g=normalize_groupings_upload(groups,round_no=1)
        out=evaluate_groups(g,model,round_no=1,seed=21,sims=1000)
        self.assertEqual(len(out["probabilities"]),18)
        self.assertGreaterEqual(len(out["ticket_summary"]),1)
        self.assertIn("salary-derived",out["source"])
        with self.assertRaises(ValueError):
            evaluate_groups(g,model,round_no=2,seed=21,sims=1000,live=pd.DataFrame())

if __name__=="__main__":
    unittest.main()
