"""Offline PGA unified integration checks; synthetic test fixtures are not predictions."""
import unittest
import pandas as pd
import numpy as np

from ai2_pga_oneclick import (
    parse_dk, normalize_otis, market_baseline, run_tournament,
    dk_roster_ids, normalize_groupings_upload,
    evaluate_groups,
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
