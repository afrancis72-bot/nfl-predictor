"""No-network test: observed injury evidence feeds one-pass game simulation."""
import unittest
import pandas as pd
from ai2_nfl_model_pool import create_observed_model_pool

class ModelPoolTests(unittest.TestCase):
    def test_out_starter_removed_but_backup_evidence_kept(self):
        game = "DAL@NYG 11/01/2026 01:00PM ET"
        data = [
            ("1","Dallas QB","QB","DAL","",20.0,"matched"),
            ("2","New York QB","QB","NYG","",20.0,"matched"),
            ("3","Backup RB","RB","DAL","",8.0,"matched"),
            ("4","Starter RB","RB","DAL","OUT",15.0,"matched"),
            ("5","Receiver WR","WR","DAL","",10.0,"matched"),
            ("6","Receiver TE","TE","NYG","",7.0,"matched"),
            ("7","DAL","DST","DAL","",float("nan"),"no unique historical match"),
            ("8","NYG","DST","NYG","",float("nan"),"no unique historical match"),
        ]
        base = pd.DataFrame([
            {"ID":i,"Name":n,"Position":p,"Team":t,
             "current_status":status,"Game Info":game,"Salary":6000,
             "historical_partial_dk":pts,"match_status":match}
            for i,n,p,t,status,pts,match in data
        ])
        raw = pd.DataFrame([
            {"player":n,"position":p,"team":t,"pass_attempts_pg":passes,
             "carries_pg":carries,"targets_pg":targets,
             "games_played":5,"latest_observed_season":2026,
             "latest_observed_week":7}
            for n,p,t,passes,carries,targets in [
                ("Dallas QB","QB","DAL",30,2,0),
                ("New York QB","QB","NYG",31,2,0),
                ("Backup RB","RB","DAL",0,6,1),
                ("Starter RB","RB","DAL",0,18,3),
                ("Receiver WR","WR","DAL",0,0,6),
                ("Receiver TE","TE","NYG",0,0,4),
            ]
        ])
        adjusted = raw.copy()
        adjusted.loc[adjusted.player.eq("Backup RB"),"carries_pg"] = 9.0
        intel = {"players":base,"raw_rates":raw,"adjusted_rates":adjusted,
                 "target_season":2026,"target_week":8,
                 "injury_meta":{"source":"test"},"source":"test"}
        pool, unmodified_rates, outs, audit = create_observed_model_pool(intel)
        self.assertNotIn("Starter RB", pool.Name.tolist())
        self.assertIn("Backup RB", pool.Name.tolist())
        self.assertEqual(outs,[{"player":"Starter RB","team":"DAL"}])
        self.assertEqual(float(unmodified_rates.loc[unmodified_rates.player.eq("Backup RB"),"carries_pg"].iloc[0]),6.0)
        self.assertEqual(pool.loc[pool.Name.eq("Backup RB"),"projected_touches"].iloc[0],10.0)
        self.assertEqual(audit["complete_games"],1)

if __name__=="__main__":
    unittest.main()
