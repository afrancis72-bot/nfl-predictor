"""New two-game RB role detection must not be a blind one-week punt."""
import unittest
import pandas as pd
from ai2_nfl_pregame_consistency import build_pregame_role_evidence
from ai2_nfl_portfolio_risk import maximum_player_appearances, uncertain_running_back_reason


def _row(week, name, carries, targets, team="DEN"):
    return dict(season=2026, week=week, player_name=name,
                recent_team=team,position="RB",carries=carries,targets=targets,
                rushing_yards=4*carries,rushing_tds=0,
                receptions=int(targets),receiving_yards=5*targets,
                receiving_tds=0)


class EmergingNFLRoleTests(unittest.TestCase):
    def setup_roles(self):
        rows=[]
        # Coleman-like role history: Week1 no workload; Week2 used;
        # missed Weeks3-6; 11/16 touches on Weeks7,8.
        for w,c,t in [(1,0,0),(2,10,3),(7,9,2),(8,12,4)]:
            rows.append(_row(w,"Emerging RB",c,t))
        # Longtime teammate loses work specifically as the emerging runner
        # takes it; this is observed, not a claim about injury diagnosis.
        for w in range(1,9):
            rows.append(_row(w,"Previous Lead",17 if w<=6 else 4,2))
        # One-week breakout after absent games MUST still be excluded.
        for w,c,t in [(1,0,0),(2,1,0),(7,1,0),(8,17,4)]:
            rows.append(_row(w,"Single Spike",c,t,team="MIA"))
        for w in range(1,9):
            rows.append(_row(w,"Competing RB",15,2,team="MIA"))
        salary=[
            dict(ID="e",Name="Emerging RB",TeamAbbrev="DEN",
                 Position="RB",Salary=5300),
            dict(ID="s",Name="Single Spike",TeamAbbrev="MIA",
                 Position="RB",Salary=4800)]
        return pd.DataFrame(rows),salary

    def test_two_consecutive_recent_opportunities_can_be_limited(self):
        history,salary=self.setup_roles()
        found=build_pregame_role_evidence(history,salary,2026,9)
        self.assertEqual(found["e"]["Role gate"],"EMERGING_LIMITED")
        self.assertTrue(found["e"]["Emerging two-game role on latest team weeks"])
        self.assertEqual(found["e"]["Meaningful opportunities in last two appearances"],2)
        self.assertIn("DECLINING OBSERVED ROLE",
                      found["e"]["Same-position teammate pregame trends"])
        self.assertEqual(found["s"]["Role gate"],"BLOCK")

    def test_future_week_stats_do_not_change_status(self):
        history,salary=self.setup_roles()
        base=build_pregame_role_evidence(history,salary,2026,9)
        modified=pd.concat([history,pd.DataFrame([
            _row(9,"Emerging RB",30,8),
            _row(9,"Previous Lead",1,0),
        ])],ignore_index=True)
        replay=build_pregame_role_evidence(modified,salary,2026,9)
        self.assertEqual(base,replay)

    def test_emerging_role_has_strict_exposure_cap(self):
        player={"Name":"Emerging RB","Position":"RB",
                "Salary":5300,"emerging_role_limited":True,
                "emerging_role_evidence":"Two real recent games"}
        self.assertIn("Emerging two-game role",
                      uncertain_running_back_reason(player))
        self.assertEqual(maximum_player_appearances(player,40,.28),3)
        self.assertEqual(maximum_player_appearances(player,50,.28),4)


if __name__ == "__main__":
    unittest.main()
