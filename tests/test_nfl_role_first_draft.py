import unittest
from ai2_nfl_role_evidence import player_opportunity_evidence

class RoleFirstDraft(unittest.TestCase):
    def test_salary_independent_replacement(self):
        p=dict(Position="WR",Salary=3400,opportunity_verified=True,draft_role_first=True,
               observed_games=8,observed_targets_pg=3.03,projected_targets=5.,
               draft_vacancy_reason="Starter OUT, 8.8 observed targets")
        self.assertTrue(player_opportunity_evidence(p)[0])
        p["Salary"]=9000
        self.assertTrue(player_opportunity_evidence(p)[0])
        p["draft_vacancy_reason"]=""
        self.assertFalse(player_opportunity_evidence(p)[0])
        p["draft_vacancy_reason"]="Starter OUT"
        p["observed_targets_pg"]=.2
        self.assertFalse(player_opportunity_evidence(p)[0])
    def test_expensive_zero_role_not_automatically_eligible(self):
        self.assertFalse(player_opportunity_evidence(dict(Position="WR",Salary=9000,draft_role_first=True,opportunity_verified=True,observed_games=8,observed_targets_pg=.2,projected_targets=.5))[0])
