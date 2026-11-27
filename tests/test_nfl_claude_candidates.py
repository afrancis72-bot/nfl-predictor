import unittest
import numpy as np
from ai2_nfl_claude_candidates import optimize_slot_lineup
from ai2_nfl_lineup import check_classic_lineup

class ClaudeCandidateTests(unittest.TestCase):
    def test_slot_solver_respects_salary_stack_and_rb_teams(self):
        specs=[('QB','KC','DEN',7000),('RB','KC','DEN',6000),('RB','BUF','MIA',6000),
               ('WR','KC','DEN',6000),('WR','DEN','KC',5500),('WR','BUF','MIA',5000),
               ('TE','KC','DEN',4500),('WR','MIA','BUF',6000),('DST','SEA','SF',4000),
               ('RB','KC','DEN',6000),('DST','DEN','KC',4000),('DST','MIA','BUF',4000)]
        pool=[dict(ID=str(i),Name=str(i),Position=p,TeamAbbrev=t,opponent=o,Salary=s)
              for i,(p,t,o,s) in enumerate(specs)]
        chosen=optimize_slot_lineup(pool,np.array([10.]*9+[100.,100.,100.]),time_limit=5)
        self.assertIsNotNone(chosen)
        lineup=[pool[i] for i in chosen]
        self.assertFalse(check_classic_lineup(lineup,salary_floor=49600))
        self.assertNotIn(10,chosen)
        self.assertNotIn(11,chosen)
        self.assertFalse(1 in chosen and 9 in chosen)
