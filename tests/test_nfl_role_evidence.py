import unittest
from ai2_nfl_role_evidence import player_opportunity_evidence

class RoleEvidenceTests(unittest.TestCase):
    def test_no_evidence_blocks_cheap_receiver(self):
        allowed, reason = player_opportunity_evidence({"Position":"WR","Salary":4100,"opportunity_verified":True})
        self.assertFalse(allowed)
        self.assertIn("Insufficient", reason)

    def test_four_targets_qualify(self):
        allowed, reason = player_opportunity_evidence({"Position":"WR","Salary":4100,"opportunity_verified":True,"recent_targets_per_game":4.5})
        self.assertTrue(allowed)
        self.assertIn("targets",reason)

    def test_running_back_role_requires_volume(self):
        self.assertFalse(player_opportunity_evidence({"Position":"RB","Salary":4600,"opportunity_verified":True,"projected_touches":3})[0])
        self.assertTrue(player_opportunity_evidence({"Position":"RB","Salary":4600,"opportunity_verified":True,"projected_touches":12})[0])

if __name__ == "__main__":
    unittest.main()
