"""Event-scoped identity mapping safety tests."""
import unittest
from ai2_market_identity import parse_identity_csv, identity_coverage

class IdentityTests(unittest.TestCase):
    def test_valid_mapping_and_coverage(self):
        mapping = parse_identity_csv(b"event_id,player_name,player_id\nG1,Player One,dk1\n")
        result = identity_coverage([
            {"event_id": "G1", "player_name": "Player One"},
            {"event_id": "G2", "player_name": "Player One"}], mapping)
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["unmatched"], 1)

    def test_duplicate_name_rejected(self):
        with self.assertRaises(ValueError):
            parse_identity_csv(b"event_id,player_name,player_id\nG1,A,1\nG1,A,2\n")

    def test_duplicate_id_rejected(self):
        with self.assertRaises(ValueError):
            parse_identity_csv(b"event_id,player_name,player_id\nG1,A,1\nG1,B,1\n")

    def test_missing_id_rejected(self):
        with self.assertRaises(ValueError):
            parse_identity_csv(b"event_id,player_name,player_id\nG1,A,\n")

if __name__ == "__main__":
    unittest.main()
