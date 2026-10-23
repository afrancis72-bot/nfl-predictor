import unittest
from unittest.mock import patch
from ai2_nfl_official_injuries import fetch_official_injury_statuses, match_injury_statuses

HTML = b"""<table><thead><tr><th>Player</th><th>Position</th><th>Practice Status</th><th>Game Status</th></tr></thead><tbody>
<tr><td>Lamar Jackson</td><td>QB</td><td>Did Not Participate</td><td>Out</td></tr>
<tr><td>Tee Higgins</td><td>WR</td><td>Full</td><td>Questionable</td></tr>
<tr><td>Healthy Player</td><td>WR</td><td>Did Not Participate</td><td></td></tr>
</tbody></table>"""

class FakeResponse:
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return HTML

class OfficialInjuriesTests(unittest.TestCase):
    @patch("ai2_nfl_official_injuries.urlopen", return_value=FakeResponse())
    def test_only_explicit_game_status(self, _):
        statuses, meta = fetch_official_injury_statuses()
        self.assertEqual(statuses["lamarjackson"], "OUT")
        self.assertEqual(statuses["teehiggins"], "QUESTIONABLE")
        self.assertNotIn("healthyplayer", statuses)
        self.assertEqual(meta["tables"], 1)

    def test_ambiguous_names_are_not_matched(self):
        rows = [{"ID":"1","Name":"Tee Higgins"},
                {"ID":"2","Name":"Tee Higgins"},
                {"ID":"3","Name":"Lamar Jackson"}]
        self.assertEqual(match_injury_statuses(
            rows, {"teehiggins":"OUT","lamarjackson":"OUT"}), {"3":"OUT"})

if __name__ == "__main__":
    unittest.main()
