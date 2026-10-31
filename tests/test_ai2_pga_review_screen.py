"""Check the PGA slate-review tab really renders in the unified PGA screen."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]

class ActualPGAInterface(unittest.TestCase):
    def test_existing_pga_ui_has_salary_review_tab(self):
        app=AppTest.from_file(str(ROOT/"ai2_pga_workspace.py"),
                              default_timeout=60).run()
        self.assertFalse(list(app.exception),[str(i.message) for i in app.exception])
        tabs=[x.label for x in app.tabs]
        self.assertIn("🏆 Tournament DFS · One Click",tabs)
        self.assertIn("🔎 Salary-Tier Review & Vetoes",tabs)
        self.assertIn("🎯 Round Matchups · 2 & 3 Balls",tabs)
        self.assertIn("⛳ Round Showdown",tabs)

if __name__=="__main__":
    unittest.main()
