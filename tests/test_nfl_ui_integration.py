"""Minimal Streamlit integration smoke test for the unified NFL tab."""
import os
import tempfile
import unittest
from pathlib import Path


class UnifiedNFLUITests(unittest.TestCase):
    def test_one_button_nfl_tab_renders_without_extra_inputs(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory:
            old = os.environ.get("AI2_DATA_DIR")
            os.environ["AI2_DATA_DIR"] = directory
            try:
                app = AppTest.from_file("unified_app.py", default_timeout=45)
                app.run()
                self.assertFalse(app.exception, [str(e.message) for e in app.exception])
                app.session_state["sport"] = "NFL"
                app.run()
                self.assertFalse(app.exception, [str(e.message) for e in app.exception])
                self.assertIn("BUILD MY 20 LINEUPS",
                              [b.label for b in app.button])
                self.assertIn("DraftKings salary CSV",
                              [u.label for u in app.get("file_uploader")])
                self.assertTrue(any("Protected approved lineups" in e.label
                                    for e in app.get("expander")))
            finally:
                if old is None:
                    os.environ.pop("AI2_DATA_DIR", None)
                else:
                    os.environ["AI2_DATA_DIR"] = old


if __name__ == "__main__":
    unittest.main()
