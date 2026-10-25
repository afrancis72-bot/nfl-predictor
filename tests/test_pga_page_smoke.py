"""Test the imported PGA dashboard through Streamlit's actual page runtime."""
from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class PGAPageSmokeTest(unittest.TestCase):
    def test_setup_and_navigation_render(self):
        app = AppTest.from_file(str(ROOT / "pages" / "2_PGA.py"), default_timeout=30).run()
        self.assertFalse(app.exception, [str(e.message) for e in app.exception])
        self.assertTrue(any("PGA" in str(t.value) for t in app.title))
        self.assertTrue(any("Setup / Inputs" in str(h.value) for h in app.header))
        self.assertTrue(any("PGA MODEL" in str(x.label) for x in app.sidebar.radio))
        self.assertTrue(
            any("Load inputs & build tournament model" in str(x.label) for x in app.button)
        )
        # Switching to DFS before setup should explain what is missing, not crash.
        app.sidebar.radio[0].set_value("🏆 Tournament DFS").run()
        self.assertFalse(app.exception, [str(e.message) for e in app.exception])
        self.assertTrue(
            any("No tournament setup" in str(x.message) for x in app.warning)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
