"""Verify that the PGA page loads within Streamlit's multipage runtime.

The test intentionally does not fetch current-week data or generate a DFS entry.
"""
from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class PGAPageSmokeTest(unittest.TestCase):
    def test_pga_setup_renders(self):
        app = AppTest.from_file(str(ROOT / "pages" / "2_PGA.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0, [str(e.message) for e in app.exception])
        self.assertTrue(any("PGA" in str(t.value) for t in app.title))
        self.assertTrue(any("Setup" in str(h.value) for h in app.header))
        self.assertTrue(any("PGA MODEL" in str(x.label) for x in app.radio))
        self.assertTrue(any("Load inputs" in str(x.label) for x in app.button))


if __name__ == "__main__":
    unittest.main(verbosity=2)
