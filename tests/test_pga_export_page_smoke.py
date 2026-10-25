"""Verify the PGA Entries export page fails closed without a built tournament."""
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class PGAEntriesPageTest(unittest.TestCase):
    def test_no_model_shows_setup_guidance(self):
        app = AppTest.from_file(
            str(ROOT / "pages" / "3_PGA_Entries.py"), default_timeout=20
        ).run()
        self.assertFalse(app.exception, [str(e.message) for e in app.exception])
        self.assertTrue(any("PGA" in str(t.value) for t in app.title))
        self.assertTrue(
            any("Build a PGA tournament model" in str(m.message) for m in app.info)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
