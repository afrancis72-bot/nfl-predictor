"""Protect imported PGA modeling modules from accidental changes.

These are Git blob hashes verified against pga-predictor/main on 2026-10-25.
Update the baselines only after deliberately reviewing a source model upgrade.
"""
from pathlib import Path
import hashlib
import unittest

ROOT = Path(__file__).resolve().parents[1] / "sports" / "pga" / "legacy"
ORIGINAL_GIT_BLOBS = {
    "pga_predictor_pro.py": "31ce6fb879e1f1ee454dabc2ef832db5ba3c14bd",
    "golf_first_v11.py": "43716eb33ba520a050e093048c8f300d02aa02bc",
    "portfolio_optimizer.py": "637b2d520a818697d784417339569dc8dadb12a4",
    "round_parlay.py": "420fafee97dbfd1f0b957658ed348674de066efe",
    "international_ingestion.py": "7d638db7c5bcce33bcf65753bc1951e9492939eb",
    "course_fit_simulation.py": "433785b24a9da012ad8243c5a0a85ff7aa102164",
    "course_fit_ceiling.py": "9b99223e150c9e5efb2fe1959439620028c59fbc",
    "round_showdown.py": "f01ced34991b10b69f59a72df503594a029c2c8d",
    "data_ingestion.py": "3e9afcb2aa8834061c9ed5ecfd783ba4f2d5fb5c",
}


class PGAOriginalParity(unittest.TestCase):
    def test_original_engine_files_identical(self):
        for name, expected_sha in ORIGINAL_GIT_BLOBS.items():
            with self.subTest(module=name):
                content = (ROOT / name).read_bytes()
                got = hashlib.sha1(
                    b"blob " + str(len(content)).encode("ascii") + b"\0" + content
                ).hexdigest()
                self.assertEqual(
                    got, expected_sha,
                    f"Original PGA engine changed: {name}; review model parity before approval."
                )

    def test_dashboard_does_not_reconfigure_streamlit(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("st.set_page_config(", app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
