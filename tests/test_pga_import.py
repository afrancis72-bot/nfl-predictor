"""Offline tests for the PGA importer; run: python -m unittest discover -s tests."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "import_pga.py"
MODULES = (
    "pga_predictor_pro.py",
    "golf_first_v11.py",
    "portfolio_optimizer.py",
    "round_parlay.py",
    "international_ingestion.py",
)

class PGAModuleImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.target_script = self.root / "unified" / "scripts" / "import_pga.py"
        self.target_script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, self.target_script)
        for i, name in enumerate(MODULES):
            (self.source / name).write_text(f"# module {i}\n", encoding="utf-8")

    def run_import(self, *args):
        return subprocess.run(
            [sys.executable, str(self.target_script), str(self.source), *args],
            capture_output=True, text=True, check=False
        )

    def test_import_and_integrity_manifest(self):
        result = self.run_import()
        self.assertEqual(result.returncode, 0, result.stderr)
        dest = self.root / "unified" / "sports" / "pga" / "engine"
        manifest = json.loads((dest / "source_manifest.json").read_text())
        for name in MODULES:
            copied = dest / name
            self.assertEqual(copied.read_bytes(), (self.source / name).read_bytes())
            self.assertEqual(manifest["files_sha256"][name],
                             hashlib.sha256(copied.read_bytes()).hexdigest())

    def test_refuse_accidental_overwrite(self):
        self.assertEqual(self.run_import().returncode, 0)
        result = self.run_import()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--force", result.stderr)

    def test_missing_source_fails_before_copy(self):
        (self.source / MODULES[0]).unlink()
        result = self.run_import()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "unified" / "sports" / "pga" / "engine").exists())

if __name__ == "__main__":
    unittest.main()
