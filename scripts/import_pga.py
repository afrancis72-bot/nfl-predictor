"""Copy PGA source modules into the unified repository without altering them.

Run from the unified repository root after cloning the PGA repository:
    python scripts/import_pga.py ../pga-predictor

This is a local maintainer tool. It never silently overwrites imported files.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

MODULES = (
    "pga_predictor_pro.py",
    "golf_first_v11.py",
    "portfolio_optimizer.py",
    "round_parlay.py",
    "international_ingestion.py",
)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to cloned pga-predictor repository")
    parser.add_argument("--force", action="store_true", help="Replace existing imported files")
    args = parser.parse_args()
    source = args.source.resolve()
    target = Path(__file__).resolve().parents[1] / "sports" / "pga" / "engine"
    missing = [name for name in MODULES if not (source / name).is_file()]
    if missing:
        parser.error("Missing source modules: " + ", ".join(missing))
    collisions = [name for name in MODULES if (target / name).exists()]
    if collisions and not args.force:
        parser.error("Files already imported; use --force to replace: " + ", ".join(collisions))
    target.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in MODULES:
        src = source / name
        dest = target / name
        shutil.copy2(src, dest)
        if digest(src) != digest(dest):
            raise RuntimeError("Integrity check failed: " + name)
        manifest[name] = digest(dest)
    (target / "source_manifest.json").write_text(
        json.dumps({"source": "pga-predictor", "files_sha256": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Imported", len(MODULES), "PGA modules with matching SHA-256 checksums.")
    print("Next: audit transitive dependencies and run source/unified parity tests.")

if __name__ == "__main__":
    main()
