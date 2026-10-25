"""PGA installation and validation status for the unified Team AI² Pro app.

Presence of source files is NOT evidence of successful Streamlit or slate testing.
"""
from dataclasses import dataclass
from pathlib import Path

LEGACY = Path(__file__).resolve().parent / "legacy"
REQUIRED = (
    "app.py", "pga_predictor_pro.py", "golf_first_v11.py",
    "portfolio_optimizer.py", "round_parlay.py",
    "international_ingestion.py", "round_showdown.py",
    "course_fit_simulation.py", "course_fit_ceiling.py",
    "data_ingestion.py",
)


@dataclass(frozen=True)
class PGAMigrationStatus:
    source_repo: str
    source_version: str
    engine_imported: bool
    parity_test_passed: bool
    lineup_export_enabled: bool
    message: str


def status() -> PGAMigrationStatus:
    missing = [name for name in REQUIRED if not (LEGACY / name).is_file()]
    imported = not missing
    return PGAMigrationStatus(
        source_repo="afrancis72-bot/pga-predictor",
        source_version="V11 golf-first / PGA dashboard V10.9.6",
        engine_imported=imported,
        parity_test_passed=False,  # Requires verified CI + live slate checks.
        lineup_export_enabled=False,  # Status flag, not a UI access-control mechanism.
        message=(
            "PGA files are installed; CI and live interface validation remain."
            if imported else "PGA source files are missing: " + ", ".join(missing)
        ),
    )


def require_validated_engine() -> None:
    current = status()
    if not (current.engine_imported and current.parity_test_passed):
        raise RuntimeError(current.message)
