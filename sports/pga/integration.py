"""PGA integration boundary for Team AI² Pro.

The original PGA V10.9.5 engine is not bundled yet. This module intentionally
fails closed instead of substituting a simplified projection model.
"""
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class PGAMigrationStatus:
    source_repo: str = "afrancis72-bot/pga-predictor"
    source_version: str = "V10.9.5"
    engine_imported: bool = False
    parity_test_passed: bool = False
    lineup_export_enabled: bool = False
    message: str = "PGA migration in progress; standalone PGA engine remains authoritative."

def status() -> PGAMigrationStatus:
    return PGAMigrationStatus()

def require_validated_engine() -> None:
    s = status()
    if not (s.engine_imported and s.parity_test_passed):
        raise RuntimeError(s.message)
