from __future__ import annotations

from src.archive.v232_pressure_specialist_fallback_replacement_audit import (
    LABEL,
    PROTOCOL,
)


def test_v232_identifies_one_pressure_specialist_candidate() -> None:
    assert LABEL == "pressure_specialist"
    assert "PRESSURE_SPECIALIST" in PROTOCOL
