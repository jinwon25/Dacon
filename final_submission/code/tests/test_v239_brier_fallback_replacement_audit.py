from src.archive.v239_brier_fallback_replacement_audit import LABEL, PROTOCOL


def test_v239_identifies_one_brier_candidate() -> None:
    assert LABEL == "brier_xgb"
    assert "BRIER_FALLBACK" in PROTOCOL
