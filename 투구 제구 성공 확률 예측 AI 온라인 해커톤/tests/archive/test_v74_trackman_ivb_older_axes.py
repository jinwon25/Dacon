from __future__ import annotations

from src.archive.v74_trackman_ivb_older_axes import independent_gate


def _audit(gain: float, month_fraction: float, domain_gain: float, eta: float) -> dict[str, float]:
    return {
        "gain": gain,
        "positive_month_fraction": month_fraction,
        "minimum_domain_gain": domain_gain,
        "source_oof_eta": eta,
    }


def test_independent_gate_uses_only_declared_older_axes() -> None:
    audits = {
        "early22_to_late22": _audit(1.0, 1.0, 0.1, 0.5),
        "full22_to_full23": _audit(2.0, 1.0, 0.2, 1.0),
        "full23_to_full24_confirmation": _audit(-100.0, 0.0, -100.0, 0.0),
    }
    assert all(independent_gate(audits).values())


def test_independent_gate_rejects_negative_older_axis() -> None:
    audits = {
        "early22_to_late22": _audit(-0.1, 1.0, 0.1, 0.5),
        "full22_to_full23": _audit(2.0, 1.0, 0.2, 1.0),
    }
    assert not independent_gate(audits)["both_older_gains_positive"]
