import numpy as np

from src.archive.v327_target1185_feasibility_audit import required_rms


def test_required_rms_inverts_bss_gain_identity() -> None:
    rate = 0.5
    gain = 8.0
    rms = required_rms(gain, rate)
    assert np.isclose(100000.0 * rms**2 / (rate * (1.0 - rate)), gain)
