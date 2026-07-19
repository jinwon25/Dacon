import numpy as np

from experiments.exact_driver_oof import apply_weighted_gate, recover_stack5


def test_recover_stack5_round_trip() -> None:
    blend = np.asarray([100.0, 200.0, 300.0])
    calibration = np.asarray([90.0, 210.0, 280.0])
    over = calibration + 1.15 * (blend - calibration)
    scada = np.asarray([120.0, 160.0, 330.0])
    stack15 = 0.85 * over + 0.15 * scada

    recovered_over, recovered_scada, recovered_stack5 = recover_stack5(
        blend, calibration, stack15, capacity=1_000.0
    )

    assert np.allclose(recovered_over, over)
    assert np.allclose(recovered_scada, scada)
    assert np.allclose(recovered_stack5, 0.95 * over + 0.05 * scada)


def test_weighted_gate_changes_only_agreement_rows() -> None:
    base = np.asarray([100.0, 200.0, 50.0, 900.0])
    member = np.asarray([110.0, 260.0, 55.0, 850.0])
    output, mask = apply_weighted_gate(
        base, member, capacity=1_000.0, alpha=0.2, max_disagreement=0.04
    )

    assert mask.tolist() == [True, False, False, False]
    assert np.allclose(output, [102.0, 200.0, 50.0, 900.0])
