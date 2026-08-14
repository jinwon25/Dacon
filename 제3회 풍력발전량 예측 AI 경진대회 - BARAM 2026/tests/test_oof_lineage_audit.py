import numpy as np

from experiments.oof_lineage_audit import (
    extrapolate_submission,
    fit_affine,
    recover_member,
)
from experiments.exact_group3_oof import apply_calibration


def test_recover_member_inverts_a_blend() -> None:
    base = np.asarray([10.0, 20.0, 30.0])
    member = np.asarray([30.0, 10.0, 50.0])
    weight = 0.05
    blended = (1.0 - weight) * base + weight * member

    np.testing.assert_allclose(recover_member(blended, base, weight), member)


def test_fit_affine_recovers_known_mapping() -> None:
    source = np.linspace(-5.0, 5.0, 101)
    target = 0.8 * source - 2.5

    slope, intercept = fit_affine(source, target)

    assert np.isclose(slope, 0.8)
    assert np.isclose(intercept, -2.5)


def test_extrapolation_clips_to_capacity() -> None:
    anchor = np.asarray([10.0, 90.0])
    direction = np.asarray([-10.0, 120.0])

    result = extrapolate_submission(anchor, direction, factor=2.0, capacity=100.0)

    np.testing.assert_allclose(result, [0.0, 100.0])


def test_calibration_strength_scales_away_from_identity() -> None:
    prediction = np.asarray([100.0, 200.0])

    result = apply_calibration(prediction, scale=1.04, offset=-400.0, strength=1.25)

    np.testing.assert_allclose(result, prediction * 1.05 - 500.0)
