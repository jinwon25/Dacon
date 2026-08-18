import numpy as np

from src.v24_leaderboard_scale_audit import (
    infer_linear_terms,
    optimal_scales,
    score_curvature,
)


def test_score_curvature_matches_direct_quadratic_term() -> None:
    target = np.asarray([0.0, 1.0, 1.0, 0.0])
    deltas = np.asarray([[0.1, 0.0], [0.0, 0.2], [-0.1, 0.1], [0.2, -0.1]])
    expected = 100000.0 * deltas.T @ deltas / (len(target) * 0.25)
    np.testing.assert_allclose(score_curvature(target, deltas), expected)


def test_inferred_linear_terms_reproduce_sequential_gains() -> None:
    curvature = np.asarray([[2.0, 0.25], [0.25, 1.0]])
    gains = np.asarray([3.0, -0.5])
    linear = infer_linear_terms(curvature, gains)
    first = linear[0] - curvature[0, 0]
    both = linear.sum() - curvature.sum()
    np.testing.assert_allclose(first, gains[0])
    np.testing.assert_allclose(both - first, gains[1])
    scales = optimal_scales(curvature, linear)
    np.testing.assert_allclose(2.0 * curvature @ scales, linear)
