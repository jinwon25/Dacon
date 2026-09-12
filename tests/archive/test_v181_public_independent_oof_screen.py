import numpy as np

from src.archive.v181_public_independent_oof_screen import convex_blend, route_mask


def test_convex_blend_changes_only_active_rows() -> None:
    parent = np.array([0.4, 0.5, 0.6])
    independent = np.array([0.6, 0.3, 0.2])
    active = np.array([True, False, True])
    result = convex_blend(parent, independent, active, 0.25)
    np.testing.assert_allclose(result, [0.45, 0.5, 0.5])


def test_route_mask_respects_exact_rows() -> None:
    axis = {
        "exact_mask": np.array([True, True, False, True]),
        "domain3": np.array(["R_CORE", "F", "R_CORE", "R_ANCHOR"]),
    }
    np.testing.assert_array_equal(route_mask(axis, "ALL"), [True, True, False, True])
    np.testing.assert_array_equal(route_mask(axis, "R_CORE"), [True, False, False, False])
    np.testing.assert_array_equal(route_mask(axis, "F"), [False, True, False, False])
