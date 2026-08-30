import numpy as np
import pytest

from src.archive.v268_pseudo_deployment_route_audit import policy_fallback


def test_pseudo_deployment_policies_have_distinct_outcome_free_support() -> None:
    base = np.array([0.70, 0.70, 0.40, 0.40])
    pseudo = np.array([0.60, 0.80, 0.55, 0.30])
    active = np.array([True, True, True, False])
    equal, equal_mask = policy_fallback(base, pseudo, active, "equal_all")
    toward, toward_mask = policy_fallback(base, pseudo, active, "toward_center")
    same, same_mask = policy_fallback(base, pseudo, active, "same_center_side")
    np.testing.assert_array_equal(equal_mask, [True, True, True, False])
    np.testing.assert_array_equal(toward_mask, [True, False, True, False])
    np.testing.assert_array_equal(same_mask, [True, True, False, False])
    np.testing.assert_allclose(equal, [0.65, 0.75, 0.475, 0.40])
    np.testing.assert_allclose(toward, [0.65, 0.70, 0.475, 0.40])
    np.testing.assert_allclose(same, [0.65, 0.75, 0.40, 0.40])


def test_pseudo_deployment_policy_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown"):
        policy_fallback(np.array([0.5]), np.array([0.5]), np.array([True]), "x")
