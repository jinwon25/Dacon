import numpy as np
import pandas as pd
import pytest

from src.archive.v212_role_adaptive_workload_h1 import apply_role_transport


def test_apply_role_transport_respects_domain_jy_and_role() -> None:
    parent = np.array([0.40, 0.40, 0.40, 0.40])
    low = np.array([0.41, 0.41, 0.41, 0.41])
    high = np.array([0.43, 0.43, 0.43, 0.43])
    axis = {
        "exact_mask": np.ones(4, dtype=bool),
        "domain3": np.array(["R_CORE", "R_CORE", "R_CORE", "F"]),
    }
    frame = pd.DataFrame({"num_runners_on": [1, 1, 0, 1], "li": [1, 1, 1, 2]})
    role = np.array([True, False, True, True])
    candidate, active = apply_role_transport(
        parent, low, high, axis, frame, role, "low_then_high"
    )
    np.testing.assert_allclose(candidate, [0.43, 0.41, 0.40, 0.40])
    np.testing.assert_array_equal(active, [True, False, False, False])


def test_apply_role_transport_high_only_and_unknown_mode() -> None:
    parent = np.array([0.40, 0.40])
    low = np.array([0.41, 0.41])
    high = np.array([0.43, 0.43])
    axis = {
        "exact_mask": np.ones(2, dtype=bool),
        "domain3": np.array(["R_CORE", "R_CORE"]),
    }
    frame = pd.DataFrame({"num_runners_on": [1, 1], "li": [1, 1]})
    role = np.array([True, False])
    candidate, _ = apply_role_transport(
        parent, low, high, axis, frame, role, "high_only"
    )
    np.testing.assert_allclose(candidate, [0.43, 0.40])
    with pytest.raises(ValueError, match="unknown role transport mode"):
        apply_role_transport(parent, low, high, axis, frame, role, "bad")
