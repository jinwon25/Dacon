from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v60_frozen_spread_scaling import apply_route, spread_scale


def test_linear_spread_scaling_is_row_local() -> None:
    probability = np.array([0.4, 0.6])
    scaled = spread_scale(probability, center=0.5, alpha=1.1, mode="linear")

    np.testing.assert_allclose(scaled, [0.39, 0.61], rtol=0.0, atol=1e-15)
    first_alone = spread_scale(
        probability[[0]], center=0.5, alpha=1.1, mode="linear"
    )[0]
    assert first_alone == scaled[0]


def test_logit_identity_is_exact() -> None:
    probability = np.array([0.2, 0.5, 0.8])
    scaled = spread_scale(probability, center=0.45, alpha=1.0, mode="logit")

    np.testing.assert_allclose(scaled, probability, rtol=0.0, atol=2e-16)


def test_apply_route_changes_only_selected_domain() -> None:
    frame = pd.DataFrame({"domain3": ["R_CORE", "R_ANCHOR", "F"]})
    parent = np.array([0.4, 0.5, 0.6])
    transformed = np.array([0.3, 0.45, 0.7])

    candidate, active = apply_route(frame, parent, transformed, "R_ANCHOR")

    np.testing.assert_array_equal(active, [False, True, False])
    np.testing.assert_allclose(candidate, [0.4, 0.45, 0.6])
