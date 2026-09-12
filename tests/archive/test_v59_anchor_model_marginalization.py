from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v59_anchor_model_marginalization import (
    apply_anchor_direct,
    disagreement_eta,
    marginal_prediction,
)


def test_marginal_prediction_modes_are_row_local() -> None:
    members = np.array([[0.2, 0.3, 0.4], [0.6, 0.5, 0.4]])

    mean = marginal_prediction(members, "mean")
    median = marginal_prediction(members, "median")

    np.testing.assert_allclose(mean, [0.3, 0.5], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(median, [0.3, 0.5], rtol=0.0, atol=1e-15)
    first_alone = marginal_prediction(members[[0]], "mean")[0]
    assert first_alone == mean[0]


def test_disagreement_eta_uses_only_each_rows_value() -> None:
    values = np.array([0.01, 0.03, 0.02])
    eta = disagreement_eta(values, 0.02, 0.20, 0.10)

    np.testing.assert_array_equal(eta, [0.20, 0.10, 0.20])
    assert disagreement_eta(values[[1]], 0.02, 0.20, 0.10)[0] == eta[1]


def test_apply_anchor_direct_changes_only_anchor_rows() -> None:
    frame = pd.DataFrame(
        {
            "v22": [0.4, 0.5],
            "v25": [0.415, 0.5],
            "domain3": ["R_ANCHOR", "R_CORE"],
        }
    )
    candidate, active = apply_anchor_direct(
        frame, np.array([0.6, 0.2]), np.array([0.2, 0.2])
    )

    np.testing.assert_array_equal(active, [True, False])
    np.testing.assert_allclose(candidate, [0.44, 0.5], rtol=0.0, atol=1e-15)
