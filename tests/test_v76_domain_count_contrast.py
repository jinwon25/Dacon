from __future__ import annotations

import numpy as np
import pandas as pd

from src.v76_domain_count_contrast import (
    apply_domain_count_contrast,
    fit_domain_count_contrast,
)


def test_contrast_has_zero_weighted_mean_within_domain() -> None:
    frame = pd.DataFrame(
        {
            "domain3": ["R_CORE"] * 4 + ["F"] * 4,
            "balls_before": [0, 0, 1, 1] * 2,
            "strikes_before": [0] * 8,
            "target": [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        }
    )
    parent = np.full(len(frame), 0.5)
    table = fit_domain_count_contrast(frame, parent, prior=2.0)
    weighted = table.assign(value=table["correction"] * table["size"])
    assert np.allclose(weighted.groupby("domain3")["value"].sum(), 0.0)


def test_apply_is_row_local_and_preserves_order() -> None:
    frame = pd.DataFrame(
        {
            "domain3": ["F", "R_CORE"],
            "balls_before": [1, 0],
            "strikes_before": [0, 0],
            "target": [0.0, 1.0],
        }
    )
    table = pd.DataFrame(
        {
            "domain3": ["R_CORE", "F"],
            "count_state": ["0-0", "1-0"],
            "correction": [0.1, -0.2],
        }
    )
    candidate, correction = apply_domain_count_contrast(
        frame, np.array([0.5, 0.5]), table
    )
    assert np.allclose(correction, [-0.2, 0.1])
    assert np.allclose(candidate, [0.3, 0.6])
