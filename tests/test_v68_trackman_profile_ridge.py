from __future__ import annotations

import numpy as np
import pandas as pd

from src.v68_trackman_profile_ridge import (
    apply_pitcher_correction,
    fit_source_only_ridge,
)


def test_source_only_ridge_recovers_simple_direction() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    source = pd.DataFrame(
        {
            "feature": x,
            "residual_eb": 0.01 * x,
            "reliability": np.ones_like(x),
        }
    )
    audit = pd.DataFrame({"feature": [-1.0, 1.0]})
    correction, diagnostic = fit_source_only_ridge(
        source, audit, ("feature",), seed=1
    )
    assert correction[0] < 0 < correction[1]
    assert diagnostic["source_oof_eta"] > 0


def test_apply_pitcher_correction_is_row_local_and_domain_gated() -> None:
    frame = pd.DataFrame(
        {
            "pitcher_id": [1, 2, 1],
            "domain3": ["R_ANCHOR", "R_ANCHOR", "F"],
        }
    )
    profile = pd.DataFrame({"pitcher_id": [1], "tm_n": [500.0]})
    output, active, correction = apply_pitcher_correction(
        frame, np.array([0.5, 0.5, 0.5]), profile, np.array([0.02])
    )
    assert np.allclose(output, [0.51, 0.5, 0.5])
    assert active.tolist() == [True, False, False]
    assert np.allclose(correction, [0.01, 0.0, 0.01])

    all_output, all_active, _ = apply_pitcher_correction(
        frame,
        np.array([0.5, 0.5, 0.5]),
        profile,
        np.array([0.02]),
        domain="ALL",
    )
    assert np.allclose(all_output, [0.51, 0.5, 0.51])
    assert all_active.tolist() == [True, False, True]
