from __future__ import annotations

import numpy as np
import pandas as pd

from src.champion_oof import _eb_correction


def test_eb_correction_is_fit_only_and_domain_local() -> None:
    fit = pd.DataFrame(
        {
            "key": ["a", "a", "b", "a"],
            "domain3": ["R_CORE", "R_CORE", "R_CORE", "F"],
            "game_month": [3, 4, 4, 4],
            "residual_v19": [0.2, -0.1, 0.3, 1.0],
        }
    )
    audit = pd.DataFrame(
        {
            "key": ["a", "b", "c", "a"],
            "domain3": ["R_CORE", "R_CORE", "R_CORE", "F"],
        }
    )
    correction = _eb_correction(
        fit,
        audit,
        columns=("key",),
        domain="R_CORE",
        alpha=1.0,
    )
    assert np.allclose(correction, [0.1 / 3.0, 0.3 / 2.0, 0.0, 0.0])


def test_recency_weight_uses_source_cutoff_only() -> None:
    fit = pd.DataFrame(
        {
            "key": ["a", "a"],
            "domain3": ["R_CORE", "R_CORE"],
            "game_month": [3, 4],
            "residual_v19": [1.0, 0.0],
        }
    )
    audit = pd.DataFrame({"key": ["a"], "domain3": ["R_CORE"]})
    correction = _eb_correction(
        fit,
        audit,
        columns=("key",),
        domain="ALL",
        alpha=0.0,
        half_life=1.0,
    )
    assert np.allclose(correction, [1.0 / 3.0])
