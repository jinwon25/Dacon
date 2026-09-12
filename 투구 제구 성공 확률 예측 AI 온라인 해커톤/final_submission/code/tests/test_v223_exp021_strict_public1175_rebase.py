from __future__ import annotations

import numpy as np
import pytest

from src.archive.v223_exp021_strict_public1175_rebase import (
    blend_strict,
    restrictions,
    select_recipe,
)


def test_blend_strict_changes_rcore_only() -> None:
    axis = {"domain3": np.array(["R_CORE", "F"])}
    result, active = blend_strict(
        np.array([0.4, 0.4]), np.array([0.6, 0.6]), axis,
        "probability", 0.5,
    )
    np.testing.assert_allclose(result, [0.5, 0.4])
    assert active.tolist() == [True, False]
    with pytest.raises(ValueError, match="unknown mode"):
        blend_strict(
            np.array([0.5]), np.array([0.5]),
            {"domain3": np.array(["R_CORE"])}, "x", 0.1,
        )


def test_select_recipe_requires_both_sources() -> None:
    def item(gain: float):
        return {
            "gain": gain, "positive_month_fraction": 1.0,
            "worst_month_gain": 0.0, "minimum_domain_gain": 0.0,
        }
    results = {
        "probability_w0.010": {"full_2022": item(1.0), "late_2023": item(2.0)},
        "logit_w0.020": {"full_2022": item(2.0), "late_2023": item(-1.0)},
    }
    assert select_recipe(results) == "probability_w0.010"


def test_v223_is_independent_and_test_free() -> None:
    audit = restrictions()
    assert audit["independent_exp021_recipe_frozen"]
    assert audit["strictly_prior_season_oof"]
    assert audit["public1175_fallback_formula_frozen"]
    assert not audit["test_csv_read"]
