from __future__ import annotations

import copy

import numpy as np

from src.champion.v124_public_quadratic_stack import patch_specifications


def _specs() -> dict:
    return {
        "v14": {"anchor_weight": 0.2, "f_trend_alpha": 0.15},
        "v16": {
            "correction_weight": 1.5,
            "effects": {"a": -0.2, "b": 0.1},
            "effect_summary": {
                "minimum": -0.2,
                "maximum": 0.1,
                "mean": -0.05,
                "mean_absolute": 0.15,
                "weighted_mean_absolute": 0.12,
                "median_effective_count": 10.0,
            },
        },
        "v20": {
            "extra_mode": {"weight": 0.045},
            "eb_recipes": [{"weight": 0.07}, {"weight": 0.16}, {"weight": 0.18}],
            "pfd": {"overlay_weight": 0.26},
        },
        "v21": {
            "recipes": [
                {"weight": 0.275}, {"weight": 0.075}, {"weight": 0.05},
                {"weight": 0.1}, {"weight": 0.2}, {"weight": 0.05},
            ]
        },
        "v22": {
            "domain_calibration": {
                "R_CORE": {"weight": 0.035},
                "R_ANCHOR": {"weight": 0.02},
                "F": {"weight": 0.02},
            },
            "asof_prior": {"weight": 0.05},
        },
        "v25": {"blend_eta": 0.15},
    }


SELECTED = {
    "v17_scale": 1.45,
    "v19_scale": 1.0,
    "v20_scale": 0.75,
    "v21_scale": 0.5,
    "v22_scale": 0.65,
    "anchor_eta": 0.36,
}


def test_patch_specifications_scales_only_registered_values() -> None:
    original = _specs()
    patched = patch_specifications(copy.deepcopy(original), SELECTED)
    assert np.isclose(patched["v14"]["anchor_weight"], 0.29)
    assert np.isclose(patched["v14"]["f_trend_alpha"], 0.2175)
    assert np.isclose(patched["v16"]["correction_weight"], 2.175)
    assert np.isclose(patched["v16"]["effects"]["a"], -0.29)
    assert np.allclose([r["weight"] for r in patched["v20"]["eb_recipes"]], [0.0525, 0.12, 0.135])
    assert np.allclose([r["weight"] for r in patched["v21"]["recipes"]], np.asarray([0.275, 0.075, 0.05, 0.1, 0.2, 0.05]) * 0.5)
    assert np.isclose(patched["v22"]["domain_calibration"]["R_CORE"]["weight"], 0.02275)
    assert np.isclose(patched["v22"]["asof_prior"]["weight"], 0.0325)
    assert np.isclose(patched["v25"]["blend_eta"], 0.36)
    assert patched["v16"]["effect_summary"]["median_effective_count"] == 10.0


def test_patch_rejects_nonunit_v19_or_wrong_parent() -> None:
    selected = dict(SELECTED)
    selected["v19_scale"] = 0.99
    try:
        patch_specifications(_specs(), selected)
    except ValueError as exc:
        assert "preserve" in str(exc)
    else:
        raise AssertionError("nonunit v19 was accepted")

    specs = _specs()
    specs["v25"]["blend_eta"] = 0.10
    try:
        patch_specifications(specs, SELECTED)
    except ValueError as exc:
        assert "v25 blend_eta" in str(exc)
    else:
        raise AssertionError("wrong parent value was accepted")
