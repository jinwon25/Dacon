import numpy as np

from src.archive.v246_swimmer_independent_residual_blend import (
    MODELS,
    SOURCE_AXES,
    WEIGHTS,
    blend,
    restrictions,
    select_candidate,
)


def test_blend_is_convex_and_clipped() -> None:
    parent = np.array([0.2, 0.8])
    independent = np.array([0.6, 0.4])
    np.testing.assert_allclose(blend(parent, independent, 0.05), [0.22, 0.78])


def test_selection_requires_both_sources_positive() -> None:
    results = {
        model: {
            f"{weight:g}": {
                axis: {"gain": -1.0} for axis in SOURCE_AXES
            }
            for weight in WEIGHTS
        }
        for model in MODELS
    }
    results["logistic"]["0.005"]["full_2022"]["gain"] = 2.0
    assert select_candidate(results) is None
    results["logistic"]["0.005"]["late_2023"]["gain"] = 1.0
    assert select_candidate(results) == ("logistic", 0.005)


def test_contract_is_train_only_and_public_blind() -> None:
    audit = restrictions()
    assert audit["official_train_only_strict_forward_oof"]
    assert audit["full_2024_locked_from_selection"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
