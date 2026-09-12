import numpy as np

from src.archive.v249_dual_tree_consensus_scope import (
    ALPHAS,
    SCOPES,
    restrictions,
    scope_masks,
    select_candidate,
)


def test_scopes_are_subsets_of_active() -> None:
    parent = np.array([0.5, 0.5, 0.5, 0.5])
    xgb = np.array([0.6, 0.6, 0.4, 0.7])
    lgbm = np.array([0.59, 0.3, 0.41, 0.6])
    active = np.array([True, True, True, False])
    masks = scope_masks(parent, xgb, lgbm, active)
    assert tuple(masks) == SCOPES
    assert all(np.all(~mask | active) for mask in masks.values())
    assert masks["same_direction_close_02"].tolist() == [True, False, True, False]


def test_selection_requires_source_month_majority() -> None:
    results = {
        scope: {
            f"{alpha:g}": {
                axis: {"gain": -1.0, "positive_month_fraction": 1.0}
                for axis in ("full_2022", "late_2023")
            }
            for alpha in ALPHAS
        }
        for scope in SCOPES
    }
    candidate = results["same_direction"]["0.25"]
    candidate["full_2022"]["gain"] = 1.0
    candidate["late_2023"]["gain"] = 0.5
    candidate["late_2023"]["positive_month_fraction"] = 1.0 / 3.0
    assert select_candidate(results) is None
    candidate["late_2023"]["positive_month_fraction"] = 2.0 / 3.0
    assert select_candidate(results) == ("same_direction", 0.25)


def test_contract_is_public_blind() -> None:
    audit = restrictions()
    assert audit["six_predeclared_consensus_scopes_only"]
    assert audit["full_2024_locked_from_selection"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
