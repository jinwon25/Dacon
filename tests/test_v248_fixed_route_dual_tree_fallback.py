import numpy as np

from src.archive.v248_fixed_route_dual_tree_fallback import (
    ALPHAS,
    ROUTE_WEIGHTS,
    compose,
    mix_fallback,
    restrictions,
    select_alpha,
)


def test_mix_fallback_is_convex() -> None:
    xgb = np.array([0.2, 0.8])
    lgbm = np.array([0.6, 0.4])
    np.testing.assert_allclose(mix_fallback(xgb, lgbm, 0.25), [0.3, 0.7])


def test_compose_uses_frozen_disjoint_weights() -> None:
    parent = np.full(4, 0.5)
    fallback = np.full(4, 0.7)
    routes = {
        name: np.arange(4) == index
        for index, name in enumerate(ROUTE_WEIGHTS)
    }
    output, active = compose(parent, fallback, routes)
    expected = np.array([0.56, 0.59, 0.53, 0.53])
    np.testing.assert_allclose(output, expected)
    assert active.all()


def test_selection_requires_both_sources_positive() -> None:
    results = {
        f"{alpha:g}": {
            "full_2022": {"gain": -1.0},
            "late_2023": {"gain": -1.0},
        }
        for alpha in ALPHAS
    }
    results["0.25"]["full_2022"]["gain"] = 1.0
    assert select_alpha(results) is None
    results["0.25"]["late_2023"]["gain"] = 0.5
    assert select_alpha(results) == 0.25


def test_contract_freezes_v244_and_is_public_blind() -> None:
    audit = restrictions()
    assert audit["v244_parent_and_routes_frozen"]
    assert audit["v244_route_weights_frozen"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
