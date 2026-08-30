import numpy as np

from src.archive.v270_pseudo_deployment_diversity_audit import diversity_fallback


def test_diversity_fallback_is_nested_equal_ensemble_on_active_rows() -> None:
    base = np.array([0.4, 0.6])
    xgb = np.array([0.5, 0.7])
    lgbm = np.array([0.7, 0.9])
    fallback, selected = diversity_fallback(
        base, xgb, lgbm, np.array([True, False])
    )
    # expert=.6, then equal blend with base=.5 on active row.
    np.testing.assert_allclose(fallback, [0.5, 0.6])
    np.testing.assert_array_equal(selected, [True, False])
