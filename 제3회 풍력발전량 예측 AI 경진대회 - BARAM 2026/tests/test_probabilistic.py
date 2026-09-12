import numpy as np

from src.metrics import evaluate_group
from src.probabilistic import (
    DEFAULT_QUANTILE_LEVELS,
    enforce_noncrossing,
    expected_metric_utility,
    metric_aware_bayes_action,
    quantile_scenario_weights,
    settlement_price,
    shrink_action,
)


def test_expected_utility_order_matches_official_group_score() -> None:
    truth = np.array([3_000.0, 8_000.0, 15_000.0])
    actions = np.array([[3_000.0, 2_000.0], [8_000.0, 7_000.0], [15_000.0, 14_000.0]])
    scenarios = truth[:, None]
    mean_y = truth.mean()
    utility = expected_metric_utility(actions, scenarios, capacity=20_000.0, mean_eligible_generation=mean_y)
    exact = evaluate_group(truth, actions[:, 0], 20_000.0).score
    shifted = evaluate_group(truth, actions[:, 1], 20_000.0).score
    assert np.mean(utility[:, 0]) > np.mean(utility[:, 1])
    # The utilities differ by exactly the same positive scaling as Score.
    assert np.isclose(np.mean(utility[:, 0] - utility[:, 1]), 2 * 20_000.0 * (exact - shifted))


def test_settlement_price_is_inclusive_at_both_cliffs() -> None:
    price = settlement_price(np.array([600.0, 600.001, 800.0, 800.001]), capacity=10_000.0)
    assert price.tolist() == [4.0, 3.0, 3.0, 0.0]


def test_quantile_repair_and_quadrature_weights() -> None:
    crossing = np.array([[3.0, 1.0, 2.0]])
    assert enforce_noncrossing(crossing).tolist() == [[1.0, 2.0, 3.0]]
    weights = quantile_scenario_weights(DEFAULT_QUANTILE_LEVELS)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0.0)
    assert not np.allclose(weights, weights[0])


def test_bayes_action_is_bounded_and_keeps_exact_reference() -> None:
    quantiles = np.full((4, len(DEFAULT_QUANTILE_LEVELS)), 10_000.0)
    result = metric_aware_bayes_action(
        quantiles,
        capacity=20_000.0,
        mean_eligible_generation=10_000.0,
        point_candidates=np.full(4, 9_000.0),
    )
    np.testing.assert_allclose(result.action, 10_000.0)
    np.testing.assert_allclose(result.expected_advantage, 0.0)
    assert np.all((result.action >= 0.0) & (result.action <= 20_000.0))


def test_shrinkage_endpoints() -> None:
    reference = np.array([100.0, 200.0])
    action = np.array([300.0, 400.0])
    np.testing.assert_allclose(shrink_action(reference, action, 0.0, 1_000.0), reference)
    np.testing.assert_allclose(shrink_action(reference, action, 1.0, 1_000.0), action)
