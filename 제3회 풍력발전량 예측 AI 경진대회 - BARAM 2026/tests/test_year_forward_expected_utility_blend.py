from __future__ import annotations

import numpy as np
import pytest

from experiments.year_forward_expected_utility_blend import (
    apply_rank_policy,
    choose_bayes_action,
    expected_official_utility,
    parse_targets,
)


def test_expected_official_utility_prefers_exact_action() -> None:
    samples = np.asarray([[10_000.0, 10_000.0, 10_000.0]])
    actions = np.asarray([[8_000.0, 10_000.0, 12_000.0]])
    utility = expected_official_utility(
        actions,
        samples,
        capacity=20_000.0,
        mean_eligible_generation=10_000.0,
    )
    assert int(np.argmax(utility[0])) == 1


def test_choose_bayes_action_keeps_optimal_reference() -> None:
    samples = np.full((3, 5), 10_000.0)
    reference = np.full(3, 10_000.0)
    action, advantage = choose_bayes_action(
        samples,
        reference,
        capacity=20_000.0,
        mean_eligible_generation=10_000.0,
        batch_size=2,
    )
    np.testing.assert_allclose(action, reference)
    np.testing.assert_allclose(advantage, 0.0)


def test_apply_rank_policy_respects_coverage_and_total_cap() -> None:
    incumbent = np.asarray([10.0, 10.0, 10.0, 10.0])
    reference = np.asarray([10.0, 10.0, 10.0, 10.0])
    action = np.asarray([20.0, 20.0, 20.0, 20.0])
    advantage = np.asarray([1.0, 2.0, 3.0, 4.0])
    candidate, gate = apply_rank_policy(
        incumbent,
        reference,
        action,
        advantage,
        top_fraction=0.5,
        weight=1.0,
        capacity=100.0,
        total_movement_cap=0.06,
    )
    assert gate.tolist() == [False, False, True, True]
    np.testing.assert_allclose(candidate, np.asarray([10.0, 10.0, 12.5, 12.5]))


def test_apply_rank_policy_rejects_bad_fraction() -> None:
    values = np.ones(2)
    with pytest.raises(ValueError, match="top fraction"):
        apply_rank_policy(
            values,
            values,
            values,
            values,
            top_fraction=0.0,
            weight=0.1,
            capacity=100.0,
            total_movement_cap=0.05,
        )


def test_parse_targets_supports_group3_and_rejects_duplicates() -> None:
    assert parse_targets("kpx_group_3") == ("kpx_group_3",)
    with pytest.raises(ValueError, match="unique"):
        parse_targets("kpx_group_3,kpx_group_3")
