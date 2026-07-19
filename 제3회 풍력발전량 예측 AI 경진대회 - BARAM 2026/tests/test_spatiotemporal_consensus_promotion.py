from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent_service.adapters import spatiotemporal_multitask_evaluation
from experiments.spatiotemporal_consensus_promotion import (
    BlendPolicy,
    apply_blend,
    issue_period_masks,
    policy_mask,
    select_development_policy,
)


def test_policy_mask_combines_seed_consensus_uncertainty_and_disagreement() -> None:
    base = np.asarray([100.0, 100.0, 100.0, 100.0])
    seeds = np.asarray(
        [
            [110.0, 110.0, 200.0, 102.0],
            [120.0, 90.0, 210.0, 103.0],
        ]
    )
    policy = BlendPolicy(
        "bounded", 0.20, True, max_seed_uncertainty=0.01, max_base_disagreement=0.01
    )
    mask, diagnostics = policy_mask(base, seeds, policy)
    assert diagnostics["seed_agreement"].tolist() == [True, False, True, True]
    # Group-3 capacity is much larger than these toy values, so agreeing rows
    # also satisfy both normalized bounds.
    assert mask.tolist() == [True, False, True, True]


def test_apply_blend_only_moves_selected_rows_and_clips() -> None:
    base = np.asarray([10.0, 20.0, 30.0])
    member = np.asarray([-100.0, 100.0, 50.0])
    result = apply_blend(base, member, np.asarray([True, False, True]), 0.50)
    assert result.tolist() == [0.0, 20.0, 40.0]


def test_issue_period_masks_do_not_split_one_issue_cycle() -> None:
    timestamps = pd.DatetimeIndex(
        ["2024-03-31 23:00", "2024-04-01 00:00", "2024-04-01 12:00", "2024-07-01 12:00"]
    )
    issues = pd.DatetimeIndex(
        ["2024-03-31 06:00", "2024-03-31 06:00", "2024-04-01 06:00", "2024-07-01 06:00"]
    )
    masks = issue_period_masks(timestamps, issues)
    assert masks["q1"].tolist() == [True, True, False, False]
    assert masks["q2"].tolist() == [False, False, True, False]
    assert masks["h2"].tolist() == [False, False, False, True]


def _record(name: str, q1: float, q2: float, alpha: float = 0.2) -> dict[str, object]:
    def period(value: float) -> dict[str, object]:
        delta = {"score": value, "one_minus_nmae": value, "ficr": value}
        return {
            "ensemble_blocked": {
                "robustness_passed": value > 0,
                "overall": {"delta": delta},
            },
            "seed_metrics": [
                {"metrics": {"delta": delta}},
                {"metrics": {"delta": delta}},
            ],
        }

    return {
        "policy": {"name": name, "alpha": alpha},
        "coverage_ratio": 0.5,
        "q1": period(q1),
        "q2": period(q2),
    }


def test_select_development_policy_maximizes_worst_q1_q2_only() -> None:
    selected = select_development_policy(
        [_record("high_average", 0.001, 0.020), _record("robust", 0.005, 0.006)]
    )
    assert selected["policy"]["name"] == "robust"


def test_select_development_policy_rejects_locked_h2_input() -> None:
    record = _record("leaky", 0.005, 0.006)
    record["h2"] = {"score": 1.0}
    with pytest.raises(ValueError, match="must not contain locked H2"):
        select_development_policy([record])


def test_select_development_policy_excludes_non_deployable_family() -> None:
    non_deployable = _record("consensus", 0.010, 0.010)
    non_deployable["deployable_from_retained_test_artifacts"] = False
    deployable = _record("global", 0.005, 0.006)
    deployable["deployable_from_retained_test_artifacts"] = True
    selected = select_development_policy([non_deployable, deployable])
    assert selected["policy"]["name"] == "global"


def test_service_adapter_uses_locked_ensemble_and_structural_family() -> None:
    report = {
        "locked_h2_ensemble": {
            "overall": {
                "delta": {"score": 0.006, "one_minus_nmae": 0.002, "ficr": 0.010}
            },
            "monthly": {
                "2024-07": {"delta": {"score": 0.001}},
                "2024-08": {"delta": {"score": 0.002}},
            },
            "issue_block_bootstrap": {"positive_fraction": 0.98, "q05": 0.0008},
        },
        "submission": {
            "sha256": "abc",
            "changed_ratio": 1.0,
            "p95_absolute_movement_kwh": 500.0,
            "candidate_validator": {"valid": True},
        },
    }
    evaluation = spatiotemporal_multitask_evaluation(report)
    assert evaluation.family == "spatiotemporal_multitask_blend"
    assert evaluation.expected_macro_score_delta == pytest.approx(0.002)
    assert evaluation.positive_months == evaluation.total_months == 2
