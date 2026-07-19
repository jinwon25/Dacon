from __future__ import annotations

import numpy as np
import pytest

import experiments.group3_covariate_shift as experiment
from experiments.group3_covariate_shift import (
    MAX_WEIGHT,
    MIN_WEIGHT,
    assess_promotion,
    bounded_mean_one_weights,
    select_policy,
)


def test_bounded_weight_normalization_preserves_clip_contract() -> None:
    raw = np.asarray([1e-6] * 80 + [1.0] * 10 + [1e6] * 10)
    weights = bounded_mean_one_weights(raw)

    assert weights.mean() == pytest.approx(1.0, abs=1e-12)
    assert weights.min() >= MIN_WEIGHT
    assert weights.max() <= MAX_WEIGHT
    assert weights[-1] >= weights[80] >= weights[0]


@pytest.mark.parametrize(
    "raw",
    [np.asarray([]), np.asarray([[1.0]]), np.asarray([0.0, 1.0]), np.asarray([np.nan])],
)
def test_bounded_weight_normalization_rejects_invalid_ratios(raw: np.ndarray) -> None:
    with pytest.raises(ValueError):
        bounded_mean_one_weights(raw)


def test_policy_month_report_uses_issue_cycle_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    n_rows = 100
    truth = np.full(n_rows, 0.20 * experiment.CAPACITY)
    current = truth + 0.02 * experiment.CAPACITY
    members = np.repeat(current[:, None], len(experiment.SEEDS), axis=1)
    changed = np.r_[0:10, 50:60]
    members[changed] -= 0.015 * experiment.CAPACITY
    # The two complete issue cycles intentionally straddle arbitrary calendar
    # rows.  The selector must report the supplied dependency-block labels,
    # rather than deriving a month independently from each row.
    month_blocks = np.asarray(["2024-04"] * 50 + ["2024-05"] * 50)

    def fake_compare(
        _truth: np.ndarray,
        reference: np.ndarray,
        candidate: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        movement = float(np.mean(np.abs(candidate - reference)) / experiment.CAPACITY)
        return {
            "delta": {
                "score": movement,
                "one_minus_nmae": movement,
                "ficr": movement,
            }
        }

    monkeypatch.setattr(experiment, "_compare", fake_compare)
    _, leaderboard = select_policy(truth, current, members, month_blocks)

    assert leaderboard
    assert set(leaderboard[0]["monthly"]) == {"2024-04", "2024-05"}
    assert all(value > 0.0 for value in leaderboard[0]["monthly"].values())


def test_promotion_rejects_tiny_unstable_locked_gain() -> None:
    locked = {
        "delta": {
            "score": 0.0000466,
            "one_minus_nmae": 0.0000318,
            "ficr": 0.0000613,
        }
    }
    monthly = {
        str(month): {"score": value}
        for month, value in enumerate([1e-5, -4e-4, -2e-4, 5e-5, 2e-5, 2e-4], start=7)
    }
    bootstrap = {"positive_fraction": 0.6815, "q05": -0.0000759}

    result = assess_promotion(locked, monthly, bootstrap, 0.0816, 0.002)

    assert result["passed"] is False
    assert result["failed_gates"] == [
        "locked_score_delta",
        "bootstrap_positive_fraction",
    ]
