from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.blocked_rolling_validation import (
    PublicProbe,
    assign_issue_blocks,
    evaluate_blocked_rolling,
    issue_block_bootstrap,
    public_transfer_audit,
)


def _synthetic_series() -> tuple[np.ndarray, ...]:
    timestamps = pd.date_range("2024-06-01 01:00", periods=24 * 184, freq="h")
    # Each issue owns a complete 24-hour target horizon.
    issue_times = pd.DatetimeIndex(
        timestamps.normalize() - pd.Timedelta(hours=11)
    )
    truth = 8_000.0 + 1_500.0 * np.sin(np.arange(len(timestamps)) / 24.0)
    reference = truth + 600.0
    candidate = truth + 500.0
    rows = np.ones(len(timestamps), dtype=bool)
    return truth, reference, candidate, timestamps, issue_times, rows


def test_issue_block_assignment_never_splits_one_issue_cycle() -> None:
    timestamps = pd.DatetimeIndex(
        ["2024-02-29 23:00", "2024-03-01 00:00", "2024-03-01 01:00"]
    )
    issue_times = pd.DatetimeIndex(["2024-02-29 13:00"] * 3)
    months, seasons = assign_issue_blocks(timestamps, issue_times)
    assert len(set(months)) == 1
    assert len(set(seasons)) == 1


def test_issue_block_bootstrap_is_reproducible_and_positive() -> None:
    truth, reference, candidate, timestamps, issue_times, rows = _synthetic_series()
    _, seasons = assign_issue_blocks(timestamps, issue_times)
    first = issue_block_bootstrap(
        truth, reference, candidate, issue_times, seasons, rows, 50, seed=10
    )
    second = issue_block_bootstrap(
        truth, reference, candidate, issue_times, seasons, rows, 50, seed=10
    )
    assert first == second
    assert first["positive_fraction"] == 1.0
    assert first["q05"] > 0.0


def test_blocked_rolling_reports_worst_month_and_disjoint_issues() -> None:
    truth, reference, candidate, timestamps, issue_times, rows = _synthetic_series()
    report = evaluate_blocked_rolling(
        truth,
        reference,
        candidate,
        timestamps,
        issue_times,
        rows,
        n_bootstrap=30,
    )
    assert report["monthly_worst_case"]["block"] in report["monthly"]
    assert report["issue_integrity"]["split_issue_cycles"] == 0
    assert all(fold["issue_overlap"] == 0 for fold in report["rolling_folds"])
    assert report["robustness_passed"] is True


def test_worst_month_guard_rejects_average_gain_with_regime_failure() -> None:
    truth, reference, candidate, timestamps, issue_times, rows = _synthetic_series()
    # Improve five months but deliberately damage the final one. The full mean
    # remains positive, while the worst-month contract must reject it.
    last_month = timestamps.to_period("M") == timestamps[-1].to_period("M")
    candidate[last_month] = truth[last_month] + 1_000.0
    report = evaluate_blocked_rolling(
        truth,
        reference,
        candidate,
        timestamps,
        issue_times,
        rows,
        n_bootstrap=30,
    )
    assert report["overall"]["delta"]["score"] > 0.0
    assert report["monthly_worst_case"]["score_delta"] < 0.0
    assert report["robustness_passed"] is False


def test_public_transfer_sign_reversal_disables_automatic_projection() -> None:
    audit = public_transfer_audit(
        [
            PublicProbe(
                "narrow",
                {"score": 0.001, "one_minus_nmae": 0.0, "ficr": 0.002},
                {"score": 0.0004, "one_minus_nmae": 0.0, "ficr": 0.0008},
            ),
            PublicProbe(
                "broad",
                {"score": 0.004, "one_minus_nmae": 0.0, "ficr": 0.008},
                {"score": -0.003, "one_minus_nmae": -0.002, "ficr": -0.004},
            ),
        ]
    )
    assert audit["summary"]["sign_reversal_count"] == 1
    assert audit["summary"]["conservative_auto_projection_ratio"] == 0.0
    assert audit["summary"]["automatic_public_projection_trusted"] is False
