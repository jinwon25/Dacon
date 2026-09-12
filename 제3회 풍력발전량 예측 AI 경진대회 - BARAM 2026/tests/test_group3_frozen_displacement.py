from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.group3_frozen_displacement import (
    CANDIDATES,
    CAPACITY,
    FrozenRule,
    apply_overlay,
    audit_availability,
    group_row_masks,
    prior_day_cutoffs,
    shift_within_issue,
    target_date_block_bootstrap,
    target_dates,
)
from src.metrics import CAPACITY_KWH, evaluate_competition, evaluate_group


def test_preregistered_candidate_family_is_fixed_at_six_unique_members() -> None:
    identifiers = [candidate.candidate_id for candidate in CANDIDATES]
    assert len(identifiers) == 6
    assert len(set(identifiers)) == 6
    assert identifiers == [
        "d1_nearest_grid",
        "d2_upstream_1p5km",
        "d3_upstream_3p0km",
        "d4_temporal_minus1h",
        "d5_temporal_plus1h",
        "d6_upstream_1p5km_mean_pm1h",
    ]


def test_forecast_midnight_belongs_to_preceding_target_day_cutoff() -> None:
    index = pd.DatetimeIndex(
        ["2024-01-02 23:00:00", "2024-01-03 00:00:00", "2024-01-03 01:00:00"]
    )
    assert target_dates(index).tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert prior_day_cutoffs(index).tolist() == [
        pd.Timestamp("2024-01-01 14:00:00"),
        pd.Timestamp("2024-01-01 14:00:00"),
        pd.Timestamp("2024-01-02 14:00:00"),
    ]


def test_availability_audit_accepts_cutoff_and_rejects_one_second_after() -> None:
    index = pd.DatetimeIndex(["2024-01-03 00:00:00", "2024-01-03 01:00:00"])
    legal = pd.DatetimeIndex(["2024-01-01 14:00:00", "2024-01-02 14:00:00"])
    report = audit_availability(index, legal, "synthetic")
    assert report["cutoff_violations"] == 0
    illegal = legal.copy().to_numpy()
    illegal[1] = np.datetime64("2024-01-02T14:00:01")
    with pytest.raises(AssertionError, match="availability after cutoff"):
        audit_availability(index, illegal, "synthetic")


def test_temporal_shift_never_crosses_issue_boundary() -> None:
    values = np.asarray([10.0, 11.0, 12.0, 20.0, 21.0])
    issues = pd.to_datetime(
        ["2024-01-01"] * 3 + ["2024-01-02"] * 2
    ).to_numpy()
    assert shift_within_issue(values, issues, -1).tolist() == [10.0, 10.0, 11.0, 20.0, 20.0]
    assert shift_within_issue(values, issues, 1).tolist() == [11.0, 12.0, 12.0, 21.0, 21.0]


def test_overlay_is_bounded_clipped_and_group3_only_by_construction() -> None:
    incumbent = np.asarray([0.0, 10_000.0, CAPACITY])
    centered = np.asarray([10_000.0, 10_000.0, 10_000.0])
    displaced = np.asarray([20_000.0, 0.0, 20_000.0])
    rule = FrozenRule("d1_nearest_grid", 0.50)
    candidate = apply_overlay(incumbent, centered, displaced, rule)
    assert candidate.tolist() == [210.0, 9790.0, CAPACITY]
    assert np.max(np.abs(candidate - incumbent)) <= 0.01 * CAPACITY


def test_group_masks_are_disjoint_exhaustive_and_ordered() -> None:
    masks = group_row_masks(
        {"kpx_group_1": 2, "kpx_group_2": 3, "kpx_group_3": 1}
    )
    assert masks["kpx_group_1"].tolist() == [True, True, False, False, False, False]
    assert masks["kpx_group_2"].tolist() == [False, False, True, True, True, False]
    assert masks["kpx_group_3"].tolist() == [False, False, False, False, False, True]
    assert np.all(np.column_stack(list(masks.values())).sum(axis=1) == 1)


def test_official_macro_delta_is_exactly_group3_delta_divided_by_three() -> None:
    truth = {
        "kpx_group_1": np.asarray([8_000.0, 12_000.0]),
        "kpx_group_2": np.asarray([7_000.0, 13_000.0]),
        "kpx_group_3": np.asarray([6_000.0, 15_000.0]),
    }
    incumbent = {name: values - 2_000.0 for name, values in truth.items()}
    candidate = {name: values.copy() for name, values in incumbent.items()}
    candidate["kpx_group_3"] = truth["kpx_group_3"].copy()
    macro_before = evaluate_competition(truth, incumbent)
    macro_after = evaluate_competition(truth, candidate)
    g3_before = evaluate_group(
        truth["kpx_group_3"], incumbent["kpx_group_3"], CAPACITY_KWH["kpx_group_3"]
    )
    g3_after = evaluate_group(
        truth["kpx_group_3"], candidate["kpx_group_3"], CAPACITY_KWH["kpx_group_3"]
    )
    assert macro_after["score"] - macro_before["score"] == pytest.approx(
        (g3_after.score - g3_before.score) / 3.0, abs=1e-14
    )
    assert np.array_equal(candidate["kpx_group_1"], incumbent["kpx_group_1"])
    assert np.array_equal(candidate["kpx_group_2"], incumbent["kpx_group_2"])


def test_target_date_bootstrap_keeps_midnight_in_preceding_block() -> None:
    index = pd.date_range("2024-01-01 01:00:00", periods=48, freq="h")
    truth = np.full(48, 10_000.0)
    incumbent = np.full(48, 8_000.0)
    candidate = truth.copy()
    result = target_date_block_bootstrap(
        truth, incumbent, candidate, index, repetitions=50, seed=7
    )
    assert result["blocks"] == 2
    assert result["p_delta_gt_0"] == 1.0
    assert result["q05"] > 0.0
