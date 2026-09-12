from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import experiments.kma_um_power_curve_gate as power_curve_gate
from experiments.kma_um_power_curve_gate import (
    CAPACITY,
    GatePolicy,
    apply_bounded_combo,
    issue_block_bootstrap,
    load_gfs_850_speed,
    make_submission_if_qualified,
    policy_gate,
)


def test_policy_gate_enforces_direction_threshold_and_base_band() -> None:
    disagreement = np.asarray([500.0, 2_000.0, -3_000.0, 4_000.0])
    reference = CAPACITY * np.asarray([0.05, 0.20, 0.40, 0.90])
    policy = GatePolicy("up", 0.25, 0.10, 0.80, 1_000.0)
    gate = policy_gate(
        disagreement, reference, np.ones(4, dtype=bool), policy
    )
    assert gate.tolist() == [False, True, False, False]


def test_bounded_combo_is_disjoint_and_caps_each_movement() -> None:
    reference = np.full(4, 8_000.0)
    kma = np.asarray([8_000.0, 8_000.0, 8_000.0, -8_000.0])
    gfs = np.asarray([8_000.0, 0.0, 8_000.0, 0.0])
    policy = GatePolicy("both", 1.0, 0.10, 1.00, None)
    control, candidate, gfs_gate, kma_gate = apply_bounded_combo(
        reference,
        kma,
        gfs,
        np.ones(4, dtype=bool),
        policy,
        policy,
        kma_alpha=1.0,
        gfs_alpha=1.0,
        maximum_movement_ratio=0.05,
    )
    assert not np.any(gfs_gate & kma_gate)
    assert np.max(np.abs(control - reference)) <= 0.05 * CAPACITY + 1e-9
    assert np.max(np.abs(candidate - reference)) <= 0.05 * CAPACITY + 1e-9


def test_issue_bootstrap_reports_positive_deterministic_improvement() -> None:
    hours = 24 * 20
    truth = np.full(hours, 10_000.0)
    reference = np.full(hours, 8_000.0)
    candidate = np.full(hours, 9_000.0)
    issues = np.repeat(pd.date_range("2024-07-01", periods=20, freq="D"), 24)
    report = issue_block_bootstrap(
        truth,
        reference,
        candidate,
        issues,
        np.ones(hours, dtype=bool),
        n_bootstrap=25,
        seed=7,
    )
    assert report["unique_issue_cycles"] == 20
    assert min(report["q05"].values()) > 0.0
    assert report["positive_all_component_fraction"] == 1.0


def test_bounded_combo_rejects_unsafe_movement_ratio() -> None:
    values = np.full(2, 5_000.0)
    policy = GatePolicy("up", 1.0, 0.10, 1.00, None)
    with pytest.raises(ValueError, match="movement ratio"):
        apply_bounded_combo(
            values,
            values,
            values,
            np.ones(2, dtype=bool),
            policy,
            policy,
            kma_alpha=0.1,
            gfs_alpha=0.1,
            maximum_movement_ratio=0.11,
        )


def test_gfs_loader_preserves_values_when_assigning_datetime_index(
    tmp_path,
) -> None:
    path = tmp_path / "gfs.csv"
    pd.DataFrame(
        {
            "forecast_kst_dtm": ["2024-01-01 01:00:00", "2024-01-01 02:00:00"],
            "grid_id": [5, 5],
            "isobaricInhPa_850_u": [3.0, 5.0],
            "isobaricInhPa_850_v": [4.0, 12.0],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")
    speed = load_gfs_850_speed(path)
    assert speed.tolist() == pytest.approx([5.0, 13.0])
    assert speed.index.equals(
        pd.DatetimeIndex(["2024-01-01 01:00:00", "2024-01-01 02:00:00"])
    )


def test_submission_writer_preserves_other_groups_and_bounds_group3(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        power_curve_gate,
        "validate_external_data_manifest",
        lambda _manifest, _root: None,
    )
    test_index = pd.date_range("2025-01-01 01:00:00", periods=6, freq="h")
    context_path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "forecast_kst_dtm": test_index,
            "data_available_kst_dtm": pd.Timestamp("2024-12-31 13:00:00"),
            "kma_um_ctx_speed10_r0": np.linspace(4.0, 10.0, len(test_index)),
        }
    ).to_csv(context_path, index=False, encoding="utf-8-sig")
    gfs_path = tmp_path / "gfs.csv"
    pd.DataFrame(
        {
            "forecast_kst_dtm": test_index,
            "grid_id": 5,
            "isobaricInhPa_850_u": np.linspace(5.0, 11.0, len(test_index)),
            "isobaricInhPa_850_v": 0.0,
        }
    ).to_csv(gfs_path, index=False, encoding="utf-8-sig")
    base_path = tmp_path / "base.csv"
    base = pd.DataFrame(
        {
            "forecast_kst_dtm": test_index,
            "kpx_group_1": np.arange(len(test_index), dtype=float) + 1_000.0,
            "kpx_group_2": np.arange(len(test_index), dtype=float) + 2_000.0,
            "kpx_group_3": np.full(len(test_index), 5_000.0),
        }
    )
    base.to_csv(base_path, index=False, encoding="utf-8-sig")
    train_wind = np.linspace(1.0, 15.0, 1_200)
    proxy = np.linspace(0.0, CAPACITY, 1_200)
    output_path = tmp_path / "candidate.csv"
    policy = GatePolicy("up", 1.0, 0.10, 1.00, None)
    report = make_submission_if_qualified(
        qualified=True,
        test_manifest=tmp_path / "manifest.json",
        test_features=context_path,
        test_gfs=gfs_path,
        base_submission=base_path,
        output_submission=output_path,
        train_kma=train_wind,
        train_gfs=train_wind,
        proxy_label=proxy,
        train_available=np.ones(len(train_wind), dtype=bool),
        kma_policy=policy,
        gfs_policy=policy,
        kma_alpha=0.2,
        gfs_alpha=0.2,
        maximum_movement_ratio=0.05,
    )
    assert report is not None
    assert report["rows"] == len(test_index)
    candidate = pd.read_csv(output_path, encoding="utf-8-sig")
    assert candidate["kpx_group_1"].equals(base["kpx_group_1"])
    assert candidate["kpx_group_2"].equals(base["kpx_group_2"])
    assert candidate["kpx_group_3"].between(0.0, CAPACITY).all()
    assert report["maximum_absolute_movement_kwh"] <= 0.05 * CAPACITY + 1e-9
