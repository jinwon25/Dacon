from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.kma_year_forward_quantile_blend import (
    add_base_physical_features,
    apply_bounded_blend,
    load_context_feature_bundle,
    parse_targets,
    select_weight,
)


def test_bounded_blend_never_exceeds_capacity_ratio() -> None:
    reference = np.asarray([1_000.0, 10_000.0, 20_000.0])
    expert = np.asarray([20_000.0, 0.0, 1_000.0])
    result = apply_bounded_blend(
        reference,
        expert,
        weight=0.50,
        capacity=21_600.0,
        maximum_movement_ratio=0.05,
    )
    assert np.max(np.abs(result - reference)) <= 0.05 * 21_600.0
    assert np.all((result >= 0.0) & (result <= 21_600.0))


def test_select_weight_uses_only_supplied_development_rows() -> None:
    capacity = 10_000.0
    truth = np.asarray([5_000.0, 5_000.0, 5_000.0, 5_000.0])
    reference = np.asarray([4_000.0, 4_000.0, 5_000.0, 5_000.0])
    expert = np.asarray([5_000.0, 5_000.0, 0.0, 0.0])
    development = np.asarray([True, True, False, False])
    selected, _ = select_weight(
        truth,
        reference,
        expert,
        capacity,
        development,
        weights=(0.10, 0.20),
    )
    assert selected["weight"] in {0.10, 0.20}


def test_select_weight_rejects_component_harm() -> None:
    capacity = 10_000.0
    truth = np.full(8, 5_000.0)
    reference = truth.copy()
    expert = np.zeros(8)
    development = np.ones(8, dtype=bool)
    try:
        select_weight(
            truth,
            reference,
            expert,
            capacity,
            development,
            weights=(0.10,),
        )
    except RuntimeError as exc:
        assert "no year-forward blend weight" in str(exc)
    else:
        raise AssertionError("harmful development blend should be rejected")


def test_select_weight_prefers_smaller_near_tied_footprint() -> None:
    capacity = 10_000.0
    truth = np.full(100, 5_000.0)
    reference = np.full(100, 4_200.0)
    expert = truth.copy()
    development = np.ones(100, dtype=bool)
    selected, records = select_weight(
        truth,
        reference,
        expert,
        capacity,
        development,
        weights=(0.25, 0.251),
    )
    assert all(record["eligible"] for record in records)
    assert selected["weight"] == 0.25


def test_parse_targets_accepts_stable_supported_subset() -> None:
    assert parse_targets("kpx_group_1") == ("kpx_group_1",)
    assert parse_targets("kpx_group_3,kpx_group_2,kpx_group_1") == (
        "kpx_group_3",
        "kpx_group_2",
        "kpx_group_1",
    )


def test_parse_targets_rejects_duplicates_and_unknowns() -> None:
    for value in ("kpx_group_1,kpx_group_1", "kpx_group_4", ""):
        try:
            parse_targets(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid target subset should be rejected: {value!r}")


def test_context_bundle_prefixes_extra_model_features(tmp_path) -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="h")
    primary = pd.DataFrame(
        {
            "forecast_kst_dtm": timestamps,
            "data_available_kst_dtm": timestamps - pd.Timedelta(hours=12),
            "kma_um_ctx_u10_r0": [1.0, 2.0, 3.0],
        }
    )
    extra = primary.copy()
    extra["kma_um_ctx_u10_r0"] = [4.0, 5.0, 6.0]
    primary_path = tmp_path / "primary.csv"
    extra_path = tmp_path / "extra.csv"
    primary.to_csv(primary_path, index=False, encoding="utf-8-sig")
    extra.to_csv(extra_path, index=False, encoding="utf-8-sig")

    features, issue = load_context_feature_bundle(
        primary_path,
        (extra_path,),
    )

    assert "kma_um_ctx_u10_r0" in features
    assert "extra1__kma_um_ctx_u10_r0" in features
    assert features["extra1__kma_um_ctx_u10_r0"].tolist() == [4.0, 5.0, 6.0]
    assert issue.notna().all()


def test_rich_physical_join_is_target_specific_and_aligned() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h")
    context = pd.DataFrame(
        {"kma_um_ctx_u10_r0": [1.0, 2.0, 3.0]},
        index=index,
    )
    base = pd.DataFrame(
        {
            "hour": [0.0, 1.0, 2.0],
            "ldaps__10u__mean": [4.0, 5.0, 6.0],
            "idw__kpx_group_1__ws": [7.0, 8.0, 9.0],
            "idw__kpx_group_3__ws": [10.0, 11.0, 12.0],
            "irrelevant": [13.0, 14.0, 15.0],
        },
        index=index,
    )
    combined = add_base_physical_features(
        context,
        base,
        "kpx_group_1",
    )
    assert "base__ldaps__10u__mean" in combined
    assert "base__idw__kpx_group_1__ws" in combined
    assert "base__idw__kpx_group_3__ws" not in combined
    assert "base__irrelevant" not in combined


def test_ldaps_wind_join_excludes_nonwind_and_other_targets() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="h")
    context = pd.DataFrame(
        {"kma_um_ctx_u10_r0": [1.0, 2.0]},
        index=index,
    )
    base = pd.DataFrame(
        {
            "hour": [0.0, 1.0],
            "ldaps__10u__mean": [3.0, 4.0],
            "ldaps__temperature__mean": [5.0, 6.0],
            "ldaps__kpx_group_1__hub_ws117__idw": [7.0, 8.0],
            "ldaps__kpx_group_3__hub_ws117__idw": [9.0, 10.0],
        },
        index=index,
    )
    combined = add_base_physical_features(
        context,
        base,
        "kpx_group_1",
        mode="ldaps_wind",
    )
    assert "base__ldaps__10u__mean" in combined
    assert "base__ldaps__kpx_group_1__hub_ws117__idw" in combined
    assert "base__ldaps__temperature__mean" not in combined
    assert "base__ldaps__kpx_group_3__hub_ws117__idw" not in combined
