from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.multimodel_expanding_quantile_blend import (
    _folds,
    fit_independent_source_ensemble,
    load_context_bundle,
    make_pooled_design,
)


def _context(path, prefix: str, start: str = "2024-02-10") -> None:
    index = pd.date_range(start, periods=4, freq="h")
    pd.DataFrame(
        {
            "forecast_kst_dtm": index,
            "data_available_kst_dtm": index - pd.Timedelta(hours=18),
            f"kma_um_ctx_{prefix}__safe_speed10": np.arange(4, dtype=float),
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def test_context_bundle_inner_joins_and_namespaces(tmp_path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _context(first, "a")
    _context(second, "b", start="2024-02-10 01:00")

    features, issue = load_context_bundle((first, second))

    assert len(features) == 3
    assert issue.index.equals(features.index)
    assert "source1__kma_um_ctx_a__safe_speed10" in features
    assert "source2__kma_um_ctx_b__safe_speed10" in features
    assert {"doy_sin", "doy_cos", "hour_sin", "hour_cos"}.issubset(
        features.columns
    )


def test_pooled_design_carries_target_baseline_and_one_hot() -> None:
    index = pd.date_range("2024-04-01", periods=3, freq="h")
    features = pd.DataFrame({"wind": [1.0, 2.0, 3.0]}, index=index)
    baselines = {
        "kpx_group_1": pd.Series([2160.0] * 3, index=index),
        "kpx_group_2": pd.Series([4320.0] * 3, index=index),
        "kpx_group_3": pd.Series([6300.0] * 3, index=index),
    }

    pooled, slices = make_pooled_design(features, baselines)

    assert pooled.shape[0] == 9
    assert np.allclose(
        pooled.iloc[slices["kpx_group_1"]]["frozen_baseline_ratio"],
        0.10,
    )
    assert np.allclose(
        pooled.iloc[slices["kpx_group_3"]]["frozen_baseline_ratio"],
        0.30,
    )
    assert pooled.iloc[slices["kpx_group_2"]][
        "pooled_group__kpx_group_2"
    ].eq(1.0).all()


def test_folds_are_expanding_and_non_overlapping() -> None:
    index = pd.date_range("2024-02-10", "2024-12-31 23:00", freq="h")
    folds = _folds(index)

    assert [fold["name"] for fold in folds] == [
        "selection_q2",
        "confirmation_q3",
        "confirmation_q4",
    ]
    assert [int(fold["train"].sum()) for fold in folds] == sorted(
        int(fold["train"].sum()) for fold in folds
    )
    for fold in folds:
        assert not np.any(fold["train"] & fold["query"])


def test_independent_source_ensemble_uses_robust_median(monkeypatch) -> None:
    index = pd.date_range("2024-04-01", periods=3, freq="h")
    sources = tuple(
        pd.DataFrame({"source_value": [value] * 3}, index=index)
        for value in (1.0, 2.0, 100.0)
    )
    baselines = {
        target: pd.Series([0.0] * 3, index=index)
        for target in ("kpx_group_1", "kpx_group_2", "kpx_group_3")
    }
    labels = pd.DataFrame(index=index)

    def fake_fit(train_features, *args, **kwargs):
        value = train_features["source_value"].iloc[0]
        return {
            target: np.full(3, value, dtype=float)
            for target in baselines
        }

    monkeypatch.setattr(
        "experiments.multimodel_expanding_quantile_blend.fit_pooled_expert",
        fake_fit,
    )
    prediction = fit_independent_source_ensemble(
        sources,
        baselines,
        labels,
        sources,
        baselines,
        alpha=0.5,
        seed=13,
        n_estimators=10,
        aggregation="median",
    )

    assert all(np.allclose(values, 2.0) for values in prediction.values())


def test_independent_source_ensemble_rejects_misaligned_indexes() -> None:
    index = pd.date_range("2024-04-01", periods=3, freq="h")
    shifted = index + pd.Timedelta(hours=1)
    sources = (
        pd.DataFrame({"wind": [1.0] * 3}, index=index),
        pd.DataFrame({"wind": [2.0] * 3}, index=shifted),
    )
    baselines = {
        target: pd.Series([0.0] * 3, index=index)
        for target in ("kpx_group_1", "kpx_group_2", "kpx_group_3")
    }

    with np.testing.assert_raises_regex(
        ValueError, "independent train-source indexes differ"
    ):
        fit_independent_source_ensemble(
            sources,
            baselines,
            pd.DataFrame(index=index),
            sources,
            baselines,
            alpha=0.5,
            seed=13,
            n_estimators=10,
        )
