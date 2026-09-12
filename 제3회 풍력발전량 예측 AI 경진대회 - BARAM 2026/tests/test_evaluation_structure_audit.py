from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.evaluation_structure_audit import (
    audit_auxiliary_oof_coverage,
    published_group_formula,
    randomized_metric_parity,
)


def test_published_formula_has_inclusive_settlement_boundaries() -> None:
    capacity = 10_000.0
    truth = np.full(4, 5_000.0)
    prediction = truth + capacity * np.asarray([0.0, 0.06, 0.08, 0.080001])
    result = published_group_formula(truth, prediction, capacity)
    assert result["n_samples"] == 4
    assert np.isclose(result["ficr"], (4.0 + 4.0 + 3.0 + 0.0) / 16.0)


def test_randomized_metric_parity_is_exact() -> None:
    result = randomized_metric_parity(repetitions=10, seed=7)
    assert result["exact_within_1e_12"]


def test_auxiliary_oof_coverage_detects_truncated_tail(
    tmp_path, monkeypatch
) -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="h")
    driver_path = tmp_path / "driver.npz"
    auxiliary_path = tmp_path / "auxiliary.npz"
    np.savez_compressed(
        driver_path,
        **{
            f"kpx_group_{group}__valid_index_ns": index.astype("int64")
            for group in (1, 2, 3)
        },
    )
    np.savez_compressed(
        auxiliary_path,
        index_ns=index[:-2].astype("int64"),
    )
    import experiments.evaluation_structure_audit as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    result = audit_auxiliary_oof_coverage(
        driver_path, [auxiliary_path]
    )
    record = result["records"][0]
    assert not result["all_full_year_complete"]
    assert record["missing_annual_rows"] == 2
    assert record["first_missing"] == str(index[-2])
