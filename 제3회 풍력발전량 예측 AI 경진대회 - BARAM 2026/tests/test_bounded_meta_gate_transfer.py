from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.bounded_meta_gate_transfer import (
    TARGET,
    build_strong_gate_candidate,
    macro_transfer_ratios,
)


def _frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_id": ["a", "b"],
            "forecast_kst_dtm": ["2026-01-01", "2026-01-02"],
            "kpx_group_1": [1.0, 2.0],
            "kpx_group_2": [3.0, 4.0],
            TARGET: values,
        }
    )


def test_macro_transfer_ratio_divides_group3_delta_by_three() -> None:
    ratios = macro_transfer_ratios(
        {"score": 0.1, "one_minus_nmae": 0.2, "ficr": 0.3},
        {"score": 0.6, "one_minus_nmae": 1.2, "ficr": 1.8},
    )
    assert ratios == pytest.approx(
        {"score": 0.5, "one_minus_nmae": 0.5, "ficr": 0.5}
    )


def test_strong_gate_candidate_doubles_only_group3_movement() -> None:
    source = _frame([100.0, 200.0])
    reference = _frame([110.0, 200.0])
    candidate = build_strong_gate_candidate(source, reference)
    np.testing.assert_allclose(candidate[TARGET], [120.0, 200.0])
    pd.testing.assert_series_equal(candidate["kpx_group_1"], reference["kpx_group_1"])
    pd.testing.assert_series_equal(candidate["kpx_group_2"], reference["kpx_group_2"])
