from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.group12_selective_gain_gate import (
    apply_selective_gate,
    make_gate_design,
    normalized_gain,
)


def test_positive_gain_means_absolute_error_improves() -> None:
    truth = np.asarray([10.0, 10.0, 10.0])
    reference = np.asarray([8.0, 8.0, 12.0])
    proposal = np.asarray([9.0, 11.0, 13.0])
    gain = normalized_gain(truth, reference, proposal, capacity=10.0)
    assert np.allclose(gain, [0.1, 0.1, -0.1])


def test_gate_applies_only_positive_lower_gain() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h")
    pair_features = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=index)
    reference = {
        "kpx_group_1": np.asarray([1.0, 2.0, 3.0]),
        "kpx_group_2": np.asarray([3.0, 2.0, 1.0]),
    }
    proposal = {
        "kpx_group_1": np.asarray([1.5, 2.5, 3.5]),
        "kpx_group_2": np.asarray([2.5, 1.5, 0.5]),
    }
    design, slices = make_gate_design(
        pair_features,
        reference,
        proposal,
    )
    lower = np.asarray([0.1, -0.1, 0.2, -0.1, 0.1, 0.2])
    candidate, gates = apply_selective_gate(
        reference,
        proposal,
        lower,
        slices,
        margin=0.0,
    )
    assert len(design) == 6
    assert gates["kpx_group_1"].tolist() == [True, False, True]
    assert gates["kpx_group_2"].tolist() == [False, True, True]
    assert np.allclose(candidate["kpx_group_1"], [1.5, 2.0, 3.5])
