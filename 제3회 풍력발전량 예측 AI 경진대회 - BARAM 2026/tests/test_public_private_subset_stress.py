from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.public_private_subset_stress import (
    complementary_subset_stress,
)


def test_complementary_subset_stress_is_reproducible() -> None:
    index = pd.date_range("2024-01-01", periods=240, freq="h")
    truth = {
        "kpx_group_1": np.full(len(index), 10_000.0),
        "kpx_group_2": np.full(len(index), 10_000.0),
    }
    reference = {
        target: np.full(len(index), 8_000.0)
        for target in truth
    }
    candidate = {
        target: np.full(len(index), 9_000.0)
        for target in truth
    }
    first = complementary_subset_stress(
        truth,
        reference,
        candidate,
        index,
        public_fraction=0.40,
        repetitions=20,
        seed=7,
        stratify_month=False,
    )
    second = complementary_subset_stress(
        truth,
        reference,
        candidate,
        index,
        public_fraction=0.40,
        repetitions=20,
        seed=7,
        stratify_month=False,
    )
    assert first == second
    assert np.isclose(first["public"]["score"]["positive_fraction"], 1.0)
