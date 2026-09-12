from __future__ import annotations

import numpy as np

from experiments.kma_group2_overlay_supplement import (
    SupplementPolicy,
    apply_supplement,
)


def test_supplement_never_overwrites_core_rows() -> None:
    reference = {
        "kpx_group_1": np.array([1.0, 1.0, 1.0]),
        "kpx_group_2": np.array([5_000.0, 6_000.0, 7_000.0]),
        "kpx_group_3": np.array([1.0, 1.0, 1.0]),
    }
    member = {
        **reference,
        "kpx_group_2": np.array([6_000.0, 7_000.0, 8_000.0]),
    }
    core = {group: values.copy() for group, values in reference.items()}
    core["kpx_group_2"][0] = 5_200.0
    policy = SupplementPolicy("up", 1.0, 0.20, 0.80, 0.0, 0.10)

    candidate, gate = apply_supplement(
        core,
        reference,
        member,
        np.array([True, False, False]),
        policy,
    )

    assert not gate[0]
    assert candidate["kpx_group_2"][0] == 5_200.0
    assert candidate["kpx_group_2"][1] == 6_100.0
