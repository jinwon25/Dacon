from __future__ import annotations

import numpy as np

from src.archive.v210_h1_workload_noncore_delta_audit import (
    apply_noncore_delta,
    restrictions,
)


def test_noncore_delta_changes_only_requested_exact_domain() -> None:
    axis = {
        "exact_mask": np.array([True, True, False, True]),
        "domain3": np.array(["F", "R_CORE", "F", "R_ANCHOR"]),
    }
    output, active = apply_noncore_delta(
        np.array([0.4, 0.4, 0.4, 0.4]),
        np.array([0.1, 0.1, 0.1, 0.1]),
        axis,
        "F",
        0.1,
    )
    assert active.tolist() == [True, False, False, False]
    assert np.allclose(output, [0.41, 0.4, 0.4, 0.4])


def test_v210_anchor_cannot_promote_and_test_batch_is_forbidden() -> None:
    audit = restrictions()
    assert audit["paired_three_seed_workload_delta_only"]
    assert audit["f_has_two_source_axes"]
    assert audit["r_anchor_is_diagnostic_only"]
    assert not audit["test_csv_read"]
    assert not audit["test_aggregate_used"]
    assert not audit["other_test_rows_required"]
    assert not audit["public_score_used_for_selection"]
