from __future__ import annotations

import numpy as np

from src.archive.v173_h1_noncore_extension_audit import paired_metrics
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


def test_noncore_change_is_scored_outside_rcore() -> None:
    axis = {
        "target": np.array([1.0, 0.0, 1.0, 0.0]),
        "parent": np.array([0.4, 0.4, 0.4, 0.4]),
        "exact_mask": np.array([True, True, True, True]),
        "domain3": np.array(["F", "F", "R_CORE", "R_CORE"]),
        "game_month": np.array([4, 4, 4, 4]),
    }
    candidate = np.array([0.5, 0.3, 0.4, 0.4])
    result = paired_metrics(
        axis, axis["parent"], candidate, np.array([True, True, False, False])
    )
    assert result["overall_gain"] > 0.0
    assert result["active_domain_gain"] > 0.0


def test_v210_anchor_cannot_promote_and_test_batch_is_forbidden() -> None:
    audit = restrictions()
    assert audit["paired_three_seed_workload_delta_only"]
    assert audit["f_has_two_source_axes"]
    assert audit["r_anchor_is_diagnostic_only"]
    assert not audit["test_csv_read"]
    assert not audit["test_aggregate_used"]
    assert not audit["other_test_rows_required"]
    assert not audit["public_score_used_for_selection"]
