from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v205_h1_workload_strict_forward_audit import (
    apply_component_delta,
    restrictions,
    strict_axis,
)


def test_strict_axis_is_full_exact_single_domain() -> None:
    frame = pd.DataFrame(
        {"game_month": [4, 5], "pitcher_id": [1, 2], "batter_id": [3, 4]}
    )
    axis = strict_axis(frame, np.array([0, 1]), np.array([0.4, 0.6]))
    assert axis["exact_mask"].tolist() == [True, True]
    assert axis["domain3"].tolist() == ["R_CORE", "R_CORE"]
    assert axis["pitcher_id"].tolist() == [1, 2]


def test_component_delta_is_a_convex_paired_dose() -> None:
    candidate = apply_component_delta(
        np.array([0.4, 0.6]), np.array([0.6, 0.4]), 0.25
    )
    assert np.allclose(candidate, [0.45, 0.55])


def test_v205_primary_is_strict_and_contaminated_axes_only_veto() -> None:
    audit = restrictions()
    assert audit["primary_axes_strictly_forward_full_year"]
    assert audit["source_scale_selection_before_full_2024"]
    assert audit["contaminated_pipeline_axes_can_only_veto"]
    assert not audit["test_csv_read"]
    assert not audit["test_aggregate_used"]
    assert not audit["other_test_rows_required"]
    assert not audit["public_score_used_for_selection"]
