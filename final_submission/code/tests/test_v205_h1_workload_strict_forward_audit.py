from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v205_h1_workload_strict_forward_audit import (
    apply_component_delta,
    load_batter_ids_by_year,
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


def test_load_batter_ids_preserves_csv_row_alignment(tmp_path) -> None:
    train_csv = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "season": [2022, 2023, 2022, 2024],
            "batter_id": [11, 21, 12, 31],
        }
    ).to_csv(train_csv, index=False)
    loaded = load_batter_ids_by_year(
        train_csv,
        np.array([2022, 2023, 2022, 2024]),
    )
    assert loaded[2022].tolist() == [11, 12]
    assert loaded[2023].tolist() == [21]
    assert loaded[2024].tolist() == [31]


def test_v205_primary_is_strict_and_contaminated_axes_only_veto() -> None:
    audit = restrictions()
    assert audit["primary_axes_strictly_forward_full_year"]
    assert audit["source_scale_selection_before_full_2024"]
    assert audit["contaminated_pipeline_axes_can_only_veto"]
    assert not audit["test_csv_read"]
    assert not audit["test_aggregate_used"]
    assert not audit["other_test_rows_required"]
    assert not audit["public_score_used_for_selection"]
