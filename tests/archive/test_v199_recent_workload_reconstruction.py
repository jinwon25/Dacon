from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v199_recent_workload_reconstruction import (
    attach_main_game_index,
    infer_denominators,
    infer_min_common_denominator,
    restrictions,
)


def test_common_denominator_recovers_two_rate_lcm() -> None:
    denominator, error, fit = infer_min_common_denominator(
        0.350877, 0.157895, max_denominator=240
    )
    assert denominator == 57
    assert error <= 5.00001e-7
    assert fit


def test_common_denominator_is_explicit_lower_bound() -> None:
    denominator, _, fit = infer_min_common_denominator(
        0.5, 0.25, max_denominator=240
    )
    assert denominator == 4
    assert fit


def test_missing_pair_has_no_fit() -> None:
    denominator, error, fit = infer_min_common_denominator(
        np.nan, 0.2, max_denominator=240
    )
    assert denominator == 0
    assert np.isnan(error)
    assert not fit


def test_infer_denominators_preserves_row_order_and_flags_extremes() -> None:
    inferred = infer_denominators(
        pd.Series([0.350877, np.nan, 1.0]),
        pd.Series([0.157895, np.nan, 0.0]),
        max_denominator=240,
    )
    assert inferred["minimum_denominator"].tolist() == [57, 0, 1]
    assert inferred["rounded_rational_fit"].tolist() == [True, False, True]
    assert inferred["uninformative_pair"].tolist() == [False, False, True]


def test_game_index_uses_first_inning_half_transition() -> None:
    frame = pd.DataFrame(
        {
            "season": [2022] * 7 + [2023],
            "inning": [1, 1, 1, 1, 2, 1, 1, 1],
            "top_bottom": ["T", "T", "B", "B", "T", "T", "T", "T"],
        }
    )
    indexed = attach_main_game_index(frame)
    assert indexed["main_game_index"].tolist() == [1, 1, 1, 1, 1, 2, 2, 3]


def test_v199_restrictions_are_strict_and_row_local() -> None:
    audit = restrictions()
    assert audit["official_train_only"]
    assert audit["deployed_inference_is_row_local"]
    assert audit["audit_labels_not_used_in_inferred_denominator"]
    assert not audit["test_csv_read"]
    assert not audit["test_aggregate_used"]
    assert not audit["other_test_rows_required"]
    assert not audit["public_score_used_for_selection"]
