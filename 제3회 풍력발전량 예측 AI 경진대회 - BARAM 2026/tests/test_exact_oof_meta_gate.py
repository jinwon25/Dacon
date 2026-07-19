import numpy as np
import pandas as pd

from experiments.exact_oof_meta_gate import (
    CAPACITY,
    actionable_mask,
    apply_meta_gate,
    meta_features,
    settlement_benefit_labels,
)


def test_meta_features_are_aligned_and_finite() -> None:
    timestamps = pd.date_range("2024-01-01 01:00:00", periods=3, freq="h")
    values = np.asarray([3_000.0, 4_000.0, 5_000.0])
    features = meta_features(values, values + 100, values, values, values + 50, timestamps)
    assert features.shape == (3, 12)
    assert np.isfinite(features).all()


def test_settlement_label_recognizes_band_entry() -> None:
    truth = np.asarray([10_000.0, 10_000.0])
    current = np.asarray([8_500.0, 9_500.0])
    member = np.asarray([10_000.0, 8_000.0])
    labels = settlement_benefit_labels(
        truth, current, member, np.asarray([True, True]), step=0.5
    )
    assert labels.tolist() == [True, False]


def test_meta_gate_changes_only_high_probability_action_rows() -> None:
    current = np.asarray([5_000.0, 6_000.0, 7_000.0])
    member = np.asarray([5_400.0, 6_400.0, 7_400.0])
    candidate, gate = apply_meta_gate(
        current,
        member,
        np.asarray([True, True, False]),
        np.asarray([0.60, 0.40, 0.90]),
    )
    assert gate.tolist() == [True, False, False]
    assert np.allclose(candidate, [5_100.0, 6_000.0, 7_000.0])


def test_action_mask_matches_original_cross_group_gate() -> None:
    group_1 = np.asarray([5_000.0, 5_000.0])
    group_2 = np.asarray([5_100.0, 9_000.0])
    base = np.asarray([3_000.0, 3_000.0])
    member = np.asarray([3_100.0, 3_100.0])
    assert actionable_mask(group_1, group_2, base, member).tolist() == [True, False]
    assert CAPACITY == 21_000.0
