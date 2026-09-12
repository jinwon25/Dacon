from __future__ import annotations

import numpy as np
import pytest

from src.archive.v209_h1_workload_multiseed_audit import (
    load_seed_predictions,
    restrictions,
)


def test_load_seed_predictions_routes_seed42_and_confirmation(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    seed42 = tmp_path / "seed42"
    multiseed = tmp_path / "multiseed"
    baseline.mkdir()
    seed42.mkdir()
    multiseed.mkdir()
    np.save(baseline / "h1_year2022_seed42.npy", np.array([0.4]))
    np.save(seed42 / "augmented_h1_year2022_seed42.npy", np.array([0.5]))
    base, augmented = load_seed_predictions(
        2022, 42, baseline, seed42, multiseed
    )
    assert base.tolist() == [0.4]
    assert augmented.tolist() == [0.5]


def test_load_seed_predictions_rejects_shape_mismatch(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    seed42 = tmp_path / "seed42"
    multiseed = tmp_path / "multiseed"
    baseline.mkdir()
    seed42.mkdir()
    multiseed.mkdir()
    np.save(baseline / "h1_year2022_seed43.npy", np.array([0.4]))
    np.save(
        multiseed / "augmented_h1_year2022_seed43.npy",
        np.array([0.5, 0.6]),
    )
    with pytest.raises(ValueError, match="paired seed shape mismatch"):
        load_seed_predictions(2022, 43, baseline, seed42, multiseed)


def test_v209_primary_is_frozen_and_diagnostics_cannot_promote() -> None:
    audit = restrictions()
    assert audit["paired_three_seed_confirmation"]
    assert audit["primary_weight_and_scope_frozen_from_v206"]
    assert audit["diagnostic_weights_cannot_promote"]
    assert not audit["test_csv_read"]
    assert not audit["test_aggregate_used"]
    assert not audit["other_test_rows_required"]
    assert not audit["public_score_used_for_selection"]
