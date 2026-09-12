from __future__ import annotations

import numpy as np
import pytest

from src.archive.v216_joint_workload_h1_multiseed_audit import (
    load_joint_seed_predictions,
    restrictions,
)


def test_joint_loader_routes_seed42_and_confirmation(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    seed42 = tmp_path / "seed42"
    multiseed = tmp_path / "multiseed"
    baseline.mkdir()
    seed42.mkdir()
    multiseed.mkdir()
    np.save(baseline / "h1_year2022_seed42.npy", np.array([0.4]))
    np.save(seed42 / "joint_h1_year2022_seed42.npy", np.array([0.5]))
    base, joint = load_joint_seed_predictions(
        2022, 42, baseline, seed42, multiseed
    )
    assert base.tolist() == [0.4]
    assert joint.tolist() == [0.5]


def test_joint_loader_rejects_shape_mismatch(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    seed42 = tmp_path / "seed42"
    multiseed = tmp_path / "multiseed"
    baseline.mkdir()
    seed42.mkdir()
    multiseed.mkdir()
    np.save(baseline / "h1_year2022_seed43.npy", np.array([0.4]))
    np.save(
        multiseed / "joint_h1_year2022_seed43.npy", np.array([0.5, 0.6])
    )
    with pytest.raises(ValueError, match="paired seed shape mismatch"):
        load_joint_seed_predictions(2022, 43, baseline, seed42, multiseed)


def test_v216_freezes_v209_gate_and_v214_recipe() -> None:
    audit = restrictions()
    assert audit["paired_three_seed_confirmation"]
    assert audit["primary_weight_and_scope_frozen_from_v206"]
    assert audit["same_recipe_as_v214_seed42"]
    assert audit["fixed_confirmation_seeds_from_v215"]
    assert audit["joint_denominator_rule_frozen_before_confirmation"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
