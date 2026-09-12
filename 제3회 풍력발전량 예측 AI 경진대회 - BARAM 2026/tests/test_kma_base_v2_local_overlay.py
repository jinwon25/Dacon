from __future__ import annotations

import numpy as np

from experiments.kma_base_v2_local_overlay import (
    OverlayPolicy,
    apply_overlay,
    select_development_policies,
)
from src.metrics import CAPACITY_KWH


def test_apply_overlay_respects_direction_ratio_and_movement() -> None:
    reference = np.array([1_000.0, 5_000.0, 10_000.0, 18_000.0])
    member = np.array([2_000.0, 4_000.0, 12_000.0, 20_000.0])
    policy = OverlayPolicy(
        group="kpx_group_3",
        direction="up",
        coverage=0.5,
        minimum_base_ratio=0.20,
        maximum_base_ratio=0.80,
        minimum_disagreement_kwh=1_500.0,
        alpha=0.10,
    )

    candidate, gate = apply_overlay(reference, member, policy)

    assert gate.tolist() == [False, False, True, False]
    assert candidate.tolist() == [1_000.0, 5_000.0, 10_200.0, 18_000.0]


def test_development_search_finds_helpful_single_group_overlay() -> None:
    rows = 240
    development = np.ones(rows, dtype=bool)
    truth = {}
    reference = {}
    member = {}
    for group, capacity in CAPACITY_KWH.items():
        truth[group] = np.full(rows, 0.5 * capacity)
        reference[group] = truth[group] + 600.0
        member[group] = reference[group].copy()
    member["kpx_group_2"] = truth["kpx_group_2"].copy()

    result = select_development_policies(
        truth,
        reference,
        member,
        development,
    )

    assert result["strict"] is not None
    assert result["strict"]["policy"].group == "kpx_group_2"
    assert result["strict"]["delta"]["score"] > 0.00015
