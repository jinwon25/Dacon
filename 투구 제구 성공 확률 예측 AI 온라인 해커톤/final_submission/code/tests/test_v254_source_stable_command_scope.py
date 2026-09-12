import numpy as np

from src.archive.v254_source_stable_command_scope_audit import scope_masks


def test_scope_masks_are_outcome_free_and_subsets_of_active() -> None:
    parent = np.array([0.4, 0.6, 0.5, 0.7])
    base = np.array([0.45, 0.65, 0.48, 0.68])
    command = np.array([0.47, 0.55, 0.52, 0.69])
    routes = {
        "deployed": np.array([True, False, False, False]),
        "pressure_boundary_agreement": np.array([False, True, False, False]),
        "nonpressure_same_hand": np.array([False, False, True, False]),
        "nonpressure_opposite_hand_high52": np.array([False, False, False, False]),
    }
    masks = scope_masks(parent, base, command, routes)
    active = np.logical_or.reduce(list(routes.values()))
    assert set(masks) == {
        "all", "deployed", "pressure", "boundary", "nonpressure", "same_hand",
        "opposite_hand", "same_parent_direction", "opposite_parent_direction",
        "same_center_direction", "command_toward_center", "command_more_extreme",
        "disagreement_le_005", "disagreement_le_010", "disagreement_le_020",
        "base_low50", "base_high50", "parent_mid", "parent_tail",
    }
    assert all(np.all(~mask | active) for mask in masks.values())
