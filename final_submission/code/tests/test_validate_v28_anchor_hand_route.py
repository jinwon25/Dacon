import numpy as np

from src.validate_v28_anchor_hand_route import _expected_from_parent


def test_expected_route_recovers_base_and_changes_only_matched_rows() -> None:
    base = np.array([0.4, 0.5, 0.6])
    direct = np.array([0.8, 0.3, 0.2])
    parent = base + 0.1 * (direct - base)
    matched = np.array([True, False, True])
    expected = _expected_from_parent(parent, direct, matched, 0.1, 0.125)
    wanted = parent.copy()
    wanted[matched] = base[matched] + 0.125 * (direct[matched] - base[matched])
    assert np.allclose(expected, wanted)
