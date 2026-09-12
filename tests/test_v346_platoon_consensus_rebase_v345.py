import numpy as np

from src.archive.v346_platoon_consensus_rebase_v345 import (
    apply_groups,
    full_row_rms,
)


def test_apply_groups_changes_only_eligible_selected_rows():
    parent = np.asarray([0.2, 0.3, 0.4, 0.99])
    direction = np.asarray([0.1, 0.1, 0.1, 0.2])
    keys = np.asarray(["A", "B", "A", "A"])
    candidate, active = apply_groups(
        parent, direction, keys, {"A"}, np.asarray([True, True, False, True])
    )
    np.testing.assert_array_equal(active, [True, False, False, True])
    np.testing.assert_allclose(candidate, [0.3, 0.3, 0.4, 0.999])


def test_full_row_rms_uses_all_rows():
    assert full_row_rms(np.zeros(4), np.asarray([0.2, 0.0, 0.0, 0.0])) == 0.1
