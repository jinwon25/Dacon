import numpy as np

from src.archive.v100_cross_family_consensus import cross_family_candidate


def test_only_same_sign_core_rows_move():
    parent = np.array([0.5, 0.5, 0.5, 0.5])
    fm = np.array([0.6, 0.6, 0.4, 0.6])
    conditional = np.array([0.2, -0.2, -0.2, 0.2])
    domains = np.array(["R_CORE", "R_CORE", "R_CORE", "F"])
    candidate, agree = cross_family_candidate(
        parent, domains, fm, conditional, 0.1
    )
    assert np.array_equal(agree, np.array([True, False, True, False]))
    assert np.array_equal(candidate[[1, 3]], parent[[1, 3]])
