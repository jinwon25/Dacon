import numpy as np

from src.archive.v103_fixed_union_robust import fixed_union_candidate


def test_fixed_union_is_row_local_and_r_core_only():
    parent = np.array([0.4, 0.5, 0.6])
    fm = np.array([0.5, 0.4, 0.7])
    correction = np.array([0.2, -0.3, 0.4])
    domain = np.array(["R_CORE", "F", "R_ANCHOR"])
    out = fixed_union_candidate(parent, domain, fm, correction, 0.1)
    assert np.allclose(out, np.array([0.52, 0.5, 0.6]))


def test_fixed_union_clips_probabilities():
    out = fixed_union_candidate(
        np.array([0.99]), np.array(["R_CORE"]), np.array([0.999]),
        np.array([1.0]), 0.1,
    )
    assert np.allclose(out, np.array([0.999]))
