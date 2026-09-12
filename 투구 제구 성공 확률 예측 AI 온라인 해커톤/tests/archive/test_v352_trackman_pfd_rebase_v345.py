import numpy as np

from src.archive.v352_trackman_pfd_rebase_v345 import (
    add_frozen_increment,
    frozen_increment,
)


def test_frozen_increment_is_candidate_minus_original_parent():
    parent = np.asarray([0.30, 0.40, 0.50])
    candidate = np.asarray([0.31, 0.38, 0.50])
    np.testing.assert_allclose(
        frozen_increment(parent, candidate), [0.01, -0.02, 0.0]
    )


def test_add_frozen_increment_is_additive_and_clipped():
    parent = np.asarray([0.20, 0.995, 0.002])
    increment = np.asarray([0.01, 0.02, -0.01])
    np.testing.assert_allclose(
        add_frozen_increment(parent, increment), [0.21, 0.999, 0.001]
    )
