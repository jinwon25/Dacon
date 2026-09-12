import numpy as np

from src.archive.v345_transition_workload_beta_cell_v335 import (
    add_frozen_beta_increment,
    full_row_rms,
)


def test_add_frozen_beta_increment_is_additive_and_clipped():
    v343 = np.asarray([0.20, 0.99, 0.40])
    parent = np.asarray([0.30, 0.30, 0.30])
    beta = np.asarray([0.35, 0.50, 0.10])
    result = add_frozen_beta_increment(v343, parent, beta)
    np.testing.assert_allclose(result, [0.25, 0.999, 0.20])


def test_full_row_rms_uses_all_rows():
    parent = np.asarray([0.1, 0.2, 0.3, 0.4])
    candidate = np.asarray([0.2, 0.2, 0.3, 0.4])
    assert full_row_rms(parent, candidate) == 0.05
