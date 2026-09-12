import numpy as np

from src.archive.v312_trackman_physical_strict_transfer import expit, logit


def test_logit_expit_round_trip() -> None:
    values = np.asarray([0.1, 0.5, 0.9])
    np.testing.assert_allclose(expit(logit(values)), values)
