import numpy as np

from src.archive.v275_pseudo_trackman_expert_route_audit import expert_predictions


def test_expert_predictions_build_declared_equal_ensembles() -> None:
    values = expert_predictions(
        np.array([0.3]), np.array([0.6]), np.array([0.9])
    )
    assert np.isclose(values["command_batter"][0], 0.75)
    assert np.isclose(values["all_equal"][0], 0.60)
