import numpy as np

from src.archive.v105_cross_architecture_ablation import combine_deltas


def test_cross_architecture_delta_policies():
    lgb = np.array([0.2, -0.4, 0.3, -0.1])
    mlp = np.array([0.4, -0.2, -0.1, 0.2])
    assert np.allclose(combine_deltas(lgb, mlp, "mlp_only"), mlp)
    assert np.allclose(combine_deltas(lgb, mlp, "mean"), [0.3, -0.3, 0.1, 0.05])
    assert np.allclose(combine_deltas(lgb, mlp, "sign_mean"), [0.3, -0.3, 0.0, 0.0])
    assert np.allclose(combine_deltas(lgb, mlp, "sign_min"), [0.2, -0.2, 0.0, 0.0])
