import numpy as np
import pandas as pd

from src.archive.v102_consensus_benefit_gate import apply_gate, GateModel


class _IdentityScaler:
    def transform(self, values):
        return values


class _ConstantModel:
    def predict(self, values):
        return np.ones(len(values))


def test_soft_gate_is_row_local_and_bounded():
    model = GateModel(_IdentityScaler(), _ConstantModel(), 2.0)
    parent = np.array([0.4, 0.5])
    raw = np.array([0.6, 0.3])
    features = pd.DataFrame({"x": [0.0, 0.0]})
    candidate, gate, _ = apply_gate(model, features, parent, raw)
    assert np.allclose(gate, 0.5)
    assert np.allclose(candidate, np.array([0.5, 0.4]))
