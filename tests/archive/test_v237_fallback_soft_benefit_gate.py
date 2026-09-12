from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.archive.v237_fallback_soft_benefit_gate import (
    GateModel,
    apply_gate,
    restrictions,
)


def test_soft_gate_cannot_expand_support_or_submitted_dose() -> None:
    features = pd.DataFrame({"x": [1.0, -1.0, 0.0]})
    scaler = StandardScaler().fit(features)
    model = Ridge(alpha=1.0).fit(scaler.transform(features), [2.0, -2.0, 0.0])
    fitted = GateModel(scaler=scaler, model=model, soft_scale=1.0)
    parent = np.array([0.50, 0.50, 0.70])
    incumbent = np.array([0.56, 0.44, 0.70])
    output, gate, _expected = apply_gate(
        fitted, features, parent, incumbent, np.array([True, True, False])
    )
    assert np.all((gate >= 0.0) & (gate <= 1.0))
    assert output[2] == incumbent[2]
    assert abs(output[0] - parent[0]) <= abs(incumbent[0] - parent[0])
    assert abs(output[1] - parent[1]) <= abs(incumbent[1] - parent[1])


def test_v237_restrictions_are_conservative_and_forward() -> None:
    audit = restrictions()
    assert audit["fallback_support_frozen"]
    assert audit["gate_never_exceeds_submitted_30pct"]
    assert audit["player_ids_excluded_from_gate"]
    assert audit["strict_forward_benefit_fit"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
