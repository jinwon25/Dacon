from __future__ import annotations

import numpy as np
import pandas as pd

import src.v154_runtime_script as runtime
from src.v148_contract import reconstruct_deployed_v148
from src.v154_maturity_delta_distillation_audit import _feature_frame


def test_distillation_and_runtime_feature_contract_match() -> None:
    frame = pd.DataFrame({
        "game_type": ["R", "F"],
        "pitcher_id": [10, 999],
        "li": [0.8, np.nan],
    })
    columns = ["game_type", "pitcher_id", "li"]
    categories = {
        "game_type": ["F", "R"],
        "pitcher_id": ["10", "20"],
    }
    spec = {
        "feature_columns": columns,
        "categorical_columns": ["game_type", "pitcher_id"],
        "categories": categories,
    }
    training = _feature_frame(frame, columns, categories)
    deployed = runtime._maturity_features(frame, spec)
    assert list(training.columns) == list(deployed.columns)
    assert training["game_type"].cat.codes.tolist() == deployed["game_type"].cat.codes.tolist()
    assert training["pitcher_id"].cat.codes.tolist() == [0, -1]
    assert deployed["pitcher_id"].cat.codes.tolist() == [0, -1]
    np.testing.assert_allclose(training["li"], deployed["li"], equal_nan=True)


def test_may_gate_overrides_bridge_without_changing_other_routes(monkeypatch) -> None:
    frame = pd.DataFrame({"game_month": [4, 5, 6, 5]})
    intermediate = np.asarray([0.40, 0.41, 0.42, 0.43])
    delta = np.asarray([0.00, 0.02, 0.00, -0.01])
    h1 = np.asarray([0.50, 0.51, 0.52, 0.53])
    c3 = np.asarray([0.01, -0.01, 0.02, 0.03])
    active = np.asarray([True, True, False, False])
    maturity = np.asarray([False, True, False, True])
    monkeypatch.setattr(runtime, "_predict_intermediate", lambda _frame: intermediate)
    monkeypatch.setattr(runtime, "_predict_maturity_delta", lambda _frame: (delta, maturity))
    monkeypatch.setattr(runtime, "_predict_h1", lambda _frame: h1)
    monkeypatch.setattr(runtime, "_predict_c3", lambda _frame: c3)
    monkeypatch.setattr(runtime, "_active_mask", lambda _frame: active)

    components = runtime.predict_components(frame)
    output = components[-1]
    expected = intermediate.copy()
    expected[active] = 0.85 * intermediate[active] + 0.15 * h1[active] + 0.5 * c3[active]
    expected[maturity] = intermediate[maturity] + delta[maturity]
    np.testing.assert_allclose(output, expected)
    assert output[2] == intermediate[2]


def test_v148_deployed_bridge_is_fifteen_percent_not_selected_five_percent() -> None:
    v142 = np.asarray([0.4, 0.6])
    v138 = np.asarray([0.6, 0.2])
    expected = reconstruct_deployed_v148(v142, v138)
    np.testing.assert_allclose(expected, [0.43, 0.54])
    assert not np.allclose(expected, v142 + 0.05 * (v138 - v142))
