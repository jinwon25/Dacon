import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.champion import v277_pseudo_deployment_runtime as runtime


def test_v277_runtime_reindexes_base_features_before_prediction(tmp_path, monkeypatch) -> None:
    (tmp_path / "feature_columns.json").write_text(
        json.dumps(["b", "a"]), encoding="utf-8"
    )
    (tmp_path / "pseudo_deployment_xgb.json").write_text("{}", encoding="utf-8")
    seen = {}

    class FakeModel:
        def load_model(self, _path):
            return None

        def predict_proba(self, features):
            seen["columns"] = list(features.columns)
            return np.column_stack([np.full(len(features), 0.6), np.full(len(features), 0.4)])

    monkeypatch.setattr(runtime.xgb, "XGBClassifier", FakeModel)
    base_runtime = SimpleNamespace(
        build=lambda frame, asset: pd.DataFrame({"a": [1.0], "b": [2.0]})
    )
    prediction = runtime.predict(pd.DataFrame({"x": [1]}), tmp_path, base_runtime, tmp_path)
    assert seen["columns"] == ["b", "a"]
    np.testing.assert_allclose(prediction, [0.4])
