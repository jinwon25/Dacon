import numpy as np
import pandas as pd

from src.archive.v42_forward_nested_state_mode import search_domain, top_configs


def test_top_configs_ignores_later_audit_labels():
    metrics = pd.DataFrame(
        [
            {"audit_year": 2022, "domain": "F", "config": "stable", "gain": 2.0},
            {"audit_year": 2022, "domain": "F", "config": "future", "gain": 1.0},
            {"audit_year": 2023, "domain": "F", "config": "stable", "gain": 2.0},
            {"audit_year": 2023, "domain": "F", "config": "future", "gain": 100.0},
        ]
    )
    assert top_configs(metrics, "F", (2022,), 1) == ["stable"]


def test_search_domain_uses_only_declared_selection_years(monkeypatch):
    monkeypatch.setattr(
        "src.archive.v42_forward_nested_state_mode.STATE_MULTIPLIERS", np.array([1.0])
    )
    monkeypatch.setattr(
        "src.archive.v42_forward_nested_state_mode.MODE_MULTIPLIERS", np.array([0.0])
    )

    def fake_state(fold, domain, config):
        return np.asarray(fold["directions"][config], dtype=float)

    monkeypatch.setattr(
        "src.archive.v42_forward_nested_state_mode._state_correction", fake_state
    )
    state_folds = {
        2022: {
            "target": np.array([1.0, 0.0]),
            "incumbent": np.array([0.5, 0.5]),
            "domain3": np.array(["F", "F"]),
            "directions": {
                "past": np.array([0.1, -0.1]),
                "future": np.array([-0.1, 0.1]),
            },
        },
        2023: {
            "target": np.array([0.0, 1.0]),
            "incumbent": np.array([0.5, 0.5]),
            "domain3": np.array(["F", "F"]),
            "directions": {
                "past": np.array([0.1, -0.1]),
                "future": np.array([-0.1, 0.1]),
            },
        },
    }
    mode = {
        "names": ["neutral"],
        "raw": np.full((2, 1), 0.5),
    }
    choice = search_domain(
        "F",
        ["past", "future"],
        state_folds,
        {2022: mode, 2023: mode},
        (2022,),
    )
    assert choice["state_config"] == "past"
    assert set(choice["selection_fold_gains"]) == {"2022"}
