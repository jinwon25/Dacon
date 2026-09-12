import pytest

from src.champion import v342_finalize_workload_h1 as v342


def test_seeded_config_changes_only_seed() -> None:
    config = v342.seeded_config(43)
    assert config["seed"] == 43
    assert config["iterations"] == v342.MODEL_CONFIG["iterations"]


def test_seeded_config_rejects_unregistered_seed() -> None:
    with pytest.raises(ValueError, match="unregistered seed"):
        v342.seeded_config(45)


def test_replace_models_preserves_deployed_correction_assets() -> None:
    deployed = {
        "models": [1, 2, 3],
        "features": ["a"],
        "ctx": {"frozen": True},
        "asof_prior": {"pitcher": {}},
        "alpha": 1.09,
    }
    output = v342.replace_models(deployed, [4, 5, 6], ["a", "workload"])
    assert output["models"] == [4, 5, 6]
    assert output["features"] == ["a", "workload"]
    assert output["ctx"] is deployed["ctx"]
    assert output["asof_prior"] is deployed["asof_prior"]
    assert output["alpha"] == 1.09
