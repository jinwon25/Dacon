from src.archive.v247_runtime_faithful_lightgbm_oof import PARAMS, restrictions


def test_lightgbm_is_regularized_and_deterministic() -> None:
    assert PARAMS["min_child_samples"] >= 1000
    assert PARAMS["reg_lambda"] > 0
    assert PARAMS["deterministic"]


def test_runtime_contract_is_public_blind() -> None:
    audit = restrictions()
    assert audit["audit_uses_release_runtime_transform"]
    assert audit["model_family_independent_from_xgboost"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
