from __future__ import annotations

from src.archive.v238_brier_fallback_xgb_oof import PARAMS, restrictions


def test_v238_uses_direct_brier_aligned_regression_contract() -> None:
    assert PARAMS["objective"] == "reg:squarederror"
    assert PARAMS["eval_metric"] == "rmse"
    assert PARAMS["min_child_weight"] == 24000
    audit = restrictions()
    assert audit["frozen_114_feature_contract_reused"]
    assert audit["single_loss_function_change"]
    assert audit["strictly_prior_season_training"]
    assert not audit["test_csv_read"]
    assert not audit["public_score_used_for_selection"]
