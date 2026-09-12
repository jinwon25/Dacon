from src.archive.v303_futures_brier_regressor import REGRESSOR_PARAMS


def test_regressor_uses_brier_aligned_loss():
    assert REGRESSOR_PARAMS["loss_function"] == "RMSE"
    assert "eval_metric" not in REGRESSOR_PARAMS
