from __future__ import annotations

import numpy as np

from src.archive.v204_validation_contract_reaudit import transfer_summary
from src.metrics import brier_skill_score, brier_skill_score_unclipped


def test_official_metric_formula_and_clipping() -> None:
    target = np.array([0.0, 0.0, 1.0, 1.0])
    prediction = np.array([0.1, 0.2, 0.8, 0.9])
    expected = 100000.0 * (1.0 - np.mean((prediction - target) ** 2) / 0.25)
    assert np.isclose(brier_skill_score_unclipped(target, prediction), expected)
    assert np.isclose(brier_skill_score(target, prediction), max(0.0, expected))


def test_historical_transfer_audit_is_not_a_tuning_contract() -> None:
    result = transfer_summary()
    assert result["n"] == 4
    assert "prohibited for candidate tuning" in result["usage"]
    assert result["sign_concordance_fraction"] == 0.5
    assert result["mean_absolute_error"] > 0.5
