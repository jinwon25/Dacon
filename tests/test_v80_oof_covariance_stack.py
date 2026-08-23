import numpy as np
import pandas as pd

from src.core.oof_bank import rebase_frozen_candidate
from src.archive.v80_oof_covariance_stack import _fit_transition


def test_rebase_preserves_frozen_probability_shift() -> None:
    current = np.array([0.4, 0.6])
    old = np.array([0.3, 0.7])
    candidate = np.array([0.35, 0.65])
    rebased = rebase_frozen_candidate(current, old, candidate)
    assert np.allclose(rebased - current, candidate - old)


def _axis(name: str, target: np.ndarray, parent: np.ndarray, candidate: np.ndarray) -> pd.DataFrame:
    n = len(target)
    return pd.DataFrame(
        {
            "axis": np.repeat(name, n),
            "target": target,
            "incumbent_probability": parent,
            "domain3": np.where((np.arange(n) // 2) % 2, "R_CORE", "R_ANCHOR"),
            "month": 4 + (np.arange(n) // 2) % 4,
            "candidate__safe": candidate,
        }
    )


def test_forward_fit_uses_nonzero_safe_candidate() -> None:
    n = 8000
    target = (np.arange(n) % 2).astype(float)
    parent = np.full(n, 0.5)
    candidate = 0.5 + 0.05 * (target - 0.5)
    source = _axis("source", target, parent, candidate)
    audit = _axis("audit", target, parent, candidate)
    config = {
        "max_total_candidate_weight": 0.25,
        "ridge": 1e-8,
        "max_group_brier_increase": 0.0,
        "min_group_rows": 500,
        "nonzero_weight_tolerance": 1e-8,
        "gates": {
            "positive_month_fraction_min": 0.75,
            "worst_month_gain_min_exclusive": -5.0,
            "minimum_domain_gain_min": 0.0,
        },
    }
    result = _fit_transition(source, audit, ["safe"], config)
    assert result["fit"]["weights"]["safe"] > 0.24
    assert result["passes_numeric_gate"]
