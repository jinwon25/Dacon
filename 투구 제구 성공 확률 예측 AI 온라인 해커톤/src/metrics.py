"""Competition metrics and strict probability/submission validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"


def _as_1d_float(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def validate_probabilities(y_prob: object) -> np.ndarray:
    """Return finite 1-D probabilities, rejecting values outside [0, 1]."""
    prob = _as_1d_float(y_prob, "y_prob")
    if ((prob < 0.0) | (prob > 1.0)).any():
        raise ValueError("y_prob must be within [0, 1]")
    return prob


def validate_binary_targets(y_true: object) -> np.ndarray:
    target = _as_1d_float(y_true, "y_true")
    if not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("y_true must contain only 0 and 1")
    return target


def brier_score(y_true: object, y_prob: object) -> float:
    target = validate_binary_targets(y_true)
    prob = validate_probabilities(y_prob)
    if len(target) != len(prob):
        raise ValueError("y_true and y_prob lengths differ")
    if len(target) == 0:
        raise ValueError("metric input is empty")
    return float(np.mean(np.square(prob - target)))


def brier_skill_score(y_true: object, y_prob: object) -> float:
    """Official score: max(0, 100000 * (1 - BS / (r * (1-r))))."""
    target = validate_binary_targets(y_true)
    score = brier_score(target, y_prob)
    base_rate = float(target.mean())
    reference = base_rate * (1.0 - base_rate)
    if reference <= 0.0:
        raise ValueError("Brier Skill Score is undefined for a constant target")
    return max(0.0, 100000.0 * (1.0 - score / reference))


def make_submission(row_ids: object, y_prob: object) -> pd.DataFrame:
    """Build a submission directly in test-row order."""
    ids = pd.Series(row_ids, copy=False).reset_index(drop=True)
    prob = validate_probabilities(y_prob)
    if len(ids) != len(prob):
        raise ValueError("row_id and prediction lengths differ")
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("row_id must be finite/non-null and unique")
    return pd.DataFrame({ID_COL: ids.to_numpy(copy=True), TARGET_COL: prob})


def validate_submission(submission: pd.DataFrame, expected_ids: object) -> None:
    """Validate exact columns, row count/order, and probabilities."""
    if list(submission.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"submission columns must be {[ID_COL, TARGET_COL]}")
    expected = pd.Series(expected_ids, copy=False).reset_index(drop=True)
    actual = submission[ID_COL].reset_index(drop=True)
    if len(actual) != len(expected) or not np.array_equal(
        actual.astype("string").to_numpy(), expected.astype("string").to_numpy()
    ):
        raise ValueError("submission row_id values or order differ from test")
    validate_probabilities(submission[TARGET_COL].to_numpy())
