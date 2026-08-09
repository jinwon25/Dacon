"""Competition metrics and strict probability/submission validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from sklearn.linear_model import LogisticRegression

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


def probability_logit(y_prob: object, eps: float = 1e-6) -> np.ndarray:
    prob = validate_probabilities(y_prob)
    clipped = np.clip(prob, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def calibration_intercept_slope(
    y_true: object,
    y_prob: object,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """Fit y ~ intercept + slope * logit(p) for validation diagnostics."""
    target = validate_binary_targets(y_true)
    logit = probability_logit(y_prob, eps=eps).reshape(-1, 1)
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    model.fit(logit, target)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def reliability_table(
    y_true: object,
    y_prob: object,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Fixed-width validation-only reliability table."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    target = validate_binary_targets(y_true)
    prob = validate_probabilities(y_prob)
    if len(target) != len(prob):
        raise ValueError("y_true and y_prob lengths differ")
    bin_id = np.minimum((prob * n_bins).astype(np.int32), n_bins - 1)
    frame = pd.DataFrame({"bin": bin_id, "prediction": prob, "target": target})
    grouped = frame.groupby("bin", observed=True)
    table = grouped.agg(
        n=("target", "size"),
        prediction_mean=("prediction", "mean"),
        target_rate=("target", "mean"),
        prediction_min=("prediction", "min"),
        prediction_max=("prediction", "max"),
    ).reset_index()
    table["weight"] = table["n"] / len(frame)
    table["calibration_error"] = table["prediction_mean"] - table["target_rate"]
    return table


def brier_decomposition(
    y_true: object,
    y_prob: object,
    n_bins: int = 20,
) -> dict[str, float]:
    """Murphy-style binned reliability/resolution/uncertainty diagnostics."""
    target = validate_binary_targets(y_true)
    prob = validate_probabilities(y_prob)
    table = reliability_table(target, prob, n_bins=n_bins)
    base_rate = float(target.mean())
    reliability = float(
        np.sum(table["weight"] * np.square(table["prediction_mean"] - table["target_rate"]))
    )
    resolution = float(
        np.sum(table["weight"] * np.square(table["target_rate"] - base_rate))
    )
    uncertainty = base_rate * (1.0 - base_rate)
    return {
        "brier": brier_score(target, prob),
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "binned_reconstruction": uncertainty - resolution + reliability,
        "n_bins": float(n_bins),
    }


def cluster_bootstrap_delta(
    y_true: object,
    candidate_prob: object,
    incumbent_prob: object,
    clusters: object,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired cluster bootstrap for candidate BS minus incumbent BS."""
    target = validate_binary_targets(y_true)
    candidate = validate_probabilities(candidate_prob)
    incumbent = validate_probabilities(incumbent_prob)
    cluster_values = pd.Series(clusters, copy=False)
    if not (len(target) == len(candidate) == len(incumbent) == len(cluster_values)):
        raise ValueError("bootstrap inputs have different lengths")
    if cluster_values.isna().any():
        raise ValueError("clusters contain missing values")
    codes, uniques = pd.factorize(cluster_values, sort=True)
    n_clusters = len(uniques)
    if n_clusters < 2:
        raise ValueError("cluster bootstrap needs at least two clusters")
    row_delta = np.square(candidate - target) - np.square(incumbent - target)
    cluster_delta_sum = np.bincount(codes, weights=row_delta, minlength=n_clusters)
    cluster_count = np.bincount(codes, minlength=n_clusters).astype(np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, 200):
        stop = min(start + 200, n_resamples)
        sampled = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        numerator = cluster_delta_sum[sampled].sum(axis=1)
        denominator = cluster_count[sampled].sum(axis=1)
        deltas[start:stop] = numerator / denominator
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    n_improving = int(np.count_nonzero(deltas < 0.0))
    # This interval only describes Monte Carlo uncertainty in the finite
    # bootstrap estimate; it is not a real-world probability guarantee.
    if n_improving == 0:
        probability_lower = 0.0
    else:
        probability_lower = float(
            beta_distribution.ppf(0.025, n_improving, n_resamples - n_improving + 1)
        )
    if n_improving == n_resamples:
        probability_upper = 1.0
    else:
        probability_upper = float(
            beta_distribution.ppf(0.975, n_improving + 1, n_resamples - n_improving)
        )
    return {
        "observed_delta": float(row_delta.mean()),
        "bootstrap_mean_delta": float(deltas.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "improvement_probability": float(n_improving / n_resamples),
        "n_improving_resamples": float(n_improving),
        "mc_probability_ci_lower": probability_lower,
        "mc_probability_ci_upper": probability_upper,
        "n_clusters": float(n_clusters),
        "n_resamples": float(n_resamples),
    }
