"""Dependence-aware paired evaluation for probabilistic pitch forecasts.

All improvements are candidate versus a fixed incumbent on exactly the same
rows.  Positive values are better.  Bootstrap replicates recompute the target
base rate, rather than treating the BSS denominator as fixed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from src.metrics import (
    brier_score,
    brier_skill_score,
    brier_skill_score_unclipped,
    validate_binary_targets,
    validate_probabilities,
)


def _paired_arrays(
    y_true: object,
    candidate_prob: object,
    incumbent_prob: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = validate_binary_targets(y_true)
    candidate = validate_probabilities(candidate_prob)
    incumbent = validate_probabilities(incumbent_prob)
    if not (len(target) == len(candidate) == len(incumbent)):
        raise ValueError("paired evaluation inputs have different lengths")
    if len(target) == 0:
        raise ValueError("paired evaluation input is empty")
    improvement = np.square(incumbent - target) - np.square(candidate - target)
    return target, candidate, incumbent, improvement


def _gain_from_sums(
    improvement_sum: np.ndarray,
    target_sum: np.ndarray,
    count: np.ndarray,
) -> np.ndarray:
    count = np.asarray(count, dtype=np.float64)
    rate = np.asarray(target_sum, dtype=np.float64) / count
    reference = rate * (1.0 - rate)
    result = 100_000.0 * (np.asarray(improvement_sum) / count) / reference
    return np.where((count > 0.0) & (reference > 0.0), result, np.nan)


def paired_score_summary(
    y_true: object,
    candidate_prob: object,
    incumbent_prob: object,
) -> dict[str, float]:
    target, candidate, incumbent, improvement = _paired_arrays(
        y_true, candidate_prob, incumbent_prob
    )
    raw_candidate = brier_skill_score_unclipped(target, candidate)
    raw_incumbent = brier_skill_score_unclipped(target, incumbent)
    return {
        "n_rows": float(len(target)),
        "base_rate": float(target.mean()),
        "candidate_brier": brier_score(target, candidate),
        "incumbent_brier": brier_score(target, incumbent),
        "mean_brier_improvement": float(improvement.mean()),
        "unclipped_bss_equivalent_gain": float(raw_candidate - raw_incumbent),
        "candidate_unclipped_bss_equivalent": float(raw_candidate),
        "incumbent_unclipped_bss_equivalent": float(raw_incumbent),
        "candidate_official_bss": brier_skill_score(target, candidate),
        "incumbent_official_bss": brier_skill_score(target, incumbent),
        "official_bss_gain": float(
            brier_skill_score(target, candidate)
            - brier_skill_score(target, incumbent)
        ),
    }


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("all bootstrap replicates are invalid")
    p05, median, p95 = np.quantile(finite, [0.05, 0.50, 0.95])
    return {
        "bootstrap_mean": float(finite.mean()),
        "p05": float(p05),
        "median": float(median),
        "p95": float(p95),
        "prob_positive": float(np.mean(finite > 0.0)),
        "n_valid_resamples": float(len(finite)),
    }


def one_way_cluster_bootstrap(
    y_true: object,
    candidate_prob: object,
    incumbent_prob: object,
    clusters: object,
    *,
    n_resamples: int = 3000,
    seed: int = 42,
    batch_size: int = 200,
) -> dict[str, float]:
    target, _, _, improvement = _paired_arrays(
        y_true, candidate_prob, incumbent_prob
    )
    cluster = pd.Series(clusters, copy=False)
    if len(cluster) != len(target) or cluster.isna().any():
        raise ValueError("clusters have wrong length or missing values")
    codes, unique = pd.factorize(cluster, sort=True)
    n_clusters = len(unique)
    if n_clusters < 2:
        raise ValueError("cluster bootstrap needs at least two clusters")
    imp_sum = np.bincount(codes, weights=improvement, minlength=n_clusters)
    target_sum = np.bincount(codes, weights=target, minlength=n_clusters)
    count = np.bincount(codes, minlength=n_clusters).astype(np.float64)

    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        sampled = rng.integers(
            0, n_clusters, size=(stop - start, n_clusters), endpoint=False
        )
        values[start:stop] = _gain_from_sums(
            imp_sum[sampled].sum(axis=1),
            target_sum[sampled].sum(axis=1),
            count[sampled].sum(axis=1),
        )
    return {
        "method": "one_way_cluster_bootstrap",
        "n_clusters": float(n_clusters),
        **_distribution_summary(values),
    }


def crossed_pigeonhole_bootstrap(
    y_true: object,
    candidate_prob: object,
    incumbent_prob: object,
    row_clusters: object,
    column_clusters: object,
    *,
    n_resamples: int = 3000,
    seed: int = 42,
    batch_size: int = 100,
) -> dict[str, float]:
    """Separately resample pitcher and batter levels for crossed dependence."""
    target, _, _, improvement = _paired_arrays(
        y_true, candidate_prob, incumbent_prob
    )
    row = pd.Series(row_clusters, copy=False)
    column = pd.Series(column_clusters, copy=False)
    if not (len(row) == len(column) == len(target)):
        raise ValueError("crossed cluster inputs have different lengths")
    if row.isna().any() or column.isna().any():
        raise ValueError("crossed clusters contain missing values")
    row_codes, row_unique = pd.factorize(row, sort=True)
    col_codes, col_unique = pd.factorize(column, sort=True)
    n_row, n_col = len(row_unique), len(col_unique)
    if n_row < 2 or n_col < 2:
        raise ValueError("crossed bootstrap needs two levels in each dimension")
    shape = (n_row, n_col)
    imp_matrix = sparse.csr_matrix(
        (improvement, (row_codes, col_codes)), shape=shape
    )
    target_matrix = sparse.csr_matrix(
        (target, (row_codes, col_codes)), shape=shape
    )
    count_matrix = sparse.csr_matrix(
        (np.ones(len(target)), (row_codes, col_codes)), shape=shape
    )

    rng = np.random.default_rng(seed)
    row_probability = np.full(n_row, 1.0 / n_row)
    col_probability = np.full(n_col, 1.0 / n_col)
    values = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        size = stop - start
        row_weight = rng.multinomial(n_row, row_probability, size=size)
        col_weight = rng.multinomial(n_col, col_probability, size=size)

        def bilinear(matrix: sparse.csr_matrix) -> np.ndarray:
            projected = matrix.T.dot(row_weight.T).T
            return np.asarray(projected * col_weight).sum(axis=1)

        values[start:stop] = _gain_from_sums(
            bilinear(imp_matrix),
            bilinear(target_matrix),
            bilinear(count_matrix),
        )
    return {
        "method": "crossed_pigeonhole_bootstrap",
        "n_row_clusters": float(n_row),
        "n_column_clusters": float(n_col),
        **_distribution_summary(values),
    }


def _circular_block_sums(values: np.ndarray, block_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    n_rows = len(array)
    if not 2 <= block_size <= n_rows:
        raise ValueError("block_size must be between 2 and the row count")
    extended = np.concatenate([array, array[: block_size - 1]])
    cumulative = np.concatenate([[0.0], np.cumsum(extended)])
    return cumulative[block_size:] - cumulative[:-block_size]


def circular_block_bootstrap(
    y_true: object,
    candidate_prob: object,
    incumbent_prob: object,
    *,
    block_size: int,
    n_resamples: int = 3000,
    seed: int = 42,
    batch_size: int = 200,
) -> dict[str, float]:
    """Circular moving-block bootstrap over chronological pitch order."""
    target, _, _, improvement = _paired_arrays(
        y_true, candidate_prob, incumbent_prob
    )
    n_rows = len(target)
    block_imp = _circular_block_sums(improvement, block_size)
    block_target = _circular_block_sums(target, block_size)
    n_blocks = int(np.ceil(n_rows / block_size))
    sampled_count = float(n_blocks * block_size)
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        starts = rng.integers(
            0, n_rows, size=(stop - start, n_blocks), endpoint=False
        )
        values[start:stop] = _gain_from_sums(
            block_imp[starts].sum(axis=1),
            block_target[starts].sum(axis=1),
            np.full(stop - start, sampled_count),
        )
    return {
        "method": "circular_moving_block_bootstrap",
        "block_size": float(block_size),
        "blocks_per_resample": float(n_blocks),
        **_distribution_summary(values),
    }


def grouped_gain_table(
    y_true: object,
    candidate_prob: object,
    incumbent_prob: object,
    groups: object,
) -> pd.DataFrame:
    target, _, _, improvement = _paired_arrays(
        y_true, candidate_prob, incumbent_prob
    )
    frame = pd.DataFrame(
        {"group": pd.Series(groups, copy=False), "target": target, "imp": improvement}
    )
    if len(frame["group"]) != len(target) or frame["group"].isna().any():
        raise ValueError("groups have wrong length or missing values")
    rows = []
    for group, part in frame.groupby("group", observed=True, sort=True):
        gain = _gain_from_sums(
            np.asarray([part["imp"].sum()]),
            np.asarray([part["target"].sum()]),
            np.asarray([len(part)]),
        )[0]
        rows.append({"group": group, "n_rows": len(part), "gain": gain})
    return pd.DataFrame(rows)


def leave_one_team_out_summary(
    y_true: object,
    candidate_prob: object,
    incumbent_prob: object,
    pitcher_team: object,
    batter_team: object,
) -> dict[str, Any]:
    target, candidate, incumbent, _ = _paired_arrays(
        y_true, candidate_prob, incumbent_prob
    )
    pitcher = pd.Series(pitcher_team, copy=False).reset_index(drop=True)
    batter = pd.Series(batter_team, copy=False).reset_index(drop=True)
    if not (len(pitcher) == len(batter) == len(target)):
        raise ValueError("team columns have wrong length")
    teams = sorted(set(pitcher.dropna()).union(set(batter.dropna())), key=str)
    rows = []
    for team in teams:
        keep = ~(pitcher.eq(team) | batter.eq(team)).to_numpy()
        summary = paired_score_summary(target[keep], candidate[keep], incumbent[keep])
        rows.append(
            {
                "left_out_team": team,
                "n_rows": int(keep.sum()),
                "gain": summary["unclipped_bss_equivalent_gain"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("gain").reset_index(drop=True)
    return {
        "minimum_gain": float(frame["gain"].min()),
        "maximum_gain": float(frame["gain"].max()),
        "worst_left_out_team": frame.iloc[0]["left_out_team"],
        "rows": frame.to_dict(orient="records"),
    }


def white_reality_check(
    y_true: object,
    improvement_matrix: object,
    *,
    block_size: int,
    n_resamples: int = 3000,
    seed: int = 42,
    batch_size: int = 200,
) -> dict[str, Any]:
    """Block-bootstrap Reality Check for a supplied final candidate family.

    Columns contain incumbent squared loss minus candidate squared loss.  This
    corrects only for the supplied family; omitted research trials remain a
    documented source of selection bias.
    """
    target = validate_binary_targets(y_true)
    matrix = np.asarray(improvement_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(target) or matrix.shape[1] < 2:
        raise ValueError("improvement_matrix must be n_rows by at least 2 candidates")
    if not np.isfinite(matrix).all():
        raise ValueError("improvement_matrix contains non-finite values")
    n_rows, n_candidates = matrix.shape
    observed_mean = matrix.mean(axis=0)
    observed_stat = float(np.sqrt(n_rows) * observed_mean.max())
    centered = matrix - observed_mean
    block_sums = np.column_stack(
        [_circular_block_sums(centered[:, j], block_size) for j in range(n_candidates)]
    )
    n_blocks = int(np.ceil(n_rows / block_size))
    sampled_count = float(n_blocks * block_size)
    rng = np.random.default_rng(seed)
    statistics = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        starts = rng.integers(
            0, n_rows, size=(stop - start, n_blocks), endpoint=False
        )
        boot_mean = block_sums[starts].sum(axis=1) / sampled_count
        statistics[start:stop] = np.sqrt(n_rows) * boot_mean.max(axis=1)
    p_value = float((1 + np.count_nonzero(statistics >= observed_stat)) / (n_resamples + 1))
    return {
        "method": "white_reality_check_circular_block",
        "block_size": int(block_size),
        "n_candidates": int(n_candidates),
        "observed_best_index": int(np.argmax(observed_mean)),
        "observed_statistic": observed_stat,
        "p_value": p_value,
        "scope_warning": "adjusts only for the supplied final candidate family",
    }
