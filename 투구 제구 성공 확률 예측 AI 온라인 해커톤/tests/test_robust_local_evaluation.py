import numpy as np
import pytest

from src.robust_local_evaluation import (
    circular_block_bootstrap,
    crossed_pigeonhole_bootstrap,
    leave_one_team_out_summary,
    one_way_cluster_bootstrap,
    paired_score_summary,
    white_reality_check,
)


def _sample():
    target = np.tile(np.array([0.0, 1.0, 0.0, 1.0]), 40)
    incumbent = np.where(target == 1.0, 0.55, 0.45)
    candidate = np.where(target == 1.0, 0.60, 0.40)
    pitcher = np.tile(np.repeat(np.arange(8), 5), 4)
    batter = np.tile(np.arange(10), 16)
    return target, candidate, incumbent, pitcher, batter


def test_paired_summary_separates_official_and_unclipped_scores():
    target = np.array([0.0, 1.0, 0.0, 1.0])
    incumbent = np.array([0.9, 0.1, 0.9, 0.1])
    candidate = np.array([0.8, 0.2, 0.8, 0.2])
    result = paired_score_summary(target, candidate, incumbent)

    assert result["official_bss_gain"] == 0.0
    assert result["unclipped_bss_equivalent_gain"] > 0.0


def test_dependence_bootstraps_are_reproducible_and_detect_uniform_gain():
    target, candidate, incumbent, pitcher, batter = _sample()
    first = one_way_cluster_bootstrap(
        target,
        candidate,
        incumbent,
        pitcher,
        n_resamples=200,
        seed=7,
    )
    second = one_way_cluster_bootstrap(
        target,
        candidate,
        incumbent,
        pitcher,
        n_resamples=200,
        seed=7,
    )
    crossed = crossed_pigeonhole_bootstrap(
        target,
        candidate,
        incumbent,
        pitcher,
        batter,
        n_resamples=200,
        seed=9,
    )
    blocked = circular_block_bootstrap(
        target,
        candidate,
        incumbent,
        block_size=16,
        n_resamples=200,
        seed=11,
    )

    assert first == second
    assert first["p05"] > 0.0
    assert crossed["p05"] > 0.0
    assert blocked["p05"] > 0.0


def test_leave_one_team_out_reports_worst_deletion():
    target, candidate, incumbent, _, _ = _sample()
    pitcher_team = np.tile([1, 2, 3, 4], 40)
    batter_team = np.tile([2, 3, 4, 1], 40)
    result = leave_one_team_out_summary(
        target, candidate, incumbent, pitcher_team, batter_team
    )

    assert result["minimum_gain"] > 0.0
    assert len(result["rows"]) == 4


def test_reality_check_validates_candidate_family_and_is_reproducible():
    target, candidate, incumbent, _, _ = _sample()
    first = np.square(incumbent - target) - np.square(candidate - target)
    second = 0.5 * first
    matrix = np.column_stack([first, second])
    result = white_reality_check(
        target, matrix, block_size=16, n_resamples=200, seed=12
    )
    repeat = white_reality_check(
        target, matrix, block_size=16, n_resamples=200, seed=12
    )

    assert result == repeat
    assert result["observed_best_index"] == 0
    assert 0.0 < result["p_value"] <= 1.0


def test_reality_check_requires_multiple_candidates():
    target, candidate, incumbent, _, _ = _sample()
    improvement = np.square(incumbent - target) - np.square(candidate - target)
    with pytest.raises(ValueError, match="at least 2 candidates"):
        white_reality_check(target, improvement[:, None], block_size=16)
