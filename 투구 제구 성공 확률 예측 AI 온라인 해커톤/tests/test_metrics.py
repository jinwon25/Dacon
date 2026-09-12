import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    brier_score,
    brier_skill_score,
    brier_skill_score_unclipped,
    make_submission,
    validate_probabilities,
    validate_submission,
)


def test_perfect_prediction_score():
    y = np.array([0, 1, 1, 0])
    assert brier_score(y, y.astype(float)) == 0.0
    assert brier_skill_score(y, y.astype(float)) == 100000.0


def test_validation_mean_constant_score_is_zero():
    y = np.array([0, 1, 1, 0, 1], dtype=float)
    pred = np.full(len(y), y.mean())
    assert brier_skill_score(y, pred) == pytest.approx(0.0, abs=1e-10)


def test_official_score_clips_but_unclipped_score_preserves_comparison_signal():
    y = np.array([0, 1, 0, 1])
    bad = np.array([0.9, 0.1, 0.9, 0.1])
    less_bad = np.array([0.8, 0.2, 0.8, 0.2])

    assert brier_skill_score(y, bad) == 0.0
    assert brier_skill_score(y, less_bad) == 0.0
    assert brier_skill_score_unclipped(y, less_bad) > brier_skill_score_unclipped(
        y, bad
    )


@pytest.mark.parametrize(
    "bad",
    [np.array([-0.1]), np.array([1.1]), np.array([np.nan]), np.array([np.inf])],
)
def test_invalid_probabilities_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_probabilities(bad)


def test_submission_preserves_row_order_and_ids():
    ids = pd.Series(["TEST_3", "TEST_1", "TEST_2"], dtype="string")
    pred = np.array([0.3, 0.1, 0.2])
    submission = make_submission(ids, pred)
    validate_submission(submission, ids)
    assert submission["row_id"].tolist() == ids.tolist()


def test_submission_rejects_reordered_ids():
    ids = pd.Series(["TEST_3", "TEST_1", "TEST_2"], dtype="string")
    submission = make_submission(ids, [0.3, 0.1, 0.2])
    bad = submission.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        validate_submission(bad, ids)
