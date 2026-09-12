import numpy as np

from src.archive.v333_failure_mode_student_rebase_v320 import (
    PROTOCOL,
    apply_blend,
    route_masks,
)


def test_v333_protocol() -> None:
    assert PROTOCOL == "V333_FAILURE_MODE_STUDENT_REBASE_V320_V1"


def test_apply_blend_changes_only_active_rows() -> None:
    parent = np.asarray([0.4, 0.5, 0.6])
    raw = np.asarray([0.8, 0.1, 0.2])
    active = np.asarray([True, False, True])
    candidate = apply_blend(parent, raw, active, 0.1)
    np.testing.assert_allclose(candidate, [0.44, 0.5, 0.56])
