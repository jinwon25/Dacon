import pytest

from src.champion.v356_build_late_hierarchy_only_package import (
    FUNCTION_ANCHOR,
    NEW_TAIL,
    OLD_TAIL,
    _validate_audit,
    patch_script,
)


def test_patch_adds_only_one_hierarchy_runtime():
    source = FUNCTION_ANCHOR + "    pass\n\n" + OLD_TAIL
    patched = patch_script(source)
    assert patched.count("def _predict_late_hierarchy") == 1
    assert NEW_TAIL in patched
    assert OLD_TAIL not in patched
    assert "def _predict_trackman_pfd" not in patched


def test_patch_rejects_trackman_parent():
    source = "def _predict_trackman_pfd():\n    pass\n\n" + FUNCTION_ANCHOR + OLD_TAIL
    with pytest.raises(ValueError, match="TrackMan"):
        patch_script(source)


def test_audit_gate_requires_positive_sources_and_robustness():
    audit = {
        "protocol": "V354_LATE_HIERARCHY_REBASE_V345_V1",
        "candidate_gate_passed": True,
        "metrics": {
            "full_2022": {"gain": 1.0},
            "late_2023": {"gain": 1.0},
            "full_2024": {"gain": 1.0},
        },
        "locked_robustness": {
            "pitcher": {"p05": 1.0},
            "crossed_pitcher_batter": {"p05": 1.0},
            "chronological_block": {"p05": 1.0},
        },
    }
    _validate_audit(audit)
    audit["locked_robustness"]["crossed_pitcher_batter"]["p05"] = -0.1
    with pytest.raises(ValueError, match="robustness"):
        _validate_audit(audit)
