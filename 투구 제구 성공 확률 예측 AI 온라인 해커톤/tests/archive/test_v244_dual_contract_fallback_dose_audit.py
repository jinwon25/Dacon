import numpy as np

from src.archive.v244_dual_contract_fallback_dose_audit import (
    SCALES,
    scaled_candidate,
    select_scale,
)


def test_scaled_candidate_is_common_delta_dose():
    base = np.array([0.4, 0.6])
    proposal = np.array([0.5, 0.4])
    np.testing.assert_allclose(
        scaled_candidate(base, proposal, 1.5), [0.55, 0.3]
    )


def test_select_scale_maximises_worst_exact_source_after_all_parent_gate():
    results = {}
    for scale in SCALES:
        key = f"{scale:g}"
        gain = 3.0 - abs(scale - 1.5)
        results[key] = {
            family: {
                axis: {"gain": gain if family == "exact_jy" else 1.0}
                for axis in ("full_2022", "late_2023")
            }
            for family in ("evidence_proxy", "exact_jy")
        }
    assert select_scale(results) == 1.5
