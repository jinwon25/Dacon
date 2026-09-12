import numpy as np
import pandas as pd

from src.archive.v61_final_gate_oof import (
    apply_final_gate,
    compare_profiles,
    gate_weights,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "pitcher_id": [1, 2, 3],
            "domain3": ["R_ANCHOR", "R_ANCHOR", "F"],
            "balls_before": [3, 0, 3],
            "strikes_before": [0, 0, 0],
            "asof_pitcher_n": [100.0, 100.0, 100.0],
            "asof_pitcher_success_rate": [0.60, 0.60, 0.60],
            "asof_batter_n": [80.0, 80.0, 80.0],
            "asof_batter_success_rate": [0.40, 0.40, 0.40],
        }
    )


def _profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "pitcher_id": [1, 2, 3],
            "tm_linked": [1, 0, 1],
            "tm_pitcher_n": [500.0, 1000.0, 500.0],
        }
    )


def test_gate_weights_match_final_formula_and_routing():
    gate = gate_weights(_frame(), _profile())
    assert np.isclose(gate[0], 0.03 * 0.5 * 1.25)
    assert gate[1] == 0.0
    assert gate[2] == 0.0


def test_apply_final_gate_only_moves_active_row_toward_prior():
    frame = _frame()
    parent = np.array([0.50, 0.50, 0.50])
    candidate, gate = apply_final_gate(frame, parent, _profile(), 0.52)
    assert gate[0] > 0.0
    assert candidate[0] != parent[0]
    np.testing.assert_array_equal(candidate[1:], parent[1:])


def test_compare_profiles_accepts_reordered_exact_gate_fields():
    frozen = _profile()
    reconstructed = frozen.iloc[::-1].reset_index(drop=True)
    result = compare_profiles(reconstructed, frozen)
    assert result["status"] == "pass"
    assert result["tm_pitcher_n_max_abs_diff"] == 0.0


def test_compare_profiles_rejects_count_drift():
    frozen = _profile()
    reconstructed = frozen.copy()
    reconstructed.loc[0, "tm_pitcher_n"] += 1
    assert compare_profiles(reconstructed, frozen)["status"] == "fail"
