import numpy as np
import pandas as pd

from src.champion.v345_build_transition_workload_beta_package import (
    BETA_FUNCTIONS,
    FUNCTION_ANCHOR,
    NEW_TAIL,
    OLD_TAIL,
    BUNDLE_MEMBER,
    bundle_zip_info,
    patch_script,
    predict_beta_bundle,
)


def test_patch_script_adds_one_beta_runtime_and_route():
    source = FUNCTION_ANCHOR + "    pass\n\n" + OLD_TAIL
    patched = patch_script(source)
    assert patched.count("def _predict_beta_cell") == 1
    assert patched.count("def _beta_cell_mask") == 1
    assert BETA_FUNCTIONS in patched
    assert NEW_TAIL in patched
    assert OLD_TAIL not in patched


def test_beta_bundle_member_uses_a_fixed_zip_timestamp():
    info = bundle_zip_info()
    assert info.filename == BUNDLE_MEMBER
    assert info.date_time == (2026, 8, 31, 0, 0, 0)


def test_beta_runtime_clips_each_pool_candidate_before_weighting():
    frame = pd.DataFrame(
        {
            "game_type": ["R"],
            "pitcher_id": [1],
            "batter_id": [2],
            "asof_pitcher_n": [0.0],
            "asof_pitcher_success_rate": [0.0],
            "asof_pitcher_prev5_game_success_rate": [1.0],
            "asof_batter_n": [0.0],
            "asof_batter_success_rate": [0.0],
        }
    )
    bundle = {
        "prior_by_game_type": {"R": 0.5},
        "global_prior": 0.5,
        "pitcher_opening_n": {},
        "pitcher_opening_success": {},
        "batter_opening_n": {},
        "batter_opening_success": {},
        "concentration": 50.0,
        "weights": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0]),
    }
    np.testing.assert_allclose(predict_beta_bundle(frame, bundle), [0.001])
