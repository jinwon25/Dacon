import numpy as np
import pandas as pd

from src.archive.v201_paired_workload_booster import workload_feature_frame
from src.champion import v343_build_transition_workload_package as v343


def test_embedded_workload_features_match_training_formula() -> None:
    frame = pd.DataFrame(
        {
            "asof_pitcher_success_rate": [0.51, 0.48, np.nan],
            "asof_pitcher_middle_rate": [0.21, 0.25, np.nan],
            "asof_pitcher_prev1_game_success_rate": [0.5, 0.428571, np.nan],
            "asof_pitcher_prev1_game_middle_rate": [0.25, 0.285714, np.nan],
            "asof_pitcher_prev3_game_success_rate": [0.52, 0.45, 1.0],
            "asof_pitcher_prev3_game_middle_rate": [0.20, 0.30, 0.0],
            "asof_pitcher_prev5_game_success_rate": [0.55, 0.47, 0.0],
            "asof_pitcher_prev5_game_middle_rate": [0.22, 0.24, 1.0],
        }
    )
    expected = workload_feature_frame(frame)
    namespace = {"np": np, "pd": pd}
    exec(v343.WORKLOAD_FUNCTIONS, namespace)
    actual = namespace["_attach_workload_features"](frame)[expected.columns]
    assert np.allclose(actual, expected, equal_nan=True)


def test_patch_script_requires_unique_route_tail() -> None:
    source = v343.FUNCTION_ANCHOR + "    pass\n\n" + v343.OLD_TAIL
    patched = v343.patch_script(source)
    assert patched.count("def _predict_workload_h1") == 1
    assert "workload_delta = workload_proposal - jy_probability" in patched
