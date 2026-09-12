import pandas as pd

from src.champion.v286_build_exact_anchor_package import (
    _restore_v244_script,
    replace_anchors,
)


def test_restore_v244_removes_pseudo_route_and_preserves_fixed_routes() -> None:
    source = '''def _predict_pseudo_deployment(frame):
    return frame

def _window_adjustment(frame):
    pass

def predict_components(frame):
    jy_probability = output.copy()
    pseudo_selected = True
    return parent, h1, c3, active, output
'''
    restored = _restore_v244_script(source)
    assert "def _predict_pseudo_deployment" not in restored
    assert "pseudo_selected" not in restored
    assert "xgb_probability = _predict_fallback_xgb(frame)" in restored
    assert "XGB_NONPRESSURE_OPPOSITE_WEIGHT" in restored


def test_replace_anchors_uses_latest_career_cumulative_state() -> None:
    history = pd.DataFrame(
        {
            "season": [2023, 2024, 2024],
            "pitcher_id": [1, 1, 1],
            "batter_id": [2, 2, 2],
            "asof_pitcher_n": [10, 100, 120],
            "asof_batter_n": [20, 80, 90],
            "asof_pitcher_success_rate": [0.5, 0.4, 0.5],
            "asof_pitcher_reverse_rate": [0.1, 0.1, 0.1],
            "asof_pitcher_middle_rate": [0.2, 0.2, 0.2],
            "asof_pitcher_ball_rate": [0.3, 0.3, 0.3],
            "asof_pitcher_strike_rate": [0.4, 0.4, 0.4],
            "asof_batter_success_rate": [0.5, 0.5, 0.6],
            "asof_batter_middle_rate": [0.2, 0.2, 0.3],
        }
    )
    lookup = {"anchors": {prefix: {} for prefix in ("p_succ", "p_rev", "p_mid", "p_ball", "p_stk", "b_succ", "b_mid")}}
    updated = replace_anchors(lookup, history)
    assert updated["anchors"]["p_succ"][1] == (120.0, 60.0)
    assert updated["anchors"]["b_succ"][2] == (90.0, 54.0)
    assert updated["features_version"] == 2
