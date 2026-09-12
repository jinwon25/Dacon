import numpy as np
import pandas as pd

from src.champion.v334_build_aggressive_moe_package import (
    NEW_ROUTE_BLOCK,
    patch_script,
)
from src.champion.v334_finalize_player_transition import _count_key


def test_count_key_is_row_local() -> None:
    frame = pd.DataFrame({"balls_before": [0, 3], "strikes_before": [2, 1]})
    assert _count_key(frame).tolist() == ["0-2", "3-1"]


def test_patch_adds_disjoint_r_routes_and_preserves_f_formula() -> None:
    source = '''def _window_adjustment(
    frame, tables, scale
):
    pass

def run(frame, output, parent, h1, c3, active):
    futures = frame["game_type"].astype(str).eq("F").to_numpy()
    if np.any(futures):
        futures_probability = _predict_recent_futures(frame)
        lowrank_probability_delta = _predict_futures_lowrank(frame)
        output[futures] = np.clip(
            output[futures]
            + 0.20 * (futures_probability[futures] - output[futures])
            + 0.50 * lowrank_probability_delta[futures],
            0.001,
            0.999,
        )
    return parent, h1, c3, active, output
'''
    patched = patch_script(source)
    assert patched.count("def _predict_player_transition") == 1
    assert NEW_ROUTE_BLOCK in patched
    assert "regular & ~anchor" in patched
    assert "+ 0.20 * (futures_probability[futures] - output[futures])" in patched
    assert "0.25 * transition_delta[rcore]" in patched
    assert "0.50 * lowrank_probability_delta[anchor]" in patched


def test_status_runtime_contract_has_explicit_unseen_and_return_logic() -> None:
    source = patch_script('''def _window_adjustment(
    frame, tables, scale
):
    pass

def run(frame, output, parent, h1, c3, active):
    futures = frame["game_type"].astype(str).eq("F").to_numpy()
    if np.any(futures):
        futures_probability = _predict_recent_futures(frame)
        lowrank_probability_delta = _predict_futures_lowrank(frame)
        output[futures] = np.clip(
            output[futures]
            + 0.20 * (futures_probability[futures] - output[futures])
            + 0.50 * lowrank_probability_delta[futures],
            0.001,
            0.999,
        )
    return parent, h1, c3, active, output
''')
    assert 'status[~seen] = "NEW"' in source
    assert 'status[seen & (last_year < 2024)] = "RETURN"' in source
    assert np.isfinite(0.25)
