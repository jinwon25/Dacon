from src.champion.v335_build_anchor_lowrank_package import (
    NEW_ROUTE_BLOCK,
    patch_script,
)


def test_patch_preserves_f_and_adds_only_anchor_route() -> None:
    source = '''def run(frame, output, parent, h1, c3, active):
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
    assert NEW_ROUTE_BLOCK in patched
    assert "regular & (" in patched
    assert "lowrank_probability_delta[anchor]" in patched
    assert "player_transition" not in patched
    assert "transition_delta" not in patched
