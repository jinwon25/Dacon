from src.champion.v290_build_exact_anchor_futures_package import _patch_script


def test_v290_patch_adds_futures_helper_and_fixed_weight_once() -> None:
    source = '''def _window_adjustment(frame):
    pass

def predict_components(frame):
    return parent, h1, c3, active, output
'''
    patched = _patch_script(source)
    assert patched.count("def _predict_recent_futures") == 1
    assert patched.count('frame["game_type"].astype(str).eq("F")') == 1
    assert "+ 0.10 * (futures_probability[futures] - output[futures])" in patched
