from src.champion.v296_build_abs_regime_package import _patch_script


def test_v296_patch_adds_regular_logit_blend_once() -> None:
    source = '''def _window_adjustment(frame):
    pass

def predict_components(frame):
    return parent, h1, c3, active, output
'''
    patched = _patch_script(source)
    assert patched.count("def _predict_abs_regular") == 1
    assert patched.count('frame["game_type"].astype(str).eq("R")') == 1
    assert "parent_logit + 0.10 * (expert_logit - parent_logit)" in patched
    assert patched.count("return parent, h1, c3, active, output") == 1
