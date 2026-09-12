from src.champion.v277_build_pseudo_deployment_package import _patch_script


def test_v277_patch_inserts_one_pseudo_helper_and_frozen_formula() -> None:
    source = '''def _window_adjustment(
    frame, tables, scale
):
    pass

def predict_components(frame):
    output = parent.copy()
    jy_probability = output.copy()
    old = True
    return parent, h1, c3, active, output
'''
    patched = _patch_script(source)
    assert patched.count("def _predict_pseudo_deployment") == 1
    assert patched.count('frame["game_month"].between(4, 9)') == 1
    assert "0.50 * xgb_probability[pseudo_selected]" in patched
    assert "0.50 * pseudo_probability[pseudo_selected]" in patched
