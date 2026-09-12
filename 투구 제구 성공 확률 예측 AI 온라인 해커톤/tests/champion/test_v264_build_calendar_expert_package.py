from src.champion.v264_build_calendar_expert_package import _patch_script


def test_patch_script_inserts_experts_and_calendar_once() -> None:
    source = '''def _predict_fallback_xgb(frame):
    return frame

def _window_adjustment(
    pass

def formula():
    jy_probability = output.copy()
    old = True
    return parent, h1, c3, active, output
'''
    patched = _patch_script(source)
    assert patched.count("def _predict_calendar_experts") == 1
    assert patched.count('frame["game_month"].between(4, 9)') == 1
    assert "0.60 * command_probability" in patched
    assert "0.25 * batter_probability" in patched
