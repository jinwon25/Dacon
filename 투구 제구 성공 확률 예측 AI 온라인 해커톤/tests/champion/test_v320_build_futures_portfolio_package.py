from src.champion.v320_build_futures_portfolio_package import (
    NEW_FUTURES_BLOCK,
    patch_script,
)


def test_patch_script_is_fail_closed_and_changes_only_expected_contract() -> None:
    source = '''def _predict_recent_futures(frame):
    return frame


def _window_adjustment(
    frame, tables, scale
):
    pass

def run(frame, output, futures, futures_probability):
    if True:
        output[futures] = np.clip(
            output[futures]
            + 0.10 * (futures_probability[futures] - output[futures]),
            0.001,
            0.999,
        )
'''
    patched = patch_script(source)
    assert patched.count("def _predict_futures_lowrank") == 1
    assert NEW_FUTURES_BLOCK in patched
    assert "+ 0.10 *" not in patched
