from pathlib import Path
import runpy


def test_uploaded_release_builder_uses_its_flat_directory() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "JY_fallback_XGB_active50_w030"
        / "build_jy_xgb_active_high50_zip.py"
    )
    namespace = runpy.run_path(str(script), run_name="jy_fallback_layout_test")

    release_root = script.parent
    assert namespace["ROOT"] == release_root
    assert namespace["ASSET"] == release_root
    assert namespace["RUNTIME"] == release_root / "fallback_xgb_frozen_runtime.py"
    assert namespace["SOURCE"] == (
        release_root
        / "parent"
        / "submit_jy_runners_high_li_bridge027_public1172.zip"
    )
    assert namespace["OUT"] == (
        release_root / "rebuilt" / "submit_jy_fallback_xgb_active50_w030.zip"
    )
