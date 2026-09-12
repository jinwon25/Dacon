from pathlib import Path
import runpy


def test_uploaded_release_builder_uses_its_flat_directory() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "fallback_xgb"
        / "build_pressure_blend_package.py"
    )
    namespace = runpy.run_path(str(script), run_name="fallback_layout_test")

    release_root = script.parent
    assert namespace["ROOT"] == release_root
    assert namespace["ASSET"] == release_root
    assert namespace["RUNTIME"] == release_root / "fallback_xgb_frozen_runtime.py"
    assert namespace["DEFAULT_SOURCE"] == (
        release_root
        / "parent"
        / "submit_row_region_gate_bridge027.zip"
    )
    assert namespace["DEFAULT_OUT"] == (
        release_root / "rebuilt" / "submit_fallback_xgb_pressure_w030.zip"
    )
