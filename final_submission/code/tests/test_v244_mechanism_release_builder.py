from pathlib import Path
import runpy


def test_v244_builder_is_isolated_from_public1175_release():
    root = Path(__file__).resolve().parents[1]
    script = root / "fallback_routes" / "build_candidate_zip.py"
    namespace = runpy.run_path(str(script), run_name="v244_layout_test")

    assert namespace["INCUMBENT"] == root / "fallback_xgb"
    assert namespace["OUT"] == (
        script.parent / "rebuilt" / "submit_fallback_routes.zip"
    )
    assert namespace["SOURCE"].name == (
        "submit_row_region_gate_bridge027.zip"
    )
