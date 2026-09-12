from pathlib import Path
import runpy


def test_v244_builder_is_isolated_from_public1175_release():
    root = Path(__file__).resolve().parents[1]
    script = root / "src" / "team_assets" / "JY_fallback_XGB_mechanism_w045_w015" / "build_candidate_zip.py"
    namespace = runpy.run_path(str(script), run_name="v244_layout_test")

    assert namespace["INCUMBENT"] == root / "src" / "team_assets" / "JY_fallback_XGB_active50_w030"
    assert namespace["OUT"] == (
        script.parent / "rebuilt" / "submit_jy_xgb_mechanism_w045_w015.zip"
    )
    assert namespace["SOURCE"].name == (
        "submit_jy_runners_high_li_bridge027_public1172.zip"
    )
