from pathlib import Path

from src.champion.v352_build_trackman_pfd_package import (
    FUNCTION_ANCHOR,
    IMPORT_ANCHOR,
    MODEL_ROOT,
    NEW_TAIL,
    OLD_TAIL,
    TRACKMAN_FUNCTIONS,
    model_zip_info,
    normalised_model_bytes,
    patch_script,
)


def test_patch_script_adds_trackman_runtime_and_anchor_route():
    source = IMPORT_ANCHOR + "\n" + FUNCTION_ANCHOR + "    pass\n\n" + OLD_TAIL
    patched = patch_script(source)
    assert "import lightgbm as lgb" in patched
    assert patched.count("def _predict_trackman_pfd") == 1
    assert TRACKMAN_FUNCTIONS in patched
    assert NEW_TAIL in patched
    assert OLD_TAIL not in patched


def test_model_member_uses_fixed_timestamp():
    info = model_zip_info("example.txt")
    assert info.filename == f"{MODEL_ROOT}/example.txt"
    assert info.date_time == (2026, 8, 31, 0, 0, 0)


def test_normalised_model_bytes_remove_crlf(tmp_path: Path):
    path = tmp_path / "model.txt"
    path.write_bytes(b"tree\r\nversion=v4\r\n")
    assert normalised_model_bytes(path) == b"tree\nversion=v4\n"
