"""Regression tests for the 2026-09-05 submission-path corrections."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_entrypoint():
    spec = importlib.util.spec_from_file_location("submission_reproduce", ROOT / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_output_does_not_overwrite(tmp_path):
    module = load_entrypoint()
    with pytest.raises(FileExistsError):
        module.fresh_output(tmp_path)


def test_preflight_rejects_changed_official_data(tmp_path):
    (tmp_path / "train.csv").write_text("not the official train", encoding="utf-8")
    with pytest.raises(ValueError, match="Official data hash mismatch"):
        load_entrypoint().check_inputs(tmp_path)


@pytest.mark.parametrize("script, flag", [
    ("fallback_xgb/build_pressure_blend_package.py", "--asset-dir"),
    ("fallback_routes/build_candidate_zip.py", "--source"),
    ("fallback_routes/build_candidate_zip.py", "--asset-dir"),
    ("h1/exp/build_asof.py", "--data-dir"),
    ("h1/exp/build_asof.py", "--output-dir"),
])
def test_required_portability_flags_exist(script, flag):
    tree = ast.parse((ROOT / script).read_text(encoding="utf-8"))
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "add_argument" and any(
                   isinstance(arg, ast.Constant) and arg.value == flag for arg in node.args)
               for node in ast.walk(tree))


@pytest.mark.parametrize("script, message", [
    ("build_fallback_xgb_model.py", "Exact CUDA training parity is unverified"),
    ("rebuild_fallback_xgb_oof.py", "Retired:"),
])
def test_unverified_or_leaky_training_fails_before_writing(script, message, tmp_path):
    output = tmp_path / "must_not_exist"
    result = subprocess.run([sys.executable, str(ROOT / "fallback_xgb" / script),
                             "--output-dir", str(output)], capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 2
    assert message in result.stderr
    assert not output.exists()
