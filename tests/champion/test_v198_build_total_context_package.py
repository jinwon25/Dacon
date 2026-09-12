from __future__ import annotations

import ast
import zipfile
from pathlib import Path

import pytest

from src.champion import v198_build_total_context_package as v198


ROOT = Path(__file__).resolve().parents[2]


def test_runtime_patch_adds_frozen_gate_and_weight() -> None:
    package = ROOT / "artifacts/v193_triyear_stack_package_20260828_01/submit_v193_triyear_stack.zip"
    if not package.is_file():
        pytest.skip("ignored v193 package unavailable")
    with zipfile.ZipFile(package) as archive:
        source = archive.read("script.py").decode("utf-8")
    patched = v198._patch_runtime(source)
    tree = ast.parse(patched)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "HIER_CONTEXT_WEIGHT":
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    assert constants == {"HIER_CONTEXT_WEIGHT": 0.05}
    assert 'np.abs(score) <= 1.0' in patched
    assert '(leverage > 0.7) & (leverage <= 1.5)' in patched
    assert "candidate=v198_total_context_stack" in patched


def test_runtime_patch_fails_closed() -> None:
    with pytest.raises(ValueError, match="expected one runtime token"):
        v198._patch_runtime("not the v193 runtime")
