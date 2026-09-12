from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.champion import v193_build_triyear_stack_package as v193


ROOT = Path(__file__).resolve().parents[1]


def _constants(source: str) -> dict[str, float]:
    tree = ast.parse(source)
    output: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {
            "STACK_SCALE", "V114_WEIGHT", "V131_WEIGHT", "V135_WEIGHT", "V160_WEIGHT"
        }:
            output[target.id] = float(ast.literal_eval(node.value))
    return output


def test_patch_runtime_matches_frozen_v192_weights() -> None:
    source = (ROOT / "src/champion/v180_signed_stack_runtime.py").read_text(
        encoding="utf-8"
    )
    patched = v193._patch_runtime(source)
    values = _constants(patched)
    assert values == {
        "STACK_SCALE": 1.0,
        "V114_WEIGHT": v193.NET_WEIGHTS[
            "v114_independent_source_stability_mask_20260823_01"
        ],
        "V131_WEIGHT": v193.NET_WEIGHTS[
            "v131_catboost_h1_independent_oof_20260823_01"
        ],
        "V135_WEIGHT": v193.NET_WEIGHTS["v135_c3_recent_window_20260823_01"],
        "V160_WEIGHT": 0.0,
    }
    assert "candidate=v193_triyear_stack" in patched


def test_patch_runtime_fails_closed_on_changed_source() -> None:
    source = (ROOT / "src/champion/v180_signed_stack_runtime.py").read_text(
        encoding="utf-8"
    )
    with pytest.raises(ValueError, match="expected one runtime token"):
        v193._patch_runtime(source.replace("STACK_SCALE = 0.25", "STACK_SCALE = 0.24"))


def test_weight_contract_matches_saved_v192_summary(tmp_path: Path) -> None:
    source = ROOT / "artifacts/v192_soft_loo_triyear_stack_20260828_01/summary.json"
    if not source.is_file():
        pytest.skip("local ignored v192 summary is unavailable")
    contract = v193._verify_weight_contract(source)
    assert contract["selected_budget"] == 0.15
    assert contract["full_2022_gain"] > 0.0
    assert contract["late_2023_gain"] > 0.0
    assert contract["full_2024_gain"] > 0.0
    assert contract["robust_gate_passed"] is True

    changed = json.loads(source.read_text(encoding="utf-8"))
    changed["restrictions"]["test_aggregate_used"] = True
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="restriction contract mismatch"):
        v193._verify_weight_contract(invalid)
