import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h1_recipe_uses_actual_depth_and_frozen_center(tmp_path):
    module = module_at("repro_contract", ROOT / "train.py")
    recipe = json.loads((ROOT / "h1/training_recipe.json").read_text(encoding="utf-8"))
    args = module.h1_training_arguments(tmp_path, tmp_path / "output", recipe["required_versions"])
    assert args[args.index("--depth") + 1] == 8
    assert args[args.index("--seeds") + 1] == 3
    assert "--skip-eval" in args
    assert args[args.index("--center") + 1] == 0.5854452601930041


def test_h1_version_mismatch_does_not_silently_fit(tmp_path):
    module = module_at("repro_contract", ROOT / "train.py")
    with pytest.raises(RuntimeError, match="separate H1 training environment"):
        module.h1_training_arguments(tmp_path, tmp_path / "output", {"catboost": "1.2.8"})
    assert not (tmp_path / "output").exists()


def test_strict_source_hashes_and_all_source_seasons(tmp_path):
    runner = module_at("strict_repro_contract", ROOT / "strict/reproduce_strict.py")
    assert runner.verify_sources() == 13
    learner = runner.configure("lgb_oof", tmp_path / "data", tmp_path / "work")
    assert learner.VALIDATION_SEASONS == [2021, 2022, 2023, 2024]
    assert learner.ARTIFACT_ROOT == tmp_path / "work/artifacts/EXP-019/r_full_residual"
    runner.configure("lgb_oof", tmp_path / "data2", tmp_path / "work2")
    assert learner.ARTIFACT_ROOT == tmp_path / "work2/artifacts/EXP-019/r_full_residual"


def test_strict_lowrank_checks_source_arrays_not_just_summary(tmp_path):
    runner = module_at("strict_prerequisite_contract", ROOT / "strict/reproduce_strict.py")
    required = runner.prerequisite_files("lowrank", tmp_path)
    for year in (2021, 2022, 2023, 2024):
        assert tmp_path / f"artifacts/EXP-020/pitcher_count_eb_atop_team/predictions_team_pc_all_{year}.npy" in required
