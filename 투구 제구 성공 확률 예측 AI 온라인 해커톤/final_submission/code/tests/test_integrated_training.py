import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from common_features.season import multi_scale_success

ROOT = Path(__file__).resolve().parents[1]


def test_multi_scale_success_uses_season_opening_state():
    frame = pd.DataFrame({
        "pitcher_id": [1, 1, 1, 1], "batter_id": [2, 2, 2, 2],
        "season": [2023, 2023, 2024, 2024],
        "asof_pitcher_n": [100, 120, 300, 340],
        "asof_pitcher_success_rate": [.5, .5, .5, .6],
        "asof_batter_n": [100, 120, 300, 340],
        "asof_batter_success_rate": [.5, .5, .5, .6],
    })
    original = frame.copy(deep=True)
    result = multi_scale_success(frame)
    assert result.shape == (4, 8)
    np.testing.assert_allclose(result["p_succ_k25"], [
        .525, (10 + 25 * .525) / 45, .525, (54 + 25 * .525) / 65], rtol=1e-7)
    assert all(dtype == np.float32 for dtype in result.dtypes)
    pd.testing.assert_frame_equal(frame, original)


def test_shared_feature_is_wired_into_fallback():
    from src.archive.v217_rebuild_fallback_xgb_oof import multi_scale_success as imported
    assert imported is multi_scale_success


def test_integrated_h1_has_no_repository_metadata():
    import json
    recipe = json.loads((ROOT / "h1/training_recipe.json").read_text(encoding="utf-8"))
    assert recipe["depth"] == 8
    assert not any(key.startswith("source_") for key in recipe)
    assert "archive_sha256" in recipe


def test_strict_command_is_exposed_without_training(capsys):
    import sys
    import pytest
    from unittest.mock import patch
    spec = importlib.util.spec_from_file_location("integrated_cli", ROOT / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with patch.object(sys, "argv", ["train.py", "--help"]), pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 0
    assert "train-strict" in capsys.readouterr().out
