import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v84_probe_wrapper import apply_fixed_v56


FIELDS = [
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "count_state",
    "hand_matchup",
    "inning_bucket",
    "base_state",
    "domain3",
    "pressure",
]


def _write_model(path: Path) -> None:
    path.mkdir()
    vocabularies = {field: {"1": 1} for field in FIELDS}
    vocabularies.update(
        {
            "count_state": {"1-1": 1},
            "hand_matchup": {"R-L": 1},
            "inning_bucket": {"middle": 1},
            "base_state": {"1": 1},
            "domain3": {"R_CORE": 1, "F": 2},
            "pressure": {"normal": 1},
        }
    )
    (path / "vocabularies.json").write_text(json.dumps(vocabularies))
    metadata = {
        "protocol": "V84_FIXED_V56_F_ROUTE_NUMPY_EXPORT_V1",
        "fields": FIELDS,
        "pair_index": [[0, 8]],
        "domain_centres": {"R_CORE": 0.0, "F": 0.0},
        "correction_clip": 0.25,
        "route": "F",
        "eta": 0.1,
    }
    (path / "metadata.json").write_text(json.dumps(metadata))
    arrays = {f"field_{index}": np.zeros((3, 1), dtype=np.float32) for index in range(10)}
    arrays["field_0"][1, 0] = 0.5
    arrays["field_8"][2, 0] = 0.4
    np.savez_compressed(path / "embeddings.npz", **arrays)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pitcher_id": [1, 1],
            "batter_id": [1, 1],
            "pitcher_team_id": [1, 1],
            "batter_team_id": [1, 1],
            "balls_before": [1, 1],
            "strikes_before": [1, 1],
            "pitcher_hand": ["R", "R"],
            "batter_hand": ["L", "L"],
            "inning": [4, 4],
            "base_state": [1, 1],
            "game_type": ["R", "F"],
        }
    )


def test_fixed_v56_changes_only_f(tmp_path: Path) -> None:
    model = tmp_path / "fm"
    _write_model(model)
    parent = np.asarray([0.5, 0.5])
    output = apply_fixed_v56(parent, _frame(), model)
    assert output[0] == parent[0]
    assert output[1] > parent[1]


def test_fixed_v56_is_row_order_independent(tmp_path: Path) -> None:
    model = tmp_path / "fm"
    _write_model(model)
    frame = _frame()
    parent = np.asarray([0.4, 0.6])
    original = apply_fixed_v56(parent, frame, model)
    shuffled = apply_fixed_v56(parent[::-1], frame.iloc[::-1].reset_index(drop=True), model)
    np.testing.assert_array_equal(original, shuffled[::-1])
