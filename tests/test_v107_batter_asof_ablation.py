import pandas as pd
import pytest

from src.v107_batter_asof_ablation import drop_batter_asof


def test_drop_batter_asof_preserves_legitimate_context() -> None:
    frame = pd.DataFrame(
        {
            "asof_batter_n": [10.0],
            "asof_batter_success_rate": [0.5],
            "asof_batter_middle_rate": [0.1],
            "shrunk_batter_rate": [0.51],
            "batter_hand": ["1"],
            "batter_team_id": ["2"],
        }
    )
    dropped = drop_batter_asof(
        frame,
        [
            "asof_batter_n",
            "asof_batter_success_rate",
            "asof_batter_middle_rate",
            "shrunk_batter_rate",
        ],
    )
    assert set(dropped.columns) == {"batter_hand", "batter_team_id"}


def test_drop_batter_asof_rejects_contract_drift() -> None:
    with pytest.raises(ValueError):
        drop_batter_asof(
            pd.DataFrame({"batter_hand": ["1"], "batter_team_id": ["2"]}),
            ["asof_batter_n"],
        )
