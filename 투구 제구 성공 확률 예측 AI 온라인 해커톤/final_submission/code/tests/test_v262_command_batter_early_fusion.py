import pandas as pd
import pytest

from src.archive.v262_command_batter_early_fusion_xgb import combine_feature_blocks


def _block(prefix: str, rows: int, columns: int) -> pd.DataFrame:
    return pd.DataFrame(
        {f"{prefix}_{index}": [float(index)] * rows for index in range(columns)}
    )


def test_combine_feature_blocks_preserves_rows_and_unique_columns() -> None:
    combined = combine_feature_blocks(
        _block("base", 3, 114),
        _block("command", 3, 98),
        _block("batter", 3, 89),
    )
    assert combined.shape == (3, 301)
    assert combined.columns.is_unique


def test_combine_feature_blocks_rejects_row_mismatch() -> None:
    with pytest.raises(ValueError, match="different row counts"):
        combine_feature_blocks(
            _block("base", 2, 114),
            _block("command", 3, 98),
            _block("batter", 3, 89),
        )
