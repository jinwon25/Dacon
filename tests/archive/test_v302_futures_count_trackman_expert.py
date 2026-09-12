import pandas as pd
import pytest

from src.archive.v302_futures_count_trackman_expert import attach_trackman


def test_attach_trackman_preserves_row_alignment():
    result = attach_trackman(pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"tm": [3, 4]}))
    assert result.to_dict("list") == {"x": [1, 2], "tm": [3, 4]}


def test_attach_trackman_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        attach_trackman(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]}))
