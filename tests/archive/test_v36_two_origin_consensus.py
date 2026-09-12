import pandas as pd
import pytest

from src.archive.v36_two_origin_consensus import select_consensus


def _rows(year_gain_a, year_gain_b):
    common = {
        "domain": "R_CORE",
        "weight": 0.1,
        "direction": "toward_parent",
        "positive_month_fraction": 1.0,
        "worst_month_gain": 1.0,
        "applied_domain_gain": 2.0,
    }
    return pd.DataFrame(
        [
            {"signal": "state::a", "gain": year_gain_a, **common},
            {"signal": "exact::b", "gain": year_gain_b, **common},
        ]
    )


def test_consensus_rejects_one_origin_reversal():
    merged, chosen = select_consensus(_rows(3.0, 4.0), _rows(-1.0, 2.0))
    assert int(merged["passes_consensus_gate"].sum()) == 1
    assert chosen["signal"] == "exact::b"


def test_consensus_never_matches_a_different_weight():
    left = _rows(3.0, 4.0)
    right = _rows(2.0, 3.0)
    right["weight"] = 0.2
    with pytest.raises(ValueError, match="no exact recipe"):
        select_consensus(left, right)
