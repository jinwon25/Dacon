import pytest
from train_c3_fresh import compare


def test_c3_nested_tables_compare_numerically():
    assert compare({"a": [{"pitcher": 0.1}]}, {"a": [{"pitcher": 0.1}]}) == 0
    assert compare({"a": 0.1}, {"a": 0.11}) == pytest.approx(0.01)


def test_c3_missing_key_is_rejected():
    with pytest.raises(ValueError, match="dictionary mismatch"):
        compare({"a": 0.1}, {})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_c3_nonfinite_value_is_rejected(value):
    with pytest.raises(ValueError, match="non-finite"):
        compare({"a": value}, {"a": value})
