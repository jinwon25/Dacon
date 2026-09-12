import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
    spec = importlib.util.spec_from_file_location("auxiliary_fit", Path(__file__).resolve().parents[1]/"train_auxiliary_lookups.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Arrays(dict):
    @property
    def files(self):
        return list(self)


def test_lookup_strings_and_numbers_exact():
    a = Arrays(keys=np.array(["SAME|0-0|R"]), values=np.array([0.125],dtype=np.float32))
    result = module().compare_arrays(a,a)
    assert all(r["exact"] and r["max_abs"] == 0 for r in result.values())


def test_lookup_dtype_change_is_not_silently_accepted():
    with pytest.raises(ValueError,match="shape/dtype"):
        module().compare_arrays(Arrays(x=np.array([1],dtype=np.int16)),Arrays(x=np.array([1],dtype=np.int64)))


def test_lookup_key_mismatch_is_rejected():
    with pytest.raises(ValueError,match="keys"):
        module().compare_arrays(Arrays(a=np.array([1])),Arrays(b=np.array([1])))
