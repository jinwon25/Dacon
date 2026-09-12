import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "fallback_xgb/build_fallback_xgb_model.py"
    spec = importlib.util.spec_from_file_location("xgb_fit_contract", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Probe:
    def __init__(self, device):
        self.device = device

    def save_config(self):
        return json.dumps({"learner": {"generic_param": {"device": self.device}}})


def test_cpu_request_is_explicit_and_does_not_probe_cuda():
    m = module()
    with patch.object(m.xgb, "train") as train:
        assert m.verify_training_device("cpu") == "cpu"
        train.assert_not_called()


def test_silent_cuda_fallback_is_rejected():
    m = module()
    with patch.object(m.xgb, "train", return_value=Probe("cpu")):
        with pytest.raises(RuntimeError, match="selected CPU"):
            m.verify_training_device("cuda:0")


def test_actual_cuda_device_is_reported():
    m = module()
    with patch.object(m.xgb, "train", return_value=Probe("cuda:0")):
        assert m.verify_training_device("cuda:0") == "cuda:0"


def test_invalid_device_rejected():
    with pytest.raises(ValueError):
        module().verify_training_device("gpu_unknown")
