import pytest

from src.archive.package_v26_anchor_weight_probe import PROBE_ETA, SOURCE_ETA, build


def test_v26_probe_is_equally_spaced_from_v22_and_v25() -> None:
    assert SOURCE_ETA == 0.075
    assert PROBE_ETA == 2.0 * SOURCE_ETA


def test_probe_rejects_non_increasing_eta(tmp_path) -> None:
    with pytest.raises(ValueError, match="probe eta"):
        build(tmp_path, tmp_path / "missing.zip", tmp_path / "out.zip", 0.05)
