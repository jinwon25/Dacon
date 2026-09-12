from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.package_standalone_champion import (
    build,
    materialize,
    official_row_independence_test,
    validate_payload,
    write_release_manifest,
)


SCRIPT = '''
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

def apply_v25_postbreak_anchor_overlay(probability, frame):
    return probability

def apply_trackman_asof_gate_overlay(probability, frame, global_rate):
    return probability

def predict_dataframe(frame):
    prediction = 1.0 / (1.0 + np.exp(-frame["x"].to_numpy(dtype=float)))
    prediction = apply_v25_postbreak_anchor_overlay(prediction, frame)
    prediction = apply_trackman_asof_gate_overlay(
        prediction, frame, 0.5
    )
    return prediction

def main():
    frame = pd.read_csv(BASE_DIR / "data" / "test.csv")
    output = BASE_DIR / "output"
    output.mkdir(exist_ok=True)
    pd.DataFrame(
        {"row_id": frame["row_id"], "control_success": predict_dataframe(frame)}
    ).to_csv(output / "submission.csv", index=False)

if __name__ == "__main__":
    main()
'''


def _source_package(path: Path, *, script: str = SCRIPT) -> None:
    hybrid = {
        "trackman_asof_gate": {
            "eta": 0.03,
            "public_score": 1158.0745556751,
        }
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("script.py", script)
        archive.writestr("requirements.txt", "numpy\n")
        archive.writestr("model/hybrid.json", json.dumps(hybrid))
        archive.writestr("model/dummy.txt", "model")


def test_materialized_payload_rebuilds_without_source_package(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    payload = tmp_path / "payload"
    first_output = tmp_path / "standalone-1.zip"
    second_output = tmp_path / "standalone-2.zip"
    _source_package(source)

    manifest = materialize(source, payload)
    assert manifest["historical_parent_package_required_after_materialization"] is False
    assert manifest["model_file_count"] == 2
    source.unlink()

    first = build(payload, first_output)
    second = build(payload, second_output)
    assert first["historical_parent_package_required"] is False
    assert first["sha256"] == second["sha256"]
    assert validate_payload(payload)["file_count"] == 4


def test_materialize_rejects_unsafe_archive(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("../escape.txt", "unsafe")
        archive.writestr("script.py", SCRIPT)
        archive.writestr("requirements.txt", "numpy\n")
        archive.writestr("model/hybrid.json", "{}")

    with pytest.raises(ValueError, match="unsafe ZIP member|bad ZIP roots"):
        materialize(source, tmp_path / "payload")


def test_official_row_independence_gate_passes_row_local_script(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    payload = tmp_path / "payload"
    package = tmp_path / "standalone.zip"
    test_csv = tmp_path / "test.csv"
    _source_package(source)
    materialize(source, payload)
    build(payload, package)
    pd.DataFrame({"row_id": ["a", "b", "c"], "x": [-1.0, 0.0, 2.0]}).to_csv(
        test_csv, index=False
    )

    result = official_row_independence_test(package, test_csv)

    assert result["status"] == "pass"
    assert result["singleton_vs_full_max_abs_diff"] == 0.0
    assert result["shuffled_vs_full_max_abs_diff"] == 0.0


def test_official_row_independence_gate_rejects_batch_dependent_script(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    payload = tmp_path / "payload"
    package = tmp_path / "standalone.zip"
    test_csv = tmp_path / "test.csv"
    batch_dependent = SCRIPT.replace(
        'prediction = 1.0 / (1.0 + np.exp(-frame["x"].to_numpy(dtype=float)))',
        'prediction = np.full(len(frame), frame["x"].mean(), dtype=float)',
    )
    _source_package(source, script=batch_dependent)
    materialize(source, payload)
    build(payload, package)
    pd.DataFrame({"row_id": ["a", "b", "c"], "x": [-1.0, 0.0, 2.0]}).to_csv(
        test_csv, index=False
    )

    with pytest.raises(ValueError, match="row-independence test failed"):
        official_row_independence_test(package, test_csv)


def test_static_gate_rejects_forbidden_batch_operation(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    forbidden = SCRIPT + '\ndef forbidden(frame):\n    return frame.groupby("team")\n'
    _source_package(source, script=forbidden)

    with pytest.raises(ValueError, match="forbidden test-batch operations"):
        materialize(source, tmp_path / "payload")


def test_release_manifest_records_verification(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    payload = tmp_path / "payload"
    _source_package(source)
    materialize(source, payload)
    result = {"official_row_independence": {"status": "pass"}}

    manifest_path = write_release_manifest(payload, result)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["historical_parent_package_required"] is False
    assert manifest["payload"]["row_independence_static_audit"]["status"] == "pass"
    assert manifest["release"]["official_row_independence"]["status"] == "pass"
