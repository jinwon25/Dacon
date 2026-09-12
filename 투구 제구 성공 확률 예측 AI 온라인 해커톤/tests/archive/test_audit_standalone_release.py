from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from src.archive.audit_standalone_release import audit_release, scale_proxy


SCRIPT = '''from pathlib import Path
import pandas as pd


def predict_dataframe(frame):
    return frame["feature"].astype(float).clip(0.0, 1.0).to_numpy()


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    test = pd.read_csv(root / "data" / "test.csv")
    output = test[["row_id"]].copy()
    output["control_success"] = predict_dataframe(test)
    (root / "output").mkdir(exist_ok=True)
    output.to_csv(root / "output" / "submission.csv", index=False)
'''


def _package(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("script.py", SCRIPT)
        archive.writestr("requirements.txt", "pandas\nnumpy\n")
        archive.writestr("model/dummy.txt", "standalone")


def test_scale_proxy_preserves_schema_and_unique_ids() -> None:
    source = pd.DataFrame(
        {
            "row_id": ["a", "b"],
            "game_type": ["R", "R"],
            "feature": [0.2, 0.8],
        }
    )
    proxy = scale_proxy(source, 5)
    assert list(proxy.columns) == list(source.columns)
    assert len(proxy) == 5
    assert proxy["row_id"].is_unique
    assert proxy.loc[1, "game_type"] == "F"


def test_audit_release_runs_archive_in_isolation(tmp_path: Path) -> None:
    package = tmp_path / "standalone.zip"
    _package(package)
    test_csv = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "row_id": ["a", "b", "c"],
            "game_type": ["R", "F", "R"],
            "feature": [0.2, 0.8, 0.4],
        }
    ).to_csv(test_csv, index=False)

    result = audit_release(package, test_csv, scale_rows=11, timeout_seconds=30)

    assert result["roots"] == ["model", "requirements.txt", "script.py"]
    assert result["crc_ok"] is True
    assert result["file_count"] == 3
    assert result["sample_run"]["rows"] == 3
    assert result["scale_run"]["rows"] == 11
    assert result["dynamic_row_independence"]["status"] == "pass"
