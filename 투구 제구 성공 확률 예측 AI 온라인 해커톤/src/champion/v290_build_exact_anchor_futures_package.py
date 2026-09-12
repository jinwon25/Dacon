"""Package exact-anchor v244 plus the fixed 10% recent-F expert."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any


PROTOCOL = "V290_BUILD_EXACT_ANCHOR_FUTURES_PACKAGE_V1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _patch_script(text: str) -> str:
    text = text.replace("\r\n", "\n")
    marker = "def _window_adjustment("
    helper = '''def _predict_recent_futures(frame: pd.DataFrame) -> np.ndarray:
    module = _load_module(
        "recent_futures_expert", MODEL_DIR / "recent_futures" / "runtime.py"
    )
    return module.predict(frame, MODEL_DIR / "recent_futures")


'''
    if marker not in text:
        raise RuntimeError("champion helper marker missing")
    if "def _predict_recent_futures(" not in text:
        text = text.replace(marker, helper + marker, 1)

    return_marker = "    return parent, h1, c3, active, output\n"
    replacement = '''    futures = frame["game_type"].astype(str).eq("F").to_numpy()
    if np.any(futures):
        futures_probability = _predict_recent_futures(frame)
        output[futures] = np.clip(
            output[futures]
            + 0.10 * (futures_probability[futures] - output[futures]),
            0.001,
            0.999,
        )
    return parent, h1, c3, active, output
'''
    if text.count(return_marker) != 1:
        raise RuntimeError("unexpected champion return marker count")
    return text.replace(return_marker, replacement, 1)


def run(
    source_package: Path,
    finalized_model_dir: Path,
    runtime_source: Path,
    v288_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_summary = json.loads(
        (finalized_model_dir / "summary.json").read_text(encoding="utf-8")
    )
    if final_summary.get("status") != "final_model_ready":
        raise ValueError("v289 final model is not ready")
    audit = json.loads(v288_summary.read_text(encoding="utf-8"))
    if audit.get("protocol") != "V288_FUTURES_MULTISEED_FIXED_AUDIT_V1":
        raise ValueError("unexpected v288 summary protocol")
    if not audit.get("eligible_for_exploratory_packaging"):
        raise ValueError("v288 is not eligible for exploratory packaging")
    if int(audit["support_disjoint"]["overlap"]) != 0:
        raise ValueError("regular and futures supports overlap")

    output_zip = output_dir / "submit_v290_exact_anchor_futures.zip"
    with tempfile.TemporaryDirectory(prefix="v290_exact_futures_") as temp_dir:
        stage = Path(temp_dir)
        with zipfile.ZipFile(source_package) as archive:
            archive.extractall(stage)
        script = stage / "script.py"
        script.write_text(
            _patch_script(script.read_text(encoding="utf-8")), encoding="utf-8"
        )
        asset = stage / "model" / "recent_futures"
        asset.mkdir(parents=True, exist_ok=True)
        for source in finalized_model_dir.iterdir():
            if source.is_file() and source.name != "summary.json":
                shutil.copy2(source, asset / source.name)
        shutil.copy2(runtime_source, asset / "runtime.py")
        metadata = {
            "protocol": PROTOCOL,
            "source_v286_sha256": _sha256(source_package),
            "v289_summary_sha256": _sha256(finalized_model_dir / "summary.json"),
            "v288_summary_sha256": _sha256(v288_summary),
            "futures_weight": 0.10,
            "exact_anchor_and_futures_support_disjoint": True,
            "test_row_aggregation": False,
            "exploratory_challenger": True,
        }
        (asset / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in stage.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())

    with zipfile.ZipFile(output_zip) as archive:
        bad_member = archive.testzip()
        entries = archive.namelist()
    if bad_member is not None:
        raise RuntimeError(f"zip CRC failure: {bad_member}")
    result = {
        "protocol": PROTOCOL,
        "status": "package_built_pending_standalone_audit",
        "package": str(output_zip),
        "bytes": output_zip.stat().st_size,
        "sha256": _sha256(output_zip),
        "file_count": len(entries),
        "crc_passed": True,
        "source_v286_sha256": _sha256(source_package),
        "futures_weight": 0.10,
        "local_full_2024_gain_vs_v244": audit["metrics"]["full_2024"]["gain"],
        "exploratory_challenger": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--finalized-model-dir", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--v288-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.source_package,
        args.finalized_model_dir,
        args.runtime_source,
        args.v288_summary,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
