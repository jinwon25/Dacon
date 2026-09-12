"""Build v290 plus the frozen 10% logit-space 2024-R ABS expert."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any


PROTOCOL = "V296_BUILD_ABS_REGIME_PACKAGE_V1"
EXPECTED_V290_SHA256 = "09DE96351304E80BC23B4A9B61739B901E6FD1A4C8BD18E131A352312BA0A4B6"
WEIGHT = 0.10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _patch_script(text: str) -> str:
    text = text.replace("\r\n", "\n")
    marker = "def _window_adjustment("
    helper = '''def _predict_abs_regular(frame: pd.DataFrame) -> np.ndarray:
    module = _load_module(
        "abs_regular_expert", MODEL_DIR / "abs_regular" / "runtime.py"
    )
    return module.predict(frame, MODEL_DIR / "abs_regular")


'''
    if marker not in text:
        raise RuntimeError("v290 helper marker missing")
    if "def _predict_abs_regular(" not in text:
        text = text.replace(marker, helper + marker, 1)

    return_marker = "    return parent, h1, c3, active, output\n"
    replacement = '''    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    if np.any(regular):
        regular_probability = _predict_abs_regular(frame)
        parent_probability = np.clip(output[regular], 1e-6, 1.0 - 1e-6)
        expert_probability = np.clip(
            regular_probability[regular], 1e-6, 1.0 - 1e-6
        )
        parent_logit = np.log(parent_probability / (1.0 - parent_probability))
        expert_logit = np.log(expert_probability / (1.0 - expert_probability))
        blended_logit = parent_logit + 0.10 * (expert_logit - parent_logit)
        output[regular] = np.clip(
            1.0 / (1.0 + np.exp(-np.clip(blended_logit, -35.0, 35.0))),
            0.001,
            0.999,
        )
    return parent, h1, c3, active, output
'''
    if text.count(return_marker) != 1:
        raise RuntimeError("unexpected v290 return marker count")
    return text.replace(return_marker, replacement, 1)


def run(
    source_package: Path,
    finalized_model_dir: Path,
    runtime_source: Path,
    v294_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_sha = _sha256(source_package)
    if source_sha != EXPECTED_V290_SHA256:
        raise ValueError(f"unexpected v290 package SHA-256: {source_sha}")
    final_summary = json.loads(
        (finalized_model_dir / "summary.json").read_text(encoding="utf-8")
    )
    if final_summary.get("protocol") != "V295_FINALIZE_ABS_REGULAR_EXPERT_V1":
        raise ValueError("unexpected v295 protocol")
    if final_summary.get("status") != "final_model_ready":
        raise ValueError("v295 final model is not ready")
    if float(final_summary["weight"]) != WEIGHT or final_summary["blend_mode"] != "logit":
        raise ValueError("v295 blend contract changed")
    sanity = json.loads(v294_summary.read_text(encoding="utf-8"))
    if sanity.get("status") != "sanity_reproduced":
        raise ValueError("v294 sanity is not eligible")

    output_zip = output_dir / "submit_v296_abs_regular_regime.zip"
    with tempfile.TemporaryDirectory(prefix="v296_abs_regular_") as temp_dir:
        stage = Path(temp_dir)
        with zipfile.ZipFile(source_package) as archive:
            archive.extractall(stage)
        script = stage / "script.py"
        script.write_text(
            _patch_script(script.read_text(encoding="utf-8")), encoding="utf-8"
        )
        asset = stage / "model" / "abs_regular"
        asset.mkdir(parents=True, exist_ok=True)
        for source in finalized_model_dir.iterdir():
            if source.is_file() and source.name != "summary.json":
                shutil.copy2(source, asset / source.name)
        shutil.copy2(runtime_source, asset / "runtime.py")
        metadata = {
            "protocol": PROTOCOL,
            "source_v290_sha256": source_sha,
            "v295_summary_sha256": _sha256(finalized_model_dir / "summary.json"),
            "v294_summary_sha256": _sha256(v294_summary),
            "active_game_type": "R",
            "weight": WEIGHT,
            "blend_mode": "logit",
            "fit_scope": "official train season=2024, game_type=R only",
            "futures_abs_start_corrected_to_2020": True,
            "honest_local_r2025_validation_available": False,
            "test_row_aggregation": False,
            "high_risk_exploratory_challenger": True,
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
        "source_v290_sha256": source_sha,
        "active_game_type": "R",
        "weight": WEIGHT,
        "blend_mode": "logit",
        "high_risk_exploratory_challenger": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--finalized-model-dir", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--v294-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.source_package,
                args.finalized_model_dir,
                args.runtime_source,
                args.v294_summary,
                args.output_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
