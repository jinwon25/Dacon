"""Build v353 with the frozen TrackMan-PFD recipe refit on all 2024 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import zipfile

from src.champion.v352_build_trackman_pfd_package import (
    MODEL_NAMES,
    MODEL_ROOT,
    model_zip_info,
    normalised_model_bytes,
    patch_script,
    sha256,
)


PROTOCOL = "V353_BUILD_REFIT_TRACKMAN_PFD_PACKAGE_V1"


def run(
    source_zip: Path,
    model_dir: Path,
    refit_summary: Path,
    validation_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    refit = json.loads(refit_summary.read_text(encoding="utf-8"))
    validation = json.loads(validation_summary.read_text(encoding="utf-8"))
    if refit.get("protocol") != "V353_REFIT_TRACKMAN_PFD_STUDENTS_V1":
        raise ValueError("unexpected v353 refit protocol")
    if refit.get("status") != "refit_complete":
        raise ValueError("v353 student refit is incomplete")
    if validation.get("protocol") != "V352_TRACKMAN_PFD_REBASE_V345_V1":
        raise ValueError("unexpected TrackMan-PFD validation protocol")
    if not validation.get("candidate_gate_passed"):
        raise ValueError("frozen TrackMan-PFD recipe did not pass validation")

    model_paths = [model_dir / name for name in MODEL_NAMES]
    missing = [str(path) for path in model_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing refit TrackMan-PFD models: {missing}")
    expected_hash = {
        row["path"].replace("\\", "/").split("/")[-1]: row["sha256"]
        for row in refit["models"]
    }
    for path in model_paths:
        if sha256(path) != expected_hash.get(path.name):
            raise ValueError(f"refit model hash mismatch: {path.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_zip = output_dir / "submit_v353.zip"
    with zipfile.ZipFile(source_zip, "r") as source:
        names = source.namelist()
        if names.count("script.py") != 1:
            raise ValueError("source ZIP must contain one root script.py")
        collisions = [
            f"{MODEL_ROOT}/{name}"
            for name in MODEL_NAMES
            if f"{MODEL_ROOT}/{name}" in names
        ]
        if collisions:
            raise ValueError(f"source ZIP already contains v353 models: {collisions}")
        patched = patch_script(source.read("script.py").decode("utf-8"))
        with zipfile.ZipFile(
            output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as target:
            for item in source.infolist():
                payload = (
                    patched.encode("utf-8")
                    if item.filename == "script.py"
                    else source.read(item.filename)
                )
                target.writestr(item, payload)
            for name, path in zip(MODEL_NAMES, model_paths):
                target.writestr(model_zip_info(name), normalised_model_bytes(path))

    with zipfile.ZipFile(output_zip, "r") as built:
        bad_crc = built.testzip()
        members = built.namelist()
    summary = {
        "protocol": PROTOCOL,
        "status": "built_pending_runtime_audit",
        "output_zip": str(output_zip),
        "output_sha256": sha256(output_zip),
        "output_bytes": output_zip.stat().st_size,
        "member_count": len(members),
        "crc_passed": bad_crc is None,
        "source_zip": str(source_zip),
        "source_zip_sha256": sha256(source_zip),
        "refit_protocol": refit["protocol"],
        "refit_fit": refit["fit"],
        "models": [
            {
                "name": path.name,
                "sha256": sha256(path),
                "normalised_bytes": len(normalised_model_bytes(path)),
            }
            for path in model_paths
        ],
        "frozen_recipe_validation_vs_v345": validation["metrics_vs_v345"],
        "recipe": refit["recipe"],
        "restrictions": {
            **refit["restrictions"],
            "runtime_audit_pending": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--refit-summary", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.source_zip,
                args.model_dir,
                args.refit_summary,
                args.validation_summary,
                args.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
