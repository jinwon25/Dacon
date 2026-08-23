"""Build the controlled Top-1100 weight and F-specialist code submissions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"unsafe ZIP member: {member.filename}")
    archive.extractall(destination)


def _write_zip(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())


def build_candidate(
    project_dir: Path,
    *,
    parent_zip: Path,
    expected_parent_sha256: str,
    inference_script: Path,
    output: Path,
    mode: str,
    f_weight: float,
    r_weight: float,
    catboost_weight: float,
) -> dict:
    if mode not in {"weight", "catboost"}:
        raise ValueError(f"unsupported mode: {mode}")
    if not parent_zip.exists():
        raise FileNotFoundError(parent_zip)
    parent_sha256 = _sha256(parent_zip)
    if parent_sha256 != expected_parent_sha256.upper():
        raise ValueError(
            "parent SHA-256 mismatch: "
            f"expected {expected_parent_sha256.upper()}, got {parent_sha256}"
        )
    if not inference_script.exists():
        raise FileNotFoundError(inference_script)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate: {output}")

    with tempfile.TemporaryDirectory(prefix="dacon_top1100_") as temp_name:
        package = Path(temp_name) / "package"
        package.mkdir(parents=True)
        with zipfile.ZipFile(parent_zip) as archive:
            _safe_extract(archive, package)

        shutil.copy2(inference_script, package / "script.py")
        hybrid_path = package / "model" / "hybrid.json"
        hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
        hybrid.update(
            {
                "candidate": output.stem,
                "trackman_weight": 0.0,
                "trackman_default_weight": 0.0,
                "trackman_weight_by_game_type": {
                    "F": float(f_weight),
                    "R": float(r_weight),
                },
                "variant_parent": parent_zip.name,
                "selection_scope": "2021-2024 OOF plus controlled public deployment evidence",
            }
        )
        for key in [
            "f_catboost_models",
            "f_catboost_model_weights",
            "f_catboost_feature_spec",
            "f_catboost_game_type",
            "f_catboost_weight",
            "f_catboost_calibration",
        ]:
            hybrid.pop(key, None)

        if mode == "catboost":
            final_dir = project_dir / "artifacts" / "f_regime" / "final"
            manifest = json.loads(
                (final_dir / "manifest.json").read_text(encoding="utf-8")
            )
            model_names = list(manifest["model_names"])
            for name in model_names:
                shutil.copy2(final_dir / name, package / "model" / name)
            feature_spec_name = "f_catboost_feature_spec.json"
            shutil.copy2(
                final_dir / feature_spec_name,
                package / "model" / feature_spec_name,
            )
            hybrid.update(
                {
                    "f_catboost_models": model_names,
                    "f_catboost_model_weights": [1.0 / len(model_names)]
                    * len(model_names),
                    "f_catboost_feature_spec": feature_spec_name,
                    "f_catboost_game_type": "F",
                    "f_catboost_weight": float(catboost_weight),
                    "f_catboost_calibration": {"method": "identity"},
                }
            )
            requirements_path = package / "requirements.txt"
            requirements = requirements_path.read_text(encoding="utf-8").splitlines()
            if not any(line.strip().lower().startswith("catboost==") for line in requirements):
                requirements.append("catboost==1.2.8")
            requirements_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")

        hybrid_path.write_text(
            json.dumps(hybrid, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_zip(package, output)

    manifest = {
        "candidate": hybrid["candidate"],
        "mode": mode,
        "parent_zip": parent_zip.name,
        "parent_sha256": parent_sha256,
        "inference_script": str(inference_script.relative_to(project_dir)),
        "inference_script_sha256": _sha256(inference_script),
        "output": output.name,
        "sha256": _sha256(output),
        "size_bytes": output.stat().st_size,
        "trackman_weight_by_game_type": hybrid["trackman_weight_by_game_type"],
        "f_catboost_weight": hybrid.get("f_catboost_weight"),
    }
    manifest_dir = project_dir / "artifacts" / "candidates" / hybrid["candidate"]
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--parent", type=Path, default=Path("submit_v10_1015.zip"))
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument(
        "--inference-script", type=Path, default=Path("src/archive/v10_overlay_script.py")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["weight", "catboost"], required=True)
    parser.add_argument("--f-weight", type=float, default=1.5)
    parser.add_argument("--r-weight", type=float, default=0.05)
    parser.add_argument("--catboost-weight", type=float, default=0.75)
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    parent = args.parent if args.parent.is_absolute() else project_dir / args.parent
    inference_script = (
        args.inference_script
        if args.inference_script.is_absolute()
        else project_dir / args.inference_script
    )
    output = args.output if args.output.is_absolute() else project_dir / args.output
    manifest = build_candidate(
        project_dir,
        parent_zip=parent,
        expected_parent_sha256=args.expected_parent_sha256,
        inference_script=inference_script,
        output=output,
        mode=args.mode,
        f_weight=args.f_weight,
        r_weight=args.r_weight,
        catboost_weight=args.catboost_weight,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
