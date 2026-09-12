"""Package a v14 refinement by making a minimal, verified edit to v13."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from src.package import verify_package


ARTIFACT_NAMES = (
    "v14_anchor_ridge.joblib",
    "v14_f_trend_lgb.txt",
    "v14_refinement_preprocess.joblib",
    "v14_refinement_spec.json",
)


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
        if target != root and root not in target.parents:
            raise ValueError(f"unsafe ZIP member: {member.filename}")
    archive.extractall(destination)


def build(
    project: Path,
    parent: Path,
    output: Path,
    expected_parent_sha256: str,
    f_trend_alpha: float,
) -> dict[str, object]:
    project = project.resolve()
    parent = parent.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not 0.0 <= f_trend_alpha <= 1.0:
        raise ValueError("f_trend_alpha must be in [0, 1]")
    parent_hash = _sha256(parent)
    if parent_hash != expected_parent_sha256.upper():
        raise ValueError(
            f"parent hash differs: expected {expected_parent_sha256}, got {parent_hash}"
        )
    artifact_dir = project / "artifacts" / "v14_refinement_final_20260815"
    missing = [name for name in ARTIFACT_NAMES if not (artifact_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"v14 artifacts missing: {missing}")
    script = project / "src" / "v10_overlay_script.py"

    with tempfile.TemporaryDirectory(prefix="v14_refinement_package_") as temp_name:
        stage = Path(temp_name)
        with zipfile.ZipFile(parent) as archive:
            _safe_extract(archive, stage)
        shutil.copy2(script, stage / "script.py")
        for name in ARTIFACT_NAMES:
            shutil.copy2(artifact_dir / name, stage / "model" / name)
        spec_path = stage / "model" / "v14_refinement_spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["candidate"] = output.stem
        spec["f_trend_alpha"] = float(f_trend_alpha)
        spec["selection_rule"] = (
            "three-transition conservative F trend"
            if f_trend_alpha <= 0.15
            else "post-break 2022->2023 and 2023->2024 F trend"
        ) + "; three-transition high-reliability anchor gate"
        spec_path.write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        hybrid_path = stage / "model" / "hybrid.json"
        hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
        hybrid.update(
            {
                "candidate": output.stem,
                "variant_parent": parent.name,
                "v14_refinement": "v14_refinement_spec.json",
            }
        )
        hybrid_path.write_text(
            json.dumps(hybrid, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())

    verify_package(output)
    result = {
        "candidate": output.stem,
        "parent": parent.name,
        "parent_sha256": parent_hash,
        "output": output.name,
        "sha256": _sha256(output),
        "size_bytes": output.stat().st_size,
        "script_sha256": _sha256(script),
        "f_trend_alpha": f_trend_alpha,
        "refinement_artifacts": {
            name: _sha256(artifact_dir / name) for name in ARTIFACT_NAMES
        },
    }
    manifest_dir = project / "artifacts" / "candidates" / output.stem
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--parent", type=Path, default=Path("submit_v13_fixed.zip"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--f-trend-alpha", type=float, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    parent = args.parent if args.parent.is_absolute() else project / args.parent
    output = args.output if args.output.is_absolute() else project / args.output
    build(
        project,
        parent,
        output,
        args.expected_parent_sha256,
        args.f_trend_alpha,
    )


if __name__ == "__main__":
    main()
