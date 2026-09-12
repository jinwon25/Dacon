"""Package the verified v11 parent with the recent exact-ASOF overlay."""

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
    "recent_exact_lgb.txt",
    "recent_exact_preprocess.joblib",
    "recent_exact_ridge.joblib",
    "recent_exact_spec.json",
    "recent_stable_ridge.joblib",
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
) -> dict[str, object]:
    project = project.resolve()
    parent = parent.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    parent_hash = _sha256(parent)
    if parent_hash != expected_parent_sha256.upper():
        raise ValueError(
            f"parent hash differs: expected {expected_parent_sha256}, got {parent_hash}"
        )
    artifact_dir = project / "artifacts" / "recent_exact_overlay_final_20260815"
    missing = [name for name in ARTIFACT_NAMES if not (artifact_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"recent overlay artifacts missing: {missing}")
    script = project / "src" / "v10_overlay_script.py"

    with tempfile.TemporaryDirectory(prefix="recent_exact_package_") as temp_name:
        stage = Path(temp_name)
        with zipfile.ZipFile(parent) as archive:
            _safe_extract(archive, stage)
        shutil.copy2(script, stage / "script.py")
        for name in ARTIFACT_NAMES:
            shutil.copy2(artifact_dir / name, stage / "model" / name)
        hybrid_path = stage / "model" / "hybrid.json"
        hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
        hybrid.update(
            {
                "candidate": output.stem,
                "variant_parent": parent.name,
                "recent_exact_overlay": "recent_exact_spec.json",
                "selection_scope": (
                    "strict 2021->2022, 2022->2023, 2023->2024 temporal transfer; "
                    "v11 F public-confirmed parent"
                ),
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
        "overlay_artifacts": {
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
    parser.add_argument("--parent", type=Path, default=Path("submit_v11.zip"))
    parser.add_argument("--output", type=Path, default=Path("submit_v13.zip"))
    parser.add_argument("--expected-parent-sha256", required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    parent = args.parent if args.parent.is_absolute() else project / args.parent
    output = args.output if args.output.is_absolute() else project / args.output
    build(project, parent, output, args.expected_parent_sha256)


if __name__ == "__main__":
    main()
