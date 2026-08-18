"""Package v16 as a minimal, lineage-checked edit of submit_v14.zip."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from src.package import verify_package


PARENT_SHA256 = "AC8E135FBA8DBF41E331A8F4E2AAA658F0AF6675DD5F3B6C89EF51E9BC02977E"


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
    artifact_dir_name: str,
    expected_parent_sha256: str = PARENT_SHA256,
) -> dict[str, object]:
    project = project.resolve()
    parent = parent.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    parent_hash = _sha256(parent)
    if parent_hash != expected_parent_sha256.upper():
        raise ValueError(
            f"unexpected parent SHA-256: expected {expected_parent_sha256}, got {parent_hash}"
        )
    artifact = (
        project
        / "artifacts"
        / artifact_dir_name
        / "v16_residual_spec.json"
    )
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    script = project / "src" / "v10_overlay_script.py"
    with tempfile.TemporaryDirectory(prefix="v16_residual_package_") as temp_name:
        stage = Path(temp_name)
        with zipfile.ZipFile(parent) as archive:
            _safe_extract(archive, stage)
        shutil.copy2(script, stage / "script.py")
        shutil.copy2(artifact, stage / "model" / artifact.name)
        hybrid_path = stage / "model" / "hybrid.json"
        hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
        hybrid.update(
            {
                "candidate": output.stem,
                "variant_parent": parent.name,
                "v16_residual": artifact.name,
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
        "residual_spec_sha256": _sha256(artifact),
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
    parser.add_argument("--parent", type=Path, default=Path("submit_v14.zip"))
    parser.add_argument("--output", type=Path, default=Path("submit_v16.zip"))
    parser.add_argument(
        "--artifact-dir", default="v16_residual_final_20260815"
    )
    parser.add_argument(
        "--expected-parent-sha256", default=PARENT_SHA256
    )
    args = parser.parse_args()
    project = args.project.resolve()
    parent = args.parent if args.parent.is_absolute() else project / args.parent
    output = args.output if args.output.is_absolute() else project / args.output
    build(
        project,
        parent,
        output,
        args.artifact_dir,
        args.expected_parent_sha256,
    )


if __name__ == "__main__":
    main()
