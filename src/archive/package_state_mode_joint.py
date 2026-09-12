"""Package the joint state/mode overlay as a lineage-checked v17 child."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from src.package import verify_package


PARENT_SHA256 = "A327740E5AB995F48832B0D514EC7330F5B73EDDCD81407E8BE3326385CCD595"


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
    artifact_dir: Path,
    expected_parent_sha256: str,
) -> dict[str, object]:
    project = project.resolve()
    parent = parent.resolve()
    output = output.resolve()
    artifact_dir = (project / artifact_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    parent_hash = _sha256(parent)
    if parent_hash != expected_parent_sha256.upper():
        raise ValueError(
            f"unexpected parent SHA-256: expected {expected_parent_sha256}, got {parent_hash}"
        )
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_names = sorted(manifest["artifacts"])
    for name in artifact_names:
        path = artifact_dir / name
        if _sha256(path) != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"artifact hash mismatch: {name}")

    overlay_script = project / "src" / "v10_overlay_script.py"
    with tempfile.TemporaryDirectory(prefix="joint_state_mode_package_") as temp_name:
        stage = Path(temp_name)
        with zipfile.ZipFile(parent) as archive:
            _safe_extract(archive, stage)
        shutil.copy2(overlay_script, stage / "script.py")
        for name in artifact_names:
            shutil.copy2(artifact_dir / name, stage / "model" / name)
        hybrid_path = stage / "model" / "hybrid.json"
        hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
        hybrid.update(
            {
                "candidate": output.stem,
                "variant_parent": parent.name,
                "joint_state_mode_spec": "joint_state_mode_spec.json",
                "joint_local_latest_gain": 46.39598090147024,
                "joint_row_local_inference": True,
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
        "script_sha256": _sha256(overlay_script),
        "artifact_count": len(artifact_names),
        "local_evidence": {
            "latest_gain_vs_v17": 46.39598090147024,
            "minimum_cluster_bootstrap_p05": 27.110003370133857,
            "positive_month_fraction": 1.0,
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
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("submissions/history/submit_v17.zip"),
    )
    parser.add_argument("--output", type=Path, default=Path("submit_v19.zip"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/state_mode_joint_final_20260816"),
    )
    parser.add_argument("--expected-parent-sha256", default=PARENT_SHA256)
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
