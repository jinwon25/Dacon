"""Package the target-1160 overlay as a lineage-checked v19 child."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from src.package import verify_package


PARENT_SHA256 = "B12070D8017AE6A78BFC056F392BACDA931F8ED7439AA9821880FC986CE962B2"


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
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    artifact_names = sorted(manifest["artifacts"])
    for name in artifact_names:
        path = artifact_dir / name
        if _sha256(path) != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"artifact hash mismatch: {name}")

    overlay_script = project / "src" / "v10_overlay_script.py"
    with tempfile.TemporaryDirectory(prefix="v20_target1160_package_") as temp_name:
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
                "v20_target1160_spec": "v20_target1160_spec.json",
                "v20_local_min_gain_vs_v19": 14.251662840875076,
                "v20_public_center_estimate": 1159.75,
                "v20_current_pitch_trackman_at_inference": False,
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
            "minimum_forward_gain_vs_v19": 14.251662840875076,
            "y2024_early_to_late_gain": 20.652933302219225,
            "public_center_estimate": 1159.75,
        },
    }
    candidate_dir = project / "artifacts" / "candidates" / output.stem
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--parent", type=Path, default=Path("submissions/history/submit_v19.zip")
    )
    parser.add_argument("--output", type=Path, default=Path("submit_v20.zip"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/v20_target1160_final_20260816"),
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
