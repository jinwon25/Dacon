"""Create a short-name controlled hybrid submission variant.

The variant reuses the verified ``submit_v2.zip`` artifacts and changes only
the frozen Trackman blend weight. This makes the next Public submission a
controlled probe instead of a new, untracked training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path

from src.package import verify_package


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _extract_safe(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe source ZIP member: {member.filename}")
            target = destination / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))


def _write_zip(package_dir: Path, output: Path) -> None:
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(package_dir.rglob("*")):
            if not path.is_file():
                continue
            name = path.relative_to(package_dir).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 9, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def run(
    project_dir: Path,
    source_name: str = "submit_v2.zip",
    output_name: str = "submit_v3.zip",
    trackman_weight: float = 0.10,
) -> dict[str, object]:
    project_dir = project_dir.resolve()
    source = project_dir / source_name
    output = project_dir / output_name
    if len(output.name) > 40:
        raise ValueError(f"final ZIP filename must be <=40 characters: {output.name}")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not 0.0 < trackman_weight < 1.0:
        raise ValueError("trackman_weight must be between 0 and 1")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    candidate = f"hybrid_recency_r_trackman_w{int(round(trackman_weight * 100)):02d}_v1"
    root = project_dir / "artifacts" / "candidates" / candidate / output.stem
    package_dir = root / "package"
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing artifacts: {root}")
    package_dir.mkdir(parents=True)
    _extract_safe(source, package_dir)

    hybrid_path = package_dir / "model" / "hybrid.json"
    hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
    previous_weight = float(hybrid["trackman_weight"])
    hybrid["candidate"] = candidate
    hybrid["trackman_weight"] = float(trackman_weight)
    hybrid["variant_parent"] = source.name
    hybrid_path.write_text(
        json.dumps(hybrid, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    _write_zip(package_dir, output)
    verify_package(output)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "candidate": candidate,
        "output_name": output.name,
        "output_name_length": len(output.name),
        "package_path": str(output),
        "package_sha256": _sha256(output),
        "source_name": source.name,
        "source_sha256": _sha256(source),
        "previous_trackman_weight": previous_weight,
        "trackman_weight": float(trackman_weight),
        "local_grid": {
            "w05_delta": -0.000046588,
            "w075_delta": -0.000059646,
            "w10_delta": -0.000071654,
        },
        "change_scope": "Only hybrid.json Trackman blend weight changed; all model artifacts are inherited from submit_v2.zip.",
        "selection_note": "Public score 763.2665303697 validated the w05 direction; w10 is a predeclared local-grid probe.",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--source", default="submit_v2.zip")
    parser.add_argument("--output", default="submit_v3.zip")
    parser.add_argument("--trackman-weight", type=float, default=0.10)
    args = parser.parse_args()
    manifest = run(
        args.project_dir,
        source_name=args.source,
        output_name=args.output,
        trackman_weight=args.trackman_weight,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
