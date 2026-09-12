"""Package a game-type Trackman blend variant without mutating the incumbent ZIP."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.package import verify_package
from src.package_hybrid_variant import _extract_safe, _sha256, _write_zip


CANDIDATE = "game_type_trackman_f100_r10_v1"


def run(project_dir: Path, source_name: str, output_name: str) -> dict[str, object]:
    project_dir = project_dir.resolve()
    source = project_dir / source_name
    output = project_dir / output_name
    if len(output.name) > 40:
        raise ValueError(f"final ZIP filename must be <=40 characters: {output.name}")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    root = project_dir / "artifacts" / "candidates" / CANDIDATE / output.stem
    package_dir = root / "package"
    if root.exists():
        raise FileExistsError(f"candidate artifact already exists: {root}")
    package_dir.mkdir(parents=True)
    _extract_safe(source, package_dir)
    # The source ZIP predates the optional row-local game-type weight resolver.
    # Copy the current inference entry point into the candidate package while
    # leaving the source ZIP immutable.
    shutil.copy2(project_dir / "script.py", package_dir / "script.py")

    hybrid_path = package_dir / "model" / "hybrid.json"
    hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
    hybrid.update(
        {
            "candidate": CANDIDATE,
            "trackman_weight": 0.0,
            "trackman_default_weight": 0.0,
            "trackman_weight_by_game_type": {"F": 1.0, "R": 0.10},
            "variant_parent": source.name,
            "selection_scope": "cached 2021-2024 walk-forward OOF",
        }
    )
    hybrid_path.write_text(json.dumps(hybrid, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _write_zip(package_dir, output)
    verify_package(output)
    manifest = {
        "candidate": CANDIDATE,
        "output_name": output.name,
        "package_path": str(output),
        "package_sha256": _sha256(output),
        "source_name": source.name,
        "source_sha256": _sha256(source),
        "trackman_default_weight": 0.0,
        "trackman_weight_by_game_type": {"F": 1.0, "R": 0.10},
        "local_validation_report": "reports/game_type_trackman_candidate.md",
        "public_score": None,
        "selection_note": "Local OOF candidate only; incumbent public score remains the deployment reference.",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--source", default="submit_v2.zip")
    parser.add_argument("--output", default="submit_game_type.zip")
    args = parser.parse_args()
    run(args.project_dir, args.source, args.output)


if __name__ == "__main__":
    main()
