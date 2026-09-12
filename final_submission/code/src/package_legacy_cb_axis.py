"""Layer the low-weight legacy CatBoost axis onto submit_v6."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from src.finalize_legacy_cb_axis import (
    BATTER_PRIOR_NAME,
    MODEL_NAME,
    PITCHER_PRIOR_NAME,
    SPEC_NAME,
)
from src.package import verify_package


AXIS_FILES = (MODEL_NAME, SPEC_NAME, PITCHER_PRIOR_NAME, BATTER_PRIOR_NAME)


def build(project: Path, base_zip: Path, axis_dir: Path, output: Path) -> Path:
    project = project.resolve()
    base_zip = base_zip.resolve()
    axis_dir = axis_dir.resolve()
    output = output.resolve()
    missing = [name for name in AXIS_FILES if not (axis_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"legacy CB artifacts missing: {missing}")
    with tempfile.TemporaryDirectory(prefix="aimers9_legacy_cb_") as temp_name:
        stage = Path(temp_name)
        with zipfile.ZipFile(base_zip) as archive:
            archive.extractall(stage)
        shutil.copy2(project / "script.py", stage / "script.py")
        shutil.copy2(project / "requirements.txt", stage / "requirements.txt")
        for name in AXIS_FILES:
            shutil.copy2(axis_dir / name, stage / "model" / name)
        if output.exists():
            output.unlink()
        with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())
    verify_package(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--base-zip", type=Path, default=Path("submit_v6.zip"))
    parser.add_argument("--axis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submit_v7.zip"))
    args = parser.parse_args()
    project = args.project_dir.resolve()
    base_zip = args.base_zip if args.base_zip.is_absolute() else project / args.base_zip
    axis_dir = args.axis_dir if args.axis_dir.is_absolute() else project / args.axis_dir
    output = args.output if args.output.is_absolute() else project / args.output
    build(project, base_zip, axis_dir, output)


if __name__ == "__main__":
    main()
