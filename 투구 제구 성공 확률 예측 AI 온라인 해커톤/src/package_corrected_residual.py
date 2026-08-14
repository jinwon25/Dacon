"""Layer the production corrected-state residual onto an immutable base ZIP."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from src.finalize_corrected_residual import (
    BATTER_PRIOR_NAME,
    MODEL_NAME,
    PITCHER_PRIOR_NAME,
    SPEC_NAME,
)
from src.package import verify_package


RESIDUAL_FILES = (MODEL_NAME, SPEC_NAME, PITCHER_PRIOR_NAME, BATTER_PRIOR_NAME)


def build(
    project_dir: Path,
    base_zip: Path,
    residual_dir: Path,
    output: Path,
) -> Path:
    project_dir = project_dir.resolve()
    base_zip = base_zip.resolve()
    residual_dir = residual_dir.resolve()
    output = output.resolve()
    missing = [name for name in RESIDUAL_FILES if not (residual_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"residual artifacts missing: {missing}")

    with tempfile.TemporaryDirectory(prefix="aimers9_residual_") as temp_name:
        stage = Path(temp_name)
        with zipfile.ZipFile(base_zip) as archive:
            archive.extractall(stage)
        shutil.copy2(project_dir / "script.py", stage / "script.py")
        for name in RESIDUAL_FILES:
            shutil.copy2(residual_dir / name, stage / "model" / name)
        output.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--base-zip", type=Path, default=Path("submit_v2.zip"))
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submit_v6.zip"))
    args = parser.parse_args()
    project = args.project_dir.resolve()
    base_zip = args.base_zip if args.base_zip.is_absolute() else project / args.base_zip
    residual_dir = (
        args.residual_dir
        if args.residual_dir.is_absolute()
        else project / args.residual_dir
    )
    output = args.output if args.output.is_absolute() else project / args.output
    build(project, base_zip, residual_dir, output)


if __name__ == "__main__":
    main()
