"""Layer the promoted advanced-domain residual onto the immutable V9 ZIP."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from src.finalize_advanced_domain_residual import MODEL_NAME, SPEC_NAME
from src.package import verify_package


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
    missing = [name for name in (MODEL_NAME, SPEC_NAME) if not (residual_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"advanced residual artifacts missing: {missing}")

    with tempfile.TemporaryDirectory(prefix="aimers9_advanced_") as temp_name:
        stage = Path(temp_name)
        with zipfile.ZipFile(base_zip) as archive:
            archive.extractall(stage)
        shutil.copy2(project_dir / "script.py", stage / "script.py")
        for name in (MODEL_NAME, SPEC_NAME):
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
    parser.add_argument("--base-zip", type=Path, default=Path("submit_v9.zip"))
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submit_v10.zip"))
    arguments = parser.parse_args()
    project = arguments.project_dir.resolve()
    build(
        project,
        arguments.base_zip if arguments.base_zip.is_absolute() else project / arguments.base_zip,
        arguments.residual_dir
        if arguments.residual_dir.is_absolute()
        else project / arguments.residual_dir,
        arguments.output if arguments.output.is_absolute() else project / arguments.output,
    )


if __name__ == "__main__":
    main()
