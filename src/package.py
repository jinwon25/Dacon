"""Build and verify submit.zip with the exact required top-level layout."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

REQUIRED_ROOT = {"model", "script.py", "requirements.txt"}
BASE_MODEL_FILES = ["lgb_model.txt", "feature_spec.json", "ensemble.json", "metadata.json"]


def build_package(project_dir: Path, output: Path) -> None:
    model_dir = project_dir / "model"
    ensemble = json.loads((model_dir / "ensemble.json").read_text(encoding="utf-8"))
    model_files = list(BASE_MODEL_FILES)
    if ensemble["feature_set"] == "trackman":
        model_files.append("trackman_context.csv")
    if float(ensemble["weights"]["random_forest"]) > 0.0:
        model_files.append("rf_model.joblib")
    missing = [name for name in model_files if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"model artifacts missing: {missing}")

    with tempfile.TemporaryDirectory(prefix="aimers9_submit_") as temp_name:
        stage = Path(temp_name)
        (stage / "model").mkdir()
        shutil.copy2(project_dir / "script.py", stage / "script.py")
        shutil.copy2(project_dir / "requirements.txt", stage / "requirements.txt")
        for name in model_files:
            shutil.copy2(model_dir / name, stage / "model" / name)
        if output.exists():
            output.unlink()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())

    verify_package(output)


def verify_package(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        roots = {name.split("/", 1)[0] for name in names}
        if roots != REQUIRED_ROOT:
            raise ValueError(f"bad ZIP roots: {roots}, expected {REQUIRED_ROOT}")
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("unsafe ZIP member path")
        bad = archive.testzip()
        if bad:
            raise ValueError(f"corrupt ZIP member: {bad}")
    print(f"Verified {path} ({path.stat().st_size / (1024**2):.3f} MB), roots={sorted(roots)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("submit.zip"))
    args = parser.parse_args()
    project = args.project_dir.resolve()
    output = args.output if args.output.is_absolute() else project / args.output
    build_package(project, output)


if __name__ == "__main__":
    main()
