"""Conditionally package submit_v6.zip from the immutable v2 parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


PARENT_SHA = "FE368AF109EF0BB8103D728F45794A6192599B691BF08DF02A22F31BD2D438A7"


def package(project: Path, candidate: str) -> Path | None:
    parent = project / "submit_v2.zip"
    gate = project / "artifacts/candidates" / candidate / "manifest.json"
    if not parent.exists() or hashlib.sha256(parent.read_bytes()).hexdigest().upper() != PARENT_SHA:
        raise RuntimeError("immutable submit_v2.zip SHA check failed")
    if not gate.exists():
        raise FileNotFoundError(gate)
    manifest = json.loads(gate.read_text(encoding="utf-8"))
    if manifest.get("status") != "promoted":
        print("No candidate passed the fixed gate; submit_v6.zip was not created.")
        return None
    raise RuntimeError("promotion manifest is missing an implementation-specific residual artifact; refuse unsafe package")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, default=Path(".")); parser.add_argument("--candidate", default="main_residual_nested_v1"); args = parser.parse_args(); package(args.project_dir.resolve(), args.candidate)


if __name__ == "__main__":
    main()
