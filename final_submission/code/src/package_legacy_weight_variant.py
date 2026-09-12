"""Create a controlled legacy-CB weight variant from a verified package."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from src.package import verify_package


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def build(source: Path, output: Path, effect_weight: float) -> Path:
    source = source.resolve()
    output = output.resolve()
    if not 0.0 < effect_weight < 1.0:
        raise ValueError("effect weight must be in (0, 1)")
    with tempfile.TemporaryDirectory(prefix="aimers9_weight_variant_") as temp_name:
        stage = Path(temp_name)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(stage)
        spec_path = stage / "model" / "legacy_cb_axis_spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["parent_effect_weight"] = float(spec["effect_weight"])
        spec["effect_weight"] = float(effect_weight)
        spec["variant_parent"] = source.name
        spec_path.write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output.exists():
            output.unlink()
        with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())
    verify_package(output)
    print(
        json.dumps(
            {
                "source": source.name,
                "source_sha256": _sha256(source),
                "output": output.name,
                "output_sha256": _sha256(output),
                "effect_weight": effect_weight,
                "change_scope": "legacy_cb_axis_spec.json effect_weight only",
            },
            indent=2,
        )
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effect-weight", type=float, required=True)
    args = parser.parse_args()
    project = args.project_dir.resolve()
    source = args.source if args.source.is_absolute() else project / args.source
    output = args.output if args.output.is_absolute() else project / args.output
    build(source, output, args.effect_weight)


if __name__ == "__main__":
    main()
