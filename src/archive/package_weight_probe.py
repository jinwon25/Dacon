"""Build a one-field legacy-axis probe from an exact verified parent ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from src.package import verify_package


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"unsafe ZIP member: {member.filename}")
    archive.extractall(destination)


def build(
    source: Path,
    output: Path,
    *,
    expected_source_sha256: str,
    effect_weight: float,
) -> dict:
    source = source.resolve()
    output = output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not 0.0 < effect_weight < 1.0:
        raise ValueError("effect weight must be in (0, 1)")
    source_hash = sha256(source)
    if source_hash != expected_source_sha256.upper():
        raise ValueError(
            f"source SHA-256 mismatch: expected {expected_source_sha256.upper()}, "
            f"got {source_hash}"
        )

    with tempfile.TemporaryDirectory(prefix="dacon_weight_probe_") as temp_name:
        stage = Path(temp_name)
        with zipfile.ZipFile(source) as archive:
            safe_extract(archive, stage)
        spec_path = stage / "model" / "legacy_cb_axis_spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        parent_effect_weight = float(spec["effect_weight"])
        spec.update(
            {
                "parent_effect_weight": parent_effect_weight,
                "effect_weight": float(effect_weight),
                "variant_parent": source.name,
                "change_scope": "legacy_cb_axis_effect_weight_only",
            }
        )
        spec_path.write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())

    verify_package(output)
    result = {
        "source": source.name,
        "source_sha256": source_hash,
        "output": output.name,
        "output_sha256": sha256(output),
        "parent_effect_weight": parent_effect_weight,
        "effect_weight": float(effect_weight),
        "change_scope": "legacy_cb_axis_spec.json effect_weight only",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effect-weight", type=float, required=True)
    args = parser.parse_args()
    project = args.project_dir.resolve()
    source = args.source if args.source.is_absolute() else project / args.source
    output = args.output if args.output.is_absolute() else project / args.output
    result = build(
        source,
        output,
        expected_source_sha256=args.expected_source_sha256,
        effect_weight=args.effect_weight,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
