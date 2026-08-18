"""Package the low-degree pitcher-hand R_ANCHOR route as a child of v27."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from src.package import verify_package
from src.package_v25_postbreak_anchor import _safe_extract


PARENT_SHA256 = "CEFC91F4F8EBBA32EE8025816319F32C95023CB59EA70741176A98FB7A172385"
DEFAULT_ETA = 0.10
MATCH_ETA = 0.125
MATCH_COLUMN = "pitcher_hand"
MATCH_VALUE = "1"

OLD_BLOCK = """    eta = float(spec[\"blend_eta\"])
    result[apply_mask] = np.clip(
        parent[apply_mask] + eta * (direct - parent[apply_mask]), low, high
    )
"""

NEW_BLOCK = """    route = spec.get(\"blend_eta_route\")
    eta = np.full(len(direct), float(spec[\"blend_eta\"]), dtype=np.float64)
    if route is not None:
        column = str(route[\"column\"])
        if column not in selected.columns:
            raise ValueError(f\"blend eta route column missing: {column}\")
        values = selected[column].astype(\"string\").fillna(\"__MISSING__\")
        matched = values.eq(str(route[\"match_value\"])).to_numpy()
        eta[matched] = float(route[\"match_eta\"])
    result[apply_mask] = np.clip(
        parent[apply_mask] + eta * (direct - parent[apply_mask]), low, high
    )
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _patch_script(text: str) -> str:
    if text.count(OLD_BLOCK) != 1:
        raise ValueError("expected exactly one frozen v27 eta block")
    return text.replace(OLD_BLOCK, NEW_BLOCK)


def build(
    project: Path,
    parent: Path,
    output: Path,
    expected_parent_sha256: str = PARENT_SHA256,
) -> dict[str, object]:
    project = project.resolve()
    parent = parent.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if _sha256(parent) != expected_parent_sha256.upper():
        raise ValueError("unexpected v27 parent SHA-256")
    verify_package(parent)

    with tempfile.TemporaryDirectory(prefix="v28_anchor_hand_route_") as name:
        stage = Path(name)
        with zipfile.ZipFile(parent) as archive:
            _safe_extract(archive, stage)
        script_path = stage / "script.py"
        script_path.write_text(
            _patch_script(script_path.read_text(encoding="utf-8")), encoding="utf-8"
        )
        spec_path = stage / "model" / "v25_postbreak_anchor_spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if float(spec["blend_eta"]) != DEFAULT_ETA:
            raise ValueError("v27 default eta is not the frozen 0.10")
        spec.update(
            {
                "protocol": "V28_LOW_DOF_ANCHOR_HAND_ROUTE_2025_V1",
                "parent": parent.name,
                "parent_sha256": _sha256(parent),
                "blend_eta_route": {
                    "column": MATCH_COLUMN,
                    "match_value": MATCH_VALUE,
                    "match_eta": MATCH_ETA,
                    "default_eta": DEFAULT_ETA,
                },
                "route_evidence": "artifacts/v29_anchor_route_20260817_01/summary.json",
                "selection_note": (
                    "single row-local pitcher-hand split; all three temporal axes positive; "
                    "no test-row aggregation"
                ),
            }
        )
        spec_path.write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        hybrid_path = stage / "model" / "hybrid.json"
        hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
        hybrid.update(
            {
                "candidate": output.stem,
                "variant_parent": parent.name,
                "v28_anchor_hand_route": True,
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
        "parent_sha256": _sha256(parent),
        "route": {
            "column": MATCH_COLUMN,
            "match_value": MATCH_VALUE,
            "default_eta": DEFAULT_ETA,
            "match_eta": MATCH_ETA,
        },
        "sha256": _sha256(output),
        "size_bytes": output.stat().st_size,
    }
    manifest_dir = project / "artifacts" / "candidates" / output.stem
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--parent", type=Path, default=Path("submit_v27.zip"))
    parser.add_argument("--output", type=Path, default=Path("submit_v28.zip"))
    parser.add_argument("--expected-parent-sha256", default=PARENT_SHA256)
    args = parser.parse_args()
    project = args.project.resolve()
    parent = args.parent if args.parent.is_absolute() else project / args.parent
    output = args.output if args.output.is_absolute() else project / args.output
    build(project, parent, output, args.expected_parent_sha256)


if __name__ == "__main__":
    main()
