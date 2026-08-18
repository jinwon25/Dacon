"""Create the official-FAQ-permitted 15% R_ANCHOR weight probe from v25."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from src.package import verify_package
from src.package_v25_postbreak_anchor import _safe_extract


PARENT_SHA256 = "60CF13B28AF01B24A49E2B03E13161F2E04A1C09647D8DA2CFD95407ED9335B1"
SOURCE_ETA = 0.075
PROBE_ETA = 0.150


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def build(
    project: Path,
    parent: Path,
    output: Path,
    probe_eta: float = PROBE_ETA,
    expected_parent_sha256: str = PARENT_SHA256,
) -> dict[str, object]:
    project = project.resolve()
    parent = parent.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not SOURCE_ETA < float(probe_eta) <= 0.25:
        raise ValueError("probe eta must be in (0.075, 0.25]")
    if _sha256(parent) != expected_parent_sha256.upper():
        raise ValueError("unexpected v25 parent SHA-256")
    verify_package(parent)

    with tempfile.TemporaryDirectory(prefix="v26_anchor_weight_probe_") as name:
        stage = Path(name)
        with zipfile.ZipFile(parent) as archive:
            _safe_extract(archive, stage)
        spec_path = stage / "model" / "v25_postbreak_anchor_spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if float(spec["blend_eta"]) != SOURCE_ETA:
            raise ValueError("v25 source eta differs from the frozen 7.5% parent")
        spec.update(
            {
                "protocol": "R_ANCHOR_WEIGHT_PROBE_2025_V1",
                "leaderboard_probe_parent": parent.name,
                "leaderboard_probe_parent_sha256": _sha256(parent),
                "blend_eta": float(probe_eta),
                "local_evidence_reference": (
                    "artifacts/v25_postbreak_anchor_audit_20260817_01 and "
                    "the exact quadratic eta sweep recorded in the research report"
                ),
                "selection_note": (
                    f"same row-local v25 signal at eta={float(probe_eta):.6f}; "
                    "Public weight probe explicitly permitted by DACON FAQ "
                    "reply 320256; no test-row aggregation"
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
                "leaderboard_weight_probe": True,
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
        "source_eta": SOURCE_ETA,
        "probe_eta": float(probe_eta),
        "sha256": _sha256(output),
        "size_bytes": output.stat().st_size,
        "official_faq": "https://dacon.io/competitions/official/236743/talkboard/417082#comment_320256",
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
    parser.add_argument("--parent", type=Path, default=Path("submit_v25.zip"))
    parser.add_argument("--output", type=Path, default=Path("submit_v26.zip"))
    parser.add_argument("--probe-eta", type=float, default=PROBE_ETA)
    parser.add_argument("--expected-parent-sha256", default=PARENT_SHA256)
    args = parser.parse_args()
    project = args.project.resolve()
    parent = args.parent if args.parent.is_absolute() else project / args.parent
    output = args.output if args.output.is_absolute() else project / args.output
    build(project, parent, output, args.probe_eta, args.expected_parent_sha256)


if __name__ == "__main__":
    main()
