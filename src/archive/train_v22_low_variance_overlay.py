"""Write the frozen, model-free v22 overlay specification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROFILES = {
    "conservative": {
        "R_CORE": {"anchor": 0.44, "weight": 0.020},
        "R_ANCHOR": {"anchor": 0.48, "weight": 0.010},
        "F": {"anchor": 0.52, "weight": 0.010},
        "prior_weight": 0.035,
        "local_evidence": {
            "y2023_to_y2024_gain_vs_v21": 3.9484317378,
            "y2023_early_to_late_gain_vs_v21": 54.714897,
            "y2024_early_to_late_gain_vs_v21": 8.149712,
            "minimum_primary_gain_vs_v21": 3.9484317378,
        },
    },
    "balanced": {
        "R_CORE": {"anchor": 0.44, "weight": 0.035},
        "R_ANCHOR": {"anchor": 0.48, "weight": 0.020},
        "F": {"anchor": 0.52, "weight": 0.020},
        "prior_weight": 0.050,
        "local_evidence": {
            "y2023_to_y2024_gain_vs_v21": 4.6215457142,
            "y2023_early_to_late_gain_vs_v21": 80.605986,
            "y2024_early_to_late_gain_vs_v21": 12.079557,
            "minimum_primary_gain_vs_v21": 4.6215457142,
        },
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def run(project: Path, output_dir: Path, profile: str) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = PROFILES[profile]
    spec = {
        "candidate": f"v22_low_variance_{profile}",
        "profile": profile,
        "parent": "submit_v21.zip",
        "domain_calibration": {
            name: selected[name] for name in ("R_CORE", "R_ANCHOR", "F")
        },
        "asof_prior": {
            "pitcher_fraction": 0.75,
            "batter_fraction": 0.25,
            "weight": selected["prior_weight"],
        },
        "missing_rate_default": 0.5,
        "local_evidence": selected["local_evidence"],
        "selection_warning": (
            "2024 full-season May and June are negative; public transfer must be "
            "measured before promotion over v21"
        ),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "external_data_used": False,
    }
    spec_path = output_dir / "v22_low_variance_spec.json"
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "protocol": "V22_LOW_VARIANCE_FINAL_2025_V1",
        "profile": profile,
        "train_sha256": _sha256(project / "data" / "train.csv"),
        "artifacts": {
            spec_path.name: {
                "size_bytes": spec_path.stat().st_size,
                "sha256": _sha256(spec_path),
            }
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--profile", choices=sorted(PROFILES), default="balanced")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/v22_low_variance_balanced_20260817")
    )
    args = parser.parse_args()
    run(args.project, args.output_dir, args.profile)


if __name__ == "__main__":
    main()
