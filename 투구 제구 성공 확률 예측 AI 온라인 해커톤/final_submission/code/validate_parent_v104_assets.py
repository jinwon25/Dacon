"""Compare freshly trained v56/v104 assets with both submitted parent trees."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


ROOTS = ("model/v124/model", "model/v124_bridge/model")
V56_BYTES = ("vocabularies.json", "metadata.json")
V104_BYTES = (
    "r_fm/older/vocabularies.json",
    "r_fm/older/metadata.json",
    "r_fm/recent/vocabularies.json",
    "r_fm/recent/metadata.json",
    "conditional/feature_spec.json",
    "conditional/baseline_lgb.txt",
    "conditional/conditional_lgb.txt",
    "conditional/bank.joblib",
)
V104_NPZ = ("r_fm/older/embeddings.npz", "r_fm/recent/embeddings.npz")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def compare_bytes(candidate: bytes, reference: bytes) -> dict[str, Any]:
    return {
        "candidate_bytes": len(candidate),
        "reference_bytes": len(reference),
        "candidate_sha256": sha256_bytes(candidate),
        "reference_sha256": sha256_bytes(reference),
        "byte_identical": candidate == reference,
    }


def compare_npz(candidate_path: Path, reference: bytes) -> dict[str, Any]:
    with np.load(candidate_path) as candidate, np.load(io.BytesIO(reference)) as expected:
        keys_equal = candidate.files == expected.files
        common = sorted(set(candidate.files) & set(expected.files))
        array_equal = keys_equal and all(
            np.array_equal(candidate[key], expected[key]) for key in common
        )
        max_abs = max(
            (
                float(
                    np.max(
                        np.abs(
                            candidate[key].astype(np.float64)
                            - expected[key].astype(np.float64)
                        )
                    )
                )
                for key in common
            ),
            default=0.0,
        )
    return {
        "keys_equal": keys_equal,
        "array_equal": array_equal,
        "max_abs": max_abs,
    }


def run(
    reference_zip: Path,
    v56_dir: Path,
    v104_assets_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "protocol": "PARENT_V56_V104_NATIVE_FIT_AUDIT_V1",
        "reference_zip_sha256": sha256_bytes(reference_zip.read_bytes()),
        "roots": {},
    }
    passed = True
    with zipfile.ZipFile(reference_zip) as archive:
        for root in ROOTS:
            checks: dict[str, Any] = {}
            for name in V56_BYTES:
                rel = f"parent/v56_fm/{name}"
                check = compare_bytes(
                    (v56_dir / name).read_bytes(), archive.read(f"{root}/{rel}")
                )
                checks[rel] = check
                passed &= bool(check["byte_identical"])
            rel = "parent/v56_fm/embeddings.npz"
            check = compare_npz(
                v56_dir / "embeddings.npz", archive.read(f"{root}/{rel}")
            )
            checks[rel] = check
            passed &= bool(check["array_equal"])
            for rel in V104_BYTES:
                check = compare_bytes(
                    (v104_assets_dir / rel).read_bytes(), archive.read(f"{root}/{rel}")
                )
                checks[rel] = check
                passed &= bool(check["byte_identical"])
            for rel in V104_NPZ:
                check = compare_npz(
                    v104_assets_dir / rel, archive.read(f"{root}/{rel}")
                )
                checks[rel] = check
                passed &= bool(check["array_equal"])
            result["roots"][root] = checks
    result["all_passed"] = passed
    report = output_dir / "parent_v104_native_fit.json"
    report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-zip", type=Path, required=True)
    parser.add_argument("--v56-dir", type=Path, required=True)
    parser.add_argument("--v104-assets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference_zip, args.v56_dir, args.v104_assets_dir, args.output_dir)


if __name__ == "__main__":
    main()
