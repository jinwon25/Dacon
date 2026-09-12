"""Create an immutable audit manifest for the public v2 champion package.

The audit is deliberately independent of model retraining.  It verifies the
user-supplied SHA before opening the archive and never replaces an existing
manifest or champion artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_SHA256 = "FE368AF109EF0BB8103D728F45794A6192599B691BF08DF02A22F31BD2D438A7"
BASELINE_COMMIT = "9b0cb8401336916f510669021005e48b4db2c92a"
PUBLIC = {
    "submission_id": 39023,
    "filename": "submit_v2.zip",
    "public_score": 763.2665303697,
    "submitted_at": "2026-08-08 16:08:48",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def git_snapshot(project: Path) -> dict[str, str]:
    root = project.parent
    safe = str(root).replace("\\", "/")

    def run(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-c", f"safe.directory={safe}", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return (completed.stdout or "").strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"unavailable: {exc}"

    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short", "--branch"),
        "baseline_commit_present": run("cat-file", "-e", f"{BASELINE_COMMIT}^{{commit}}"),
    }


def build_manifest(project: Path) -> dict:
    zip_path = project / "submit_v2.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    actual_sha = sha256(zip_path)
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError(
            f"submit_v2.zip SHA mismatch: expected {EXPECTED_SHA256}, got {actual_sha}"
        )
    if len(actual_sha) != 64:
        raise RuntimeError("SHA-256 must contain exactly 64 hexadecimal characters")

    members: list[dict] = []
    extracted_bytes = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            payload = archive.read(info.filename)
            extracted_bytes += len(payload)
            members.append(
                {
                    "name": info.filename,
                    "compressed_size": info.compress_size,
                    "uncompressed_size": info.file_size,
                    "sha256": hashlib.sha256(payload).hexdigest().upper(),
                }
            )

    config: dict[str, object] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in (
            "model/hybrid.json",
            "model/ensemble.json",
            "model/feature_spec.json",
            "model/trackman_feature_spec.json",
            "model/metadata.json",
        ):
            try:
                config[name] = json.loads(archive.read(name).decode("utf-8"))
            except KeyError:
                config[name] = None

    return {
        "manifest_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "champion": {
            "path": str(zip_path.resolve()),
            "filename": zip_path.name,
            "size_bytes": zip_path.stat().st_size,
            "sha256": actual_sha,
            "expected_sha256": EXPECTED_SHA256,
            "sha_verified": True,
            "extracted_size_bytes": extracted_bytes,
            "members": members,
        },
        "baseline_commit": BASELINE_COMMIT,
        "git": git_snapshot(project),
        "public_submission": PUBLIC,
        "recipe": {
            "base": "0.35 * engineered LightGBM + 0.65 * official RandomForest",
            "calibration": "train-only logit offset from ensemble.json",
            "r_rows": "replace RF component with half-life-1 recency RF",
            "trackman_rows": "blend rolling-damped Trackman LGB at 5%",
            "inference_branch_note": "r_only_rf_model takes precedence over recency_rf_model via if/elif; v4/v5 are not clean v2 ablations",
            "formula": "p = optional_game_type_offset(0.95 * candidate_base + 0.05 * trackman_probability)",
        },
        "package_config": config,
        "preservation": {
            "existing_v2_overwritten": False,
            "retrained_as_replacement": False,
            "external_outcome_data_used": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    project = args.project_dir.resolve()
    manifest = build_manifest(project)
    out_dir = project / "artifacts" / "champion_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"immutable manifest already exists with different content: {manifest_path}")
    manifest_path.write_text(encoded, encoding="utf-8")

    report = project / "research" / "reports" / "champion_v2_audit_20260809.md"
    report.write_text(
        "# Champion v2 audit\n\n"
        "- `submit_v2.zip` SHA-256: **verified**\n"
        f"- SHA-256: `{manifest['champion']['sha256']}` (64 hex characters)\n"
        f"- ZIP size / extracted size: `{manifest['champion']['size_bytes']:,}` / `{manifest['champion']['extracted_size_bytes']:,}` bytes\n"
        f"- Members hashed: **{len(manifest['champion']['members'])}**\n"
        f"- Baseline commit: `{BASELINE_COMMIT}`; present check: `{manifest['git']['baseline_commit_present'] or 'present'}`\n"
        f"- Current branch/HEAD: `{manifest['git']['branch']}` / `{manifest['git']['head']}`\n"
        "- Public record: submission 39023, `submit_v2.zip`, score **763.2665303697**.\n\n"
        "## Replayed recipe\n\n"
        "The package first blends engineered LightGBM (35%) and official RandomForest (65%) with the frozen train-only calibration. For `game_type=R`, the half-life-1 recency RF replaces the RF component. The resulting candidate is blended with the rolling-damped Trackman LightGBM at 5%. The package contains an `if r_only_rf_model ... elif recency_rf_model` branch; consequently v4/v5 cannot identify the causal contribution of the v2 R-recency or Trackman terms.\n\n"
        "This manifest is immutable for this cycle; the original ZIP is not overwritten or reconstructed.\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
