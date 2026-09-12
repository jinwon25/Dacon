"""Build the gated game-type regime candidate without changing incumbent files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from src.archive.data import read_main
from src.archive.domain_drift import (
    CANDIDATE_NAME,
    fit_game_type_regime_offsets,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_zip(package_dir: Path, output: Path) -> None:
    members = [
        path
        for path in package_dir.rglob("*")
        if path.is_file()
    ]
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(members):
            name = path.relative_to(package_dir).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 6, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def run(project_dir: Path, output_name: str) -> dict[str, object]:
    project_dir = project_dir.resolve()
    gate_path = project_dir / "research" / "reports" / "domain_drift_results.csv"
    if not gate_path.exists():
        raise FileNotFoundError("run domain_drift.py before packaging")
    import pandas as pd

    gate = pd.read_csv(gate_path)
    candidate_rows = gate.loc[gate["candidate"] == CANDIDATE_NAME]
    if len(candidate_rows) != 4 or not candidate_rows["passes_statistical_gate"].all():
        raise RuntimeError("candidate did not pass the fixed statistical gate")

    output = project_dir / output_name
    candidate_root = (
        project_dir
        / "artifacts"
        / "candidates"
        / CANDIDATE_NAME
        / output.stem
    )
    package_dir = candidate_root / "package"
    if package_dir.exists() or output.exists():
        raise FileExistsError(
            "candidate output already exists; use a new output name to preserve it"
    )
    (package_dir / "model").mkdir(parents=True, exist_ok=False)
    incumbent = project_dir / "submit.zip"
    with zipfile.ZipFile(incumbent) as archive:
        for member in sorted(archive.namelist()):
            if member.startswith("model/") and not member.endswith("/"):
                destination = package_dir / member
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
    shutil.copy2(project_dir / "script.py", package_dir / "script.py")
    shutil.copy2(project_dir / "requirements.txt", package_dir / "requirements.txt")

    train = read_main(project_dir / "data" / "train.csv")
    offsets, diagnostics = fit_game_type_regime_offsets(train, 2025)
    offset_artifact = {
        "method": "game_type_logit_offset",
        "candidate": CANDIDATE_NAME,
        "forecast_season": 2025,
        "offsets": offsets,
        "fit_scope": "official train seasons 2019-2024 only",
        "row_local_apply_key": "game_type",
        "test_aggregate_used": False,
        "detection_rule": {
            "minimum_rate_residual_gap": 0.05,
            "minimum_group_rows": 5000,
            "recent_window": "one-year sign change or persistent two-year sign change",
        },
        "diagnostics": diagnostics.to_dict("records"),
    }
    (package_dir / "model" / "game_type_offsets.json").write_text(
        json.dumps(offset_artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_zip(package_dir, output)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "candidate": CANDIDATE_NAME,
        "package_path": str(output),
        "package_size_bytes": output.stat().st_size,
        "package_sha256": _sha256(output),
        "incumbent_package_sha256": _sha256(incumbent),
        "offsets": offsets,
        "local_2024_delta_brier": float(
            candidate_rows.loc[
                candidate_rows["outer_validation_season"] == 2024, "delta_brier"
            ].iloc[0]
        ),
        "recency_weighted_delta": float(
            candidate_rows["recency_weighted_delta"].iloc[0]
        ),
        "worst_fold_delta": float(candidate_rows["worst_fold_delta"].iloc[0]),
        "passes_statistical_gate": True,
        "submission_priority": 1,
        "hypothesis": (
            "Preserve the incumbent and carry only a large, persistent train-only "
            "game-type regime residual into 2025."
        ),
    }
    candidate_root.mkdir(parents=True, exist_ok=True)
    (candidate_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output", default="submit_candidate_game_type_regime_v1.zip"
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project_dir, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
