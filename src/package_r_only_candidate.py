"""Train and package controlled R-only RF candidates.

The R-only model is fit on regular-season rows only. At inference it replaces
25% of the incumbent probability for ``game_type=R``. Two packages are made:
one isolates the R-only branch and one adds the previously validated 5%
Trackman blend.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from src.data import TARGET_COL, read_main
from src.package import verify_package
from src.train import make_official_rf


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _extract_safe(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe source ZIP member: {member.filename}")
            target = destination / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))


def _write_zip(package_dir: Path, output: Path) -> None:
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(package_dir.rglob("*")):
            if not path.is_file():
                continue
            name = path.relative_to(package_dir).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 9, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def run(
    project_dir: Path,
    source_name: str = "submit_v2.zip",
    r_only_weight: float = 0.25,
) -> list[dict[str, object]]:
    project_dir = project_dir.resolve()
    source = project_dir / source_name
    if not source.is_file():
        raise FileNotFoundError(source)
    if not 0.0 < r_only_weight < 1.0:
        raise ValueError("r_only_weight must be between 0 and 1")

    outputs = [
        ("submit_v4.zip", 0.0, "r_only_rf_w25_v1"),
        ("submit_v5.zip", 0.05, "r_only_rf_w25_trackman_w05_v1"),
    ]
    for output_name, _, _ in outputs:
        if len(output_name) > 40:
            raise ValueError(f"final ZIP filename must be <=40 characters: {output_name}")
        if (project_dir / output_name).exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output_name}")

    with zipfile.ZipFile(source) as source_archive:
        ensemble = json.loads(source_archive.read("model/ensemble.json"))
    official_features = list(ensemble["official_features"])
    train = read_main(project_dir / "data" / "train.csv")
    regular = train["game_type"].astype("string").eq("R").to_numpy()
    target = train.loc[regular, TARGET_COL].to_numpy(dtype=np.int8)
    print(f"[R-only package] Training RF on {regular.sum():,} regular-season rows...")
    r_only_rf = make_official_rf(official_features, random_state=42)
    r_only_rf.fit(train.loc[regular, official_features], target)
    del train, target
    gc.collect()

    manifests: list[dict[str, object]] = []
    for output_name, trackman_weight, candidate in outputs:
        output = project_dir / output_name
        root = project_dir / "artifacts" / "candidates" / candidate / output.stem
        package_dir = root / "package"
        if root.exists():
            raise FileExistsError(f"refusing to overwrite existing artifacts: {root}")
        package_dir.mkdir(parents=True)
        _extract_safe(source, package_dir)
        shutil.copy2(project_dir / "script.py", package_dir / "script.py")
        shutil.copy2(project_dir / "requirements.txt", package_dir / "requirements.txt")
        joblib.dump(r_only_rf, package_dir / "model" / "rf_r_only.joblib", compress=3)

        hybrid_path = package_dir / "model" / "hybrid.json"
        hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
        hybrid.update(
            {
                "candidate": candidate,
                "r_only_rf_model": "rf_r_only.joblib",
                "r_only_weight": float(r_only_weight),
                "r_only_apply_game_type": "R",
                "r_only_rf_calibration": {"method": "identity"},
                "trackman_weight": float(trackman_weight),
                "variant_parent": source.name,
            }
        )
        hybrid_path.write_text(
            json.dumps(hybrid, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _write_zip(package_dir, output)
        verify_package(output)
        manifest = {
            "created_at": datetime.now().astimezone().isoformat(),
            "candidate": candidate,
            "output_name": output.name,
            "output_name_length": len(output.name),
            "package_path": str(output),
            "package_sha256": _sha256(output),
            "source_name": source.name,
            "source_sha256": _sha256(source),
            "r_only_weight": float(r_only_weight),
            "trackman_weight": float(trackman_weight),
            "change_scope": (
                "Regular-season-only RF is blended at 25%; the second package additionally retains Trackman at 5%."
            ),
            "local_expected_delta": {
                "r_only_25": -0.000112170,
                "r_only_25_trackman_5": -0.000172396,
            },
            "public_probe_note": (
                "Prepared after Trackman 10% lost 1.2818936509 Public points; v4 isolates R-only RF and v5 tests its interaction with 5% Trackman."
            ),
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifests.append(manifest)
    del r_only_rf
    gc.collect()
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--source", default="submit_v2.zip")
    parser.add_argument("--r-only-weight", type=float, default=0.25)
    args = parser.parse_args()
    manifests = run(args.project_dir, args.source, args.r_only_weight)
    print(json.dumps(manifests, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
