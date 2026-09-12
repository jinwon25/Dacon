"""Audit the private Git LFS model and OOF-evidence release for champion 1161."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET = "control_success"
EXPECTED_MODEL_ROOTS = {"model", "requirements.txt", "script.py"}
REQUIRED_OOF_ARRAYS = {
    "raw_index",
    "target",
    "parent",
    "exact_mask",
    "season",
    "game_month",
    "domain3",
    "pitcher_id",
    "batter_id",
    "common_parent",
}
TRAIN_COLUMNS = [
    "row_id",
    TARGET,
    "season",
    "game_month",
    "game_type",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_id",
    "batter_id",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest().upper()


def sha256_strings(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype("string").fillna("__MISSING__"):
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest().upper()


def domain3(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype("string").fillna("__MISSING__").eq("R")
    anchor = frame["pitcher_team_id"].eq(13) | frame["batter_team_id"].eq(13)
    return np.where(
        ~regular.to_numpy(),
        "F",
        np.where(anchor.to_numpy(), "R_ANCHOR", "R_CORE"),
    )


def audit_model(model_zip: Path, model_manifest: Path) -> dict[str, Any]:
    model_zip = model_zip.resolve()
    manifest = json.loads(model_manifest.read_text(encoding="utf-8"))
    release = manifest["canonical_release"]
    actual_sha = sha256_file(model_zip)
    actual_bytes = model_zip.stat().st_size
    if actual_sha != str(release["sha256"]).upper():
        raise ValueError("model ZIP SHA-256 does not match canonical release")
    if actual_bytes != int(release["bytes"]):
        raise ValueError("model ZIP size does not match canonical release")
    if not manifest.get("promoted_to_champion"):
        raise ValueError("model manifest is not promoted_to_champion")
    public = manifest["public_result"]
    if float(public["public_score"]) != 1161.2020600422:
        raise ValueError("unexpected champion Public score")
    if int(public["submission_id"]) != 59988:
        raise ValueError("unexpected champion submission ID")
    if not manifest.get("standalone_no_parent_zip_dependency"):
        raise ValueError("model manifest does not guarantee standalone execution")
    if not manifest.get("row_local_inference") or manifest.get("test_aggregate_used"):
        raise ValueError("model manifest violates the row-local contract")

    with zipfile.ZipFile(model_zip) as archive:
        bad_member = archive.testzip()
        names = archive.namelist()
        roots = {
            name.split("/", 1)[0]
            for name in names
            if name and not name.endswith("/")
        }
        script = archive.read("script.py").decode("utf-8")
    if bad_member is not None:
        raise ValueError(f"model ZIP CRC failure: {bad_member}")
    if roots != EXPECTED_MODEL_ROOTS:
        raise ValueError(f"unexpected model ZIP roots: {sorted(roots)}")
    if "from src" in script or "import src" in script:
        raise ValueError("standalone model imports repository src")

    return {
        "sha256": actual_sha,
        "bytes": int(actual_bytes),
        "file_count": int(sum(not name.endswith("/") for name in names)),
        "roots": sorted(roots),
        "crc_ok": True,
        "public_score": float(public["public_score"]),
        "submission_id": int(public["submission_id"]),
        "official_runtime_seconds": float(public["official_runtime_seconds"]),
    }


def _aligned_equal(left: pd.Series, right: np.ndarray) -> bool:
    numeric = pd.to_numeric(left, errors="coerce").to_numpy()
    return np.array_equal(numeric, right)


def audit_axis(
    oof_dir: Path,
    spec: dict[str, Any],
    train: pd.DataFrame | None,
) -> dict[str, Any]:
    path = (oof_dir / spec["file"]).resolve()
    if sha256_file(path) != str(spec["file_sha256"]).upper():
        raise ValueError(f"OOF file SHA-256 mismatch: {path.name}")
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"OOF file size mismatch: {path.name}")

    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != REQUIRED_OOF_ARRAYS:
            raise ValueError(
                f"unexpected OOF arrays in {path.name}: {sorted(saved.files)}"
            )
        arrays = {name: saved[name] for name in saved.files}

    rows = int(spec["rows"])
    if any(array.shape != (rows,) for array in arrays.values()):
        raise ValueError(f"OOF arrays are not one-dimensional/aligned: {path.name}")
    raw_index = arrays["raw_index"]
    target = arrays["target"]
    prediction = arrays["parent"]
    exact = arrays["exact_mask"]
    if raw_index.dtype.kind not in "iu" or len(np.unique(raw_index)) != rows:
        raise ValueError(f"invalid or duplicate raw_index: {path.name}")
    if np.any(raw_index < 0):
        raise ValueError(f"negative raw_index: {path.name}")
    if not np.isin(target, (0.0, 1.0)).all():
        raise ValueError(f"non-binary target: {path.name}")
    if not np.isfinite(prediction).all() or np.any(
        (prediction < 0.0) | (prediction > 1.0)
    ):
        raise ValueError(f"invalid prediction probability: {path.name}")
    if not np.isfinite(arrays["common_parent"]).all() or np.any(
        (arrays["common_parent"] < 0.0) | (arrays["common_parent"] > 1.0)
    ):
        raise ValueError(f"invalid common-parent probability: {path.name}")
    if exact.dtype.kind != "b" or int(exact.sum()) != int(spec["exact_rows"]):
        raise ValueError(f"exact_mask mismatch: {path.name}")
    if not np.all(arrays["season"] == int(spec["evaluation_season"])):
        raise ValueError(f"evaluation season mismatch: {path.name}")
    exact_domains = sorted(np.unique(arrays["domain3"][exact]).astype(str).tolist())
    if exact_domains != sorted(str(value) for value in spec["exact_domains"]):
        raise ValueError(f"exact-domain mismatch: {path.name}")
    if sha256_array(raw_index) != str(spec["raw_index_sha256"]).upper():
        raise ValueError(f"raw-index digest mismatch: {path.name}")
    if sha256_array(target) != str(spec["target_sha256"]).upper():
        raise ValueError(f"target digest mismatch: {path.name}")
    if sha256_array(prediction) != str(spec["prediction_sha256"]).upper():
        raise ValueError(f"prediction digest mismatch: {path.name}")

    train_alignment = "not_checked"
    if train is not None:
        if int(raw_index.max(initial=-1)) >= len(train):
            raise ValueError(f"raw_index outside train.csv: {path.name}")
        selected = train.iloc[raw_index].reset_index(drop=True)
        if not np.array_equal(selected[TARGET].to_numpy(np.float64), target):
            raise ValueError(f"target/train alignment mismatch: {path.name}")
        if not _aligned_equal(selected["season"], arrays["season"]):
            raise ValueError(f"season/train alignment mismatch: {path.name}")
        if not _aligned_equal(selected["game_month"], arrays["game_month"]):
            raise ValueError(f"month/train alignment mismatch: {path.name}")
        if not _aligned_equal(selected["pitcher_id"], arrays["pitcher_id"]):
            raise ValueError(f"pitcher/train alignment mismatch: {path.name}")
        if not _aligned_equal(selected["batter_id"], arrays["batter_id"]):
            raise ValueError(f"batter/train alignment mismatch: {path.name}")
        if not np.array_equal(domain3(selected).astype(str), arrays["domain3"].astype(str)):
            raise ValueError(f"domain/train alignment mismatch: {path.name}")
        if sha256_strings(selected["row_id"]) != str(spec["row_id_sha256"]).upper():
            raise ValueError(f"row_id digest mismatch: {path.name}")
        train_alignment = "pass"

    return {
        "axis": spec["axis"],
        "role": spec["role"],
        "fidelity": spec["fidelity"],
        "sha256": str(spec["file_sha256"]).upper(),
        "bytes": int(path.stat().st_size),
        "rows": rows,
        "exact_rows": int(exact.sum()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "train_alignment": train_alignment,
    }


def audit_private_release(
    model_zip: Path,
    model_manifest: Path,
    oof_dir: Path,
    *,
    train_csv: Path | None = None,
) -> dict[str, Any]:
    model = audit_model(model_zip, model_manifest)
    oof_dir = oof_dir.resolve()
    manifest = json.loads((oof_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != "CHAMPION_1161_PRIVATE_OOF_EVIDENCE_V1":
        raise ValueError("unexpected private OOF protocol")
    if manifest["visibility"] != "PRIVATE_OFFICIAL_DACON_TEAM_ONLY":
        raise ValueError("private OOF visibility guard is missing")
    if str(manifest["model"]["sha256"]).upper() != model["sha256"]:
        raise ValueError("OOF/model lineage SHA-256 mismatch")
    if not manifest.get("row_local") or manifest.get("test_aggregate_used"):
        raise ValueError("OOF manifest violates the row-local contract")
    if not manifest.get("contains_competition_targets_and_player_ids"):
        raise ValueError("OOF sensitivity flag is missing")

    config_path = (oof_dir / manifest["model"]["config"]).resolve()
    if sha256_file(config_path) != str(manifest["model"]["config_sha256"]).upper():
        raise ValueError("champion config SHA-256 mismatch")

    train: pd.DataFrame | None = None
    train_sha: str | None = None
    if train_csv is not None:
        train_path = train_csv.resolve()
        train_sha = sha256_file(train_path)
        if train_sha != str(manifest["train_csv_sha256"]).upper():
            raise ValueError("train.csv SHA-256 mismatch")
        train = pd.read_csv(train_path, usecols=TRAIN_COLUMNS, low_memory=False)
        if len(train) != int(manifest["train_rows"]):
            raise ValueError("train.csv row count mismatch")

    axes = [audit_axis(oof_dir, spec, train) for spec in manifest["axes"]]
    return {
        "status": "pass",
        "model": model,
        "oof_protocol": manifest["protocol"],
        "oof_axis_count": len(axes),
        "train_sha256": train_sha,
        "axes": axes,
        "privacy": manifest["visibility"],
        "caveat": manifest["caveat"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-zip", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--oof-dir", required=True, type=Path)
    parser.add_argument("--train-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_private_release(
        args.model_zip,
        args.model_manifest,
        args.oof_dir,
        train_csv=args.train_csv,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
