"""Train, package, and audit the user-authorized v116 DACON public probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.core.packaging import (
    _safe_extract,
    _safe_remove_generated,
    _sha256,
    _zip_directory,
    run_package,
)
from src.v113_fine_pitch_failure_prior_v104 import (
    COMPONENTS,
    PITCH_TYPE_NORMALISATION,
    failure_selection_delta,
    reconstruct_failure_components,
)
from src.v117_failure_prior_component import (
    build_failure_bank,
    predict_failure_components,
)


PROTOCOL = "V117_V116_PUBLIC_PROBE_V1"
ID_COL = "row_id"
TARGET_COL = "control_success"
ANCHOR_TEAM = 13


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def build_history(
    raw: pd.DataFrame,
    trackman_csv: Path,
    alignment_npz: Path,
    pitch_types: list[str],
) -> pd.DataFrame:
    work = raw.copy()
    work["count_state"] = (
        work["balls_before"].astype(np.int8) * 3
        + work["strikes_before"].astype(np.int8)
    )
    failure = reconstruct_failure_components(work)
    trackman = pd.read_csv(
        trackman_csv, usecols=["season", "tagged_pitch_type"], low_memory=False
    )
    with np.load(alignment_npz, allow_pickle=False) as saved:
        main_index = saved["main_index"].astype(np.int64)
        trackman_index = saved["trackman_index"].astype(np.int64)
        aligned_season = saved["season"].astype(np.int16)
    if len(np.unique(main_index)) != len(main_index):
        raise ValueError("main alignment is not one-to-one")
    if not np.array_equal(work.iloc[main_index]["season"].to_numpy(np.int16), aligned_season):
        raise ValueError("main/alignment season mismatch")
    if not np.array_equal(
        trackman.iloc[trackman_index]["season"].to_numpy(np.int16), aligned_season
    ):
        raise ValueError("TrackMan/alignment season mismatch")
    fine = (
        trackman.iloc[trackman_index]["tagged_pitch_type"]
        .astype(str)
        .replace(PITCH_TYPE_NORMALISATION)
    )
    fine = fine.where(fine.isin(pitch_types[:-1]), pitch_types[-1]).to_numpy(str)
    history = work.iloc[main_index][
        ["season", "pitcher_id", "batter_hand", "count_state"]
    ].reset_index(drop=True)
    history["pitch_type_fine"] = fine
    for component in COMPONENTS:
        history[f"failure__{component}"] = failure.iloc[main_index][component].to_numpy(float)
    return history.dropna(
        subset=[f"failure__{component}" for component in COMPONENTS]
    ).reset_index(drop=True)


def train_export_bank(
    history: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
    v116_summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    recipe = config["recipe"]
    pitch_types = [
        "Fastball", "Slider", "Curveball", "ChangeUp", "Splitter", "Sinker",
        "Cutter", "Other",
    ]
    bank = build_failure_bank(
        history,
        pitch_types,
        outcome_shrink=float(recipe["outcome_shrink"]),
        selection_shrink=float(recipe["selection_shrink"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bank, output_dir / "bank.joblib", compress=3)
    components = [str(value) for value in recipe["components"]]
    weights = np.asarray(recipe["weights"], dtype=np.float64)
    fitted = np.asarray(
        [float(v116_summary["selected_weights"][name]) for name in components]
    )
    if not np.allclose(weights, fitted, atol=1e-15, rtol=0.0):
        raise ValueError("package weights differ from the locked v116 result")
    spec = {
        "protocol": "V117_V116_FAILURE_PRIOR_EXPORT_V1",
        "components": components,
        "weights": weights.tolist(),
        "route_domains": ["R_CORE"],
        "history_rows": int(len(history)),
        "history_seasons": sorted(int(value) for value in history["season"].unique()),
        "current_pitch_type_used": False,
        "row_local_inference": True,
        "test_aggregate_used": False,
    }
    (output_dir / "spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = test.reset_index(drop=True).copy()
    rows["count_state"] = (
        rows["balls_before"].astype(np.int8) * 3
        + rows["strikes_before"].astype(np.int8)
    )
    actual = predict_failure_components(rows, bank)
    expected = np.column_stack(
        [
            failure_selection_delta(
                history,
                rows,
                component,
                pitch_types,
                outcome_shrink=float(recipe["outcome_shrink"]),
                selection_shrink=float(recipe["selection_shrink"]),
            )
            for component in components
        ]
    )
    parity = float(np.max(np.abs(actual - expected))) if len(rows) else 0.0
    if parity > 1e-12:
        raise ValueError(f"failure-prior bank parity failed: {parity}")
    return {
        **spec,
        "bank_bytes": int((output_dir / "bank.joblib").stat().st_size),
        "direct_formula_max_abs": parity,
        "component_rms_on_smoke": {
            name: float(np.sqrt(np.mean(np.square(actual[:, index]))))
            for index, name in enumerate(components)
        },
    }


def build_package(
    parent_zip: Path,
    assets: Path,
    wrapper: Path,
    component: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    staging = output_dir / "_staging"
    parent_stage = output_dir / "_parent"
    _safe_remove_generated(staging, output_dir)
    _safe_remove_generated(parent_stage, output_dir)
    staging.mkdir(parents=True)
    parent_stage.mkdir()
    with zipfile.ZipFile(parent_zip) as archive:
        _safe_extract(archive, parent_stage)
    (staging / "model" / "components").mkdir(parents=True)
    shutil.copytree(parent_stage / "model", staging / "model" / "parent")
    shutil.copyfile(
        parent_stage / "script.py", staging / "model" / "components" / "parent_script.py"
    )
    shutil.copyfile(component, staging / "model" / "components" / "failure_prior.py")
    shutil.copytree(assets / "failure_prior", staging / "model" / "failure_prior")
    shutil.copyfile(wrapper, staging / "script.py")
    shutil.copyfile(parent_stage / "requirements.txt", staging / "requirements.txt")
    package = output_dir / "submit_v116_failure_prior_probe.zip"
    if package.exists():
        _safe_remove_generated(package, output_dir)
    result = _zip_directory(staging, package)
    _safe_remove_generated(staging, output_dir)
    _safe_remove_generated(parent_stage, output_dir)
    return package, result


def _mixed_smoke(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    mixed = pd.concat([source] * 4, ignore_index=True)
    mixed[ID_COL] = [f"v116_probe_{index:03d}" for index in range(len(mixed))]
    group = np.arange(len(mixed)) % 4
    mixed.loc[group == 2, "pitcher_team_id"] = ANCHOR_TEAM
    mixed.loc[group == 3, "game_type"] = "F"
    sample = pd.DataFrame({ID_COL: mixed[ID_COL], TARGET_COL: 0.0})
    return mixed, sample


def _direct_correction(
    mixed: pd.DataFrame, bank: dict[str, Any], weights: np.ndarray
) -> np.ndarray:
    rows = mixed.copy()
    rows["count_state"] = (
        rows["balls_before"].astype(np.int8) * 3
        + rows["strikes_before"].astype(np.int8)
    )
    return predict_failure_components(rows, bank) @ weights


def audit_package(
    package: Path,
    parent_zip: Path,
    assets: Path,
    config: dict[str, Any],
    data_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    mixed, sample = _mixed_smoke(data_dir)
    parent_out, parent_seconds, _ = run_package(parent_zip, mixed, sample, timeout=timeout)
    candidate_out, candidate_seconds, stdout = run_package(
        package, mixed, sample, timeout=timeout
    )
    parent = mixed[ID_COL].map(parent_out.set_index(ID_COL)[TARGET_COL]).to_numpy(float)
    candidate = mixed[ID_COL].map(candidate_out.set_index(ID_COL)[TARGET_COL]).to_numpy(float)
    regular = mixed["game_type"].eq("R").to_numpy()
    anchor = mixed["pitcher_team_id"].eq(ANCHOR_TEAM).to_numpy() | mixed[
        "batter_team_id"
    ].eq(ANCHOR_TEAM).to_numpy()
    active = regular & ~anchor
    protected_max_abs = float(np.max(np.abs(candidate[~active] - parent[~active])))
    active_shift_count = int(np.count_nonzero(np.abs(candidate[active] - parent[active]) > 0.0))
    if protected_max_abs > 1e-12 or active_shift_count == 0:
        raise ValueError(
            f"v116 route audit failed: protected={protected_max_abs}, shifted={active_shift_count}"
        )
    bank = joblib.load(assets / "failure_prior" / "bank.joblib")
    weights = np.asarray(config["recipe"]["weights"], dtype=np.float64)
    correction = _direct_correction(mixed, bank, weights)
    expected = np.clip(parent + np.where(active, correction, 0.0), 0.001, 0.999)
    formula_max_abs = float(np.max(np.abs(candidate - expected)))
    if formula_max_abs > 1e-12:
        raise ValueError(f"v116 package formula parity failed: {formula_max_abs}")

    shuffled_test = mixed.sample(frac=1.0, random_state=117).reset_index(drop=True)
    shuffled_sample = pd.DataFrame({ID_COL: shuffled_test[ID_COL], TARGET_COL: 0.0})
    shuffled, _, _ = run_package(package, shuffled_test, shuffled_sample, timeout=timeout)
    base_map = candidate_out.set_index(ID_COL)[TARGET_COL].sort_index()
    shuffled_map = shuffled.set_index(ID_COL)[TARGET_COL].sort_index()
    shuffled_max_abs = float(np.max(np.abs(base_map - shuffled_map)))
    if shuffled_max_abs > 1e-12:
        raise ValueError(f"v116 shuffled-row parity failure: {shuffled_max_abs}")

    parts = []
    for indices in np.array_split(np.arange(len(mixed)), 2):
        local_test = mixed.iloc[indices].reset_index(drop=True)
        local_sample = pd.DataFrame({ID_COL: local_test[ID_COL], TARGET_COL: 0.0})
        output, _, _ = run_package(package, local_test, local_sample, timeout=timeout)
        parts.append(output)
    partition_map = pd.concat(parts).set_index(ID_COL)[TARGET_COL].sort_index()
    partition_max_abs = float(np.max(np.abs(base_map - partition_map)))
    if partition_max_abs > 1e-12:
        raise ValueError(f"v116 partition parity failure: {partition_max_abs}")

    scale_rows = 245789
    scale = mixed.iloc[np.arange(scale_rows) % len(mixed)].reset_index(drop=True)
    scale[ID_COL] = [f"v116_scale_{index:06d}" for index in range(scale_rows)]
    scale_sample = pd.DataFrame({ID_COL: scale[ID_COL], TARGET_COL: 0.0})
    scale_output, scale_seconds, scale_stdout = run_package(
        package, scale, scale_sample, timeout=timeout
    )
    values = scale_output[TARGET_COL].to_numpy(float)
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("v116 scale proxy produced invalid probabilities")
    return {
        "mixed_rows": int(len(mixed)),
        "active_rows": int(active.sum()),
        "active_shift_count": active_shift_count,
        "protected_max_abs": protected_max_abs,
        "formula_max_abs": formula_max_abs,
        "shuffled_max_abs": shuffled_max_abs,
        "partition_max_abs": partition_max_abs,
        "parent_smoke_runtime_seconds": parent_seconds,
        "candidate_smoke_runtime_seconds": candidate_seconds,
        "scale_proxy_rows": scale_rows,
        "scale_proxy_runtime_seconds": scale_seconds,
        "scale_proxy_mean": float(values.mean()),
        "scale_proxy_min": float(values.min()),
        "scale_proxy_max": float(values.max()),
        "smoke_stdout": stdout,
        "scale_stdout": scale_stdout,
    }


def run(
    train_csv: Path,
    trackman_csv: Path,
    alignment_npz: Path,
    data_dir: Path,
    parent_zip: Path,
    v116_dir: Path,
    config_path: Path,
    output_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    train_csv = train_csv.resolve()
    trackman_csv = trackman_csv.resolve()
    alignment_npz = alignment_npz.resolve()
    data_dir = data_dir.resolve()
    parent_zip = parent_zip.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    v116_summary = json.loads((v116_dir / "summary.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(train_csv, low_memory=False)
    test = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    history = build_history(raw, trackman_csv, alignment_npz, [
        "Fastball", "Slider", "Curveball", "ChangeUp", "Splitter", "Sinker",
        "Cutter", "Other",
    ])

    assets = output_dir / "_training_assets"
    _safe_remove_generated(assets, output_dir)
    bank_audit = train_export_bank(
        history, test, config, v116_summary, assets / "failure_prior"
    )
    package, package_result = build_package(
        parent_zip,
        assets,
        Path(__file__).with_name("v117_v116_probe_wrapper.py"),
        Path(__file__).with_name("v117_failure_prior_component.py"),
        output_dir,
    )
    package_audit = audit_package(
        package, parent_zip, assets, config, data_dir, timeout
    )
    result = {
        "protocol": PROTOCOL,
        "config": config,
        "train_csv_sha256": _file_sha256(train_csv),
        "trackman_csv_sha256": _file_sha256(trackman_csv),
        "alignment_npz_sha256": _file_sha256(alignment_npz),
        "parent_zip_sha256": _sha256(parent_zip),
        "failure_prior_export": bank_audit,
        "package": package_result,
        "audit": package_audit,
        "eligible_for_user_authorized_public_probe": True,
        "local_promotion_gate_passed": False,
        "authorization_basis": config["authorization"],
        "test_aggregate_used": False,
        "row_local_inference": True,
        "standalone_no_parent_zip_dependency": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--alignment-npz", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--parent-zip", type=Path, required=True)
    parser.add_argument("--v116-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    run(
        args.train_csv, args.trackman_csv, args.alignment_npz, args.data_dir,
        args.parent_zip, args.v116_dir, args.config, args.output_dir, args.timeout,
    )


if __name__ == "__main__":
    main()
