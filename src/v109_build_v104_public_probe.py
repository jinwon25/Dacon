"""Train, package, and audit the user-authorized v104 DACON public probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as torch_f

from src.core.diagnostics import v27_parent
from src.core.axes import _cached_v25_axes
from src.core.banks import _metadata
from src.v53_factorization_offset import (
    BATCH_SIZE,
    DROPOUT,
    EPOCHS,
    FIELDS,
    LEARNING_RATE,
    PAIR_INDEX,
    PAIR_NAMES,
    SEED,
    TARGET,
    WEIGHT_DECAY,
    FieldEncoder,
    PairwiseFM,
    _logit,
    _predict_raw,
    domain_centres,
    prepare_fields,
)
from src.core.packaging import (
    _safe_extract,
    _safe_remove_generated,
    _sha256,
    _zip_directory,
    run_package,
)
from src.v84_fixed_v56_export import numpy_raw
from src.v109_v104_feature_component import (
    CATEGORICAL,
    CONDITIONAL_COLUMNS,
    align_categories,
    apply_model_spec,
    build_bank,
    feature_frame,
)
from src.v109_v104_probe_wrapper import stability_mask


PROTOCOL = "V109_V104_PUBLIC_PROBE_V1"
RANK = 16
CORRECTION_CLIP = 0.25
ID_COL = "row_id"
TARGET_COL = "control_success"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def train_export_independent_fm(
    source: pd.DataFrame,
    parent: np.ndarray,
    output_dir: Path,
    source_window: str,
) -> dict[str, Any]:
    prepared = prepare_fields(source).reset_index(drop=True)
    parent = np.asarray(parent, dtype=np.float64)
    if len(prepared) != len(parent):
        raise ValueError(f"FM source parent mismatch: {source_window}")
    encoder = FieldEncoder.fit(prepared)
    code = encoder.transform(prepared)
    target = prepared[TARGET].to_numpy(np.float32)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(6)
    model = PairwiseFM(encoder.cardinalities, RANK, DROPOUT)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    code_tensor = torch.from_numpy(code)
    target_tensor = torch.from_numpy(target)
    offset_tensor = torch.from_numpy(_logit(parent).astype(np.float32))
    losses: list[float] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(
            len(code_tensor), generator=torch.Generator().manual_seed(SEED + epoch)
        )
        loss_sum = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            correction = model(code_tensor[index])
            loss = torch_f.binary_cross_entropy_with_logits(
                offset_tensor[index] + correction, target_tensor[index]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
        losses.append(loss_sum / len(code_tensor))
        print(
            f"[v109 FM] source={source_window} epoch={epoch} logloss={losses[-1]:.8f}",
            flush=True,
        )

    raw = _predict_raw(model, code, BATCH_SIZE)
    centres = domain_centres(raw, prepared["domain3"].astype(str).to_numpy())
    embeddings = [
        layer.weight.detach().cpu().numpy().astype(np.float32)
        for layer in model.embeddings
    ]
    parity = float(np.max(np.abs(raw - numpy_raw(code, embeddings))))
    if parity > 2e-6:
        raise ValueError(f"FM NumPy parity failure: {source_window}/{parity}")

    output_dir.mkdir(parents=True, exist_ok=True)
    vocabularies = {
        field: vocabulary
        for field, vocabulary in zip(FIELDS, encoder.vocabularies, strict=True)
    }
    (output_dir / "vocabularies.json").write_text(
        json.dumps(vocabularies, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "embeddings.npz",
        **{f"field_{i}": value for i, value in enumerate(embeddings)},
    )
    metadata = {
        "protocol": "V109_INDEPENDENT_R_FM_NUMPY_V1",
        "source_window": source_window,
        "fields": list(FIELDS),
        "pair_names": [list(pair) for pair in PAIR_NAMES],
        "pair_index": [list(pair) for pair in PAIR_INDEX],
        "rank": RANK,
        "seed": SEED,
        "epochs": EPOCHS,
        "losses": losses,
        "domain_centres": centres,
        "correction_clip": CORRECTION_CLIP,
        "numpy_export_max_abs_parity": parity,
        "source_rows": int(len(prepared)),
        "row_local_inference": True,
        "test_aggregate_used": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return metadata


def train_export_conditional(
    raw: pd.DataFrame,
    test: pd.DataFrame,
    v97_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    years = list(range(int(raw["season"].min()), int(raw["season"].max()) + 1))
    by_year: dict[int, pd.DataFrame] = {}
    target_parts: list[np.ndarray] = []
    season_parts: list[np.ndarray] = []
    for year in years:
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        bank = build_bank(
            raw.loc[raw["season"].lt(year)], v97_config["conditional_strength"]
        )
        by_year[year] = feature_frame(rows, bank)
        target_parts.append(rows[TARGET].to_numpy(np.int8))
        season_parts.append(np.full(len(rows), year, dtype=np.int16))
        print(f"[v109 conditional] features year={year} rows={len(rows)}", flush=True)

    final_bank = build_bank(raw, v97_config["conditional_strength"])
    fit = pd.concat([by_year[year] for year in years], ignore_index=True)
    audit = feature_frame(test.reset_index(drop=True), final_bank)
    fit, audit, categories = align_categories(fit, audit)
    baseline_fit = fit.drop(columns=list(CONDITIONAL_COLUMNS))
    baseline_audit = audit.drop(columns=list(CONDITIONAL_COLUMNS))
    target = np.concatenate(target_parts)
    seasons = np.concatenate(season_parts)
    audit_year = max(years) + 1
    weights = np.exp2(
        -((audit_year - 1) - seasons) / float(v97_config["recency_half_life"])
    )

    models: dict[str, lgb.LGBMClassifier] = {}
    for name, fit_frame in (("conditional", fit), ("baseline", baseline_fit)):
        print(f"[v109 conditional] fit {name} rows={len(fit_frame)}", flush=True)
        model = lgb.LGBMClassifier(**v97_config["model"])
        model.fit(
            fit_frame,
            target,
            sample_weight=weights,
            categorical_feature=list(CATEGORICAL),
        )
        models[name] = model

    output_dir.mkdir(parents=True, exist_ok=True)
    # LightGBM's Windows C API cannot write directly to a Unicode path.  Let
    # Python perform the UTF-8 file write so Korean workspace paths remain safe.
    for name, model in models.items():
        # LightGBM records byte offsets in tree_sizes.  Path.write_text() uses
        # CRLF on Windows, which changes those offsets and corrupts model_file
        # loading even though model_str loading appears valid after universal
        # newline conversion.  Preserve the booster's LF bytes verbatim.
        (output_dir / f"{name}_lgb.txt").write_bytes(
            model.booster_.model_to_string().encode("utf-8")
        )
    joblib.dump(final_bank, output_dir / "bank.joblib", compress=3)
    spec = {
        "protocol": "V109_CONDITIONAL_PAIRED_LGB_EXPORT_V1",
        "conditional": {
            "feature_columns": list(fit.columns),
            "categories": categories,
        },
        "baseline": {
            "feature_columns": list(baseline_fit.columns),
            "categories": categories,
        },
        "conditional_columns": list(CONDITIONAL_COLUMNS),
        "audit_year": audit_year,
        "source_years": years,
        "row_local_inference": True,
        "test_aggregate_used": False,
    }
    (output_dir / "feature_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    parity: dict[str, float] = {}
    for name, audit_frame in (("conditional", audit), ("baseline", baseline_audit)):
        expected = models[name].predict_proba(audit_frame)[:, 1]
        loaded = lgb.Booster(
            model_str=(output_dir / f"{name}_lgb.txt").read_text(encoding="utf-8")
        )
        rebuilt = feature_frame(test.reset_index(drop=True), final_bank)
        if name == "baseline":
            rebuilt = rebuilt.drop(columns=list(CONDITIONAL_COLUMNS))
        rebuilt = apply_model_spec(rebuilt, spec[name])
        actual = loaded.predict(rebuilt)
        parity[name] = float(np.max(np.abs(expected - actual)))
        if parity[name] > 1e-12:
            raise ValueError(f"LightGBM export parity failure: {name}/{parity[name]}")
    return {
        "source_rows": int(len(fit)),
        "source_years": years,
        "audit_rows": int(len(test)),
        "feature_counts": {
            "conditional": int(fit.shape[1]),
            "baseline": int(baseline_fit.shape[1]),
        },
        "export_max_abs_parity": parity,
        "bank_global_rate": float(final_bank["global"]),
    }


def build_package(
    parent_zip: Path,
    assets: Path,
    wrapper: Path,
    feature_component: Path,
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
    shutil.copyfile(feature_component, staging / "model" / "components" / "v104_features.py")
    shutil.copytree(assets / "r_fm", staging / "model" / "r_fm")
    shutil.copytree(assets / "conditional", staging / "model" / "conditional")
    shutil.copyfile(wrapper, staging / "script.py")
    shutil.copyfile(parent_stage / "requirements.txt", staging / "requirements.txt")
    package = output_dir / "submit_v104_probe.zip"
    result = _zip_directory(staging, package)
    _safe_remove_generated(staging, output_dir)
    _safe_remove_generated(parent_stage, output_dir)
    return package, result


def _mixed_smoke(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    mixed = pd.concat([source] * 4, ignore_index=True)
    mixed[ID_COL] = [f"v104_probe_{index:03d}" for index in range(len(mixed))]
    # Keep every history-bearing field internally consistent with the official
    # 2025 smoke rows.  The champion intentionally rejects synthetic rows whose
    # as-of counts predate its stored 2024 history.  Route protection can be
    # exercised safely by changing game_type only; the untouched official rows
    # already cover both active and inactive v104 stability masks.
    group = np.arange(len(mixed)) % 4
    mixed.loc[group == 3, "game_type"] = "F"
    sample = pd.DataFrame({ID_COL: mixed[ID_COL], TARGET_COL: 0.0})
    return mixed, sample


def audit_package(
    package: Path,
    parent_zip: Path,
    data_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    mixed, sample = _mixed_smoke(data_dir)
    parent_out, parent_seconds, _ = run_package(
        parent_zip, mixed, sample, timeout=timeout
    )
    candidate_out, candidate_seconds, stdout = run_package(
        package, mixed, sample, timeout=timeout
    )
    parent = mixed[ID_COL].map(parent_out.set_index(ID_COL)[TARGET_COL]).to_numpy(float)
    candidate = mixed[ID_COL].map(candidate_out.set_index(ID_COL)[TARGET_COL]).to_numpy(float)
    regular = mixed["game_type"].eq("R").to_numpy()
    core = regular & mixed["pitcher_team_id"].ne(13).to_numpy() & mixed[
        "batter_team_id"
    ].ne(13).to_numpy()
    active = core & stability_mask(mixed)
    protected_max_abs = float(np.max(np.abs(candidate[~active] - parent[~active])))
    active_shift_count = int(np.count_nonzero(np.abs(candidate[active] - parent[active]) > 0.0))
    if protected_max_abs > 1e-12 or active_shift_count == 0:
        raise ValueError(
            f"v104 route/mask audit failed: protected={protected_max_abs}, shifted={active_shift_count}"
        )

    shuffled_test = mixed.sample(frac=1.0, random_state=109).reset_index(drop=True)
    shuffled_sample = sample.sample(frac=1.0, random_state=110).reset_index(drop=True)
    shuffled, _, _ = run_package(package, shuffled_test, shuffled_sample, timeout=timeout)
    base_map = candidate_out.set_index(ID_COL)[TARGET_COL].sort_index()
    shuffled_map = shuffled.set_index(ID_COL)[TARGET_COL].sort_index()
    shuffled_max_abs = float(np.max(np.abs(base_map - shuffled_map)))
    if shuffled_max_abs > 1e-12:
        raise ValueError(f"v104 shuffled-row parity failure: {shuffled_max_abs}")

    parts = []
    for index in np.array_split(np.arange(len(mixed)), 2):
        local_test = mixed.iloc[index].reset_index(drop=True)
        local_sample = pd.DataFrame({ID_COL: local_test[ID_COL], TARGET_COL: 0.0})
        output, _, _ = run_package(package, local_test, local_sample, timeout=timeout)
        parts.append(output)
    partition_map = pd.concat(parts).set_index(ID_COL)[TARGET_COL].sort_index()
    partition_max_abs = float(np.max(np.abs(base_map - partition_map)))
    if partition_max_abs > 1e-12:
        raise ValueError(f"v104 partition parity failure: {partition_max_abs}")

    scale_rows = 245789
    scale = mixed.iloc[np.arange(scale_rows) % len(mixed)].reset_index(drop=True)
    scale[ID_COL] = [f"v104_scale_{index:06d}" for index in range(scale_rows)]
    scale_sample = pd.DataFrame({ID_COL: scale[ID_COL], TARGET_COL: 0.0})
    scale_output, scale_seconds, scale_stdout = run_package(
        package, scale, scale_sample, timeout=timeout
    )
    values = scale_output[TARGET_COL].to_numpy(float)
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("v104 scale proxy produced invalid probabilities")
    return {
        "mixed_rows": int(len(mixed)),
        "active_mask_rows": int(active.sum()),
        "active_shift_count": active_shift_count,
        "protected_max_abs": protected_max_abs,
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
    project: Path,
    train_csv: Path,
    data_dir: Path,
    parent_zip: Path,
    v97_config_path: Path,
    config_path: Path,
    output_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    project = project.resolve()
    train_csv = train_csv.resolve()
    data_dir = data_dir.resolve()
    parent_zip = parent_zip.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    v97_config = json.loads(v97_config_path.read_text(encoding="utf-8"))
    raw = pd.read_csv(train_csv, low_memory=False)
    test = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")

    assets = output_dir / "_training_assets"
    _safe_remove_generated(assets, output_dir)
    (assets / "r_fm").mkdir(parents=True)
    full23 = raw.loc[raw["season"].eq(2023)].reset_index(drop=True)
    meta23 = _metadata(project, 2023)
    if not np.array_equal(meta23["target"], full23[TARGET].to_numpy(float)):
        raise ValueError("full-2023 parent/target mismatch")
    axes = _cached_v25_axes(project, raw)
    late24 = axes["replication_late_2024"].reset_index(drop=True)
    late24_parent = v27_parent(late24)
    fm_audit = {
        "older": train_export_independent_fm(
            full23, meta23["parent"], assets / "r_fm" / "older", "full_2023"
        ),
        "recent": train_export_independent_fm(
            late24, late24_parent, assets / "r_fm" / "recent", "late_2024"
        ),
    }
    conditional_audit = train_export_conditional(
        raw, test, v97_config, assets / "conditional"
    )
    package, package_result = build_package(
        parent_zip,
        assets,
        Path(__file__).with_name("v109_v104_probe_wrapper.py"),
        Path(__file__).with_name("v109_v104_feature_component.py"),
        output_dir,
    )
    package_audit = audit_package(package, parent_zip, data_dir, timeout)
    result = {
        "protocol": PROTOCOL,
        "config": config,
        "train_csv_sha256": _file_sha256(train_csv),
        "parent_zip_sha256": _sha256(parent_zip),
        "fm_training": fm_audit,
        "conditional_training": conditional_audit,
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
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--parent-zip", type=Path, required=True)
    parser.add_argument("--v97-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    run(
        args.project,
        args.train_csv,
        args.data_dir,
        args.parent_zip,
        args.v97_config,
        args.config,
        args.output_dir,
        args.timeout,
    )


if __name__ == "__main__":
    main()
