from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from agent_service.compliance import validate_external_data_manifest
from agent_service.config import load_config
from agent_service.submission import CandidateValidator
from experiments.kma_um_power_curve_gate import (
    CAPACITY,
    COMPONENTS,
    TARGET,
    GatePolicy,
    _compare,
    _fit_direct_for_test,
    apply_bounded_combo,
    build_proxy_labels,
    fit_direct_power,
    issue_block_bootstrap,
    load_gfs_850_speed,
)


Q1_START = pd.Timestamp("2024-01-01 00:00:00")
Q2_START = pd.Timestamp("2024-04-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")
END = pd.Timestamp("2025-01-01 00:00:00")
KEYS = ("forecast_kst_dtm", "data_available_kst_dtm")
CONTEXT_COLUMNS = (
    "kma_um_ctx_u10_r0",
    "kma_um_ctx_v10_r0",
    "kma_um_ctx_speed10_r0",
    "kma_um_ctx_shear10_850_r0",
    "kma_um_ctx_cos10_850_r0",
    "kma_um_ctx_run_change10",
)
REGIME_FAMILIES = (
    "direction4",
    "shear3",
    "alignment2",
    "run_change3",
    "direction4_shear2",
    "direction4_alignment2",
)


@dataclass(frozen=True)
class AdjustmentPolicy:
    regime_family: str
    direction: str
    coverage: float
    minimum_base_ratio: float
    maximum_base_ratio: float
    minimum_regime_difference_kwh: float | None
    beta: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_context_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=[*KEYS, *CONTEXT_COLUMNS],
    )
    for key in KEYS:
        frame[key] = pd.to_datetime(frame[key])
    if frame.duplicated(list(KEYS)).any():
        raise ValueError("KMA context rows are not unique by forecast and issue")
    if frame["forecast_kst_dtm"].duplicated().any():
        raise ValueError("KMA context contains multiple issues for a target")
    for column in CONTEXT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    values = frame.loc[:, CONTEXT_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("KMA regime context contains missing or non-finite values")
    return frame.set_index("forecast_kst_dtm").sort_index()


def _direction4(context: pd.DataFrame) -> np.ndarray:
    angle = np.arctan2(
        context["kma_um_ctx_v10_r0"].to_numpy(dtype=float),
        context["kma_um_ctx_u10_r0"].to_numpy(dtype=float),
    )
    return np.floor((angle + np.pi) / (0.5 * np.pi)).astype(int) % 4


def _quantile_edges(values: np.ndarray, quantiles: tuple[float, ...]) -> list[float]:
    edges = [float(value) for value in np.quantile(values, quantiles)]
    if not np.isfinite(edges).all():
        raise ValueError("regime quantile edge is non-finite")
    return edges


def regime_codes(
    family: str,
    train_context: pd.DataFrame,
    train_mask: np.ndarray,
    predict_context: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if family not in REGIME_FAMILIES:
        raise ValueError(f"unknown regime family: {family}")
    if len(train_context) != len(train_mask):
        raise ValueError("regime training mask is misaligned")
    train_direction = _direction4(train_context)
    predict_direction = _direction4(predict_context)
    metadata: dict[str, object] = {"family": family}

    if family == "direction4":
        return train_direction, predict_direction, metadata

    if family in {"shear3", "direction4_shear2"}:
        train_values = train_context["kma_um_ctx_shear10_850_r0"].to_numpy(
            dtype=float
        )
        predict_values = predict_context[
            "kma_um_ctx_shear10_850_r0"
        ].to_numpy(dtype=float)
        quantiles = (0.5,) if family.endswith("shear2") else (1 / 3, 2 / 3)
        edges = _quantile_edges(train_values[train_mask], quantiles)
        train_shear = np.digitize(train_values, edges)
        predict_shear = np.digitize(predict_values, edges)
        metadata["edges"] = edges
        if family == "shear3":
            return train_shear, predict_shear, metadata
        return (
            train_direction * 2 + train_shear,
            predict_direction * 2 + predict_shear,
            metadata,
        )

    if family in {"alignment2", "direction4_alignment2"}:
        threshold = 0.95
        train_alignment = (
            train_context["kma_um_ctx_cos10_850_r0"].to_numpy(dtype=float)
            >= threshold
        ).astype(int)
        predict_alignment = (
            predict_context["kma_um_ctx_cos10_850_r0"].to_numpy(dtype=float)
            >= threshold
        ).astype(int)
        metadata["threshold"] = threshold
        if family == "alignment2":
            return train_alignment, predict_alignment, metadata
        return (
            train_direction * 2 + train_alignment,
            predict_direction * 2 + predict_alignment,
            metadata,
        )

    train_values = np.abs(
        train_context["kma_um_ctx_run_change10"].to_numpy(dtype=float)
    )
    predict_values = np.abs(
        predict_context["kma_um_ctx_run_change10"].to_numpy(dtype=float)
    )
    edges = _quantile_edges(train_values[train_mask], (1 / 3, 2 / 3))
    metadata["edges"] = edges
    return (
        np.digitize(train_values, edges),
        np.digitize(predict_values, edges),
        metadata,
    )


def fit_regime_power(
    train_wind: np.ndarray,
    proxy_label: np.ndarray,
    train_context: pd.DataFrame,
    train_mask: np.ndarray,
    predict_wind: np.ndarray,
    predict_context: pd.DataFrame,
    family: str,
    *,
    minimum_regime_rows: int,
    shrinkage_rows: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if int(train_mask.sum()) < 1_000:
        raise ValueError("regime power curve needs at least 1,000 training rows")
    if minimum_regime_rows < 50:
        raise ValueError("minimum_regime_rows must be at least 50")
    if shrinkage_rows <= 0:
        raise ValueError("shrinkage_rows must be positive")
    global_model = IsotonicRegression(
        y_min=0.0, y_max=CAPACITY, out_of_bounds="clip"
    )
    global_model.fit(train_wind[train_mask], proxy_label[train_mask])
    global_prediction = np.asarray(
        global_model.predict(predict_wind), dtype=float
    )
    train_codes, predict_codes, metadata = regime_codes(
        family, train_context, train_mask, predict_context
    )
    prediction = global_prediction.copy()
    regime_rows: dict[str, object] = {}
    for code in sorted(np.unique(train_codes[train_mask])):
        local_train = train_mask & (train_codes == code)
        count = int(local_train.sum())
        local_predict = predict_codes == code
        if count < minimum_regime_rows or not local_predict.any():
            regime_rows[str(int(code))] = {
                "train_rows": count,
                "prediction_rows": int(local_predict.sum()),
                "fitted": False,
                "local_weight": 0.0,
            }
            continue
        local_model = IsotonicRegression(
            y_min=0.0, y_max=CAPACITY, out_of_bounds="clip"
        )
        local_model.fit(train_wind[local_train], proxy_label[local_train])
        local_prediction = np.asarray(
            local_model.predict(predict_wind[local_predict]), dtype=float
        )
        local_weight = float(count / (count + shrinkage_rows))
        prediction[local_predict] = (
            local_weight * local_prediction
            + (1.0 - local_weight) * global_prediction[local_predict]
        )
        regime_rows[str(int(code))] = {
            "train_rows": count,
            "prediction_rows": int(local_predict.sum()),
            "fitted": True,
            "local_weight": local_weight,
        }
    if not np.isfinite(prediction).all():
        raise ValueError("regime power curve produced non-finite predictions")
    return prediction, global_prediction, {
        **metadata,
        "minimum_regime_rows": minimum_regime_rows,
        "shrinkage_rows": shrinkage_rows,
        "regimes": regime_rows,
    }


def adjustment_gate(
    regime_difference: np.ndarray,
    reference: np.ndarray,
    base_kma_gate: np.ndarray,
    policy: AdjustmentPolicy,
) -> np.ndarray:
    if policy.direction == "up":
        direction = regime_difference > 0.0
    elif policy.direction == "down":
        direction = regime_difference < 0.0
    elif policy.direction == "both":
        direction = np.ones(len(reference), dtype=bool)
    else:
        raise ValueError(f"unknown adjustment direction: {policy.direction}")
    threshold = (
        -np.inf
        if policy.minimum_regime_difference_kwh is None
        else policy.minimum_regime_difference_kwh
    )
    ratio = reference / CAPACITY
    return (
        base_kma_gate
        & direction
        & (np.abs(regime_difference) >= threshold)
        & (ratio >= policy.minimum_base_ratio)
        & (ratio <= policy.maximum_base_ratio)
    )


def apply_regime_adjustment(
    public_reference: np.ndarray,
    incumbent: np.ndarray,
    global_direct: np.ndarray,
    regime_direct: np.ndarray,
    base_kma_gate: np.ndarray,
    policy: AdjustmentPolicy,
    *,
    kma_alpha: float,
    maximum_total_movement_ratio: float,
    maximum_incremental_movement_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    regime_difference = regime_direct - global_direct
    gate = adjustment_gate(
        regime_difference, public_reference, base_kma_gate, policy
    )
    total_bound = maximum_total_movement_ratio * CAPACITY
    incremental_bound = maximum_incremental_movement_ratio * CAPACITY
    desired = public_reference + np.clip(
        kma_alpha * (regime_direct - public_reference),
        -total_bound,
        total_bound,
    )
    increment = np.clip(
        policy.beta * (desired - incumbent),
        -incremental_bound,
        incremental_bound,
    )
    candidate = incumbent.copy()
    candidate[gate] = incumbent[gate] + increment[gate]
    candidate[gate] = np.clip(
        candidate[gate],
        public_reference[gate] - total_bound,
        public_reference[gate] + total_bound,
    )
    candidate = np.clip(candidate, 0.0, CAPACITY)
    return candidate, gate


def select_adjustment_policy(
    truth: np.ndarray,
    public_reference: np.ndarray,
    incumbent: np.ndarray,
    global_direct: np.ndarray,
    regime_direct: np.ndarray,
    base_kma_gate: np.ndarray,
    selection: np.ndarray,
    family: str,
    *,
    kma_alpha: float,
    maximum_total_movement_ratio: float,
    maximum_incremental_movement_ratio: float,
    minimum_changed_rows: int,
    maximum_changed_ratio: float,
) -> dict[str, object] | None:
    difference = regime_direct - global_direct
    rows: list[dict[str, object]] = []
    for direction_name in ("both", "up", "down"):
        if direction_name == "up":
            direction = difference > 0.0
        elif direction_name == "down":
            direction = difference < 0.0
        else:
            direction = np.ones(len(difference), dtype=bool)
        pool_mask = selection & base_kma_gate & direction
        if not pool_mask.any():
            continue
        pool = np.abs(difference[pool_mask])
        for coverage in (1.0, 0.50, 0.25, 0.10, 0.05):
            threshold = (
                None
                if coverage == 1.0
                else float(np.quantile(pool, 1.0 - coverage))
            )
            for minimum_ratio, maximum_ratio in (
                (0.10, 0.80),
                (0.20, 0.80),
                (0.40, 0.80),
            ):
                for beta in (0.10, 0.20, 0.35, 0.50, 0.75, 1.00):
                    policy = AdjustmentPolicy(
                        family,
                        direction_name,
                        coverage,
                        minimum_ratio,
                        maximum_ratio,
                        threshold,
                        beta,
                    )
                    candidate, gate = apply_regime_adjustment(
                        public_reference,
                        incumbent,
                        global_direct,
                        regime_direct,
                        base_kma_gate,
                        policy,
                        kma_alpha=kma_alpha,
                        maximum_total_movement_ratio=maximum_total_movement_ratio,
                        maximum_incremental_movement_ratio=(
                            maximum_incremental_movement_ratio
                        ),
                    )
                    changed_rows = int((gate & selection).sum())
                    if changed_rows < minimum_changed_rows:
                        continue
                    if changed_rows / int(selection.sum()) > maximum_changed_ratio:
                        continue
                    comparison = _compare(
                        truth, incumbent, candidate, selection
                    )
                    delta = comparison["delta"]
                    if min(delta[component] for component in COMPONENTS) <= 0.0:
                        continue
                    rows.append(
                        {
                            "policy": policy,
                            "comparison": comparison,
                            "changed_rows": changed_rows,
                            "mean_absolute_increment_kwh": float(
                                np.mean(
                                    np.abs(candidate[selection] - incumbent[selection])
                                )
                            ),
                        }
                    )
    if not rows:
        return None
    rows.sort(
        key=lambda row: (
            row["comparison"]["delta"]["score"],
            min(
                row["comparison"]["delta"]["one_minus_nmae"],
                row["comparison"]["delta"]["ficr"],
            ),
            -row["mean_absolute_increment_kwh"],
        ),
        reverse=True,
    )

    def serialize(row: dict[str, object]) -> dict[str, object]:
        return {
            **row,
            "policy": row["policy"].to_dict(),
        }

    return {
        "selected": rows[0],
        "top": [serialize(row) for row in rows[:10]],
        "eligible_policies": len(rows),
    }


def _gate_policy(raw: dict[str, object]) -> GatePolicy:
    return GatePolicy(
        direction=str(raw["direction"]),
        coverage=float(raw["coverage"]),
        minimum_base_ratio=float(raw["minimum_base_ratio"]),
        maximum_base_ratio=float(raw["maximum_base_ratio"]),
        minimum_disagreement_kwh=(
            None
            if raw.get("minimum_disagreement_kwh") is None
            else float(raw["minimum_disagreement_kwh"])
        ),
    )


def _recreate_incumbent(
    public_reference: np.ndarray,
    kma_direct: np.ndarray,
    gfs_direct: np.ndarray,
    available: np.ndarray,
    kma_policy: GatePolicy,
    gfs_policy: GatePolicy,
    *,
    kma_alpha: float,
    gfs_alpha: float,
    maximum_total_movement_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    _, candidate, _, kma_gate = apply_bounded_combo(
        public_reference,
        kma_direct - public_reference,
        gfs_direct - public_reference,
        available,
        kma_policy,
        gfs_policy,
        kma_alpha=kma_alpha,
        gfs_alpha=gfs_alpha,
        maximum_movement_ratio=maximum_total_movement_ratio,
    )
    return candidate, kma_gate


def _month_deltas(
    truth: np.ndarray,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    index: pd.DatetimeIndex,
    h2: np.ndarray,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for month in range(7, 13):
        mask = h2 & np.asarray(index.month == month)
        if int(mask.sum()) < 24:
            continue
        output[str(month)] = _compare(
            truth, incumbent, candidate, mask
        )["delta"]
    return output


def _write_submission(
    *,
    qualified: bool,
    train_context: pd.DataFrame,
    train_wind: np.ndarray,
    train_gfs: np.ndarray,
    proxy_label: np.ndarray,
    train_mask: np.ndarray,
    family: str,
    policy: AdjustmentPolicy,
    minimum_regime_rows: int,
    shrinkage_rows: float,
    kma_policy: GatePolicy,
    gfs_policy: GatePolicy,
    kma_alpha: float,
    gfs_alpha: float,
    maximum_total_movement_ratio: float,
    maximum_incremental_movement_ratio: float,
    test_manifest: Path,
    test_context_path: Path,
    test_gfs_path: Path,
    gate_reference_submission: Path,
    incumbent_submission: Path,
    output_submission: Path,
) -> dict[str, object] | None:
    if not qualified:
        return None
    validate_external_data_manifest(test_manifest, Path.cwd().resolve())
    test_context = load_context_features(test_context_path)
    test_wind_series = test_context["kma_um_ctx_speed10_r0"]
    test_gfs_series = load_gfs_850_speed(test_gfs_path)
    incumbent_frame = pd.read_csv(incumbent_submission, encoding="utf-8-sig")
    gate_reference_frame = pd.read_csv(
        gate_reference_submission, encoding="utf-8-sig"
    )
    timestamp_column = "forecast_kst_dtm"
    if timestamp_column not in incumbent_frame or TARGET not in incumbent_frame:
        raise ValueError("incumbent submission lacks timestamp or group-3 output")
    if not incumbent_frame[[timestamp_column]].equals(
        gate_reference_frame[[timestamp_column]]
    ):
        raise ValueError("gate-reference and incumbent submission IDs are misaligned")
    test_index = pd.DatetimeIndex(
        pd.to_datetime(incumbent_frame[timestamp_column])
    )
    predict_context = test_context.reindex(test_index)
    predict_wind = test_wind_series.reindex(test_index).to_numpy(dtype=float)
    predict_gfs = test_gfs_series.reindex(test_index).to_numpy(dtype=float)
    if (
        predict_context.isna().any().any()
        or not np.isfinite(predict_wind).all()
        or not np.isfinite(predict_gfs).all()
    ):
        raise ValueError("test KMA/GFS regime coverage is incomplete")
    regime_direct, global_direct, regime_metadata = fit_regime_power(
        train_wind,
        proxy_label,
        train_context,
        train_mask,
        predict_wind,
        predict_context,
        family,
        minimum_regime_rows=minimum_regime_rows,
        shrinkage_rows=shrinkage_rows,
    )
    global_gfs = _fit_direct_for_test(
        train_wind=train_gfs,
        proxy_label=proxy_label,
        train_mask=train_mask,
        test_wind=predict_gfs,
    )
    public_reference = gate_reference_frame[TARGET].to_numpy(dtype=float)
    recreated, kma_gate = _recreate_incumbent(
        public_reference,
        global_direct,
        global_gfs,
        np.ones(len(test_index), dtype=bool),
        kma_policy,
        gfs_policy,
        kma_alpha=kma_alpha,
        gfs_alpha=gfs_alpha,
        maximum_total_movement_ratio=maximum_total_movement_ratio,
    )
    incumbent = incumbent_frame[TARGET].to_numpy(dtype=float)
    lineage_error = float(np.max(np.abs(recreated - incumbent)))
    if lineage_error > 1e-6:
        raise ValueError(
            f"test incumbent lineage mismatch: {lineage_error:.9f} kWh"
        )
    candidate, gate = apply_regime_adjustment(
        public_reference,
        incumbent,
        global_direct,
        regime_direct,
        kma_gate,
        policy,
        kma_alpha=kma_alpha,
        maximum_total_movement_ratio=maximum_total_movement_ratio,
        maximum_incremental_movement_ratio=maximum_incremental_movement_ratio,
    )
    output = incumbent_frame.copy()
    output[TARGET] = candidate
    output_submission.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_submission, index=False, encoding="utf-8-sig")
    audit = CandidateValidator(load_config(Path.cwd())).audit(output_submission)
    if not audit.valid:
        raise RuntimeError(f"CandidateValidator rejected output: {audit.errors}")
    movement = candidate - incumbent
    return {
        "path": output_submission.as_posix(),
        "sha256": hashlib.sha256(output_submission.read_bytes()).hexdigest(),
        "rows": len(output),
        "changed_rows": int((np.abs(movement) > 1e-9).sum()),
        "maximum_absolute_increment_kwh": float(np.max(np.abs(movement))),
        "mean_absolute_increment_kwh": float(np.mean(np.abs(movement))),
        "incumbent_lineage_max_absolute_error_kwh": lineage_error,
        "regime_metadata": regime_metadata,
        "candidate_validator": audit.to_dict(),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    validate_external_data_manifest(Path(args.manifest), Path.cwd().resolve())
    context = load_context_features(Path(args.context_features))
    gfs_series = load_gfs_850_speed(Path(args.gfs), args.gfs_grid_id)
    oof = np.load(args.power_curve_oof)
    driver = np.load(args.driver_cache)
    index = pd.DatetimeIndex(pd.to_datetime(oof["index_ns"]))
    driver_index = pd.DatetimeIndex(
        pd.to_datetime(driver[f"{TARGET}__valid_index_ns"])
    )
    if not driver_index.equals(index):
        raise ValueError("driver and power-curve OOF indexes differ")
    aligned_context = context.reindex(index)
    context_available = (
        aligned_context.loc[:, CONTEXT_COLUMNS].notna().all(axis=1).to_numpy()
    )
    evaluation_year = np.asarray(
        (index >= Q1_START) & (index < END)
    )
    if np.any(~context_available & evaluation_year):
        raise ValueError("KMA regime context does not cover the 2024 OOF period")
    # The exact driver lineage retains a 2025-01-01 00:00 boundary diagnostic
    # outside the 2024 KMA archive. It is never train/selection/locked eligible,
    # but finite placeholders keep vectorized model prediction well-defined.
    aligned_context = aligned_context.copy()
    for column in CONTEXT_COLUMNS:
        aligned_context[column] = aligned_context[column].fillna(
            context[column].median()
        )
    kma = aligned_context["kma_um_ctx_speed10_r0"].to_numpy(dtype=float)
    gfs = gfs_series.reindex(index).to_numpy(dtype=float)
    issues = pd.to_datetime(oof["issue_ns"])
    truth = oof["truth"].astype(float)
    public_reference = oof["public_fine"].astype(float)
    static_incumbent = oof["static_candidate"].astype(float)
    rolling_incumbent = oof["rolling_candidate"].astype(float)
    available = (
        oof["available"].astype(bool)
        & context_available
        & np.isfinite(kma)
        & np.isfinite(gfs)
        & pd.notna(issues)
    )
    q1 = available & np.asarray(
        (index >= Q1_START) & (index < Q2_START)
    )
    q2 = available & np.asarray(
        (index >= Q2_START) & (index < H2_START)
    )
    h1 = available & np.asarray(
        (index >= Q1_START) & (index < H2_START)
    )
    h2 = available & np.asarray((index >= H2_START) & (index < END))
    within_year = available & np.asarray(
        (index >= Q1_START) & (index < END)
    )
    if int(within_year.sum()) < args.minimum_common_rows:
        raise ValueError("regime experiment lacks minimum 2024 coverage")

    source_report = json.loads(
        Path(args.power_curve_report).read_text(encoding="utf-8")
    )
    proxy_name = str(source_report["selected_proxy"])
    proxy_labels = build_proxy_labels(driver)
    proxy_label = proxy_labels[proxy_name]
    kma_policy = _gate_policy(source_report["selected_kma_gate"])
    gfs_policy = _gate_policy(source_report["selected_gfs_control_gate"])
    combo = source_report["selected_bounded_combo"]
    kma_alpha = float(combo["kma_alpha"])
    gfs_alpha = float(combo["gfs_alpha"])

    q1_global_kma = fit_direct_power(
        kma, proxy_label, q1, available, public_reference
    )
    q1_global_gfs = fit_direct_power(
        gfs, proxy_label, q1, available, public_reference
    )
    recreated_static, q1_kma_gate = _recreate_incumbent(
        public_reference,
        q1_global_kma,
        q1_global_gfs,
        available,
        kma_policy,
        gfs_policy,
        kma_alpha=kma_alpha,
        gfs_alpha=gfs_alpha,
        maximum_total_movement_ratio=args.maximum_total_movement_ratio,
    )
    static_lineage_error = float(
        np.max(np.abs(recreated_static[within_year] - static_incumbent[within_year]))
    )
    if static_lineage_error > args.maximum_lineage_error_kwh:
        raise ValueError(
            f"static KMA incumbent lineage mismatch: {static_lineage_error}"
        )

    development: list[dict[str, object]] = []
    retained: dict[str, Any] | None = None
    for family in REGIME_FAMILIES:
        regime_direct, global_direct, metadata = fit_regime_power(
            kma,
            proxy_label,
            aligned_context,
            q1,
            kma,
            aligned_context,
            family,
            minimum_regime_rows=args.minimum_regime_rows,
            shrinkage_rows=args.shrinkage_rows,
        )
        global_error = float(
            np.max(np.abs(global_direct[within_year] - q1_global_kma[within_year]))
        )
        if global_error > 1e-6:
            raise ValueError("regime global curve does not match source curve")
        selection = select_adjustment_policy(
            truth,
            public_reference,
            static_incumbent,
            global_direct,
            regime_direct,
            q1_kma_gate,
            q2,
            family,
            kma_alpha=kma_alpha,
            maximum_total_movement_ratio=args.maximum_total_movement_ratio,
            maximum_incremental_movement_ratio=(
                args.maximum_incremental_movement_ratio
            ),
            minimum_changed_rows=args.minimum_selection_rows,
            maximum_changed_ratio=args.maximum_changed_h2_ratio,
        )
        if selection is None:
            development.append(
                {
                    "regime_family": family,
                    "selection_status": "no_all-component-positive_policy",
                    "regime_metadata": metadata,
                }
            )
            continue
        selected = selection["selected"]
        serialized_selected = {
            **selected,
            "policy": selected["policy"].to_dict(),
        }
        development.append(
            {
                "regime_family": family,
                "selection_status": "passed",
                "regime_metadata": metadata,
                "selected": serialized_selected,
                "top": selection["top"],
                "eligible_policies": selection["eligible_policies"],
            }
        )
        key = (
            selected["comparison"]["delta"]["score"],
            min(
                selected["comparison"]["delta"]["one_minus_nmae"],
                selected["comparison"]["delta"]["ficr"],
            ),
            -selected["mean_absolute_increment_kwh"],
        )
        if retained is None or key > retained["key"]:
            retained = {
                "key": key,
                "family": family,
                "policy": selected["policy"],
                "q1_regime_direct": regime_direct,
                "q1_global_direct": global_direct,
                "q1_metadata": metadata,
                "selection": serialized_selected,
            }
    if retained is None:
        report = {
            "family": "kma_group3_regime_conditioned_power_curve",
            "source_manifest": args.manifest,
            "public_result_trigger": {
                "submission_id": 1501731,
                "score_delta_vs_incumbent": -0.0001599771,
                "decision": "close group-2 supplement and switch to group-3",
            },
            "development": development,
            "decision": {
                "tier": "rejected",
                "submission_eligible": False,
                "reason": "no Q2 policy improved score, 1-NMAE, and FICR together",
            },
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return report

    policy: AdjustmentPolicy = retained["policy"]
    static_candidate, static_gate = apply_regime_adjustment(
        public_reference,
        static_incumbent,
        retained["q1_global_direct"],
        retained["q1_regime_direct"],
        q1_kma_gate,
        policy,
        kma_alpha=kma_alpha,
        maximum_total_movement_ratio=args.maximum_total_movement_ratio,
        maximum_incremental_movement_ratio=(
            args.maximum_incremental_movement_ratio
        ),
    )
    static_locked = _compare(
        truth, static_incumbent, static_candidate, h2
    )

    h1_regime_direct, h1_global_kma, h1_metadata = fit_regime_power(
        kma,
        proxy_label,
        aligned_context,
        h1,
        kma,
        aligned_context,
        retained["family"],
        minimum_regime_rows=args.minimum_regime_rows,
        shrinkage_rows=args.shrinkage_rows,
    )
    h1_global_gfs = fit_direct_power(
        gfs, proxy_label, h1, available, public_reference
    )
    recreated_rolling, h1_kma_gate = _recreate_incumbent(
        public_reference,
        h1_global_kma,
        h1_global_gfs,
        available,
        kma_policy,
        gfs_policy,
        kma_alpha=kma_alpha,
        gfs_alpha=gfs_alpha,
        maximum_total_movement_ratio=args.maximum_total_movement_ratio,
    )
    rolling_lineage_error = float(
        np.max(
            np.abs(
                recreated_rolling[within_year] - rolling_incumbent[within_year]
            )
        )
    )
    if rolling_lineage_error > args.maximum_lineage_error_kwh:
        raise ValueError(
            f"rolling KMA incumbent lineage mismatch: {rolling_lineage_error}"
        )
    locked_candidate, locked_gate = apply_regime_adjustment(
        public_reference,
        rolling_incumbent,
        h1_global_kma,
        h1_regime_direct,
        h1_kma_gate,
        policy,
        kma_alpha=kma_alpha,
        maximum_total_movement_ratio=args.maximum_total_movement_ratio,
        maximum_incremental_movement_ratio=(
            args.maximum_incremental_movement_ratio
        ),
    )
    locked = _compare(truth, rolling_incumbent, locked_candidate, h2)
    monthly = _month_deltas(
        truth, rolling_incumbent, locked_candidate, index, h2
    )
    bootstrap = issue_block_bootstrap(
        truth,
        rolling_incumbent,
        locked_candidate,
        issues,
        h2,
        n_bootstrap=args.n_bootstrap,
        seed=20260726,
    )
    movement = locked_candidate - rolling_incumbent
    changed_h2 = int((locked_gate & h2).sum())
    gates = {
        "q2_all_components_positive": min(
            retained["selection"]["comparison"]["delta"][component]
            for component in COMPONENTS
        )
        > 0.0,
        "static_q1_curve_h2_all_components_positive": min(
            static_locked["delta"][component] for component in COMPONENTS
        )
        > 0.0,
        "rolling_h1_curve_h2_all_components_positive": min(
            locked["delta"][component] for component in COMPONENTS
        )
        > 0.0,
        "all_locked_month_scores_nonnegative": all(
            value["score"] >= 0.0 for value in monthly.values()
        ),
        "all_locked_month_ficr_nonnegative": all(
            value["ficr"] >= 0.0 for value in monthly.values()
        ),
        "bootstrap_q05_all_components_positive": min(
            bootstrap["q05"].values()
        )
        > 0.0,
        "bootstrap_positive_fraction_passed": bootstrap[
            "positive_all_component_fraction"
        ]
        >= args.minimum_bootstrap_positive_fraction,
        "minimum_locked_group3_score_gain_passed": locked["delta"]["score"]
        >= args.minimum_locked_group3_score_gain,
        "maximum_changed_h2_ratio_passed": (
            changed_h2 / int(h2.sum())
        )
        <= args.maximum_changed_h2_ratio,
        "maximum_incremental_movement_passed": float(
            np.max(np.abs(movement[h2]))
        )
        <= args.maximum_incremental_movement_ratio * CAPACITY + 1e-9,
    }
    qualified = bool(all(gates.values()))

    full_regime_direct, full_global_kma, full_metadata = fit_regime_power(
        kma,
        proxy_label,
        aligned_context,
        within_year,
        kma,
        aligned_context,
        retained["family"],
        minimum_regime_rows=args.minimum_regime_rows,
        shrinkage_rows=args.shrinkage_rows,
    )
    del full_regime_direct, full_global_kma
    submission = None
    if args.write_submission_if_qualified:
        submission = _write_submission(
            qualified=qualified,
            train_context=aligned_context,
            train_wind=kma,
            train_gfs=gfs,
            proxy_label=proxy_label,
            train_mask=within_year,
            family=retained["family"],
            policy=policy,
            minimum_regime_rows=args.minimum_regime_rows,
            shrinkage_rows=args.shrinkage_rows,
            kma_policy=kma_policy,
            gfs_policy=gfs_policy,
            kma_alpha=kma_alpha,
            gfs_alpha=gfs_alpha,
            maximum_total_movement_ratio=args.maximum_total_movement_ratio,
            maximum_incremental_movement_ratio=(
                args.maximum_incremental_movement_ratio
            ),
            test_manifest=Path(args.test_manifest),
            test_context_path=Path(args.test_features),
            test_gfs_path=Path(args.test_gfs),
            gate_reference_submission=Path(args.gate_reference_submission),
            incumbent_submission=Path(args.incumbent_submission),
            output_submission=Path(args.output_submission),
        )

    report = {
        "family": "kma_group3_regime_conditioned_power_curve",
        "source_manifest": args.manifest,
        "public_result_trigger": {
            "submission_id": 1501731,
            "score_delta_vs_incumbent": -0.0001599771,
            "one_minus_nmae_delta_vs_incumbent": -0.0000119700,
            "ficr_delta_vs_incumbent": -0.0003079842,
            "decision": "close group-2 supplement and switch to group-3",
        },
        "contract": {
            "q1_model_train": "2024-01-01 through 2024-03-31",
            "q2_policy_selection": "2024-04-01 through 2024-07-01 00:00",
            "locked_h2": "H1-refit; 2024-07-01 01:00 through 2024-12-31 23:00",
            "regime_families": list(REGIME_FAMILIES),
            "minimum_regime_rows": args.minimum_regime_rows,
            "shrinkage_rows": args.shrinkage_rows,
            "maximum_total_movement_ratio": args.maximum_total_movement_ratio,
            "maximum_incremental_movement_ratio": (
                args.maximum_incremental_movement_ratio
            ),
            "maximum_changed_ratio_in_selection_and_locked_h2": (
                args.maximum_changed_h2_ratio
            ),
            "public_score_used_for_selection": False,
        },
        "source_power_curve": {
            "report": args.power_curve_report,
            "oof": args.power_curve_oof,
            "selected_proxy": proxy_name,
            "kma_gate": kma_policy.to_dict(),
            "gfs_gate": gfs_policy.to_dict(),
            "kma_alpha": kma_alpha,
            "gfs_alpha": gfs_alpha,
            "static_lineage_max_absolute_error_kwh": static_lineage_error,
            "rolling_lineage_max_absolute_error_kwh": rolling_lineage_error,
        },
        "development": development,
        "selected_regime_family": retained["family"],
        "selected_policy": policy.to_dict(),
        "selected_q2": retained["selection"],
        "regime_metadata": {
            "q1": retained["q1_metadata"],
            "h1": h1_metadata,
            "full_2024": full_metadata,
        },
        "locked": {
            "static_q1_curve_h2": static_locked,
            "rolling_h1_curve_h2": locked,
            "monthly": monthly,
            "bootstrap": bootstrap,
            "changed_rows": changed_h2,
            "changed_ratio": float(changed_h2 / int(h2.sum())),
            "maximum_absolute_increment_kwh": float(
                np.max(np.abs(movement[h2]))
            ),
            "mean_absolute_increment_kwh": float(
                np.mean(np.abs(movement[h2]))
            ),
        },
        "promotion_gates": gates,
        "decision": {
            "tier": "candidate" if qualified else "rejected",
            "submission_requested": bool(args.write_submission_if_qualified),
            "submission_created": submission is not None,
            "submission_eligible": submission is not None and qualified,
            "reason": (
                "regime-conditioned group-3 candidate passed every locked gate"
                if qualified
                else "regime-conditioned group-3 candidate failed at least one locked gate"
            ),
        },
        "submission": submission,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/manifest.json",
    )
    parser.add_argument(
        "--context-features",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/features.csv",
    )
    parser.add_argument("--gfs", default="data/train/gfs_train.csv")
    parser.add_argument("--gfs-grid-id", type=int, default=5)
    parser.add_argument(
        "--driver-cache",
        default="artifacts_final/lineage/exact_driver_oof.npz",
    )
    parser.add_argument(
        "--power-curve-report",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/power_curve_gate_revalidated_20260725.json",
    )
    parser.add_argument(
        "--power-curve-oof",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/power_curve_oof_20260725.npz",
    )
    parser.add_argument("--minimum-common-rows", type=int, default=8_000)
    parser.add_argument("--minimum-regime-rows", type=int, default=150)
    parser.add_argument("--shrinkage-rows", type=float, default=400.0)
    parser.add_argument("--minimum-selection-rows", type=int, default=24)
    parser.add_argument(
        "--maximum-total-movement-ratio", type=float, default=0.05
    )
    parser.add_argument(
        "--maximum-incremental-movement-ratio", type=float, default=0.01
    )
    parser.add_argument(
        "--maximum-changed-h2-ratio", type=float, default=0.10
    )
    parser.add_argument(
        "--minimum-locked-group3-score-gain", type=float, default=0.002
    )
    parser.add_argument(
        "--minimum-bootstrap-positive-fraction", type=float, default=0.90
    )
    parser.add_argument("--maximum-lineage-error-kwh", type=float, default=0.01)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument(
        "--output",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/group3_regime_power_curve_20260726.json",
    )
    parser.add_argument(
        "--test-manifest",
        default="artifacts_final/external_weather/kma_um_regional_context_2025/manifest.json",
    )
    parser.add_argument(
        "--test-features",
        default="artifacts_final/external_weather/kma_um_regional_context_2025/features.csv",
    )
    parser.add_argument("--test-gfs", default="data/test/gfs_test.csv")
    parser.add_argument(
        "--gate-reference-submission",
        default="submissions/blend_best_crossg3_traj_meta_finesweep.csv",
    )
    parser.add_argument(
        "--incumbent-submission",
        default="artifacts_final/candidates/kma_group2_overlay_alpha2375_20260725.csv",
    )
    parser.add_argument(
        "--output-submission",
        default="artifacts_final/candidates/kma_group3_regime_power_curve_20260726.csv",
    )
    parser.add_argument("--write-submission-if-qualified", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "family": report["family"],
                "selected_regime_family": report.get(
                    "selected_regime_family"
                ),
                "selected_policy": report.get("selected_policy"),
                "locked": report.get("locked"),
                "promotion_gates": report.get("promotion_gates"),
                "decision": report["decision"],
                "submission": report.get("submission"),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
