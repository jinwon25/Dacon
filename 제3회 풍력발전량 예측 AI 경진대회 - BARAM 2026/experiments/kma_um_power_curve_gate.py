from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from agent_service.compliance import validate_external_data_manifest
from experiments.exact_oof_meta_gate_sweep import _prepare_validation
from experiments.kma_um_meta_risk_gate import _rolling_fine_surfaces
from src.metrics import CAPACITY_KWH, MetricResult, evaluate_group


TARGET = "kpx_group_3"
CAPACITY = CAPACITY_KWH[TARGET]
Q1_START = pd.Timestamp("2024-01-01 00:00:00")
Q2_START = pd.Timestamp("2024-04-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")
END = pd.Timestamp("2025-01-01 00:00:00")
KEYS = ["forecast_kst_dtm", "data_available_kst_dtm"]
COMPONENTS = ("score", "one_minus_nmae", "ficr")


@dataclass(frozen=True)
class GatePolicy:
    direction: str
    coverage: float
    minimum_base_ratio: float
    maximum_base_ratio: float
    minimum_disagreement_kwh: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _delta(before: MetricResult, after: MetricResult) -> dict[str, float]:
    return {
        "score": float(after.score - before.score),
        "one_minus_nmae": float(after.one_minus_nmae - before.one_minus_nmae),
        "ficr": float(after.ficr - before.ficr),
    }


def _compare(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    mask_or_positions: np.ndarray,
) -> dict[str, object]:
    before = evaluate_group(
        truth[mask_or_positions], reference[mask_or_positions], CAPACITY
    )
    after = evaluate_group(
        truth[mask_or_positions], candidate[mask_or_positions], CAPACITY
    )
    return {
        "reference": before.to_dict(),
        "candidate": after.to_dict(),
        "delta": _delta(before, after),
    }


def load_context_speed(path: Path) -> tuple[pd.Series, pd.Series]:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=KEYS + ["kma_um_ctx_speed10_r0"],
    )
    for key in KEYS:
        frame[key] = pd.to_datetime(frame[key])
    if frame.duplicated(KEYS).any():
        raise ValueError("KMA context rows are not unique by forecast and issue")
    if frame["forecast_kst_dtm"].duplicated().any():
        raise ValueError("KMA context contains multiple issues for a forecast target")
    values = pd.to_numeric(frame["kma_um_ctx_speed10_r0"], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError("KMA 10 m wind speed contains missing or non-finite values")
    index = pd.DatetimeIndex(frame["forecast_kst_dtm"])
    speed = pd.Series(values.to_numpy(dtype=float), index=index, name="kma_speed10")
    issues = pd.Series(
        frame["data_available_kst_dtm"].to_numpy(),
        index=index,
        name="data_available_kst_dtm",
    )
    return speed.sort_index(), issues.sort_index()


def load_gfs_850_speed(path: Path, grid_id: int = 5) -> pd.Series:
    columns = [
        "forecast_kst_dtm",
        "grid_id",
        "isobaricInhPa_850_u",
        "isobaricInhPa_850_v",
    ]
    frame = pd.read_csv(path, encoding="utf-8-sig", usecols=columns)
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    frame = frame.loc[frame["grid_id"] == grid_id]
    if frame.empty:
        raise ValueError(f"GFS grid {grid_id} is absent")
    if frame["forecast_kst_dtm"].duplicated().any():
        raise ValueError("GFS 850 hPa rows are not unique by forecast time")
    speed = np.hypot(
        pd.to_numeric(frame["isobaricInhPa_850_u"], errors="coerce"),
        pd.to_numeric(frame["isobaricInhPa_850_v"], errors="coerce"),
    )
    if not np.isfinite(speed).all():
        raise ValueError("GFS 850 hPa wind speed contains non-finite values")
    return pd.Series(
        np.asarray(speed, dtype=float),
        index=pd.DatetimeIndex(frame["forecast_kst_dtm"]),
        name="gfs_speed850",
    ).sort_index()


def build_proxy_labels(driver: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    ratios = {
        target: driver[f"{target}__valid_truth"].astype(float) / capacity
        for target, capacity in CAPACITY_KWH.items()
    }
    stack = np.column_stack(list(ratios.values()))
    group_12 = np.column_stack(
        [ratios["kpx_group_1"], ratios["kpx_group_2"]]
    )
    return {
        "group3": driver[f"{TARGET}__valid_truth"].astype(float),
        "mean_group123_ratio": CAPACITY * stack.mean(axis=1),
        "mean_group12_ratio": CAPACITY * group_12.mean(axis=1),
        "median_group123_ratio": CAPACITY * np.median(stack, axis=1),
    }


def fit_direct_power(
    wind_speed: np.ndarray,
    proxy_label: np.ndarray,
    train: np.ndarray,
    available: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    if train.sum() < 1_000:
        raise ValueError("power curve needs at least 1,000 training rows")
    model = IsotonicRegression(
        y_min=0.0, y_max=CAPACITY, out_of_bounds="clip"
    )
    model.fit(wind_speed[train], proxy_label[train])
    prediction = np.asarray(reference, dtype=float).copy()
    prediction[available] = model.predict(wind_speed[available])
    if not np.isfinite(prediction[available]).all():
        raise ValueError("isotonic power curve produced non-finite predictions")
    return prediction


def policy_gate(
    disagreement: np.ndarray,
    reference: np.ndarray,
    available: np.ndarray,
    policy: GatePolicy,
) -> np.ndarray:
    if policy.direction == "up":
        direction = disagreement > 0.0
    elif policy.direction == "down":
        direction = disagreement < 0.0
    elif policy.direction == "both":
        direction = np.ones(len(disagreement), dtype=bool)
    else:
        raise ValueError(f"unknown policy direction: {policy.direction}")
    magnitude = np.abs(disagreement)
    threshold = (
        -np.inf
        if policy.minimum_disagreement_kwh is None
        else policy.minimum_disagreement_kwh
    )
    ratio = reference / CAPACITY
    return (
        available
        & direction
        & (magnitude >= threshold)
        & (ratio >= policy.minimum_base_ratio)
        & (ratio <= policy.maximum_base_ratio)
    )


def _select_gate_policy(
    truth: np.ndarray,
    reference: np.ndarray,
    direct_power: np.ndarray,
    available: np.ndarray,
    selection: np.ndarray,
) -> dict[str, object] | None:
    disagreement = direct_power - reference
    rows: list[dict[str, object]] = []
    for direction_name in ("both", "up", "down"):
        if direction_name == "up":
            direction = disagreement > 0.0
        elif direction_name == "down":
            direction = disagreement < 0.0
        else:
            direction = np.ones(len(reference), dtype=bool)
        pool_mask = selection & available & direction
        if not pool_mask.any():
            continue
        pool = np.abs(disagreement[pool_mask])
        for coverage in (1.0, 0.75, 0.50, 0.25, 0.10, 0.05):
            threshold = (
                None
                if coverage == 1.0
                else float(np.quantile(pool, 1.0 - coverage))
            )
            for minimum_ratio, maximum_ratio in (
                (0.10, 1.00),
                (0.10, 0.80),
                (0.10, 0.60),
                (0.20, 1.00),
                (0.40, 1.00),
            ):
                policy = GatePolicy(
                    direction_name,
                    coverage,
                    minimum_ratio,
                    maximum_ratio,
                    threshold,
                )
                gate = policy_gate(disagreement, reference, available, policy)
                for alpha in (
                    0.005,
                    0.01,
                    0.02,
                    0.03,
                    0.05,
                    0.075,
                    0.10,
                    0.15,
                    0.20,
                    0.30,
                    0.40,
                    0.50,
                ):
                    candidate = reference.copy()
                    candidate[gate] = np.clip(
                        reference[gate] + alpha * disagreement[gate],
                        0.0,
                        CAPACITY,
                    )
                    comparison = _compare(
                        truth, reference, candidate, selection
                    )
                    delta = comparison["delta"]
                    if min(delta[component] for component in COMPONENTS) < 0.0:
                        continue
                    rows.append(
                        {
                            "policy": policy,
                            "screen_alpha": alpha,
                            "comparison": comparison,
                            "changed_rows": int((gate & selection).sum()),
                        }
                    )
    if not rows:
        return None
    rows.sort(
        key=lambda row: tuple(
            row["comparison"]["delta"][component]
            for component in COMPONENTS
        ),
        reverse=True,
    )
    return {"selected": rows[0], "top": rows[:10]}


def apply_bounded_combo(
    reference: np.ndarray,
    kma_disagreement: np.ndarray,
    gfs_disagreement: np.ndarray,
    available: np.ndarray,
    kma_policy: GatePolicy,
    gfs_policy: GatePolicy,
    *,
    kma_alpha: float,
    gfs_alpha: float,
    maximum_movement_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < maximum_movement_ratio <= 0.10:
        raise ValueError("maximum movement ratio must be in (0, 0.10]")
    gfs_gate = policy_gate(
        gfs_disagreement, reference, available, gfs_policy
    )
    kma_gate = policy_gate(
        kma_disagreement, reference, available, kma_policy
    ) & ~gfs_gate
    bound = maximum_movement_ratio * CAPACITY
    control = reference.copy()
    control[gfs_gate] = np.clip(
        reference[gfs_gate]
        + np.clip(gfs_alpha * gfs_disagreement[gfs_gate], -bound, bound),
        0.0,
        CAPACITY,
    )
    candidate = control.copy()
    candidate[kma_gate] = np.clip(
        control[kma_gate]
        + np.clip(kma_alpha * kma_disagreement[kma_gate], -bound, bound),
        0.0,
        CAPACITY,
    )
    return control, candidate, gfs_gate, kma_gate


def _select_bounded_combo(
    truth: np.ndarray,
    reference: np.ndarray,
    kma_disagreement: np.ndarray,
    gfs_disagreement: np.ndarray,
    available: np.ndarray,
    selection: np.ndarray,
    kma_policy: GatePolicy,
    gfs_policy: GatePolicy,
    *,
    maximum_movement_ratio: float,
    minimum_component_gain: float,
) -> dict[str, object] | None:
    alpha_grid = (0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
    rows = []
    for gfs_alpha in alpha_grid:
        for kma_alpha in alpha_grid:
            control, candidate, gfs_gate, kma_gate = apply_bounded_combo(
                reference,
                kma_disagreement,
                gfs_disagreement,
                available,
                kma_policy,
                gfs_policy,
                kma_alpha=kma_alpha,
                gfs_alpha=gfs_alpha,
                maximum_movement_ratio=maximum_movement_ratio,
            )
            total = _compare(truth, reference, candidate, selection)
            incremental = _compare(truth, control, candidate, selection)
            if min(total["delta"][key] for key in COMPONENTS) <= 0.0:
                continue
            if min(incremental["delta"][key] for key in COMPONENTS) < (
                minimum_component_gain
            ):
                continue
            rows.append(
                {
                    "gfs_alpha": gfs_alpha,
                    "kma_alpha": kma_alpha,
                    "total_vs_public_fine": total,
                    "incremental_kma_vs_gfs_control": incremental,
                    "selection_gfs_rows": int((gfs_gate & selection).sum()),
                    "selection_kma_rows": int((kma_gate & selection).sum()),
                }
            )
    if not rows:
        return None
    rows.sort(
        key=lambda row: (
            row["total_vs_public_fine"]["delta"]["score"],
            row["incremental_kma_vs_gfs_control"]["delta"]["score"],
            -row["gfs_alpha"],
            -row["kma_alpha"],
        ),
        reverse=True,
    )
    return {"selected": rows[0], "top": rows[:10]}


def issue_block_bootstrap(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    issue_times: np.ndarray,
    mask: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    local_positions = np.flatnonzero(mask)
    local_issues = pd.Series(issue_times[mask])
    if local_issues.isna().any():
        raise ValueError("issue bootstrap received missing issue times")
    codes, unique = pd.factorize(local_issues, sort=True)
    groups = [local_positions[codes == code] for code in range(len(unique))]
    rng = np.random.default_rng(seed)
    values = np.empty((n_bootstrap, len(COMPONENTS)), dtype=float)
    for iteration in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        positions = np.concatenate([groups[value] for value in sampled])
        values[iteration] = np.asarray(
            list(_compare(truth, reference, candidate, positions)["delta"].values())
        )
    return {
        "n_bootstrap": n_bootstrap,
        "unique_issue_cycles": int(len(unique)),
        "q05": {
            component: float(value)
            for component, value in zip(
                COMPONENTS, np.quantile(values, 0.05, axis=0)
            )
        },
        "median": {
            component: float(value)
            for component, value in zip(
                COMPONENTS, np.quantile(values, 0.50, axis=0)
            )
        },
        "positive_all_component_fraction": float(
            np.mean(np.min(values, axis=1) > 0.0)
        ),
    }


def _serialize_gate_selection(selection: dict[str, object]) -> dict[str, object]:
    def row(value: dict[str, object]) -> dict[str, object]:
        return {
            "policy": value["policy"].to_dict(),
            "screen_alpha": value["screen_alpha"],
            "comparison": value["comparison"],
            "changed_rows": value["changed_rows"],
        }

    return {
        "selected": row(selection["selected"]),
        "top": [row(value) for value in selection["top"]],
    }


def _load_or_build_fine_surface(
    cache_path: Path,
    index: pd.DatetimeIndex,
    truth: np.ndarray,
    current: np.ndarray,
    member: np.ndarray,
    meta_features: np.ndarray,
    action: np.ndarray,
) -> np.ndarray:
    if cache_path.is_file():
        retained = np.load(cache_path)
        retained_index = pd.DatetimeIndex(
            pd.to_datetime(retained["valid_index_ns"])
        )
        if retained_index.equals(index):
            candidate = retained["valid_candidate"].astype(float)
            if len(candidate) == len(index) and np.isfinite(candidate).all():
                return candidate
    candidate, gate, probability = _rolling_fine_surfaces(
        index, truth, current, member, meta_features, action
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        valid_index_ns=index.astype("int64").to_numpy(),
        valid_candidate=candidate.astype("float32"),
        valid_gate=gate,
        valid_probability=probability.astype("float32"),
    )
    return candidate


def _fit_direct_for_test(
    train_wind: np.ndarray,
    proxy_label: np.ndarray,
    train_mask: np.ndarray,
    test_wind: np.ndarray,
) -> np.ndarray:
    model = IsotonicRegression(
        y_min=0.0, y_max=CAPACITY, out_of_bounds="clip"
    )
    model.fit(train_wind[train_mask], proxy_label[train_mask])
    result = model.predict(test_wind)
    if not np.isfinite(result).all():
        raise ValueError("test power curve produced non-finite values")
    return result


def make_submission_if_qualified(
    *,
    qualified: bool,
    test_manifest: Path,
    test_features: Path,
    test_gfs: Path,
    base_submission: Path,
    output_submission: Path,
    train_kma: np.ndarray,
    train_gfs: np.ndarray,
    proxy_label: np.ndarray,
    train_available: np.ndarray,
    kma_policy: GatePolicy,
    gfs_policy: GatePolicy,
    kma_alpha: float,
    gfs_alpha: float,
    maximum_movement_ratio: float,
) -> dict[str, object] | None:
    if not qualified:
        return None
    validate_external_data_manifest(test_manifest, Path.cwd().resolve())
    test_kma_series, _ = load_context_speed(test_features)
    test_gfs_series = load_gfs_850_speed(test_gfs)
    submission = pd.read_csv(base_submission, encoding="utf-8-sig")
    timestamp_column = next(
        (
            column
            for column in ("forecast_kst_dtm", "kst_dtm")
            if column in submission.columns
        ),
        None,
    )
    if timestamp_column is None or TARGET not in submission.columns:
        raise ValueError("base submission is missing timestamp or group-3 output")
    test_index = pd.DatetimeIndex(pd.to_datetime(submission[timestamp_column]))
    if test_index.duplicated().any():
        raise ValueError("base submission contains duplicate timestamps")
    test_kma = test_kma_series.reindex(test_index).to_numpy(dtype=float)
    test_gfs_values = test_gfs_series.reindex(test_index).to_numpy(dtype=float)
    if not np.isfinite(test_kma).all() or not np.isfinite(test_gfs_values).all():
        raise ValueError("test KMA/GFS coverage is incomplete")
    final_train = train_available & np.isfinite(proxy_label)
    kma_direct = _fit_direct_for_test(
        train_kma, proxy_label, final_train, test_kma
    )
    gfs_direct = _fit_direct_for_test(
        train_gfs, proxy_label, final_train, test_gfs_values
    )
    reference = submission[TARGET].to_numpy(dtype=float)
    available = np.ones(len(reference), dtype=bool)
    control, candidate, gfs_gate, kma_gate = apply_bounded_combo(
        reference,
        kma_direct - reference,
        gfs_direct - reference,
        available,
        kma_policy,
        gfs_policy,
        kma_alpha=kma_alpha,
        gfs_alpha=gfs_alpha,
        maximum_movement_ratio=maximum_movement_ratio,
    )
    output = submission.copy()
    output[TARGET] = candidate
    if not np.isfinite(candidate).all() or not (
        (candidate >= 0.0) & (candidate <= CAPACITY)
    ).all():
        raise ValueError("candidate submission is outside group-3 bounds")
    output_submission.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_submission, index=False, encoding="utf-8-sig")
    movement = candidate - reference
    return {
        "path": output_submission.as_posix(),
        "sha256": hashlib.sha256(output_submission.read_bytes()).hexdigest(),
        "rows": int(len(output)),
        "gfs_rows": int(gfs_gate.sum()),
        "kma_incremental_rows": int(kma_gate.sum()),
        "changed_rows": int((np.abs(movement) > 1e-9).sum()),
        "maximum_absolute_movement_kwh": float(np.max(np.abs(movement))),
        "mean_absolute_movement_kwh": float(np.mean(np.abs(movement))),
        "gfs_control_mean_kwh": float(np.mean(control)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--fine-cache",
        default="artifacts_final/meta_gate/fine_sweep_rolling_oof.npz",
    )
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
    parser.add_argument("--minimum-common-rows", type=int, default=8_000)
    parser.add_argument("--minimum-selection-component-gain", type=float, default=0.0001)
    parser.add_argument("--minimum-locked-score-gain", type=float, default=0.001)
    parser.add_argument("--minimum-incremental-score-gain", type=float, default=0.0005)
    parser.add_argument("--minimum-bootstrap-positive-fraction", type=float, default=0.90)
    parser.add_argument("--maximum-movement-ratio", type=float, default=0.05)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument(
        "--output",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/power_curve_gate.json",
    )
    parser.add_argument(
        "--oof-cache",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/power_curve_oof.npz",
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
        "--base-submission",
        default="submissions/blend_best_crossg3_traj_meta_finesweep.csv",
    )
    parser.add_argument(
        "--output-submission",
        default="submissions/blend_best_kma_um_power_curve_gate.csv",
    )
    parser.add_argument("--write-submission-if-qualified", action="store_true")
    args = parser.parse_args()

    validate_external_data_manifest(Path(args.manifest), Path.cwd().resolve())
    kma_series, issue_series = load_context_speed(Path(args.context_features))
    gfs_series = load_gfs_850_speed(Path(args.gfs), args.gfs_grid_id)
    driver = np.load(args.driver_cache)
    (
        _labels,
        index,
        truth,
        _group_1,
        _group_2,
        _exact_base,
        member,
        current,
        meta_features,
        action,
    ) = _prepare_validation(Path(args.labels), Path(args.driver_cache))
    public_fine = _load_or_build_fine_surface(
        Path(args.fine_cache),
        index,
        truth,
        current,
        member,
        meta_features,
        action,
    )
    kma = kma_series.reindex(index).to_numpy(dtype=float)
    gfs = gfs_series.reindex(index).to_numpy(dtype=float)
    issues = issue_series.reindex(index).to_numpy()
    available = np.isfinite(kma) & np.isfinite(gfs) & pd.notna(issues)
    within_year = np.asarray((index >= Q1_START) & (index < END))
    if int((available & within_year).sum()) < args.minimum_common_rows:
        raise ValueError("KMA power-curve screen lacks minimum 2024 coverage")
    q1 = available & np.asarray((index >= Q1_START) & (index < Q2_START))
    q2 = available & np.asarray((index >= Q2_START) & (index < H2_START))
    h1 = available & np.asarray((index >= Q1_START) & (index < H2_START))
    h2 = available & np.asarray((index >= H2_START) & (index < END))
    proxy_labels = build_proxy_labels(driver)

    development = []
    retained: dict[str, object] = {}
    for proxy_name, proxy_label in proxy_labels.items():
        kma_direct = fit_direct_power(kma, proxy_label, q1, available, public_fine)
        gfs_direct = fit_direct_power(gfs, proxy_label, q1, available, public_fine)
        kma_selection = _select_gate_policy(
            truth, public_fine, kma_direct, available, q2
        )
        gfs_selection = _select_gate_policy(
            truth, public_fine, gfs_direct, available, q2
        )
        if kma_selection is None or gfs_selection is None:
            development.append(
                {"proxy": proxy_name, "selection_status": "no_gate_policy"}
            )
            continue
        kma_policy = kma_selection["selected"]["policy"]
        gfs_policy = gfs_selection["selected"]["policy"]
        combo = _select_bounded_combo(
            truth,
            public_fine,
            kma_direct - public_fine,
            gfs_direct - public_fine,
            available,
            q2,
            kma_policy,
            gfs_policy,
            maximum_movement_ratio=args.maximum_movement_ratio,
            minimum_component_gain=args.minimum_selection_component_gain,
        )
        if combo is None:
            development.append(
                {
                    "proxy": proxy_name,
                    "selection_status": "no_component_margin_combo",
                    "kma_gate_screen": _serialize_gate_selection(kma_selection),
                    "gfs_gate_screen": _serialize_gate_selection(gfs_selection),
                }
            )
            continue
        row = {
            "proxy": proxy_name,
            "selection_status": "passed",
            "kma_gate_screen": _serialize_gate_selection(kma_selection),
            "gfs_gate_screen": _serialize_gate_selection(gfs_selection),
            "bounded_combo": combo,
        }
        development.append(row)
        selected = combo["selected"]
        key = (
            selected["total_vs_public_fine"]["delta"]["score"],
            selected["incremental_kma_vs_gfs_control"]["delta"]["score"],
        )
        if not retained or key > retained["key"]:
            retained = {
                "key": key,
                "proxy_name": proxy_name,
                "proxy_label": proxy_label,
                "kma_policy": kma_policy,
                "gfs_policy": gfs_policy,
                "combo": selected,
                "q1_kma_direct": kma_direct,
                "q1_gfs_direct": gfs_direct,
            }
    if not retained:
        raise RuntimeError("no proxy family passed the Q2 component-margin gate")

    combo = retained["combo"]
    kma_policy = retained["kma_policy"]
    gfs_policy = retained["gfs_policy"]
    static_control, static_candidate, _, _ = apply_bounded_combo(
        public_fine,
        retained["q1_kma_direct"] - public_fine,
        retained["q1_gfs_direct"] - public_fine,
        available,
        kma_policy,
        gfs_policy,
        kma_alpha=combo["kma_alpha"],
        gfs_alpha=combo["gfs_alpha"],
        maximum_movement_ratio=args.maximum_movement_ratio,
    )
    h1_kma_direct = fit_direct_power(
        kma, retained["proxy_label"], h1, available, public_fine
    )
    h1_gfs_direct = fit_direct_power(
        gfs, retained["proxy_label"], h1, available, public_fine
    )
    locked_control, locked_candidate, gfs_gate, kma_gate = apply_bounded_combo(
        public_fine,
        h1_kma_direct - public_fine,
        h1_gfs_direct - public_fine,
        available,
        kma_policy,
        gfs_policy,
        kma_alpha=combo["kma_alpha"],
        gfs_alpha=combo["gfs_alpha"],
        maximum_movement_ratio=args.maximum_movement_ratio,
    )
    static_total = _compare(truth, public_fine, static_candidate, h2)
    static_incremental = _compare(truth, static_control, static_candidate, h2)
    locked_total = _compare(truth, public_fine, locked_candidate, h2)
    locked_incremental = _compare(truth, locked_control, locked_candidate, h2)
    monthly_total = {}
    monthly_incremental = {}
    for month in range(7, 13):
        month_mask = h2 & np.asarray(index.month == month)
        monthly_total[str(month)] = _compare(
            truth, public_fine, locked_candidate, month_mask
        )["delta"]
        monthly_incremental[str(month)] = _compare(
            truth, locked_control, locked_candidate, month_mask
        )["delta"]
    bootstrap_total = issue_block_bootstrap(
        truth,
        public_fine,
        locked_candidate,
        issues,
        h2,
        n_bootstrap=args.n_bootstrap,
        seed=20260720,
    )
    bootstrap_incremental = issue_block_bootstrap(
        truth,
        locked_control,
        locked_candidate,
        issues,
        h2,
        n_bootstrap=args.n_bootstrap,
        seed=20260720,
    )
    movement = locked_candidate - public_fine
    gates = {
        "q2_proxy_selected_with_component_margin": min(
            combo["incremental_kma_vs_gfs_control"]["delta"][key]
            for key in COMPONENTS
        )
        >= args.minimum_selection_component_gain,
        "static_h2_total_all_components_positive": min(
            static_total["delta"][key] for key in COMPONENTS
        )
        > 0.0,
        "static_h2_incremental_all_components_positive": min(
            static_incremental["delta"][key] for key in COMPONENTS
        )
        > 0.0,
        "rolling_h2_total_all_components_positive": min(
            locked_total["delta"][key] for key in COMPONENTS
        )
        > 0.0,
        "rolling_h2_incremental_all_components_positive": min(
            locked_incremental["delta"][key] for key in COMPONENTS
        )
        > 0.0,
        "all_locked_month_total_ficr_nonnegative": all(
            row["ficr"] >= 0.0 for row in monthly_total.values()
        ),
        "total_bootstrap_q05_all_components_positive": min(
            bootstrap_total["q05"].values()
        )
        > 0.0,
        "incremental_bootstrap_q05_all_components_positive": min(
            bootstrap_incremental["q05"].values()
        )
        > 0.0,
        "bootstrap_positive_fractions_pass": min(
            bootstrap_total["positive_all_component_fraction"],
            bootstrap_incremental["positive_all_component_fraction"],
        )
        >= args.minimum_bootstrap_positive_fraction,
        "minimum_locked_score_gain_passed": locked_total["delta"]["score"]
        >= args.minimum_locked_score_gain,
        "minimum_incremental_score_gain_passed": locked_incremental["delta"][
            "score"
        ]
        >= args.minimum_incremental_score_gain,
        "maximum_movement_bound_passed": float(np.max(np.abs(movement[h2])))
        <= args.maximum_movement_ratio * CAPACITY + 1e-9,
    }
    qualified = bool(all(gates.values()))
    oof_cache = Path(args.oof_cache)
    oof_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        oof_cache,
        index_ns=index.astype("int64").to_numpy(),
        issue_ns=pd.to_datetime(issues).astype("int64"),
        truth=truth.astype("float32"),
        public_fine=public_fine.astype("float32"),
        static_control=static_control.astype("float32"),
        static_candidate=static_candidate.astype("float32"),
        rolling_control=locked_control.astype("float32"),
        rolling_candidate=locked_candidate.astype("float32"),
        available=available,
        q2=q2,
        h2=h2,
    )
    oof_cache_record = {
        "path": oof_cache.as_posix(),
        "sha256": hashlib.sha256(oof_cache.read_bytes()).hexdigest(),
        "rows": int(len(index)),
    }
    submission = None
    if args.write_submission_if_qualified:
        submission = make_submission_if_qualified(
            qualified=qualified,
            test_manifest=Path(args.test_manifest),
            test_features=Path(args.test_features),
            test_gfs=Path(args.test_gfs),
            base_submission=Path(args.base_submission),
            output_submission=Path(args.output_submission),
            train_kma=kma,
            train_gfs=gfs,
            proxy_label=retained["proxy_label"],
            train_available=available & within_year,
            kma_policy=kma_policy,
            gfs_policy=gfs_policy,
            kma_alpha=combo["kma_alpha"],
            gfs_alpha=combo["gfs_alpha"],
            maximum_movement_ratio=args.maximum_movement_ratio,
        )
    report = {
        "family": "kma_um_independent_bounded_joint_power_curve",
        "source_manifest": args.manifest,
        "public_reference": {
            "submission": args.base_submission,
            "policy": "rolling fine meta gate p=0.545 alpha=0.50",
        },
        "selection_contract": {
            "q1_power_curve_train": "2024-01-01 through 2024-03-31",
            "q2_proxy_gate_and_alpha_selection": "2024-04-01 through 2024-07-01 00:00",
            "locked_h2": "H1-refit curves; 2024-07-01 01:00 through 2024-12-31 23:00",
            "proxy_candidates": list(proxy_labels),
            "minimum_q2_incremental_component_gain": args.minimum_selection_component_gain,
            "maximum_per_row_movement_ratio": args.maximum_movement_ratio,
            "common_rows": int((available & within_year).sum()),
        },
        "development": development,
        "selected_proxy": retained["proxy_name"],
        "selected_kma_gate": kma_policy.to_dict(),
        "selected_gfs_control_gate": gfs_policy.to_dict(),
        "selected_bounded_combo": combo,
        "static_q1_curve_locked_h2": {
            "total_vs_public_fine": static_total,
            "incremental_kma_vs_gfs_control": static_incremental,
        },
        "rolling_h1_curve_locked_h2": {
            "total_vs_public_fine": locked_total,
            "incremental_kma_vs_gfs_control": locked_incremental,
            "monthly_total_vs_public_fine": monthly_total,
            "monthly_incremental_kma_vs_gfs_control": monthly_incremental,
            "gfs_control_rows": int((gfs_gate & h2).sum()),
            "kma_incremental_rows": int((kma_gate & h2).sum()),
            "changed_ratio": float(((np.abs(movement) > 1e-9) & h2).sum() / h2.sum()),
            "maximum_absolute_movement_kwh": float(np.max(np.abs(movement[h2]))),
            "mean_absolute_movement_kwh": float(np.mean(np.abs(movement[h2]))),
        },
        "issue_block_bootstrap": {
            "total_vs_public_fine": bootstrap_total,
            "incremental_kma_vs_gfs_control": bootstrap_incremental,
        },
        "promotion_gates": gates,
        "oof_cache": oof_cache_record,
        "decision": {
            "locked_gate_passed": qualified,
            "expand_2025_kma_collection": qualified,
            "submission_requested": bool(args.write_submission_if_qualified),
            "submission_created": submission is not None,
            "submission_eligible": submission is not None and qualified,
            "reason": (
                "collect 2025 KMA UM and create only the bounded candidate after provenance validation"
                if qualified and submission is None
                else (
                    "qualified bounded KMA UM candidate created"
                    if submission is not None
                    else "KMA UM power curve failed at least one locked promotion gate"
                )
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
    print(
        json.dumps(
            {
                "family": report["family"],
                "selected_proxy": report["selected_proxy"],
                "selected_bounded_combo": report["selected_bounded_combo"],
                "rolling_h1_curve_locked_h2": report[
                    "rolling_h1_curve_locked_h2"
                ],
                "issue_block_bootstrap": report["issue_block_bootstrap"],
                "promotion_gates": gates,
                "decision": report["decision"],
                "submission": submission,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
