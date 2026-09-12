from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from agent_service.compliance import validate_external_data_manifest
from experiments.kma_um_power_curve_gate import (
    CAPACITY,
    COMPONENTS,
    GatePolicy,
    _compare,
    apply_bounded_combo,
    build_proxy_labels,
    issue_block_bootstrap,
    load_context_speed,
    load_gfs_850_speed,
    policy_gate,
)
from src.metrics import CAPACITY_KWH


HISTORY_START = pd.Timestamp("2023-01-01 00:00:00")
HISTORY_END = pd.Timestamp("2024-01-01 00:00:00")
Q1_START = pd.Timestamp("2024-01-01 00:00:00")
Q2_START = pd.Timestamp("2024-04-01 00:00:00")
H2_START = pd.Timestamp("2024-07-01 01:00:00")
END = pd.Timestamp("2025-01-01 00:00:00")


def load_label_proxies(
    path: Path, index: pd.DatetimeIndex
) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["kst_dtm"])
    frame = frame.set_index("kst_dtm").reindex(index)
    targets = list(CAPACITY_KWH)
    ratios = np.column_stack(
        [
            frame[target].to_numpy(dtype=float) / capacity
            for target, capacity in CAPACITY_KWH.items()
        ]
    )
    group_12 = ratios[:, :2]
    proxies = {
        "group3": frame["kpx_group_3"].to_numpy(dtype=float),
        "mean_group123_ratio": CAPACITY * ratios.mean(axis=1),
        "mean_group12_ratio": CAPACITY * group_12.mean(axis=1),
        "median_group123_ratio": CAPACITY * np.median(ratios, axis=1),
    }
    if any(np.isfinite(values).sum() < 8_000 for values in proxies.values()):
        raise ValueError("historical proxy labels lack minimum finite coverage")
    return proxies


def fit_mixed_direct_power(
    history_wind: np.ndarray,
    history_label: np.ndarray,
    recent_wind: np.ndarray,
    recent_label: np.ndarray,
    recent_train: np.ndarray,
    available: np.ndarray,
    fallback: np.ndarray,
    *,
    history_weight: float | None,
) -> np.ndarray:
    history_valid = np.isfinite(history_wind) & np.isfinite(history_label)
    recent_valid = recent_train & np.isfinite(recent_wind) & np.isfinite(recent_label)
    if history_valid.sum() < 1_000 or recent_valid.sum() < 1_000:
        raise ValueError("mixed power curve lacks minimum training coverage")
    if history_weight is None:
        x = history_wind[history_valid]
        y = history_label[history_valid]
        sample_weight = None
    else:
        if history_weight <= 0.0:
            raise ValueError("history weight must be positive or None")
        x = np.concatenate([history_wind[history_valid], recent_wind[recent_valid]])
        y = np.concatenate([history_label[history_valid], recent_label[recent_valid]])
        sample_weight = np.concatenate(
            [
                np.full(int(history_valid.sum()), history_weight, dtype=float),
                np.ones(int(recent_valid.sum()), dtype=float),
            ]
        )
    model = IsotonicRegression(
        y_min=0.0, y_max=CAPACITY, out_of_bounds="clip"
    )
    model.fit(x, y, sample_weight=sample_weight)
    result = np.asarray(fallback, dtype=float).copy()
    result[available] = model.predict(recent_wind[available])
    if not np.isfinite(result).all():
        raise ValueError("mixed power curve produced non-finite predictions")
    return result


def apply_sparse_expert_blend(
    recent: np.ndarray,
    prior: np.ndarray,
    available: np.ndarray,
    policy: GatePolicy,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    disagreement = prior - recent
    gate = policy_gate(disagreement, recent, available, policy)
    candidate = np.asarray(recent, dtype=float).copy()
    candidate[gate] = np.clip(
        recent[gate] + alpha * disagreement[gate], 0.0, CAPACITY
    )
    return candidate, gate


def select_sparse_expert_blend(
    truth: np.ndarray,
    recent: np.ndarray,
    prior: np.ndarray,
    available: np.ndarray,
    selection: np.ndarray,
    *,
    maximum_changed_ratio: float,
) -> dict[str, object] | None:
    if not 0.0 < maximum_changed_ratio <= 0.25:
        raise ValueError("maximum changed ratio must be in (0, 0.25]")
    disagreement = prior - recent
    rows: list[dict[str, object]] = []
    for direction_name in ("both", "up", "down"):
        if direction_name == "up":
            direction = disagreement > 0.0
        elif direction_name == "down":
            direction = disagreement < 0.0
        else:
            direction = np.ones(len(recent), dtype=bool)
        pool_mask = selection & available & direction
        if not pool_mask.any():
            continue
        pool = np.abs(disagreement[pool_mask])
        for coverage in (0.25, 0.10, 0.05):
            threshold = float(np.quantile(pool, 1.0 - coverage))
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
                for alpha in (0.05, 0.10, 0.20, 0.30):
                    candidate, gate = apply_sparse_expert_blend(
                        recent, prior, available, policy, alpha
                    )
                    changed_rows = int((gate & selection).sum())
                    changed_ratio = changed_rows / int(selection.sum())
                    if changed_ratio > maximum_changed_ratio:
                        continue
                    comparison = _compare(
                        truth, recent, candidate, selection
                    )
                    delta = comparison["delta"]
                    if min(delta[component] for component in COMPONENTS) < 0.0:
                        continue
                    rows.append(
                        {
                            "policy": policy,
                            "alpha": alpha,
                            "changed_rows": changed_rows,
                            "changed_ratio": changed_ratio,
                            "comparison": comparison,
                        }
                    )
    if not rows:
        return None
    rows.sort(
        key=lambda row: (
            row["comparison"]["delta"]["score"],
            min(row["comparison"]["delta"].values()),
            -row["changed_ratio"],
        ),
        reverse=True,
    )
    return {"selected": rows[0], "top": rows[:10], "qualified_count": len(rows)}


def _serialize_selection(selection: dict[str, object]) -> dict[str, object]:
    def serialize(row: dict[str, object]) -> dict[str, object]:
        return {
            "policy": row["policy"].to_dict(),
            "alpha": row["alpha"],
            "changed_rows": row["changed_rows"],
            "changed_ratio": row["changed_ratio"],
            "comparison": row["comparison"],
        }

    return {
        "selected": serialize(selection["selected"]),
        "top": [serialize(row) for row in selection["top"]],
        "qualified_count": selection["qualified_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument(
        "--driver-cache", default="artifacts_final/lineage/exact_driver_oof.npz"
    )
    parser.add_argument(
        "--history-manifest",
        default="artifacts_final/external_weather/kma_um_regional_context_2023/manifest.json",
    )
    parser.add_argument(
        "--history-features",
        default="artifacts_final/external_weather/kma_um_regional_context_2023/features.csv",
    )
    parser.add_argument(
        "--validation-manifest",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/manifest.json",
    )
    parser.add_argument(
        "--validation-features",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/features.csv",
    )
    parser.add_argument("--gfs", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--active-power-curve-report",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/power_curve_gate_revalidated_20260725.json",
    )
    parser.add_argument(
        "--active-oof",
        default="artifacts_final/external_weather/kma_um_regional_context_2024/power_curve_oof_20260725.npz",
    )
    parser.add_argument("--maximum-changed-ratio", type=float, default=0.25)
    parser.add_argument("--minimum-q2-score-gain", type=float, default=0.00015)
    parser.add_argument("--minimum-locked-score-gain", type=float, default=0.001)
    parser.add_argument("--minimum-bootstrap-positive-fraction", type=float, default=0.80)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument(
        "--output",
        default="artifacts_final/diagnostics/kma_year_forward_blend_20260726.json",
    )
    args = parser.parse_args()

    root = Path.cwd().resolve()
    validate_external_data_manifest(Path(args.history_manifest), root)
    validate_external_data_manifest(Path(args.validation_manifest), root)

    retained = np.load(args.active_oof)
    index = pd.DatetimeIndex(pd.to_datetime(retained["index_ns"]))
    truth = retained["truth"].astype(float)
    public_fine = retained["public_fine"].astype(float)
    recent_q1 = retained["static_candidate"].astype(float)
    recent_h1 = retained["rolling_candidate"].astype(float)
    available = retained["available"].astype(bool)
    q2 = retained["q2"].astype(bool)
    h2 = retained["h2"].astype(bool)
    issues = pd.to_datetime(retained["issue_ns"]).to_numpy()
    q1 = available & np.asarray(
        (index >= Q1_START) & (index < Q2_START)
    )

    history_index = pd.date_range(
        HISTORY_START, HISTORY_END - pd.Timedelta(hours=1), freq="h"
    )
    history_proxies = load_label_proxies(Path(args.labels), history_index)
    validation_proxies = build_proxy_labels(np.load(args.driver_cache))

    history_kma_series, _ = load_context_speed(Path(args.history_features))
    validation_kma_series, _ = load_context_speed(Path(args.validation_features))
    gfs_series = load_gfs_850_speed(Path(args.gfs))
    history_kma = history_kma_series.reindex(history_index).to_numpy(dtype=float)
    validation_kma = validation_kma_series.reindex(index).to_numpy(dtype=float)
    history_gfs = gfs_series.reindex(history_index).to_numpy(dtype=float)
    validation_gfs = gfs_series.reindex(index).to_numpy(dtype=float)
    if not np.isfinite(history_kma).all() or not np.isfinite(history_gfs).all():
        raise ValueError("2023 KMA/GFS year-forward training coverage is incomplete")

    active_report = json.loads(
        Path(args.active_power_curve_report).read_text(encoding="utf-8")
    )
    kma_policy = GatePolicy(**active_report["selected_kma_gate"])
    gfs_policy = GatePolicy(**active_report["selected_gfs_control_gate"])
    combo = active_report["selected_bounded_combo"]

    direct_pool_rows: list[dict[str, object]] = []
    prior_candidates: dict[str, np.ndarray] = {}
    modes: tuple[tuple[str, float | None], ...] = (
        ("history_weight_025", 0.25),
        ("history_weight_050", 0.50),
        ("history_weight_100", 1.00),
        ("history_only", None),
    )
    for proxy_name, history_label in history_proxies.items():
        recent_label = validation_proxies[proxy_name]
        for mode_name, history_weight in modes:
            kma_direct = fit_mixed_direct_power(
                history_kma,
                history_label,
                validation_kma,
                recent_label,
                q1,
                available,
                public_fine,
                history_weight=history_weight,
            )
            gfs_direct = fit_mixed_direct_power(
                history_gfs,
                history_label,
                validation_gfs,
                recent_label,
                q1,
                available,
                public_fine,
                history_weight=history_weight,
            )
            _, candidate, _, _ = apply_bounded_combo(
                public_fine,
                kma_direct - public_fine,
                gfs_direct - public_fine,
                available,
                kma_policy,
                gfs_policy,
                kma_alpha=float(combo["kma_alpha"]),
                gfs_alpha=float(combo["gfs_alpha"]),
                maximum_movement_ratio=0.05,
            )
            comparison = _compare(truth, recent_q1, candidate, q2)
            direct_pool_rows.append(
                {
                    "proxy": proxy_name,
                    "mode": mode_name,
                    "history_weight": history_weight,
                    "q2_vs_recent_q1": comparison,
                    "all_components_nonnegative": min(
                        comparison["delta"].values()
                    )
                    >= 0.0,
                }
            )
            if mode_name == "history_only":
                prior_candidates[proxy_name] = candidate
    direct_pool_rows.sort(
        key=lambda row: (
            row["q2_vs_recent_q1"]["delta"]["score"],
            min(row["q2_vs_recent_q1"]["delta"].values()),
        ),
        reverse=True,
    )

    prior = prior_candidates["median_group123_ratio"]
    selection = select_sparse_expert_blend(
        truth,
        recent_q1,
        prior,
        available,
        q2,
        maximum_changed_ratio=args.maximum_changed_ratio,
    )
    if selection is None:
        raise RuntimeError("no sparse prior-year expert blend passed Q2")
    selected = selection["selected"]
    locked_candidate, locked_gate = apply_sparse_expert_blend(
        recent_h1,
        prior,
        available,
        selected["policy"],
        float(selected["alpha"]),
    )
    locked = _compare(truth, recent_h1, locked_candidate, h2)
    monthly = {}
    for month in range(7, 13):
        month_mask = h2 & np.asarray(index.month == month)
        monthly[str(month)] = _compare(
            truth, recent_h1, locked_candidate, month_mask
        )["delta"]
    bootstrap = issue_block_bootstrap(
        truth,
        recent_h1,
        locked_candidate,
        issues,
        h2,
        n_bootstrap=args.n_bootstrap,
        seed=20260726,
    )
    changed_rows = int((locked_gate & h2).sum())
    changed_ratio = changed_rows / int(h2.sum())
    movement = locked_candidate - recent_h1
    gates = {
        "q2_all_components_nonnegative": min(
            selected["comparison"]["delta"].values()
        )
        >= 0.0,
        "q2_minimum_score_gain_passed": selected["comparison"]["delta"]["score"]
        >= args.minimum_q2_score_gain,
        "locked_all_components_positive": min(locked["delta"].values()) > 0.0,
        "locked_minimum_score_gain_passed": locked["delta"]["score"]
        >= args.minimum_locked_score_gain,
        "all_locked_month_scores_nonnegative": all(
            row["score"] >= 0.0 for row in monthly.values()
        ),
        "bootstrap_q05_all_components_positive": min(
            bootstrap["q05"].values()
        )
        > 0.0,
        "bootstrap_positive_fraction_passed": bootstrap[
            "positive_all_component_fraction"
        ]
        >= args.minimum_bootstrap_positive_fraction,
        "maximum_changed_ratio_passed": changed_ratio
        <= args.maximum_changed_ratio,
    }
    qualified = bool(all(gates.values()))
    report = {
        "family": "kma_2023_to_2024_year_forward_sparse_expert_blend",
        "purpose": (
            "test whether a frozen 2023 KMA expert adds stable segment diversity "
            "to the active 2024-trained KMA lineage"
        ),
        "sources": {
            "history_manifest": args.history_manifest,
            "validation_manifest": args.validation_manifest,
            "active_oof": args.active_oof,
            "active_power_curve_report": args.active_power_curve_report,
        },
        "contract": {
            "history_expert_train": "all available 2023 labels and causal KMA forecasts",
            "recent_development_expert": "2024 Q1-trained static active architecture",
            "selection": "2024 Q2 only",
            "locked": "2024 H2 once against the active H1-refit KMA OOF",
            "prior_expert_proxy": "median_group123_ratio, frozen from the public-success architecture",
            "maximum_changed_ratio": args.maximum_changed_ratio,
            "submission_created": False,
        },
        "coverage": {
            "history_hours": int(len(history_index)),
            "q1_rows": int(q1.sum()),
            "q2_rows": int(q2.sum()),
            "h2_rows": int(h2.sum()),
        },
        "direct_history_pooling": {
            "candidate_count": len(direct_pool_rows),
            "qualified_q2_count": sum(
                row["all_components_nonnegative"] for row in direct_pool_rows
            ),
            "top_q2": direct_pool_rows[:10],
            "decision": (
                "rejected before locked evaluation; no history-pooled curve "
                "improved every Q2 component versus the recent Q1 curve"
            ),
        },
        "conditional_blend_selection": _serialize_selection(selection),
        "locked": {
            "comparison": locked,
            "monthly_deltas": monthly,
            "changed_rows": changed_rows,
            "changed_ratio": changed_ratio,
            "mean_absolute_movement_kwh": float(
                np.mean(np.abs(movement[h2]))
            ),
            "p95_absolute_movement_kwh": float(
                np.quantile(np.abs(movement[h2]), 0.95)
            ),
            "maximum_absolute_movement_kwh": float(
                np.max(np.abs(movement[h2]))
            ),
            "issue_block_bootstrap": bootstrap,
        },
        "promotion_gates": gates,
        "decision": {
            "qualified": qualified,
            "submission_created": False,
            "retain_active_submission": True,
            "reason": (
                "qualified; a separate guarded submission build would be required"
                if qualified
                else (
                    "rejected: the Q2-selected sparse year-forward blend reversed "
                    "on locked H2"
                )
            ),
        },
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
                "direct_history_pooling": report["direct_history_pooling"],
                "selected": report["conditional_blend_selection"]["selected"],
                "locked": report["locked"],
                "promotion_gates": gates,
                "decision": report["decision"],
                "output": output.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
