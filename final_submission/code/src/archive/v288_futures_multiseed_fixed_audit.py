"""Five-seed audit of a fixed 10% recent-F expert plus exact-anchor v244.

The 10% dose is frozen from the first v287 screen as the smallest tested dose;
it is not re-optimized here.  The F correction and exact-anchor correction have
disjoint support, so their combined transfer can be audited without changing
either mechanism.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics
from src.archive.v287_recent_futures_direct_expert import (
    CAT_COLUMNS,
    PARAMS,
    TARGET,
    build_features,
)


PROTOCOL = "V288_FUTURES_MULTISEED_FIXED_AUDIT_V1"
FUTURES_WEIGHT = 0.10
SEEDS = (2870, 2872, 2874, 2876, 2878)
EXTRA_CAT_COLUMNS = ["count_code", "same_hand", "pressure_code"]


def _fit(
    fit_features: pd.DataFrame,
    fit_target: np.ndarray,
    audit_features: pd.DataFrame,
    seed: int,
    output_path: Path,
) -> np.ndarray:
    cat_columns = [
        column
        for column in CAT_COLUMNS + EXTRA_CAT_COLUMNS
        if column in fit_features.columns
    ]
    model = CatBoostClassifier(**PARAMS, random_seed=seed)
    model.fit(fit_features, fit_target, cat_features=cat_columns)
    model.save_model(output_path)
    return model.predict_proba(audit_features)[:, 1].astype(np.float64)


def _axis(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "target": frame[TARGET].to_numpy(np.float64),
        "exact_mask": np.ones(len(frame), dtype=bool),
        "game_month": frame["game_month"].to_numpy(),
        "pitcher_id": frame["pitcher_id"].to_numpy(),
        "batter_id": frame["batter_id"].to_numpy(),
    }


def run(
    train_csv: Path,
    v285_axes: Path,
    v287_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    is_f = train["game_type"].astype(str).eq("F")
    early_2023 = train.loc[
        train["season"].eq(2023) & is_f & train["game_month"].lt(8)
    ].reset_index(drop=True)
    late_f_2023 = train.loc[
        train["season"].eq(2023) & is_f & train["game_month"].ge(8)
    ].reset_index(drop=True)
    all_f_2023 = train.loc[train["season"].eq(2023) & is_f].reset_index(drop=True)
    late_all_2023 = train.loc[
        train["season"].eq(2023) & train["game_month"].ge(8)
    ].reset_index(drop=True)
    full_2024 = train.loc[train["season"].eq(2024)].reset_index(drop=True)
    full_f_2024 = full_2024.loc[
        full_2024["game_type"].astype(str).eq("F")
    ].reset_index(drop=True)

    source_fit = build_features(early_2023)
    source_audit = build_features(late_f_2023).reindex(columns=source_fit.columns)
    transfer_fit = build_features(all_f_2023)
    transfer_audit = build_features(full_f_2024).reindex(columns=transfer_fit.columns)
    source_predictions = [
        np.load(v287_dir / "expert_late_2023.npy", allow_pickle=False).astype(np.float64)
    ]
    transfer_predictions = [
        np.load(v287_dir / "expert_full_2024.npy", allow_pickle=False).astype(np.float64)
    ]
    for seed in SEEDS[1:]:
        source_predictions.append(
            _fit(
                source_fit,
                early_2023[TARGET].to_numpy(np.int8),
                source_audit,
                seed,
                output_dir / f"early23_to_late23_seed{seed}.cbm",
            )
        )
        transfer_predictions.append(
            _fit(
                transfer_fit,
                all_f_2023[TARGET].to_numpy(np.int8),
                transfer_audit,
                seed + 1,
                output_dir / f"full23_to_full24_seed{seed + 1}.cbm",
            )
        )

    source_stack = np.column_stack(source_predictions)
    transfer_stack = np.column_stack(transfer_predictions)
    source_expert = source_stack.mean(axis=1)
    transfer_expert = transfer_stack.mean(axis=1)
    np.save(output_dir / "expert_late_2023_multiseed.npy", source_expert.astype(np.float32))
    np.save(output_dir / "expert_full_2024_multiseed.npy", transfer_expert.astype(np.float32))

    with np.load(v285_axes, allow_pickle=False) as saved:
        base_late = saved["baseline_late_2023"].astype(np.float64)
        base_2024 = saved["baseline_full_2024"].astype(np.float64)
        exact_late = saved["candidate_late_2023"].astype(np.float64)
        exact_2024 = saved["candidate_full_2024"].astype(np.float64)
        exact_active_late = saved["active_late_2023"].astype(bool)
        exact_active_2024 = saved["active_full_2024"].astype(bool)
    f_late = late_all_2023["game_type"].astype(str).eq("F").to_numpy()
    f_2024 = full_2024["game_type"].astype(str).eq("F").to_numpy()

    def compose(
        baseline: np.ndarray,
        exact: np.ndarray,
        f_mask: np.ndarray,
        expert: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidate = exact.copy()
        candidate[f_mask] = np.clip(
            baseline[f_mask]
            + FUTURES_WEIGHT * (expert - baseline[f_mask]),
            0.001,
            0.999,
        )
        return candidate, f_mask | (np.abs(exact - baseline) > 0.0)

    candidate_late, active_late = compose(base_late, exact_late, f_late, source_expert)
    candidate_2024, active_2024 = compose(base_2024, exact_2024, f_2024, transfer_expert)
    metrics = {
        "late_2023": paired_metrics(
            _axis(late_all_2023), base_late, candidate_late, active_late
        ),
        "full_2024": paired_metrics(
            _axis(full_2024), base_2024, candidate_2024, active_2024
        ),
    }

    seed_family = []
    seed_gains = []
    for index, prediction in enumerate(transfer_predictions):
        candidate, _active = compose(base_2024, exact_2024, f_2024, prediction)
        seed_family.append(candidate)
        seed_gains.append(
            {
                "seed": SEEDS[index],
                "gain": paired_metrics(
                    _axis(full_2024), base_2024, candidate, active_2024
                )["gain"],
            }
        )
    seed_family.append(candidate_2024)
    robustness = _robustness(
        _axis(full_2024),
        base_2024,
        candidate_2024,
        active_2024,
        seed_family,
    )
    robust_pass = bool(
        robustness["pitcher"]["p05"] > 0.0
        and robustness["crossed_pitcher_batter"]["p05"] > 0.0
        and robustness["chronological_block"]["p05"] > 0.0
        and robustness["reality_check"]["p_value"] < 0.10
    )
    numeric_promote = bool(
        metrics["late_2023"]["gain"] > 0.0
        and metrics["full_2024"]["gain"] >= 4.0
        and robust_pass
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "numeric_promote" if numeric_promote else "mechanistic_challenger",
        "futures_weight": FUTURES_WEIGHT,
        "seeds": list(SEEDS),
        "metrics": metrics,
        "individual_seed_full_2024_gains": seed_gains,
        "locked_robustness": robustness,
        "numeric_promotion_gate_passed": numeric_promote,
        "eligible_for_exploratory_packaging": bool(
            metrics["late_2023"]["gain"] > 0.0
            and metrics["full_2024"]["gain"] >= 4.0
        ),
        "support_disjoint": {
            "exact_anchor_regular_routes": int(exact_active_2024.sum()),
            "futures_rows": int(f_2024.sum()),
            "overlap": int(np.sum(exact_active_2024 & f_2024)),
        },
        "restrictions": {
            "official_train_only": True,
            "futures_weight_frozen_at_smallest_v287_dose": True,
            "exact_anchor_weight_and_routes_frozen": True,
            "full_2024_not_used_to_retune_weight": True,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_selection": False,
        },
    }
    np.savez_compressed(
        output_dir / "combined_axes.npz",
        baseline_late_2023=base_late,
        candidate_late_2023=candidate_late,
        active_late_2023=active_late,
        baseline_full_2024=base_2024,
        candidate_full_2024=candidate_2024,
        active_full_2024=active_2024,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v287-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.train_csv, args.v285_axes, args.v287_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
