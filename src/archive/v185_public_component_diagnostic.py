"""Research-only decomposition of public forward-OOF component predictions.

No public prediction is eligible for packaging.  The purpose of this audit is
to identify which independently reproducible model layer transfers across
2022, late-2023 and 2024 before another official-data model is trained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import _load_year_context, metrics
from src.archive.v177_forward_context_residual_eb import exact_jy_parents
from src.archive.v178_jy_signed_stack_rebase import (
    apply_direction,
    load_direction,
    load_weights,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V185_PUBLIC_COMPONENT_DIAGNOSTIC_V1"
V178_SCALE = 0.25
DIAGNOSTIC_WEIGHT = 0.005
MAIN_WEIGHTS = np.array([0.27358084, 0.26512224, 0.46129691])
STACK_INTERCEPT = 0.0300329767
STACK_COEFFICIENTS = np.array([0.93505266, -0.00520129, 0.01091677, -0.02528331])


def linear_public_stack(
    p_v2: np.ndarray,
    p_v3_55: np.ndarray,
    p_v3_30: np.ndarray,
    risks: np.ndarray,
) -> np.ndarray:
    main = np.column_stack([p_v2, p_v3_55, p_v3_30]) @ MAIN_WEIGHTS
    design = np.column_stack([main, np.asarray(risks, dtype=np.float64)])
    return np.clip(STACK_INTERCEPT + design @ STACK_COEFFICIENTS, 1e-6, 1 - 1e-6)


def diagnostic_blend(
    base: np.ndarray,
    independent: np.ndarray,
    active: np.ndarray,
    weight: float = DIAGNOSTIC_WEIGHT,
) -> np.ndarray:
    output = np.asarray(base, dtype=np.float64).copy()
    active = np.asarray(active, dtype=bool)
    output[active] += float(weight) * (
        np.asarray(independent, dtype=np.float64)[active] - output[active]
    )
    return np.clip(output, 0.001, 0.999)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _load_anchor(path: Path, year: int) -> np.ndarray:
    key = f"p{str(year)[-2:]}"
    with np.load(path, allow_pickle=False) as saved:
        return saved[key].astype(np.float64)


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    v158_path: Path,
    v165_summary: Path,
    library_root: Path,
    unified_root: Path,
    anchor_root: Path,
    local_oof_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _context, raw_frames, correction = _load_year_context(train_csv)
    late23 = raw_frames[2023]["game_month"].ge(8).to_numpy()
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    parents, parity = exact_jy_parents(
        axes, raw_frames, correction, v104_path, h1_path, c3_path,
        v160_path, bridge_oof,
    )
    frozen_weights = load_weights(v165_summary)
    with np.load(v158_path, allow_pickle=False) as saved:
        v158_base = {
            name: saved[name].astype(np.float64)
            for name in ("full_2022", "late_2023", "full_2024")
        }
    directions = {
        name: load_direction(name, v158_base[name], frozen_weights, library_root)
        for name in v158_base
    }
    v178 = {
        name: apply_direction(parents[name], directions[name], V178_SCALE)
        for name in parents
    }

    year_components: dict[int, dict[str, np.ndarray]] = {}
    for year in (2022, 2023, 2024):
        with np.load(unified_root / f"{year}.npz", allow_pickle=False) as saved:
            truth = raw_frames[year]["control_success"].to_numpy(np.int8)
            pitcher = raw_frames[year]["pitcher_id"].to_numpy(np.int64)
            if not np.array_equal(saved["y"].astype(np.int8), truth):
                raise ValueError(f"public label alignment failed: {year}")
            if not np.array_equal(saved["pitcher_id"].astype(np.int64), pitcher):
                raise ValueError(f"public pitcher alignment failed: {year}")
            p2 = saved["p_v2"].astype(np.float64)
            p55 = saved["p_v3_55"].astype(np.float64)
            p30 = saved["p_v3_30"].astype(np.float64)
            risks = saved["risks"].astype(np.float64)
        year_components[year] = {
            "public_v2": p2,
            "public_v3_decay55": p55,
            "public_v3_decay30": p30,
            "public_main_average": np.column_stack([p2, p55, p30]) @ MAIN_WEIGHTS,
            "public_linear_stack": linear_public_stack(p2, p55, p30, risks),
        }
        local = np.load(
            local_oof_root / f"hierarchical_residual_{year}.npy",
            allow_pickle=False,
        ).astype(np.float64)
        year_components[year]["local_v184"] = local

    for name in ("hierarchical_stack", "adaptive_gate", "psych_latent", "psych_regime_film"):
        path = anchor_root / f"{name}.npz"
        for year in (2023, 2024):
            year_components[year][f"anchor_{name}"] = _load_anchor(path, year)

    axis_components: dict[str, dict[str, np.ndarray]] = {
        "full_2022": year_components[2022],
        "late_2023": {
            name: prediction[late23]
            for name, prediction in year_components[2023].items()
        },
        "full_2024": year_components[2024],
    }
    all_names = sorted(set.union(*(set(values) for values in axis_components.values())))
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for name in all_names:
        detail[name] = {}
        for axis_name in ("full_2022", "late_2023", "full_2024"):
            if name not in axis_components[axis_name]:
                continue
            independent = axis_components[axis_name][name]
            active = np.asarray(axes[axis_name]["exact_mask"], dtype=bool)
            candidate = diagnostic_blend(v178[axis_name], independent, active)
            total = metrics(axes[axis_name], parents[axis_name], candidate)
            incremental = metrics(axes[axis_name], v178[axis_name], candidate)
            diagnostics = {
                "target_mean": float(np.mean(axes[axis_name]["target"])),
                "parent_mean": float(np.mean(parents[axis_name])),
                "v178_mean": float(np.mean(v178[axis_name])),
                "independent_mean": float(np.mean(independent)),
                "direction_correlation_with_parent_error": _safe_corr(
                    independent[active] - parents[axis_name][active],
                    np.asarray(axes[axis_name]["target"])[active] - parents[axis_name][active],
                ),
                "direction_correlation_with_local": _safe_corr(
                    independent[active] - parents[axis_name][active],
                    axis_components[axis_name]["local_v184"][active] - parents[axis_name][active],
                ),
            }
            detail[name][axis_name] = {
                "total": total,
                "incremental_over_v178": incremental,
                "diagnostics": diagnostics,
            }
        incremental = {
            axis_name: values["incremental_over_v178"]["gain"]
            for axis_name, values in detail[name].items()
        }
        rows.append(
            {
                "component": name,
                "n_axes": len(incremental),
                "full_2022_incremental": incremental.get("full_2022", np.nan),
                "late_2023_incremental": incremental.get("late_2023", np.nan),
                "full_2024_incremental": incremental.get("full_2024", np.nan),
                "minimum_incremental": min(incremental.values()),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["minimum_incremental", "full_2024_incremental"], ascending=False,
        kind="stable",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / "component_ranking.csv", index=False, encoding="utf-8-sig")
    summary = {
        "protocol": PROTOCOL,
        "status": "research_only",
        "diagnostic_weight": DIAGNOSTIC_WEIGHT,
        "ranking": ranking.to_dict(orient="records"),
        "detail": detail,
        "parity": parity,
        "restrictions": {
            "public_predictions_research_only": True,
            "eligible_for_packaging": False,
            "test_csv_read": False,
            "public_score_used": False,
            "locked_2024_used_for_diagnosis_only": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--v158-path", type=Path, required=True)
    parser.add_argument("--v165-summary", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--unified-root", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--local-oof-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        args.train_csv, args.contract_dir, args.v104_path, args.h1_path,
        args.c3_path, args.v160_path, args.bridge_oof, args.v158_path,
        args.v165_summary, args.library_root, args.unified_root,
        args.anchor_root, args.local_oof_root, args.output_dir,
    )
    print(json.dumps(summary["ranking"], ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
