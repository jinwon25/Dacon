"""Extend v142 only onto recent-sign C3 rows excluded by sign-all consensus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.v103_fixed_union_robust import _axis_metrics
from src.v127_hoo_current_state_rebase import apply_correction
from src.v130_hoo_independent_oof_blend import post4
from src.v133_hoo_h1_c3_forward import CONTEXT_COLS, c3_adjustment
from src.v135_c3_recent_window import SOURCE_AXES, _slice_axis, _source_years
from src.v97_conditional_direct_forward import _load_contract_axis


PROTOCOL = "V146_HIERARCHICAL_C3_EXTENSION_V1"
WINDOWS = ("last1", "last2", "last3", "expanding")


def _point_pass(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return bool(
        metrics["positive_month_fraction"]
        >= float(gate["positive_month_fraction_min"])
        and metrics["worst_month_gain"]
        > float(gate["worst_month_gain_min_exclusive"])
        and metrics["minimum_domain_gain"] >= float(gate["active_domain_gain_min"])
    )


def _extra_from_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recent = matrix[:, :2]
    recent_agreed = np.all(recent > 0.0, axis=1) | np.all(recent < 0.0, axis=1)
    all_agreed = np.all(matrix > 0.0, axis=1) | np.all(matrix < 0.0, axis=1)
    extra_mask = recent_agreed & ~all_agreed
    extra = recent.mean(axis=1) * extra_mask
    return extra, extra_mask, all_agreed


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_dir: Path,
    v133_dir: Path,
    v139_dir: Path,
    v141_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    context = pd.read_csv(train_csv, usecols=list(CONTEXT_COLS), low_memory=False)
    season = context["season"].to_numpy(np.int16)
    frames = {
        year: context.loc[season == year].reset_index(drop=True)
        for year in range(2019, 2025)
    }
    with np.load(v133_dir / "oof_predictions.npz", allow_pickle=False) as saved:
        h1 = {year: saved[f"h1_{year}"].astype(np.float64) for year in range(2020, 2025)}
    post = {
        year: post4(context.loc[season < year].reset_index(drop=True), frames[year])
        for year in range(2020, 2025)
    }
    residual = {
        year: frames[year]["control_success"].to_numpy(np.float64) - (h1[year] + post[year])
        for year in range(2020, 2025)
    }
    window_c3: dict[str, dict[int, np.ndarray]] = {window: {} for window in WINDOWS}
    for window in WINDOWS:
        for year in (2022, 2023, 2024):
            sources = _source_years(year, window)
            history = pd.concat([frames[value] for value in sources], ignore_index=True)
            history_residual = np.concatenate([residual[value] for value in sources])
            window_c3[window][year] = c3_adjustment(
                history, history_residual, frames[year], config["c3"]
            )
    extras = {}
    masks = {}
    all_masks = {}
    for year in (2022, 2023, 2024):
        matrix = np.column_stack([window_c3[window][year] for window in WINDOWS])
        extras[year], masks[year], all_masks[year] = _extra_from_matrix(matrix)
        print(
            f"[v146] year={year} extra_fraction={masks[year].mean():.6f} "
            f"sign_all_fraction={all_masks[year].mean():.6f} "
            f"extra_mean_abs={np.mean(np.abs(extras[year])):.8f}",
            flush=True,
        )

    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in (*SOURCE_AXES, "full_2024")
    }
    with np.load(v104_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v104 = {name: saved[name].astype(np.float64) for name in saved.files}
    with np.load(v139_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v139 = {name: saved[name].astype(np.float64) for name in saved.files}
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    correction = {
        "full_2022": extras[2022],
        "late_2023": extras[2023][late23],
    }
    metric_axes = {name: {**axes[name], "parent": v104[name]} for name in SOURCE_AXES}
    trials = []
    candidates = {}
    for eta in (float(value) for value in config["extra_eta_grid"]):
        by_axis = {
            name: apply_correction(v139[name], axes[name], correction[name], config["route"], eta)
            for name in SOURCE_AXES
        }
        metrics = {
            name: _axis_metrics(metric_axes[name], by_axis[name]) for name in SOURCE_AXES
        }
        trial = {
            "eta": eta,
            "source_gate_passed": bool(
                all(_point_pass(item, config["source_gate"]) for item in metrics.values())
            ),
            "minimum_gain": float(min(item["gain"] for item in metrics.values())),
            "mean_gain": float(np.mean([item["gain"] for item in metrics.values()])),
            "minimum_month_fraction": float(
                min(item["positive_month_fraction"] for item in metrics.values())
            ),
            "worst_month_gain": float(min(item["worst_month_gain"] for item in metrics.values())),
            "metrics": metrics,
        }
        trials.append(trial)
        candidates[eta] = by_axis
    passing = [trial for trial in trials if trial["source_gate_passed"]]
    if not passing:
        raise ValueError("no hierarchical C3 dose passed source gate")
    selected = max(passing, key=lambda row: (row["minimum_gain"], row["mean_gain"], -row["eta"]))
    eta = float(selected["eta"])

    with np.load(v141_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v142 = saved["full_2024"].astype(np.float64)
        v124 = saved["v124_full_2024"].astype(np.float64)
    candidate24 = apply_correction(v142, axes["full_2024"], extras[2024], config["route"], eta)
    late24 = frames[2024]["game_month"].ge(8).to_numpy()
    axis_v124 = {**axes["full_2024"], "parent": v124}
    axis_v142 = {**axes["full_2024"], "parent": v142}
    vs_v124 = {
        "full_2024": _axis_metrics(axis_v124, candidate24),
        "late_2024": _axis_metrics(_slice_axis(axis_v124, late24), candidate24[late24]),
    }
    incremental = {
        "full_2024": _axis_metrics(axis_v142, candidate24),
        "late_2024": _axis_metrics(_slice_axis(axis_v142, late24), candidate24[late24]),
    }
    veto = config["locked_incremental_veto"]
    locked_pass = bool(
        incremental["full_2024"]["gain"] >= float(veto["full_gain_min"])
        and incremental["late_2024"]["gain"] >= float(veto["late_gain_min"])
        and incremental["full_2024"]["worst_month_gain"]
        > float(veto["worst_month_gain_min_exclusive"])
        and incremental["late_2024"]["worst_month_gain"]
        > float(veto["worst_month_gain_min_exclusive"])
    )
    eligible = bool(eta > 0.0 and locked_pass)
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2022=candidates[eta]["full_2022"],
        late_2023=candidates[eta]["late_2023"],
        full_2024=candidate24,
        late_2024=candidate24[late24],
        extra_c3_full_2024=extras[2024],
        extra_mask_full_2024=masks[2024],
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_packaging" if eligible else "reject",
        "selected_eta": eta,
        "source_trials": trials,
        "selected_source": selected,
        "locked_metrics_vs_v124": vs_v124,
        "locked_incremental_vs_v142": incremental,
        "locked_veto_passed": locked_pass,
        "coverage": {
            "full_2022_extra_fraction": float(masks[2022].mean()),
            "full_2024_extra_fraction": float(masks[2024].mean()),
            "full_2024_sign_all_fraction": float(all_masks[2024].mean()),
        },
        "eligible_for_packaging": eligible,
        **config["restrictions"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-dir", type=Path, required=True)
    parser.add_argument("--v133-dir", type=Path, required=True)
    parser.add_argument("--v139-dir", type=Path, required=True)
    parser.add_argument("--v141-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv,
        args.contract_dir,
        args.v104_dir,
        args.v133_dir,
        args.v139_dir,
        args.v141_dir,
        args.config,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
