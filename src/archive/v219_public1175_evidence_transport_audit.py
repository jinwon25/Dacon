"""Transport the joint-workload H1 delta onto the preserved Public1175 OOF axis.

The release evaluator did not store exact JY1172 OOF predictions.  It evaluated
the fallback against the older v84/v148 ``parent`` arrays instead.  This audit
therefore keeps two objects separate: v218 supplies only the paired joint-H1
delta reconstructed on the exact JY formula, while v84/v148 supply the frozen
Public1175 evidence parent.  Candidate dose is selected on 2022/late-2023 and
is then evaluated once on locked 2024.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_jy_exact_contract_reaudit import metrics
from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v218_public1175_joint_h1_incremental_audit import (
    PUBLISHED_FALLBACK,
    align_regular_prediction,
    apply_fallback,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V219_PUBLIC1175_EVIDENCE_TRANSPORT_AUDIT_V1"
SCALES = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
AXIS_NAMES = ("full_2022", "late_2023", "full_2024")


def evaluator_bss_gain(
    target: np.ndarray, base: np.ndarray, candidate: np.ndarray
) -> float:
    target = np.asarray(target, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    rate = float(np.mean(target))
    denominator = rate * (1.0 - rate)
    if denominator <= 0.0:
        return float("nan")
    base_bss = 1.0e5 * (1.0 - np.mean((target - base) ** 2) / denominator)
    candidate_bss = 1.0e5 * (
        1.0 - np.mean((target - candidate) ** 2) / denominator
    )
    return float(candidate_bss - base_bss)


def pressure_gate(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    core = regular & ~(
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    return core & (
        (frame["num_runners_on"].to_numpy(np.float64) > 0.0)
        | (frame["li"].to_numpy(np.float64) >= 1.5)
    )


def transport(parent: np.ndarray, delta: np.ndarray, scale: float) -> np.ndarray:
    if np.asarray(parent).shape != np.asarray(delta).shape:
        raise ValueError("parent/delta shape mismatch")
    return np.clip(
        np.asarray(parent, dtype=np.float64)
        + float(scale) * np.asarray(delta, dtype=np.float64),
        0.001,
        0.999,
    )


def select_source_scale(source_results: dict[str, dict[str, Any]]) -> float | None:
    eligible: list[tuple[float, float, float]] = []
    for key, result in source_results.items():
        full22 = result["full_2022"]
        late23 = result["late_2023"]
        if (
            full22["gain"] > 0.0
            and late23["gain"] > 0.0
            and full22["positive_month_fraction"] >= 0.70
            and late23["positive_month_fraction"] >= (2.0 / 3.0)
            and full22["worst_month_gain"] > -5.0
            and late23["worst_month_gain"] > -5.0
        ):
            scale = float(key)
            minimum = min(full22["gain"], late23["gain"])
            mean = 0.5 * (full22["gain"] + late23["gain"])
            eligible.append((minimum, mean, scale))
    if not eligible:
        return None
    return max(eligible)[2]


def restrictions() -> dict[str, bool]:
    return {
        "preserved_evaluator_parent_axis": True,
        "exact_jy_joint_delta_only": True,
        "source_only_scale_selection": True,
        "locked_2024_not_used_for_selection": True,
        "fallback_weight_and_threshold_frozen": True,
        "fallback_oof_cpu_reconstruction": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    v218_axes_path: Path,
    fallback_oof_dir: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "season", "game_type", "pitcher_team_id", "batter_team_id",
        "num_runners_on", "li", "game_month", "pitcher_id",
    ]
    train = pd.read_csv(train_csv, usecols=columns, low_memory=False)
    season = train["season"].to_numpy(np.int16)
    frames = {
        year: train.loc[season == year].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    axis_frames = {
        "full_2022": frames[2022],
        "late_2023": frames[2023].loc[late23].reset_index(drop=True),
        "full_2024": frames[2024],
    }
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    evidence_parent = {
        name: axes[name]["parent"].astype(np.float64) for name in AXIS_NAMES
    }
    with np.load(v218_axes_path, allow_pickle=False) as saved:
        jy_delta = {
            name: (
                saved[f"jy_candidate_{name}"].astype(np.float64)
                - saved[f"jy_parent_{name}"].astype(np.float64)
            )
            for name in AXIS_NAMES
        }
    for name in AXIS_NAMES:
        if len(jy_delta[name]) != len(evidence_parent[name]):
            raise ValueError(f"joint delta axis mismatch: {name}")

    full_xgb = {
        year: align_regular_prediction(
            frames[year], fallback_oof_dir / f"hyunku_fallback_xgb_{year}.npy"
        )
        for year in (2022, 2023, 2024)
    }
    xgb = {
        "full_2022": full_xgb[2022],
        "late_2023": full_xgb[2023][late23],
        "full_2024": full_xgb[2024],
    }
    pressure = {name: pressure_gate(axis_frames[name]) for name in AXIS_NAMES}
    public1175: dict[str, np.ndarray] = {}
    baseline_active: dict[str, np.ndarray] = {}
    fallback_parity: dict[str, dict[str, Any]] = {}
    for name in AXIS_NAMES:
        public1175[name], baseline_active[name] = apply_fallback(
            evidence_parent[name], xgb[name], pressure[name]
        )
        measured = metrics(axes[name], evidence_parent[name], public1175[name])
        evaluator_gain = evaluator_bss_gain(
            axes[name]["target"], evidence_parent[name], public1175[name]
        )
        expected = PUBLISHED_FALLBACK[name]
        active_rows = int(baseline_active[name].sum())
        fallback_parity[name] = {
            **measured,
            "active_rows": active_rows,
            "expected_active_rows": expected["active_rows"],
            "active_row_match": active_rows == expected["active_rows"],
            "expected_gpu_gain": expected["gain"],
            "evaluator_cpu_gain": evaluator_gain,
            "cpu_minus_gpu_gain": float(evaluator_gain - expected["gain"]),
            "gain_abs_difference": float(abs(evaluator_gain - expected["gain"])),
        }
    near_parity = all(
        item["active_row_match"] and item["gain_abs_difference"] <= 1.0
        for item in fallback_parity.values()
    )

    candidates: dict[str, dict[str, np.ndarray]] = {}
    active: dict[str, dict[str, np.ndarray]] = {}
    scale_results: dict[str, dict[str, Any]] = {}
    for scale in SCALES:
        key = f"{scale:.2f}"
        candidates[key] = {}
        active[key] = {}
        scale_results[key] = {}
        for name in AXIS_NAMES:
            transported = transport(evidence_parent[name], jy_delta[name], scale)
            candidates[key][name], active[key][name] = apply_fallback(
                transported, xgb[name], pressure[name]
            )
            result = metrics(axes[name], public1175[name], candidates[key][name])
            result["changed_rows"] = int(np.sum(
                np.abs(candidates[key][name] - public1175[name]) > 1e-15
            ))
            result["fallback_gate_entries"] = int(np.sum(
                active[key][name] & ~baseline_active[name]
            ))
            result["fallback_gate_exits"] = int(np.sum(
                baseline_active[name] & ~active[key][name]
            ))
            scale_results[key][name] = result

    source_results = {
        key: {
            name: values[name] for name in ("full_2022", "late_2023")
        }
        for key, values in scale_results.items()
    }
    selected_scale = select_source_scale(source_results)
    selected_key = None if selected_scale is None else f"{selected_scale:.2f}"
    locked = None if selected_key is None else scale_results[selected_key]["full_2024"]
    locked_point_pass = bool(
        locked is not None
        and locked["gain"] > 0.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -3.0
        and locked["minimum_domain_gain"] >= 0.0
    )
    robustness = None
    robust_pass = False
    if selected_key is not None:
        changed = np.abs(
            candidates[selected_key]["full_2024"] - public1175["full_2024"]
        ) > 1e-15
        robustness = _robustness(
            axes["full_2024"],
            public1175["full_2024"],
            candidates[selected_key]["full_2024"],
            changed,
            [
                public1175["full_2024"],
                candidates[selected_key]["full_2024"],
            ],
        )
        robust_pass = bool(
            robustness["pitcher"]["p05"] > 0.0
            and robustness["crossed_pitcher_batter"]["p05"] > 0.0
            and robustness["chronological_block"]["p05"] > 0.0
            and robustness["reality_check"]["p_value"] <= 0.10
        )
    confirm = bool(near_parity and locked_point_pass and robust_pass)

    arrays: dict[str, np.ndarray] = {}
    for name in AXIS_NAMES:
        arrays[f"evidence_parent_{name}"] = evidence_parent[name]
        arrays[f"joint_delta_{name}"] = jy_delta[name]
        arrays[f"public1175_proxy_{name}"] = public1175[name]
        if selected_key is not None:
            arrays[f"selected_candidate_{name}"] = candidates[selected_key][name]
    np.savez_compressed(output_dir / "evidence_transport_axes.npz", **arrays)
    summary = {
        "protocol": PROTOCOL,
        "status": "confirmed_for_full_fit" if confirm else (
            "source_scale_reject" if selected_key is None else (
                "locked_point_reject" if not locked_point_pass else "robust_reject"
            )
        ),
        "interpretation": (
            "OOF proxy: exact active support, CPU reconstruction near published "
            "GPU gain; not an exact saved JY1172 OOF replay."
        ),
        "fallback_evidence_near_parity": fallback_parity,
        "fallback_near_parity_passed": near_parity,
        "scale_results": scale_results,
        "selected_scale_from_sources": selected_scale,
        "locked_2024": locked,
        "locked_point_passed": locked_point_pass,
        "robustness": robustness,
        "robust_gate_passed": robust_pass,
        "eligible_for_full_fit": confirm,
        "eligible_for_packaging": False,
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v218-axes-path", type=Path, required=True)
    parser.add_argument("--fallback-oof-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.v218_axes_path, args.fallback_oof_dir,
        args.contract_dir, args.bridge_oof, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
