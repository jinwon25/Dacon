"""Fixed Ridge soft-benefit gate for the v100 consensus direction."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.archive.v78_environment_stable_residual import build_features
from src.core.contract import _diagnostics, _load_contract_axis


PROTOCOL = "V102_CONSENSUS_BENEFIT_GATE_V1"


@dataclass
class GateModel:
    scaler: StandardScaler
    model: Ridge
    soft_scale: float


def gate_features(
    frame: pd.DataFrame,
    parent: np.ndarray,
    raw_candidate: np.ndarray,
    conditional_correction: np.ndarray,
) -> pd.DataFrame:
    output = build_features(frame)
    shift = np.asarray(raw_candidate, dtype=np.float64) - np.asarray(parent, dtype=np.float64)
    conditional = np.asarray(conditional_correction, dtype=np.float64)
    output["parent"] = parent
    output["consensus_shift"] = shift
    output["consensus_abs_shift"] = np.abs(shift)
    output["consensus_nonzero"] = (shift != 0.0).astype(np.float64)
    output["conditional_correction"] = conditional
    output["conditional_abs_correction"] = np.abs(conditional)
    output["shift_x_conditional"] = shift * conditional
    if not np.isfinite(output.to_numpy(np.float64)).all():
        raise ValueError("non-finite benefit-gate feature")
    return output


def _month_weights(frame: pd.DataFrame, active: np.ndarray) -> np.ndarray:
    month = pd.to_numeric(frame["game_month"], errors="raise").to_numpy(np.int16)
    weight = np.zeros(len(frame), dtype=np.float64)
    values, counts = np.unique(month[active], return_counts=True)
    for value, count in zip(values, counts, strict=True):
        weight[active & (month == value)] = 1.0 / float(count)
    weight *= float(active.sum()) / weight.sum()
    return weight


def fit_gate(
    features: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    frame: pd.DataFrame,
    alpha: float,
    quantile: float,
) -> GateModel:
    active = frame["domain3"].astype(str).eq("R_CORE").to_numpy() & (
        np.asarray(candidate) != np.asarray(parent)
    )
    benefit = (
        np.square(np.asarray(parent) - np.asarray(target))
        - np.square(np.asarray(candidate) - np.asarray(target))
    ) * 1_000_000.0
    weight = _month_weights(frame, active)
    x = features.to_numpy(np.float64)
    scaler = StandardScaler().fit(x[active], sample_weight=weight[active])
    model = Ridge(alpha=float(alpha)).fit(
        scaler.transform(x[active]), benefit[active], sample_weight=weight[active]
    )
    fitted = model.predict(scaler.transform(x[active]))
    positive = fitted[fitted > 0.0]
    scale = float(np.quantile(positive, float(quantile))) if len(positive) else 1.0
    return GateModel(scaler, model, max(scale, 1e-9))


def apply_gate(
    model: GateModel,
    features: pd.DataFrame,
    parent: np.ndarray,
    raw_candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = model.model.predict(model.scaler.transform(features.to_numpy(np.float64)))
    gate = np.clip(expected / model.soft_scale, 0.0, 1.0)
    candidate = np.clip(
        np.asarray(parent) + gate * (np.asarray(raw_candidate) - np.asarray(parent)),
        0.001,
        0.999,
    )
    return candidate, gate, expected


def _metric(
    frame: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    gate: np.ndarray,
) -> dict[str, Any]:
    axis = {
        "target": np.asarray(target, dtype=np.float64),
        "parent": np.asarray(parent, dtype=np.float64),
        "direct": np.asarray(candidate, dtype=np.float64),
        "domain3": frame["domain3"].astype(str).to_numpy(),
        "game_month": frame["game_month"].to_numpy(np.int16),
    }
    result = _diagnostics(axis, ("R_CORE",), 1.0)
    core = axis["domain3"] == "R_CORE"
    result["gate_mean_r_core"] = float(np.mean(gate[core]))
    result["gate_nonzero_fraction_r_core"] = float(np.mean(gate[core] > 0.0))
    return result


def run(
    train_csv: Path,
    contract_dir: Path,
    v98_dir: Path,
    v100_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_csv, low_memory=False)
    exact = {
        2022: _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        2023: _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        2024: _load_contract_axis(contract_dir / "v84_full_2024.npz"),
    }
    with np.load(v100_dir / "selected_axes.npz") as saved:
        candidate = {
            2022: saved["candidate22"].astype(np.float64),
            2023: saved["candidate23"].astype(np.float64),
            2024: saved["candidate24"].astype(np.float64),
        }
    correction = {
        2022: np.load(v98_dir / "ablation_full_2022.npz")["correction"].astype(np.float64),
        2024: np.load(v98_dir / "ablation_full_2024.npz")["correction"].astype(np.float64),
    }
    common23 = _load_contract_axis(contract_dir / "common_full_2023.npz")
    correction23_full = np.load(v98_dir / "ablation_full_2023.npz")["correction"].astype(np.float64)
    location = {int(value): index for index, value in enumerate(common23["raw_index"])}
    take = np.asarray([location[int(value)] for value in exact[2023]["raw_index"]], dtype=np.int64)
    correction[2023] = correction23_full[take]

    frames = {
        2022: raw.loc[raw["season"].eq(2022)].reset_index(drop=True),
        2023: raw.loc[raw["season"].eq(2023) & raw["game_month"].ge(8)].reset_index(drop=True),
        2024: raw.loc[raw["season"].eq(2024)].reset_index(drop=True),
    }
    for year in frames:
        frames[year]["domain3"] = exact[year]["domain3"]
    features = {
        year: gate_features(
            frames[year], exact[year]["parent"], candidate[year], correction[year]
        )
        for year in frames
    }
    early = frames[2022]["game_month"].le(7).to_numpy()
    late = frames[2022]["game_month"].ge(8).to_numpy()
    alpha = float(config["ridge_alpha"])
    quantile = float(config["soft_scale_quantile"])

    first = fit_gate(
        features[2022].loc[early].reset_index(drop=True),
        exact[2022]["target"][early], exact[2022]["parent"][early],
        candidate[2022][early], frames[2022].loc[early].reset_index(drop=True),
        alpha, quantile,
    )
    first_candidate, first_gate, _ = apply_gate(
        first, features[2022].loc[late].reset_index(drop=True),
        exact[2022]["parent"][late], candidate[2022][late],
    )
    first_metric = _metric(
        frames[2022].loc[late].reset_index(drop=True),
        exact[2022]["target"][late], exact[2022]["parent"][late],
        first_candidate, first_gate,
    )

    second = fit_gate(
        features[2022], exact[2022]["target"], exact[2022]["parent"],
        candidate[2022], frames[2022], alpha, quantile,
    )
    second_candidate, second_gate, _ = apply_gate(
        second, features[2023], exact[2023]["parent"], candidate[2023]
    )
    second_metric = _metric(
        frames[2023], exact[2023]["target"], exact[2023]["parent"],
        second_candidate, second_gate,
    )
    sources = [first_metric, second_metric]
    source_passed = bool(
        all(item["gain"] > 0.0 for item in sources)
        and all(
            item["positive_month_fraction"]
            >= float(config["source_minimum_positive_month_fraction"])
            for item in sources
        )
        and all(
            item["worst_month_gain"] > float(config["worst_month_gain_strictly_above"])
            for item in sources
        )
    )

    joined_features = pd.concat([features[2022], features[2023]], ignore_index=True)
    joined_frame = pd.concat([frames[2022], frames[2023]], ignore_index=True)
    joined_target = np.concatenate([exact[2022]["target"], exact[2023]["target"]])
    joined_parent = np.concatenate([exact[2022]["parent"], exact[2023]["parent"]])
    joined_candidate = np.concatenate([candidate[2022], candidate[2023]])
    final_model = fit_gate(
        joined_features, joined_target, joined_parent, joined_candidate,
        joined_frame, alpha, quantile,
    )
    locked_candidate, locked_gate, expected24 = apply_gate(
        final_model, features[2024], exact[2024]["parent"], candidate[2024]
    )
    locked_metric = _metric(
        frames[2024], exact[2024]["target"], exact[2024]["parent"],
        locked_candidate, locked_gate,
    )
    locked_passed = bool(
        locked_metric["gain"] > 0.0
        and locked_metric["positive_month_fraction"]
        >= float(config["locked_minimum_positive_month_fraction"])
        and locked_metric["worst_month_gain"]
        > float(config["worst_month_gain_strictly_above"])
    )
    np.savez_compressed(
        output_dir / "locked_full_2024.npz",
        target=exact[2024]["target"], parent=exact[2024]["parent"],
        raw_candidate=candidate[2024], candidate=locked_candidate,
        gate=locked_gate, expected_benefit=expected24,
        domain3=exact[2024]["domain3"], game_month=exact[2024]["game_month"],
        pitcher_id=exact[2024]["pitcher_id"], batter_id=exact[2024]["batter_id"],
    )
    result = {
        "protocol": PROTOCOL,
        "source": {
            "early2022_to_late2022": first_metric,
            "full2022_to_late2023": second_metric,
        },
        "source_gate_passed": source_passed,
        "locked_full_2024": locked_metric,
        "locked_gate_passed": locked_passed,
        "point_gates_passed": bool(source_passed and locked_passed),
        "eligible_for_packaging": False,
        "packaging_reason": (
            "bootstrap and Reality Check pending"
            if source_passed and locked_passed else "source or locked point gate failed"
        ),
        "test_csv_read": False,
        "test_aggregate_used": False,
        "row_local_inference": True,
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
    parser.add_argument("--v98-dir", type=Path, required=True)
    parser.add_argument("--v100-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv, args.contract_dir, args.v98_dir,
        args.v100_dir, args.config, args.output_dir,
    )


if __name__ == "__main__":
    main()
