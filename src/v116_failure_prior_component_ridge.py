"""Cross-origin ridge dosing of the frozen v113 failure-prior components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.axis_metrics import _axis_metrics, _robust_axis
from src.v104_source_stability_mask import _point_pass
from src.core.contract import _load_contract_axis


PROTOCOL = "V116_FAILURE_PRIOR_COMPONENT_RIDGE_V1"
AXES = ("full_2022", "late_2023", "full_2024", "late_2024")
SOURCE_AXES = ("full_2022", "late_2023")


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in axis.items()}


def fit_ridge(
    matrices: list[np.ndarray],
    residuals: list[np.ndarray],
    ridge_lambda: float,
    anchor: np.ndarray,
) -> np.ndarray:
    """Fit a ridge dose toward a frozen external three-component anchor."""
    x = np.concatenate([np.asarray(value, float) for value in matrices], axis=0)
    y = np.concatenate([np.asarray(value, float) for value in residuals])
    gram = x.T @ x + float(ridge_lambda) * np.eye(x.shape[1])
    rhs = x.T @ y + float(ridge_lambda) * np.asarray(anchor, float)
    return np.linalg.solve(gram, rhs).astype(np.float64)


def apply_components(parent: np.ndarray, matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.clip(
        np.asarray(parent, float) + np.asarray(matrix, float) @ np.asarray(weights, float),
        0.001,
        0.999,
    )


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_dir: Path,
    v113_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(train_csv, low_memory=False)
    frames = {
        "full_2022": raw.loc[raw["season"].eq(2022)].reset_index(drop=True),
        "late_2023": raw.loc[
            raw["season"].eq(2023) & raw["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": raw.loc[raw["season"].eq(2024)].reset_index(drop=True),
    }
    exact = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in ("full_2022", "late_2023", "full_2024")
    }
    late24 = exact["full_2024"]["game_month"].astype(np.int16) >= 8
    exact["late_2024"] = _slice_axis(exact["full_2024"], late24)
    frames["late_2024"] = frames["full_2024"].loc[late24].reset_index(drop=True)

    with np.load(v104_dir / "selected_axes.npz") as saved:
        parent = {name: saved[name].astype(np.float64) for name in AXES}
    components = [str(value) for value in config["components"]]
    route_domains = [str(value) for value in config["route_domains"]]
    with np.load(v113_dir / "component_axes.npz") as saved:
        matrices = {
            name: np.column_stack(
                [saved[f"{component}__{name}"].astype(np.float64) for component in components]
            )
            for name in AXES
        }
    for name in AXES:
        route = np.isin(exact[name]["domain3"].astype(str), route_domains)
        matrices[name][~route] = 0.0
        if not (len(exact[name]["target"]) == len(parent[name]) == len(matrices[name])):
            raise ValueError(f"axis length mismatch: {name}")

    anchor = np.asarray(config["anchor_weights"], dtype=np.float64)
    lambdas = [float(value) for value in config["ridge_lambda"]]
    source_masks = {
        name: exact[name]["exact_mask"].astype(bool) for name in SOURCE_AXES
    }
    residuals = {
        name: exact[name]["target"] - parent[name] for name in SOURCE_AXES
    }
    cross_rows: list[dict[str, Any]] = []
    for ridge_lambda in lambdas:
        cross_gain: dict[str, float] = {}
        fold_weights: dict[str, list[float]] = {}
        for train_name, test_name in (
            ("full_2022", "late_2023"),
            ("late_2023", "full_2022"),
        ):
            mask = source_masks[train_name]
            weights = fit_ridge(
                [matrices[train_name][mask]],
                [residuals[train_name][mask]],
                ridge_lambda,
                anchor,
            )
            fold_weights[train_name] = weights.tolist()
            candidate = apply_components(parent[test_name], matrices[test_name], weights)
            cross_gain[test_name] = float(
                _axis_metrics(
                    {**exact[test_name], "parent": parent[test_name]}, candidate
                )["gain"]
            )
        values = list(cross_gain.values())
        cross_rows.append(
            {
                "ridge_lambda": ridge_lambda,
                "both_cross_gains_positive": bool(min(values) > 0.0),
                "minimum_cross_gain": float(min(values)),
                "mean_cross_gain": float(np.mean(values)),
                "cross_gain": cross_gain,
                "fold_weights": fold_weights,
            }
        )
    ranking = sorted(
        cross_rows,
        key=lambda row: (
            row["both_cross_gains_positive"],
            row["minimum_cross_gain"],
            row["mean_cross_gain"],
        ),
        reverse=True,
    )
    selected_lambda = float(ranking[0]["ridge_lambda"])
    pooled_weights_by_lambda: dict[float, np.ndarray] = {}
    for ridge_lambda in lambdas:
        pooled_weights_by_lambda[ridge_lambda] = fit_ridge(
            [matrices[name][source_masks[name]] for name in SOURCE_AXES],
            [residuals[name][source_masks[name]] for name in SOURCE_AXES],
            ridge_lambda,
            anchor,
        )
    selected_weights = pooled_weights_by_lambda[selected_lambda]
    candidates = {
        name: apply_components(parent[name], matrices[name], selected_weights)
        for name in AXES
    }
    rebased = {name: {**exact[name], "parent": parent[name]} for name in AXES}
    metrics = {name: _axis_metrics(rebased[name], candidates[name]) for name in AXES}
    gate = config["selection_gate"]
    point_pass = {name: _point_pass(value, gate) for name, value in metrics.items()}
    family = {
        name: [
            apply_components(parent[name], matrices[name], pooled_weights_by_lambda[value])
            for value in lambdas
        ]
        for name in AXES
    }
    robust = {
        name: _robust_axis(
            frames[name], rebased[name], candidates[name], family[name], config,
            500 + 10 * index,
        )
        for index, name in enumerate(AXES)
    }
    robust_pass = {
        name: bool(
            all(
                value[key]["p05"] > 0.0
                for key in ("pitcher", "crossed_pitcher_batter", "chronological_block")
            )
            and value["reality_check"]["p_value"] <= float(gate["reality_check_alpha"])
        )
        for name, value in robust.items()
    }
    eligible = bool(
        ranking[0]["both_cross_gains_positive"]
        and all(point_pass.values())
        and all(robust_pass.values())
    )
    np.savez_compressed(output_dir / "selected_axes.npz", **candidates)
    np.save(output_dir / "selected_weights.npy", selected_weights)
    result = {
        "protocol": PROTOCOL,
        "status": "promote" if eligible else "reject",
        "cross_origin_ranking": ranking,
        "selected_lambda": selected_lambda,
        "selected_weights": dict(zip(components, selected_weights.tolist())),
        "anchor_weights": dict(zip(components, anchor.tolist())),
        "route_domains": route_domains,
        "metrics": metrics,
        "point_gate_pass": point_pass,
        "robust": robust,
        "robust_gate_pass": robust_pass,
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
    parser.add_argument("--v113-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv, args.contract_dir, args.v104_dir, args.v113_dir,
        args.config, args.output_dir,
    )


if __name__ == "__main__":
    main()
