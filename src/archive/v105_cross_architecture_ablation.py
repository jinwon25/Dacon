"""Paired MLP ablation and cross-architecture conditional-delta consensus."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.contract import _diagnostics, _load_contract_axis
from src.archive.v97_conditional_direct_forward import _bank, _feature_frame
from src.archive.v99_conditional_mlp_axis import prepare, train_predict
from src.core.axis_metrics import _robust_axis


PROTOCOL = "V105_CROSS_ARCHITECTURE_ABLATION_V1"


def combine_deltas(lgb: np.ndarray, mlp: np.ndarray, policy: str) -> np.ndarray:
    left = np.asarray(lgb, dtype=np.float64)
    right = np.asarray(mlp, dtype=np.float64)
    if policy == "mlp_only":
        return right.copy()
    if policy == "mean":
        return 0.5 * (left + right)
    agree = (np.signbit(left) == np.signbit(right)) & (left != 0.0) & (right != 0.0)
    result = np.zeros_like(left)
    if policy == "sign_mean":
        result[agree] = 0.5 * (left[agree] + right[agree])
        return result
    if policy == "sign_min":
        result[agree] = np.sign(left[agree]) * np.minimum(
            np.abs(left[agree]), np.abs(right[agree])
        )
        return result
    raise ValueError(f"unknown policy: {policy}")


def _align(
    common: dict[str, np.ndarray], exact: dict[str, np.ndarray], values: np.ndarray
) -> np.ndarray:
    location = {int(value): index for index, value in enumerate(common["raw_index"])}
    return values[
        np.asarray([location[int(value)] for value in exact["raw_index"]], dtype=np.int64)
    ]


def _candidate(
    exact: dict[str, np.ndarray], correction: np.ndarray, eta: float
) -> np.ndarray:
    parent = exact["parent"].astype(np.float64)
    active = exact["domain3"].astype(str) == "R_CORE"
    result = parent.copy()
    result[active] = np.clip(
        parent[active] + float(eta) * correction[active], 0.001, 0.999
    )
    return result


def _metrics(
    exact: dict[str, np.ndarray], candidate: np.ndarray
) -> dict[str, Any]:
    mask = exact["exact_mask"].astype(bool)
    axis = {
        "target": exact["target"][mask].astype(np.float64),
        "parent": exact["parent"][mask].astype(np.float64),
        "direct": candidate[mask].astype(np.float64),
        "domain3": exact["domain3"][mask],
        "game_month": exact["game_month"][mask],
    }
    return _diagnostics(axis, ("R_CORE",), 1.0)


def _optimal_eta(
    axes: list[tuple[dict[str, np.ndarray], np.ndarray]], cap: float
) -> float:
    numerator = 0.0
    denominator = 0.0
    for exact, correction in axes:
        active = exact["exact_mask"].astype(bool) & (
            exact["domain3"].astype(str) == "R_CORE"
        )
        direction = correction[active]
        residual = exact["target"][active].astype(np.float64) - exact["parent"][active].astype(np.float64)
        numerator += float(np.dot(residual, direction))
        denominator += float(np.dot(direction, direction))
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, float(cap)))


def _point_pass(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return bool(
        metrics["gain"] > 0.0
        and metrics["positive_month_fraction"]
        >= float(gate["minimum_positive_month_fraction"])
        and metrics["worst_month_gain"]
        > float(gate["worst_month_gain_strictly_above"])
        and metrics["minimum_domain_gain"] >= float(gate["minimum_domain_gain"])
    )


def run(
    train_csv: Path,
    contract_dir: Path,
    v97_config_path: Path,
    v98_dir: Path,
    v99_config_path: Path,
    v99_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    v97_config = json.loads(v97_config_path.read_text(encoding="utf-8"))
    v99_config = json.loads(v99_config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_csv, low_memory=False)
    audit_years = [int(value) for value in config["audit_years"]]
    feature_years = list(range(int(raw["season"].min()), max(audit_years) + 1))
    conditional_columns = [str(value) for value in config["conditional_columns"]]
    baseline_features: dict[int, pd.DataFrame] = {}
    for year in feature_years:
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        history = raw.loc[raw["season"].lt(year)]
        conditional = _feature_frame(rows, _bank(history, v97_config["conditional_strength"]))
        baseline_features[year] = conditional.drop(columns=conditional_columns)
        print(f"[v105 features] year={year} rows={len(rows)}", flush=True)

    common: dict[int, dict[str, np.ndarray]] = {}
    mlp_delta: dict[int, np.ndarray] = {}
    lgb_delta: dict[int, np.ndarray] = {}
    for year in audit_years:
        fit_years = [value for value in feature_years if value < year]
        fit = pd.concat([baseline_features[value] for value in fit_years], ignore_index=True)
        audit = baseline_features[year]
        fit_mask = raw["season"].lt(year).to_numpy()
        prepared = prepare(
            fit, audit,
            raw.loc[fit_mask, "control_success"].to_numpy(np.float32),
            raw.loc[fit_mask, "season"].to_numpy(np.int16),
            year, float(v99_config["recency_half_life"]),
        )
        print(f"[v105 baseline MLP] audit_year={year} fit_rows={len(fit)}", flush=True)
        baseline = train_predict(prepared, v99_config["network"])
        with np.load(v99_dir / f"mlp_full_{year}.npz") as saved:
            conditional = saved["prediction"].astype(np.float64)
            conditional_index = saved["raw_index"].astype(np.int64)
        with np.load(v98_dir / f"ablation_full_{year}.npz") as saved:
            lgb_delta[year] = saved["correction"].astype(np.float64)
            lgb_index = saved["raw_index"].astype(np.int64)
        common[year] = _load_contract_axis(contract_dir / f"common_full_{year}.npz")
        expected_index = common[year]["raw_index"].astype(np.int64)
        if not np.array_equal(conditional_index, expected_index) or not np.array_equal(lgb_index, expected_index):
            raise ValueError(f"common-axis order mismatch for {year}")
        mlp_delta[year] = conditional - baseline
        np.savez_compressed(
            output_dir / f"paired_mlp_full_{year}.npz",
            raw_index=expected_index,
            baseline=baseline,
            conditional=conditional,
            correction=mlp_delta[year],
        )
        del fit, audit, prepared, baseline, conditional
        gc.collect()

    exact = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(contract_dir / "v84_full_2024.npz"),
    }
    frames = {
        "full_2022": raw.loc[raw["season"].eq(2022)].reset_index(drop=True),
        "late_2023": raw.loc[
            raw["season"].eq(2023) & raw["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": raw.loc[raw["season"].eq(2024)].reset_index(drop=True),
    }
    late24_mask = exact["full_2024"]["game_month"].astype(np.int16) >= 8
    exact["late_2024"] = {
        key: np.asarray(value)[late24_mask]
        for key, value in exact["full_2024"].items()
    }
    frames["late_2024"] = frames["full_2024"].loc[late24_mask].reset_index(drop=True)
    year_for_axis = {"full_2022": 2022, "late_2023": 2023, "full_2024": 2024}
    aligned: dict[str, dict[str, np.ndarray]] = {}
    for axis_name, year in year_for_axis.items():
        aligned[axis_name] = {
            policy: _align(
                common[year], exact[axis_name],
                combine_deltas(lgb_delta[year], mlp_delta[year], policy),
            )
            for policy in config["policies"]
        }
    aligned["late_2024"] = {
        policy: values[late24_mask] for policy, values in aligned["full_2024"].items()
    }

    gate = config["selection_gate"]
    ranking_rows = []
    candidates: dict[str, dict[str, np.ndarray]] = {}
    source_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for policy in config["policies"]:
        eta = _optimal_eta(
            [(exact[name], aligned[name][policy]) for name in ("full_2022", "late_2023")],
            float(config["eta_cap"]),
        )
        candidates[policy] = {
            name: _candidate(exact[name], aligned[name][policy], eta)
            for name in exact
        }
        source_metrics[policy] = {
            name: _metrics(exact[name], candidates[policy][name])
            for name in ("full_2022", "late_2023")
        }
        items = list(source_metrics[policy].values())
        ranking_rows.append(
            {
                "policy": policy,
                "eta": eta,
                "source_gate_passed": all(_point_pass(item, gate) for item in items),
                "minimum_gain": min(item["gain"] for item in items),
                "worst_month_gain": min(item["worst_month_gain"] for item in items),
                "minimum_month_fraction": min(item["positive_month_fraction"] for item in items),
            }
        )
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["source_gate_passed", "minimum_gain", "worst_month_gain"], ascending=False
    ).reset_index(drop=True)
    ranking.to_csv(output_dir / "source_policy_ranking.csv", index=False)
    selected = str(ranking.iloc[0]["policy"])
    selected_eta = float(ranking.iloc[0]["eta"])
    source_pass = bool(ranking.iloc[0]["source_gate_passed"])
    locked_metrics = {
        name: _metrics(exact[name], candidates[selected][name])
        for name in ("full_2024", "late_2024")
    }
    locked_point_pass = {name: _point_pass(item, gate) for name, item in locked_metrics.items()}
    family_by_axis = {
        name: [candidates[policy][name] for policy in config["policies"]]
        for name in exact
    }
    robust = {
        name: _robust_axis(
            frames[name], exact[name], candidates[selected][name], family_by_axis[name],
            config, 10 * index,
        )
        for index, name in enumerate(("full_2022", "late_2023", "full_2024", "late_2024"))
    }
    robust_pass = {
        name: bool(
            all(result[key]["p05"] > 0.0 for key in (
                "pitcher", "crossed_pitcher_batter", "chronological_block"
            ))
            and result["reality_check"]["p_value"] <= float(gate["reality_check_alpha"])
        )
        for name, result in robust.items()
    }
    eligible = bool(source_pass and all(locked_point_pass.values()) and all(robust_pass.values()))
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{name: candidates[selected][name] for name in candidates[selected]},
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote" if eligible else "reject",
        "source_ranking": ranking.to_dict(orient="records"),
        "selected_policy": selected,
        "selected_eta": selected_eta,
        "selected_source_metrics": source_metrics[selected],
        "selected_source_gate_passed": source_pass,
        "locked_metrics": locked_metrics,
        "locked_point_gate_pass": locked_point_pass,
        "robust": robust,
        "robust_gate_pass": robust_pass,
        "eligible_for_packaging": eligible,
        "packaging_reason": "all gates passed" if eligible else "point or robustness gate failed",
        **config["restrictions"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v97-config", type=Path, required=True)
    parser.add_argument("--v98-dir", type=Path, required=True)
    parser.add_argument("--v99-config", type=Path, required=True)
    parser.add_argument("--v99-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv, args.contract_dir, args.v97_config, args.v98_dir,
        args.v99_config, args.v99_dir, args.config, args.output_dir,
    )


if __name__ == "__main__":
    main()
