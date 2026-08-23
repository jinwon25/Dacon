"""Recent-form-centred spread calibration above exact v104 OOF.

The transform is row local: a pitcher's previous-game success summaries define
the centre, while a single frozen alpha changes only the champion's deviation
from that centre.  Alpha and route are selected using full-2022 and late-2023;
2024 is a locked audit.  No statistic is computed from an audit/test batch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.v103_fixed_union_robust import _axis_metrics, _robust_axis
from src.v97_conditional_direct_forward import _load_contract_axis


PROTOCOL = "V120_RECENT_ANCHOR_SPREAD_V1"
SOURCE_AXES = ("full_2022", "late_2023")
ALL_AXES = ("full_2022", "late_2023", "full_2024", "late_2024")
RECENT_COLUMNS = (
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
)


def recent_anchor(frame: pd.DataFrame, mode: str) -> np.ndarray:
    """Build a finite row-local anchor with career-rate fallback."""

    recent = np.column_stack(
        [pd.to_numeric(frame[column], errors="coerce").to_numpy(float) for column in RECENT_COLUMNS]
    )
    career = pd.to_numeric(
        frame["asof_pitcher_success_rate"], errors="coerce"
    ).to_numpy(float)
    fallback = np.where(np.isfinite(career), career, 0.5)
    if mode == "prev1":
        anchor = recent[:, 0]
    elif mode in {"mean135", "weighted135"}:
        weights = (
            np.ones(3, dtype=np.float64)
            if mode == "mean135"
            else np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
        )
        finite = np.isfinite(recent)
        numerator = np.sum(np.where(finite, recent * weights, 0.0), axis=1)
        denominator = np.sum(finite * weights, axis=1)
        anchor = np.divide(
            numerator,
            denominator,
            out=np.full(len(frame), np.nan, dtype=np.float64),
            where=denominator > 0.0,
        )
    else:
        raise ValueError(f"unknown anchor mode: {mode}")
    return np.clip(np.where(np.isfinite(anchor), anchor, fallback), 0.001, 0.999)


def route_mask(domain: np.ndarray, route: str) -> np.ndarray:
    values = np.asarray(domain).astype(str)
    if route == "ALL":
        return np.ones(len(values), dtype=bool)
    if route == "R_CORE":
        return values == "R_CORE"
    if route == "R_CORE_R_ANCHOR":
        return np.isin(values, ["R_CORE", "R_ANCHOR"])
    raise ValueError(f"unknown route: {route}")


def fit_alpha(
    axes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    lower: float,
    upper: float,
) -> float:
    """Fit y-a = alpha*(p-a) on aligned active exact rows."""

    numerator = 0.0
    denominator = 0.0
    for target, parent, anchor, active in axes:
        mask = np.asarray(active, dtype=bool)
        direction = np.asarray(parent, dtype=float)[mask] - np.asarray(anchor, dtype=float)[mask]
        response = np.asarray(target, dtype=float)[mask] - np.asarray(anchor, dtype=float)[mask]
        numerator += float(np.dot(direction, response))
        denominator += float(np.dot(direction, direction))
    if denominator <= 0.0:
        return 1.0
    return float(np.clip(numerator / denominator, float(lower), float(upper)))


def apply_spread(
    parent: np.ndarray, anchor: np.ndarray, active: np.ndarray, alpha: float
) -> np.ndarray:
    output = np.asarray(parent, dtype=float).copy()
    mask = np.asarray(active, dtype=bool)
    output[mask] = np.clip(
        np.asarray(anchor, dtype=float)[mask]
        + float(alpha)
        * (np.asarray(parent, dtype=float)[mask] - np.asarray(anchor, dtype=float)[mask]),
        0.001,
        0.999,
    )
    return output


def _point_pass(metrics: dict[str, Any], gate: dict[str, Any], *, locked: bool) -> bool:
    minimum_gain = float(gate.get("gain_min", 0.0)) if locked else 0.0
    return bool(
        metrics["gain"] >= minimum_gain
        and metrics["positive_month_fraction"] >= float(gate["positive_month_fraction_min"])
        and metrics["worst_month_gain"] > float(gate["worst_month_gain_min_exclusive"])
        and metrics["minimum_domain_gain"] >= float(gate["active_domain_gain_min"])
    )


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_dir: Path,
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
    with np.load(v104_dir / "selected_axes.npz", allow_pickle=False) as saved:
        parents = {name: saved[name].astype(np.float64) for name in ALL_AXES}
    late_mask = exact["full_2024"]["game_month"].astype(np.int16) >= 8
    exact["late_2024"] = {
        key: np.asarray(value)[late_mask] for key, value in exact["full_2024"].items()
    }
    frames["late_2024"] = frames["full_2024"].loc[late_mask].reset_index(drop=True)
    for name in ALL_AXES:
        if len(frames[name]) != len(exact[name]["target"]) or len(parents[name]) != len(frames[name]):
            raise ValueError(f"axis alignment mismatch: {name}")

    anchors = {
        mode: {name: recent_anchor(frames[name], mode) for name in ALL_AXES}
        for mode in config["anchor_modes"]
    }
    candidates: dict[str, dict[str, np.ndarray]] = {}
    source_rows: list[dict[str, Any]] = []
    source_metrics: dict[str, dict[str, Any]] = {}
    source_gate = config["source_gate"]
    for mode in config["anchor_modes"]:
        for route in config["routes"]:
            key = f"{mode}__{route}"
            per_axis_active = {
                name: exact[name]["exact_mask"].astype(bool)
                & route_mask(exact[name]["domain3"], route)
                for name in SOURCE_AXES
            }
            individual_alpha = {
                name: fit_alpha(
                    [(
                        exact[name]["target"], parents[name], anchors[mode][name],
                        per_axis_active[name],
                    )],
                    float(config["alpha_min"]), float(config["alpha_max"]),
                )
                for name in SOURCE_AXES
            }
            pooled_alpha = fit_alpha(
                [(
                    exact[name]["target"], parents[name], anchors[mode][name],
                    per_axis_active[name],
                ) for name in SOURCE_AXES],
                float(config["alpha_min"]), float(config["alpha_max"]),
            )
            same_direction = bool(
                (individual_alpha[SOURCE_AXES[0]] - 1.0)
                * (individual_alpha[SOURCE_AXES[1]] - 1.0) > 0.0
            )
            candidates[key] = {}
            metrics: dict[str, Any] = {}
            for name in ALL_AXES:
                active = route_mask(exact[name]["domain3"], route)
                candidate = apply_spread(
                    parents[name], anchors[mode][name], active, pooled_alpha
                )
                candidates[key][name] = candidate
                if name in SOURCE_AXES:
                    metrics[name] = _axis_metrics(exact[name], candidate)
            passed = same_direction and all(
                _point_pass(metrics[name], source_gate, locked=False)
                for name in SOURCE_AXES
            )
            source_metrics[key] = metrics
            source_rows.append({
                "key": key,
                "anchor_mode": mode,
                "route": route,
                "pooled_alpha": pooled_alpha,
                "alpha_full_2022": individual_alpha["full_2022"],
                "alpha_late_2023": individual_alpha["late_2023"],
                "same_alpha_direction": same_direction,
                "source_gate_passed": passed,
                "minimum_gain": min(metrics[name]["gain"] for name in SOURCE_AXES),
                "minimum_month_fraction": min(
                    metrics[name]["positive_month_fraction"] for name in SOURCE_AXES
                ),
                "worst_month_gain": min(
                    metrics[name]["worst_month_gain"] for name in SOURCE_AXES
                ),
            })
    ranking = pd.DataFrame(source_rows).sort_values(
        ["source_gate_passed", "minimum_gain", "worst_month_gain"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    ranking.to_csv(output_dir / "source_ranking.csv", index=False)
    selected = str(ranking.iloc[0]["key"])
    selected_source_pass = bool(ranking.iloc[0]["source_gate_passed"])
    locked_metrics = {
        name: _axis_metrics(exact[name], candidates[selected][name])
        for name in ("full_2024", "late_2024")
    }
    locked_pass = {
        name: _point_pass(metrics, config["locked_gate"], locked=True)
        for name, metrics in locked_metrics.items()
    }
    family = {
        name: [candidates[key][name] for key in candidates] for name in ALL_AXES
    }
    if selected_source_pass:
        robust = {
            name: _robust_axis(
                frames[name], exact[name], candidates[selected][name], family[name],
                {
                    "bootstrap_resamples": 3000,
                    "block_size": 2000,
                    "seed": 20260823,
                },
                100 + 10 * index,
            )
            for index, name in enumerate(ALL_AXES)
        }
        robust_pass = {
            name: bool(all(
                result[item]["p05"] > 0.0
                for item in ("pitcher", "crossed_pitcher_batter", "chronological_block")
            ))
            for name, result in robust.items()
        }
    else:
        robust = {}
        robust_pass = {name: False for name in ALL_AXES}
    eligible = bool(
        selected_source_pass
        and all(locked_pass.values())
        and all(robust_pass.values())
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{name: candidates[selected][name] for name in ALL_AXES},
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote" if eligible else "reject",
        "selected": ranking.iloc[0].to_dict(),
        "source_ranking": ranking.to_dict(orient="records"),
        "selected_source_metrics": source_metrics[selected],
        "locked_metrics": locked_metrics,
        "locked_point_gate_pass": locked_pass,
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.train_csv, args.contract_dir, args.v104_dir, args.config, args.output_dir)


if __name__ == "__main__":
    main()
