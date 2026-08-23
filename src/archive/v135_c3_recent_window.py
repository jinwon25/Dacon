"""Select a C3 OOF-history window on source years, then open locked 2024."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.axis_metrics import _axis_metrics
from src.champion.v127_hoo_current_state_rebase import apply_correction
from src.champion.v130_hoo_independent_oof_blend import apply_blend, post4
from src.champion.v133_hoo_h1_c3_forward import CONTEXT_COLS, c3_adjustment
from src.core.contract import _load_contract_axis


PROTOCOL = "V135_C3_RECENT_WINDOW_V1"
SOURCE_AXES = ("full_2022", "late_2023")


def _source_years(year: int, window: str) -> list[int]:
    available = list(range(2020, year))
    if window == "expanding":
        return available
    width = int(window.removeprefix("last"))
    return available[-width:]


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in axis.items()}


def _point_pass(metrics: dict[str, Any], gate: dict[str, Any], *, locked: bool) -> bool:
    minimum_gain = float(gate.get("gain_min", 0.0)) if locked else 0.0
    return bool(
        metrics["gain"] >= minimum_gain
        and metrics["positive_month_fraction"]
        >= float(gate["positive_month_fraction_min"])
        and metrics["worst_month_gain"]
        > float(gate["worst_month_gain_min_exclusive"])
        and metrics["minimum_domain_gain"] >= float(gate["active_domain_gain_min"])
    )


def _candidate(
    parent: np.ndarray,
    axis: dict[str, np.ndarray],
    family: str,
    route: str,
    dose: float,
    h1_post: np.ndarray,
    c3: np.ndarray,
    v131_weight: float,
) -> np.ndarray:
    if family == "v131_plus_c3":
        base = apply_blend(parent, axis, h1_post, route, v131_weight)
        return apply_correction(base, axis, c3, route, dose)
    if family == "h1_post4_c3_blend":
        return apply_blend(parent, axis, h1_post + c3, route, dose)
    raise ValueError(f"unknown family: {family}")


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_dir: Path,
    v133_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    context = pd.read_csv(train_csv, usecols=list(CONTEXT_COLS), low_memory=False)
    season = context["season"].to_numpy(np.int16)
    years = (2019, 2020, 2021, 2022, 2023, 2024)
    frames = {
        year: context.loc[season == year].reset_index(drop=True) for year in years
    }
    with np.load(v133_dir / "oof_predictions.npz", allow_pickle=False) as saved:
        h1 = {
            year: saved[f"h1_{year}"].astype(np.float64) for year in range(2020, 2025)
        }
    post = {
        year: post4(context.loc[season < year].reset_index(drop=True), frames[year])
        for year in range(2020, 2025)
    }
    residual = {
        year: frames[year]["control_success"].to_numpy(np.float64)
        - (h1[year] + post[year])
        for year in range(2020, 2025)
    }

    c3: dict[str, dict[int, np.ndarray]] = {}
    for window in config["windows"]:
        c3[window] = {}
        for year in (2022, 2023, 2024):
            sources = _source_years(year, window)
            history = pd.concat([frames[value] for value in sources], ignore_index=True)
            history_residual = np.concatenate([residual[value] for value in sources])
            c3[window][year] = c3_adjustment(
                history, history_residual, frames[year], config["c3"]
            )
            print(
                f"[v135] window={window} fold={year} source={sources} "
                f"mean_abs={np.mean(np.abs(c3[window][year])):.6g}",
                flush=True,
            )

    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in (*SOURCE_AXES, "full_2024")
    }
    with np.load(v104_dir / "selected_axes.npz", allow_pickle=False) as saved:
        parent = {name: saved[name].astype(np.float64) for name in saved.files}
    metric_axes = {name: {**axis, "parent": parent[name]} for name, axis in axes.items()}
    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    h1_post = {
        "full_2022": h1[2022] + post[2022],
        "late_2023": (h1[2023] + post[2023])[late23],
    }

    trials: list[dict[str, Any]] = []
    metrics_by_key: dict[str, dict[str, Any]] = {}
    candidates_by_key: dict[str, dict[str, np.ndarray]] = {}
    for window in config["windows"]:
        correction = {
            "full_2022": c3[window][2022],
            "late_2023": c3[window][2023][late23],
        }
        for family in config["families"]:
            doses = (
                config["c3_eta_grid"]
                if family == "v131_plus_c3"
                else config["blend_weight_grid"]
            )
            for route in config["routes"]:
                for dose in doses:
                    candidates = {
                        name: _candidate(
                            parent[name], axes[name], family, route, float(dose),
                            h1_post[name], correction[name],
                            float(config["v131_blend_weight"]),
                        )
                        for name in SOURCE_AXES
                    }
                    metrics = {
                        name: _axis_metrics(metric_axes[name], candidates[name])
                        for name in SOURCE_AXES
                    }
                    passed = all(
                        _point_pass(item, config["source_gate"], locked=False)
                        for item in metrics.values()
                    )
                    key = f"{window}__{family}__{route}__d{float(dose):g}"
                    trials.append(
                        {
                            "key": key,
                            "window": window,
                            "family": family,
                            "route": route,
                            "dose": float(dose),
                            "source_gate_passed": bool(passed),
                            "minimum_gain": float(min(x["gain"] for x in metrics.values())),
                            "mean_gain": float(np.mean([x["gain"] for x in metrics.values()])),
                            "minimum_month_fraction": float(
                                min(x["positive_month_fraction"] for x in metrics.values())
                            ),
                            "worst_month_gain": float(
                                min(x["worst_month_gain"] for x in metrics.values())
                            ),
                        }
                    )
                    metrics_by_key[key] = metrics
                    candidates_by_key[key] = candidates
    ranking = pd.DataFrame(trials).sort_values(
        ["source_gate_passed", "minimum_gain", "mean_gain", "worst_month_gain"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    ranking.to_csv(output_dir / "source_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    key = str(selected["key"])
    window = str(selected["window"])
    family = str(selected["family"])
    route = str(selected["route"])
    dose = float(selected["dose"])
    print(f"[v135] source selected: {key}", flush=True)

    candidate24 = _candidate(
        parent["full_2024"], axes["full_2024"], family, route, dose,
        h1[2024] + post[2024], c3[window][2024],
        float(config["v131_blend_weight"]),
    )
    late24 = frames[2024]["game_month"].ge(8).to_numpy()
    locked_metrics = {
        "full_2024": _axis_metrics(metric_axes["full_2024"], candidate24),
        "late_2024": _axis_metrics(
            _slice_axis(metric_axes["full_2024"], late24), candidate24[late24]
        ),
    }
    locked_pass = {
        name: _point_pass(item, config["locked_gate"], locked=True)
        for name, item in locked_metrics.items()
    }
    eligible = bool(selected["source_gate_passed"] and all(locked_pass.values()))
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2022=candidates_by_key[key]["full_2022"],
        late_2023=candidates_by_key[key]["late_2023"],
        full_2024=candidate24,
        late_2024=candidate24[late24],
        c3_full_2024=c3[window][2024],
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_robust_audit" if eligible else "reject",
        "external_repository": "hoo743-ui/LG_Aimers09",
        "external_material_used": "public recent-two-OOF C3 construction only",
        "n_trials": int(len(ranking)),
        "selected": selected,
        "source_ranking_top30": ranking.head(30).to_dict(orient="records"),
        "selected_source_metrics": metrics_by_key[key],
        "locked_metrics": locked_metrics,
        "locked_point_gate_pass": locked_pass,
        "eligible_for_robust_audit": eligible,
        "eligible_for_packaging": False,
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv,
        args.contract_dir,
        args.v104_dir,
        args.v133_dir,
        args.config,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
