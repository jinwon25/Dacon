"""Source-select fixed consensus estimators across forward C3 history windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.axis_metrics import _axis_metrics
from src.champion.v130_catboost_independent_oof_blend import post4
from src.champion.v133_catboost_h1_c3_forward import CONTEXT_COLS, c3_adjustment
from src.archive.v135_c3_recent_window import (
    SOURCE_AXES,
    _candidate,
    _point_pass,
    _slice_axis,
    _source_years,
)
from src.core.contract import _load_contract_axis


PROTOCOL = "V136_C3_WINDOW_CONSENSUS_V1"
WINDOWS = ("last1", "last2", "last3", "expanding")


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_dir: Path,
    v133_dir: Path,
    config_path: Path,
    output_dir: Path,
    expected_protocol: str = PROTOCOL,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != expected_protocol:
        raise ValueError(f"config protocol must be {expected_protocol}")
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
        year: frames[year]["control_success"].to_numpy(np.float64)
        - (h1[year] + post[year])
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

    consensus: dict[str, dict[int, np.ndarray]] = {}
    for label, spec in config["consensus"].items():
        consensus[label] = {}
        for year in (2022, 2023, 2024):
            matrix = np.column_stack(
                [window_c3[member][year] for member in spec["members"]]
            )
            if spec["method"] == "mean":
                consensus[label][year] = matrix.mean(axis=1)
            elif spec["method"] == "median":
                consensus[label][year] = np.median(matrix, axis=1)
            elif spec["method"] == "sign_mean":
                mean = matrix.mean(axis=1)
                agreed = np.all(matrix > 0.0, axis=1) | np.all(matrix < 0.0, axis=1)
                consensus[label][year] = mean * agreed
            elif spec["method"] == "dispersion_shrink":
                mean = matrix.mean(axis=1)
                mean_absolute = np.mean(np.abs(matrix), axis=1)
                agreement = np.abs(mean) / np.maximum(mean_absolute, 1e-12)
                consensus[label][year] = mean * agreement
            else:
                raise ValueError(f"unknown consensus method: {spec['method']}")
            print(
                f"[v136] consensus={label} fold={year} "
                f"mean_abs={np.mean(np.abs(consensus[label][year])):.6g}",
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
    for label in config["consensus"]:
        correction = {
            "full_2022": consensus[label][2022],
            "late_2023": consensus[label][2023][late23],
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
                    key = f"{label}__{family}__{route}__d{float(dose):g}"
                    trials.append(
                        {
                            "key": key,
                            "consensus": label,
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
    label = str(selected["consensus"])
    family = str(selected["family"])
    route = str(selected["route"])
    dose = float(selected["dose"])
    print(f"[v136] source selected: {key}", flush=True)
    candidate24 = _candidate(
        parent["full_2024"], axes["full_2024"], family, route, dose,
        h1[2024] + post[2024], consensus[label][2024],
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
        consensus_c3_full_2024=consensus[label][2024],
    )
    result = {
        "protocol": expected_protocol,
        "status": "promote_to_robust_audit" if eligible else "reject",
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
