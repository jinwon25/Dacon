"""Refit the fixed v148 C3 recipe on exact deployed-H1 OOF residuals.

No route, model, window, shrinkage, or blend weight is selected here.  The
script swaps the proxy H1 residual basis for the already deployed H1 recipe,
then evaluates that single consistency correction across forward source axes
and the exact v148 locked axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v135_c3_recent_window import _source_years
from src.champion.v130_hoo_independent_oof_blend import post4
from src.champion.v133_hoo_h1_c3_forward import CONTEXT_COLS, c3_adjustment
from src.core.axis_metrics import _axis_metrics
from src.core.contract import _load_contract_axis
from src.core.robustness import evaluate_robustness


PROTOCOL = "V158_EXACT_H1_C3_CONTRACT_V1"
SOURCE_AXES = ("full_2022", "late_2023")


def _consensus(
    window_values: dict[str, dict[int, np.ndarray]],
    year: int,
    recent_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = ("last1", "last2", "last3", "expanding")
    matrix = np.column_stack([window_values[name][year] for name in order])
    agreed = np.all(matrix > 0.0, axis=1) | np.all(matrix < 0.0, axis=1)
    sign_all = matrix.mean(axis=1) * agreed
    mean_recent = matrix[:, :2].mean(axis=1)
    mixed = (1.0 - float(recent_weight)) * sign_all + float(recent_weight) * mean_recent
    return sign_all, mean_recent, mixed


def _build_c3(
    frames: dict[int, pd.DataFrame],
    residual: dict[int, np.ndarray],
    config: dict[str, Any],
) -> dict[str, dict[int, np.ndarray]]:
    output: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in config["windows"]
    }
    for window in config["windows"]:
        for year in (2022, 2023, 2024):
            sources = _source_years(year, str(window))
            history = pd.concat([frames[value] for value in sources], ignore_index=True)
            history_residual = np.concatenate([residual[value] for value in sources])
            output[str(window)][year] = c3_adjustment(
                history, history_residual, frames[year], config["c3"]
            )
            print(
                f"[v158] window={window} fold={year} sources={sources} "
                f"mean_abs={np.mean(np.abs(output[str(window)][year])):.6g}",
                flush=True,
            )
    return output


def _compose(
    parent: np.ndarray,
    axis: dict[str, np.ndarray],
    h1_post: np.ndarray,
    c3: np.ndarray,
    config: dict[str, Any],
    *,
    route: str | None = None,
) -> np.ndarray:
    active = axis["exact_mask"].astype(bool)
    selected_route = str(config["route"]) if route is None else str(route)
    if selected_route != "ALL":
        active &= axis["domain3"].astype(str) == selected_route
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(
        (1.0 - float(config["h1_weight"])) * output[active]
        + float(config["h1_weight"]) * h1_post[active]
        + float(config["c3_weight"]) * c3[active],
        0.001,
        0.999,
    )
    return output


def _delta_candidate(
    parent: np.ndarray,
    axis: dict[str, np.ndarray],
    h1_delta: np.ndarray,
    c3_delta: np.ndarray,
    config: dict[str, Any],
    component: str,
) -> np.ndarray:
    active = axis["exact_mask"].astype(bool)
    active &= axis["domain3"].astype(str) == str(config["route"])
    shift = np.zeros(len(parent), dtype=np.float64)
    if component in ("h1", "joint"):
        shift += float(config["h1_weight"]) * h1_delta
    if component in ("c3", "joint"):
        shift += float(config["c3_weight"]) * c3_delta
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(output[active] + shift[active], 0.001, 0.999)
    return output


def _metrics(
    axis: dict[str, np.ndarray], base: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    return _axis_metrics({**axis, "parent": base}, candidate)


def _source_pass(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return bool(
        metrics["gain"] > float(gate["gain_min_each"])
        and metrics["positive_month_fraction"]
        >= float(gate["positive_month_fraction_min"])
        and metrics["worst_month_gain"]
        > float(gate["worst_month_gain_min_exclusive"])
        and metrics["minimum_domain_gain"] >= float(gate["minimum_domain_gain"])
    )


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_dir: Path,
    v157_dir: Path,
    proxy_dir: Path,
    proxy_sign_dir: Path,
    proxy_recent_dir: Path,
    v148_oof: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    if list(config["windows"]) != ["last1", "last2", "last3", "expanding"]:
        raise ValueError("window order must match the deployed runtime")
    output_dir.mkdir(parents=True, exist_ok=True)

    context = pd.read_csv(train_csv, usecols=list(CONTEXT_COLS), low_memory=False)
    season = context["season"].to_numpy(np.int16)
    frames = {
        year: context.loc[season == year].reset_index(drop=True)
        for year in range(2019, 2025)
    }
    with np.load(v157_dir / "oof_predictions.npz", allow_pickle=False) as saved:
        exact_h1 = {
            year: saved[f"exact_h1_{year}"].astype(np.float64)
            for year in range(2020, 2025)
        }
    with np.load(proxy_dir / "oof_predictions.npz", allow_pickle=False) as saved:
        proxy_h1 = {
            year: saved[f"h1_{year}"].astype(np.float64)
            for year in range(2020, 2025)
        }
    post = {
        year: post4(context.loc[season < year].reset_index(drop=True), frames[year])
        for year in range(2020, 2025)
    }
    residual_exact = {
        year: frames[year]["control_success"].to_numpy(np.float64)
        - (exact_h1[year] + post[year])
        for year in range(2020, 2025)
    }
    residual_proxy = {
        year: frames[year]["control_success"].to_numpy(np.float64)
        - (proxy_h1[year] + post[year])
        for year in range(2020, 2025)
    }
    print("[v158] rebuilding fixed windows from exact H1 residuals", flush=True)
    exact_windows = _build_c3(frames, residual_exact, config)
    print("[v158] rebuilding proxy windows for parity", flush=True)
    proxy_windows = _build_c3(frames, residual_proxy, config)
    exact_c3 = {}
    proxy_c3 = {}
    exact_parts = {}
    proxy_parts = {}
    for year in (2022, 2023, 2024):
        exact_parts[year] = _consensus(
            exact_windows, year, float(config["mean_recent_weight"])
        )
        proxy_parts[year] = _consensus(
            proxy_windows, year, float(config["mean_recent_weight"])
        )
        exact_c3[year] = exact_parts[year][2]
        proxy_c3[year] = proxy_parts[year][2]

    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in (*SOURCE_AXES, "full_2024")
    }
    with np.load(v104_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v104 = {name: saved[name].astype(np.float64) for name in saved.files}

    # Verify that the proxy reconstruction exactly reproduces the two OOF
    # components that were bridged into v148.
    proxy_sign_formula = _compose(
        v104["full_2024"],
        axes["full_2024"],
        proxy_h1[2024] + post[2024],
        proxy_parts[2024][0],
        config,
        route="ALL",
    )
    proxy_recent_formula = _compose(
        v104["full_2024"],
        axes["full_2024"],
        proxy_h1[2024] + post[2024],
        proxy_parts[2024][1],
        config,
        route="ALL",
    )
    with np.load(proxy_sign_dir / "selected_axes.npz", allow_pickle=False) as saved:
        stored_sign = saved["full_2024"].astype(np.float64)
    with np.load(proxy_recent_dir / "selected_axes.npz", allow_pickle=False) as saved:
        stored_recent = saved["full_2024"].astype(np.float64)
    parity = {
        "sign_all_max_abs": float(np.max(np.abs(proxy_sign_formula - stored_sign))),
        "mean_recent_max_abs": float(
            np.max(np.abs(proxy_recent_formula - stored_recent))
        ),
    }
    if parity["sign_all_max_abs"] > 1e-12 or parity["mean_recent_max_abs"] > 1e-12:
        raise ValueError(f"proxy C3 reconstruction parity failure: {parity}")

    late23_mask = frames[2023]["game_month"].ge(8).to_numpy()
    source_values = {
        "full_2022": {
            "h1_exact": exact_h1[2022] + post[2022],
            "h1_proxy": proxy_h1[2022] + post[2022],
            "c3_exact": exact_c3[2022],
            "c3_proxy": proxy_c3[2022],
        },
        "late_2023": {
            "h1_exact": (exact_h1[2023] + post[2023])[late23_mask],
            "h1_proxy": (proxy_h1[2023] + post[2023])[late23_mask],
            "c3_exact": exact_c3[2023][late23_mask],
            "c3_proxy": proxy_c3[2023][late23_mask],
        },
    }
    source_metrics = {}
    source_candidates = {}
    for name in SOURCE_AXES:
        values = source_values[name]
        base = _compose(
            v104[name], axes[name], values["h1_proxy"], values["c3_proxy"], config
        )
        candidate = _compose(
            v104[name], axes[name], values["h1_exact"], values["c3_exact"], config
        )
        source_candidates[name] = {"base": base, "candidate": candidate}
        source_metrics[name] = _metrics(axes[name], base, candidate)
    source_gate = all(
        _source_pass(item, config["source_gate"])
        for item in source_metrics.values()
    )

    axis148 = _load_contract_axis(v148_oof)
    if not np.array_equal(
        axis148["raw_index"].astype(np.int64),
        axes["full_2024"]["raw_index"].astype(np.int64),
    ):
        raise ValueError("v148/full-2024 alignment mismatch")
    h1_delta = exact_h1[2024] - proxy_h1[2024]
    c3_delta = exact_c3[2024] - proxy_c3[2024]
    components = {
        component: _delta_candidate(
            axis148["parent"].astype(np.float64),
            axis148,
            h1_delta,
            c3_delta,
            config,
            component,
        )
        for component in ("h1", "c3", "joint")
    }
    component_metrics = {
        name: _metrics(axis148, axis148["parent"], candidate)
        for name, candidate in components.items()
    }
    robust = evaluate_robustness(
        axis148,
        components["joint"],
        [
            components["h1"],
            components["c3"],
            components["joint"],
            axis148["parent"].astype(np.float64),
        ],
        config["robustness"],
    )
    locked = component_metrics["joint"]
    gate = config["locked_gate"]
    promotion = bool(
        source_gate
        and locked["gain"] >= float(gate["v148_gain_min"])
        and locked["positive_month_fraction"]
        >= float(gate["positive_month_fraction_min"])
        and locked["worst_month_gain"]
        > float(gate["worst_month_gain_min_exclusive"])
        and locked["minimum_domain_gain"] >= float(gate["minimum_domain_gain"])
        and robust["pitcher"]["p05"] > float(gate["bootstrap_p05_min"])
        and robust["crossed_pitcher_batter"]["p05"]
        > float(gate["bootstrap_p05_min"])
        and robust["chronological_block"]["p05"]
        > float(gate["bootstrap_p05_min"])
        and robust["reality_check"]["p_value"]
        <= float(gate["reality_check_alpha"])
    )

    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2022=source_candidates["full_2022"]["candidate"],
        late_2023=source_candidates["late_2023"]["candidate"],
        full_2024=components["joint"],
        h1_only_2024=components["h1"],
        c3_only_2024=components["c3"],
        exact_c3_2024=exact_c3[2024],
        proxy_c3_2024=proxy_c3[2024],
    )
    np.savez_compressed(
        output_dir / "c3_components.npz",
        **{
            f"exact_{label}_{year}": exact_parts[year][index]
            for year in (2022, 2023, 2024)
            for index, label in enumerate(("sign_all", "mean_recent", "mixed"))
        },
        **{
            f"proxy_{label}_{year}": proxy_parts[year][index]
            for year in (2022, 2023, 2024)
            for index, label in enumerate(("sign_all", "mean_recent", "mixed"))
        },
    )
    result = {
        "protocol": PROTOCOL,
        "status": "eligible_for_packaging" if promotion else "rejected",
        "proxy_reconstruction_parity": parity,
        "source_incremental_metrics": source_metrics,
        "source_gate_passed": source_gate,
        "v148_component_metrics": component_metrics,
        "v148_joint_robustness": robust,
        "promotion_gate_passed": promotion,
        "eligible_for_packaging": promotion,
        "fixed_recipe": {
            "h1_weight": config["h1_weight"],
            "c3_weight": config["c3_weight"],
            "mean_recent_weight": config["mean_recent_weight"],
            "route": config["route"],
            "windows": config["windows"],
            "c3": config["c3"],
        },
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
    parser.add_argument("--v157-dir", type=Path, required=True)
    parser.add_argument("--proxy-dir", type=Path, required=True)
    parser.add_argument("--proxy-sign-dir", type=Path, required=True)
    parser.add_argument("--proxy-recent-dir", type=Path, required=True)
    parser.add_argument("--v148-oof", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv,
        args.contract_dir,
        args.v104_dir,
        args.v157_dir,
        args.proxy_dir,
        args.proxy_sign_dir,
        args.proxy_recent_dir,
        args.v148_oof,
        args.config,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
