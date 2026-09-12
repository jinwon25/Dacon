"""Strict-forward reconstruction of the public Hoo H1+C3 residual recipe."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.axis_metrics import _axis_metrics
from src.champion.v127_hoo_current_state_rebase import apply_correction
from src.champion.v130_hoo_independent_oof_blend import apply_blend, post4
from src.champion.v131_hoo_h1_independent_oof import _fit_year, _prepare_features
from src.core.contract import _load_contract_axis


PROTOCOL = "V133_HOO_H1_C3_FORWARD_V1"
SOURCE_AXES = ("full_2022", "late_2023")
CONTEXT_COLS = (
    "season",
    "game_month",
    "pitcher_id",
    "pitcher_hand",
    "batter_hand",
    "balls_before",
    "strikes_before",
    "num_runners_on",
    "control_success",
)


def _pitcher_contrast(
    pitcher: np.ndarray,
    context: np.ndarray,
    residual: np.ndarray,
    shrink: float,
) -> dict[int, float]:
    table = pd.DataFrame(
        {"pitcher": pitcher, "context": context, "residual": residual}
    ).groupby(["pitcher", "context"])["residual"].agg(["mean", "size"]).unstack()
    for stat in ("mean", "size"):
        for value in (0, 1):
            if (stat, value) not in table:
                table[(stat, value)] = np.nan if stat == "mean" else 0.0
    n0 = table[("size", 0)].fillna(0.0)
    n1 = table[("size", 1)].fillna(0.0)
    difference = table[("mean", 1)] - table[("mean", 0)]
    effective = (n0 * n1) / (n0 + n1).replace(0.0, np.nan)
    shrunk = (difference * effective / (effective + float(shrink))).dropna()
    return {int(key): float(value) for key, value in shrunk.items()}


def c3_adjustment(
    history: pd.DataFrame,
    residual: np.ndarray,
    query: pd.DataFrame,
    recipe: dict[str, Any],
) -> np.ndarray:
    pitcher = history["pitcher_id"].to_numpy(np.int64)
    same_hand = (
        history["pitcher_hand"].to_numpy(np.int8)
        == history["batter_hand"].to_numpy(np.int8)
    ).astype(np.int8)
    two_strike = (
        history["strikes_before"].to_numpy(np.int8) == 2
    ).astype(np.int8)
    runner_on = (history["num_runners_on"].to_numpy(np.int8) > 0).astype(np.int8)
    tables = (
        _pitcher_contrast(
            pitcher, same_hand, residual, float(recipe["same_hand_shrink"])
        ),
        _pitcher_contrast(
            pitcher, two_strike, residual, float(recipe["two_strike_shrink"])
        ),
        _pitcher_contrast(
            pitcher, runner_on, residual, float(recipe["runner_on_shrink"])
        ),
    )

    query_pitcher = query["pitcher_id"].to_numpy(np.int64)
    query_contexts = (
        (
            query["pitcher_hand"].to_numpy(np.int8)
            == query["batter_hand"].to_numpy(np.int8)
        ).astype(np.int8),
        (query["strikes_before"].to_numpy(np.int8) == 2).astype(np.int8),
        (query["num_runners_on"].to_numpy(np.int8) > 0).astype(np.int8),
    )
    scale = float(recipe["contrast_scale"])
    output = np.zeros(len(query), dtype=np.float64)
    for table, context in zip(tables, query_contexts):
        magnitude = np.asarray([table.get(int(key), 0.0) for key in query_pitcher])
        output += np.where(context == 1, scale * magnitude, -scale * magnitude)
    return output


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
    if family == "h1_post4_c3_blend":
        return apply_blend(parent, axis, h1_post + c3, route, dose)
    if family == "v131_plus_c3":
        base = apply_blend(parent, axis, h1_post, route, v131_weight)
        return apply_correction(base, axis, c3, route, dose)
    if family == "c3_only":
        return apply_correction(parent, axis, c3, route, dose)
    raise ValueError(f"unknown family: {family}")


def run(
    train_csv: Path,
    trackman_csv: Path,
    contract_dir: Path,
    v104_dir: Path,
    v131_dir: Path,
    external_root: Path,
    v131_config_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    v131_config = json.loads(v131_config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train, target, season, _base, _dx_features, h1_features = _prepare_features(
        train_csv, trackman_csv, external_root
    )
    context = train[list(CONTEXT_COLS)].copy()
    with np.load(v131_dir / "oof_predictions.npz", allow_pickle=False) as saved:
        oof = {
            key: saved[key].astype(np.float64)
            for key in ("h1_2022", "h1_2023", "h1_2024")
        }
    for year in (2020, 2021):
        oof[f"h1_{year}"] = _fit_year(
            train, target, season, year, h1_features, v131_config["model"], "H1-C3"
        )
    del train
    gc.collect()

    year_frames = {
        year: context.loc[season == year].reset_index(drop=True)
        for year in (2019, 2020, 2021, 2022, 2023, 2024)
    }
    post = {
        year: post4(
            context.loc[season < year].reset_index(drop=True), year_frames[year]
        )
        for year in (2020, 2021, 2022, 2023, 2024)
    }
    residual = {
        year: year_frames[year]["control_success"].to_numpy(np.float64)
        - (oof[f"h1_{year}"] + post[year])
        for year in (2020, 2021, 2022, 2023, 2024)
    }
    c3: dict[int, np.ndarray] = {}
    for year in (2022, 2023, 2024):
        source_years = list(range(2020, year))
        history = pd.concat([year_frames[value] for value in source_years], ignore_index=True)
        history_residual = np.concatenate([residual[value] for value in source_years])
        c3[year] = c3_adjustment(
            history, history_residual, year_frames[year], config["c3"]
        )
        print(
            f"[v133] C3 fold={year} source={source_years} "
            f"mean_abs={np.mean(np.abs(c3[year])):.6g}",
            flush=True,
        )

    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in (*SOURCE_AXES, "full_2024")
    }
    with np.load(v104_dir / "selected_axes.npz", allow_pickle=False) as saved:
        parent = {name: saved[name].astype(np.float64) for name in saved.files}
    metric_axes = {name: {**axis, "parent": parent[name]} for name, axis in axes.items()}
    late23 = year_frames[2023]["game_month"].ge(8).to_numpy()
    source = {
        "full_2022": {
            "h1_post": oof["h1_2022"] + post[2022],
            "c3": c3[2022],
        },
        "late_2023": {
            "h1_post": (oof["h1_2023"] + post[2023])[late23],
            "c3": c3[2023][late23],
        },
    }

    trials: list[dict[str, Any]] = []
    metrics_by_key: dict[str, dict[str, Any]] = {}
    candidates_by_key: dict[str, dict[str, np.ndarray]] = {}
    for family in config["families"]:
        doses = (
            config["blend_weight_grid"]
            if family == "h1_post4_c3_blend"
            else config["c3_eta_grid"]
        )
        for route in config["routes"]:
            for dose in doses:
                candidates = {
                    name: _candidate(
                        parent[name], axes[name], family, route, float(dose),
                        source[name]["h1_post"], source[name]["c3"],
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
                key = f"{family}__{route}__d{float(dose):g}"
                trials.append(
                    {
                        "key": key,
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
    family = str(selected["family"])
    route = str(selected["route"])
    dose = float(selected["dose"])
    print(f"[v133] source selected: {key}", flush=True)

    candidate24 = _candidate(
        parent["full_2024"], axes["full_2024"], family, route, dose,
        oof["h1_2024"] + post[2024], c3[2024],
        float(config["v131_blend_weight"]),
    )
    late24 = year_frames[2024]["game_month"].ge(8).to_numpy()
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
        output_dir / "oof_predictions.npz",
        **{f"h1_{year}": oof[f"h1_{year}"] for year in range(2020, 2025)},
        **{f"c3_{year}": c3[year] for year in (2022, 2023, 2024)},
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2022=candidates_by_key[key]["full_2022"],
        late_2023=candidates_by_key[key]["late_2023"],
        full_2024=candidate24,
        late_2024=candidate24[late24],
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_robust_audit" if eligible else "reject",
        "external_repository": "hoo743-ui/LG_Aimers09",
        "external_material_used": "public C3 formula only; all residuals rebuilt from official OOF",
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
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-dir", type=Path, required=True)
    parser.add_argument("--v131-dir", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--v131-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv,
        args.trackman_csv,
        args.contract_dir,
        args.v104_dir,
        args.v131_dir,
        args.external_root,
        args.v131_config,
        args.config,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
