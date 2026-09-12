"""Blend independently published CatBoost OOF above exact v104."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.axis_metrics import _axis_metrics
from src.core.contract import _load_contract_axis


PROTOCOL = "V130_CATBOOST_INDEPENDENT_OOF_BLEND_V1"
POST_SHRINK = (300.0, 2000.0, 800.0, 2000.0)
POST_WEIGHT = (0.20, 0.825, 0.280, 0.45)


def nested_deviation(
    parent: np.ndarray, child: np.ndarray, target: np.ndarray, shrink: float
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(child, kind="stable")
    ys, ps, cs = target[order], parent[order], child[order]
    unique, starts = np.unique(cs, return_index=True)
    count = np.diff(np.append(starts, len(cs)))
    cell_mean = np.add.reduceat(ys, starts) / count
    parent_key = ps[starts]
    parent_order = np.argsort(parent, kind="stable")
    yp, pp = target[parent_order], parent[parent_order]
    pu, pstarts = np.unique(pp, return_index=True)
    pcount = np.diff(np.append(pstarts, len(pp)))
    pmean = np.add.reduceat(yp, pstarts) / pcount
    difference = cell_mean - pmean[np.searchsorted(pu, parent_key)]
    return unique, count * difference / (count + float(shrink))


def lookup(unique: np.ndarray, value: np.ndarray, key: np.ndarray) -> np.ndarray:
    index = np.clip(np.searchsorted(unique, key), 0, max(len(unique) - 1, 0))
    valid = (unique[index] == key) if len(unique) else np.zeros(len(key), dtype=bool)
    output = np.zeros(len(key), dtype=np.float64)
    output[valid] = value[index[valid]]
    return output


def post4(history: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    def keys(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        pitcher = frame["pitcher_id"].to_numpy(np.int64)
        batter_hand = frame["batter_hand"].to_numpy(np.int64)
        balls = frame["balls_before"].to_numpy(np.int64)
        strikes = frame["strikes_before"].to_numpy(np.int64)
        runner = (frame["num_runners_on"].to_numpy(np.int64) > 0).astype(np.int64)
        platoon = pitcher * 10 + batter_hand
        advantage = platoon * 10 + (strikes > balls).astype(np.int64)
        count = balls * 4 + strikes
        return [
            (pitcher, platoon),
            (platoon, advantage),
            (advantage, advantage * 100 + count),
            (platoon, platoon * 10 + runner),
        ]

    history_keys = keys(history)
    query_keys = keys(query)
    target = history["control_success"].to_numpy(np.float64)
    parts = []
    for (parent, child), (_query_parent, query_child), shrink in zip(
        history_keys, query_keys, POST_SHRINK
    ):
        parts.append(lookup(*nested_deviation(parent, child, target, shrink), query_child))
    return np.column_stack(parts) @ np.asarray(POST_WEIGHT, dtype=np.float64)


def apply_blend(
    parent: np.ndarray,
    axis: dict[str, np.ndarray],
    external: np.ndarray,
    route: str,
    weight: float,
) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    active = axis["exact_mask"].astype(bool)
    if route != "ALL":
        active &= axis["domain3"].astype(str) == route
    output[active] = np.clip(
        (1.0 - float(weight)) * output[active]
        + float(weight) * np.asarray(external, dtype=np.float64)[active],
        0.001,
        0.999,
    )
    return output


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in axis.items()}


def _point_pass(metrics: dict[str, Any], gate: dict[str, Any], *, locked: bool) -> bool:
    gain_min = float(gate.get("gain_min", 0.0)) if locked else 0.0
    return bool(
        metrics["gain"] >= gain_min
        and metrics["positive_month_fraction"] >= float(gate["positive_month_fraction_min"])
        and metrics["worst_month_gain"] > float(gate["worst_month_gain_min_exclusive"])
        and metrics["minimum_domain_gain"] >= float(gate["active_domain_gain_min"])
    )


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_dir: Path,
    component_root: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, low_memory=False)
    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in ("full_2022", "late_2023", "full_2024")
    }
    with np.load(v104_dir / "selected_axes.npz", allow_pickle=False) as saved:
        parent = {name: saved[name].astype(np.float64) for name in saved.files}
    metric_axes = {name: {**axis, "parent": parent[name]} for name, axis in axes.items()}
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    raw_year_frames = {
        year: train.loc[train["season"].eq(year)].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    external: dict[str, dict[str, np.ndarray]] = {}
    for label, filename in (("d", "champ_oof.npz"), ("dx", "champ_oof_x.npz")):
        with np.load(component_root / "exp" / filename, allow_pickle=False) as saved:
            full = {year: saved[f"p{year}"].astype(np.float64) for year in (2022, 2023, 2024)}
        post = {
            year: post4(
                train.loc[train["season"].lt(year)].reset_index(drop=True),
                raw_year_frames[year],
            )
            for year in (2022, 2023, 2024)
        }
        external[f"{label}_raw"] = {
            "full_2022": full[2022],
            "late_2023": full[2023][raw_year_frames[2023]["game_month"].ge(8).to_numpy()],
            "full_2024": full[2024],
        }
        external[f"{label}_post4"] = {
            "full_2022": full[2022] + post[2022],
            "late_2023": (full[2023] + post[2023])[
                raw_year_frames[2023]["game_month"].ge(8).to_numpy()
            ],
            "full_2024": full[2024] + post[2024],
        }
    for family in config["families"]:
        for name in frames:
            if len(external[str(family)][name]) != len(frames[name]):
                raise ValueError(f"external OOF length mismatch: {family}/{name}")

    trials: list[dict[str, Any]] = []
    trial_metrics: dict[str, dict[str, Any]] = {}
    trial_candidates: dict[str, dict[str, np.ndarray]] = {}
    for family in config["families"]:
        family = str(family)
        for route in config["routes"]:
            route = str(route)
            for weight in config["weight_grid"]:
                candidates = {
                    name: apply_blend(
                        parent[name], axes[name], external[family][name], route, float(weight)
                    )
                    for name in ("full_2022", "late_2023")
                }
                metrics = {
                    name: _axis_metrics(metric_axes[name], candidates[name])
                    for name in candidates
                }
                passed = all(
                    _point_pass(item, config["source_gate"], locked=False)
                    for item in metrics.values()
                )
                key = f"{family}__{route}__w{float(weight):g}"
                trials.append({
                    "key": key,
                    "family": family,
                    "route": route,
                    "weight": float(weight),
                    "source_gate_passed": bool(passed),
                    "minimum_gain": float(min(item["gain"] for item in metrics.values())),
                    "mean_gain": float(np.mean([item["gain"] for item in metrics.values()])),
                    "minimum_month_fraction": float(min(
                        item["positive_month_fraction"] for item in metrics.values()
                    )),
                    "worst_month_gain": float(min(item["worst_month_gain"] for item in metrics.values())),
                })
                trial_metrics[key] = metrics
                trial_candidates[key] = candidates
    ranking = pd.DataFrame(trials).sort_values(
        ["source_gate_passed", "minimum_gain", "mean_gain", "worst_month_gain"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    ranking.to_csv(output_dir / "source_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    selected_key = str(selected["key"])
    family = str(selected["family"])
    route = str(selected["route"])
    weight = float(selected["weight"])
    candidate24 = apply_blend(
        parent["full_2024"], axes["full_2024"], external[family]["full_2024"],
        route, weight,
    )
    late24 = frames["full_2024"]["game_month"].ge(8).to_numpy()
    metric_late24 = _slice_axis(metric_axes["full_2024"], late24)
    locked_metrics = {
        "full_2024": _axis_metrics(metric_axes["full_2024"], candidate24),
        "late_2024": _axis_metrics(metric_late24, candidate24[late24]),
    }
    locked_pass = {
        name: _point_pass(item, config["locked_gate"], locked=True)
        for name, item in locked_metrics.items()
    }
    eligible = bool(selected["source_gate_passed"] and all(locked_pass.values()))
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        full_2022=trial_candidates[selected_key]["full_2022"],
        late_2023=trial_candidates[selected_key]["late_2023"],
        full_2024=candidate24,
        late_2024=candidate24[late24],
        external_full_2024=external[family]["full_2024"],
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_robust_audit" if eligible else "reject",
        "external_arrays": ["exp/champ_oof.npz", "exp/champ_oof_x.npz"],
        "n_trials": int(len(ranking)),
        "selected": selected,
        "source_ranking_top30": ranking.head(30).to_dict(orient="records"),
        "selected_source_metrics": trial_metrics[selected_key],
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
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv, args.contract_dir, args.v104_dir, args.component_root,
        args.config, args.output_dir,
    )


if __name__ == "__main__":
    main()
