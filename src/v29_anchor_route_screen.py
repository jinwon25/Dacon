"""Screen low-degree subgroup routing of the v27 R_ANCHOR signal.

The direct-probability model is frozen.  This experiment only asks whether a
single binary baseball-context split (handedness or count state) deserves two
different blend strengths.  Every row remains independent at inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v23_multiyear_direct_screen import _joint_domain
from src.v23_postbreak_gam_screen import SPECS, _fit_predict
from src.v23_structural_residual_screen import _derived, _load_axis
from src.v25_postbreak_anchor_audit import _early_to_late_2024
from src.train_v25_postbreak_anchor import MODEL_NAME


PARENT_ETA = 0.10
ETAS = (0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20)
STABLE_COUNTS = frozenset(("0-0", "1-0", "1-1", "2-1", "3-0", "3-2"))


def _bss(target: np.ndarray, prediction: np.ndarray) -> float:
    rate = float(np.mean(target))
    return 100_000.0 * (
        1.0 - float(np.mean(np.square(prediction - target))) / (rate * (1.0 - rate))
    )


def _candidate(
    frame: pd.DataFrame,
    direct: np.ndarray,
    route: np.ndarray,
    eta_true: float,
    eta_false: float,
) -> tuple[np.ndarray, np.ndarray]:
    base = frame["v22"].to_numpy(np.float64)
    domain = frame["domain3"].astype(str).eq("R_ANCHOR").to_numpy()
    delta = np.asarray(direct, dtype=np.float64) - base
    parent = base.copy()
    parent[domain] = np.clip(base[domain] + PARENT_ETA * delta[domain], 0.001, 0.999)
    eta = np.where(route, eta_true, eta_false)
    candidate = base.copy()
    candidate[domain] = np.clip(base[domain] + eta[domain] * delta[domain], 0.001, 0.999)
    return parent, candidate


def _diagnostics(
    frame: pd.DataFrame,
    direct: np.ndarray,
    route: np.ndarray,
    eta_true: float,
    eta_false: float,
) -> dict[str, object]:
    target = frame["target"].to_numpy(np.float64)
    parent, candidate = _candidate(frame, direct, route, eta_true, eta_false)
    domain = frame["domain3"].astype(str).eq("R_ANCHOR").to_numpy()

    def gain(mask: np.ndarray) -> float:
        return _bss(target[mask], candidate[mask]) - _bss(target[mask], parent[mask])

    months = []
    for month in sorted(frame["game_month"].unique()):
        mask = frame["game_month"].eq(month).to_numpy()
        if np.any(mask & domain):
            months.append({"month": int(month), "gain": gain(mask)})
    groups = []
    for column in ("pitcher_hand", "batter_hand", "count_state"):
        values = frame[column].astype(str)
        for value in sorted(values[domain].unique()):
            mask = domain & values.eq(value).to_numpy()
            groups.append({"column": column, "value": value, "gain": gain(mask)})
    return {
        "gain": gain(np.ones(len(frame), dtype=bool)),
        "domain_gain": gain(domain),
        "positive_month_fraction": float(np.mean([x["gain"] > 0 for x in months])),
        "worst_month_gain": float(min(x["gain"] for x in months)),
        "minimum_group_gain": float(min(x["gain"] for x in groups)),
        "months": months,
    }


def _routes(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    pitcher = frame["pitcher_hand"].astype(str)
    batter = frame["batter_hand"].astype(str)
    count = frame["count_state"].astype(str)
    return {
        "pitcher_hand_1": pitcher.eq("1").to_numpy(),
        "batter_hand_1": batter.eq("1").to_numpy(),
        "either_hand_1": (pitcher.eq("1") | batter.eq("1")).to_numpy(),
        "both_hand_1": (pitcher.eq("1") & batter.eq("1")).to_numpy(),
        "stable_count": count.isin(STABLE_COUNTS).to_numpy(),
        "first_pitch": count.eq("0-0").to_numpy(),
    }


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    spec = next(item for item in SPECS if item.name == MODEL_NAME)

    def fit_or_load(name: str, fit: pd.DataFrame, audit: pd.DataFrame) -> np.ndarray:
        cache = output_dir / f"{name}_direct.npy"
        if cache.exists():
            prediction = np.load(cache).astype(np.float64)
            if len(prediction) != len(audit):
                raise ValueError(f"cached {name} prediction length mismatch")
            print(f"[cache] {name}", flush=True)
            return prediction
        print(f"[fit] {name}", flush=True)
        prediction = _fit_predict(fit, audit, spec)
        np.save(cache, prediction)
        print(f"[saved] {name}", flush=True)
        return prediction

    selection = _load_axis(project, "y2023_early_to_late", raw)
    year23 = raw.loc[raw["season"].eq(2023)].reset_index(drop=True)
    year23 = _derived(year23, _joint_domain(year23))
    selection_fit = year23.loc[year23["game_month"].le(7)].reset_index(drop=True)
    selection_direct = fit_or_load("selection_late_2023", selection_fit, selection)

    outer = _load_axis(project, "y2023_to_y2024", raw)
    outer_direct = fit_or_load("outer_full_2024", year23, outer)

    replication_fit, replication = _early_to_late_2024(project, raw)
    replication_direct = fit_or_load(
        "replication_late_2024", replication_fit, replication
    )
    axes = {
        "selection_late_2023": (selection, selection_direct),
        "outer_full_2024": (outer, outer_direct),
        "replication_late_2024": (replication, replication_direct),
    }

    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path).sort_values(
            ["min_axis_gain", "mean_axis_gain"], ascending=False
        )
        print("[cache] route metrics", flush=True)
    else:
        rows: list[dict[str, object]] = []
        for route_name in _routes(selection):
            for eta_true in ETAS:
                for eta_false in ETAS:
                    if eta_true == eta_false:
                        continue
                    key = f"{route_name}__t{eta_true:.3f}__f{eta_false:.3f}"
                    axis_detail = {}
                    for axis_name, (frame, direct) in axes.items():
                        route = _routes(frame)[route_name]
                        axis_detail[axis_name] = _diagnostics(
                            frame, direct, route, eta_true, eta_false
                        )
                    gains = [float(x["gain"]) for x in axis_detail.values()]
                    rows.append(
                        {
                            "candidate": key,
                            "route": route_name,
                            "eta_true": eta_true,
                            "eta_false": eta_false,
                            "min_axis_gain": min(gains),
                            "mean_axis_gain": float(np.mean(gains)),
                            "selection_gain": gains[0],
                            "outer_gain": gains[1],
                            "replication_gain": gains[2],
                            "min_month_fraction": min(
                                float(x["positive_month_fraction"])
                                for x in axis_detail.values()
                            ),
                            "worst_month_gain": min(
                                float(x["worst_month_gain"])
                                for x in axis_detail.values()
                            ),
                            "minimum_group_gain": min(
                                float(x["minimum_group_gain"])
                                for x in axis_detail.values()
                            ),
                        }
                    )
        metrics = pd.DataFrame(rows).sort_values(
            ["min_axis_gain", "mean_axis_gain"], ascending=False
        )
        metrics.to_csv(metrics_path, index=False)
    eligible = metrics.loc[
        metrics["min_axis_gain"].gt(0.0)
        & metrics["mean_axis_gain"].ge(1.0)
        & metrics["min_month_fraction"].ge(2.0 / 3.0)
        & metrics["worst_month_gain"].gt(-5.0)
        & metrics["minimum_group_gain"].gt(-10.0)
    ]
    chosen = (eligible.iloc[0] if len(eligible) else metrics.iloc[0]).to_dict()
    chosen_detail = {
        axis_name: _diagnostics(
            frame,
            direct,
            _routes(frame)[str(chosen["route"])],
            float(chosen["eta_true"]),
            float(chosen["eta_false"]),
        )
        for axis_name, (frame, direct) in axes.items()
    }
    gates = {
        "all_axes_positive": bool(float(chosen["min_axis_gain"]) > 0.0),
        "mean_gain_at_least_1": bool(float(chosen["mean_axis_gain"]) >= 1.0),
        "month_fraction_at_least_two_thirds": bool(
            float(chosen["min_month_fraction"]) >= 2.0 / 3.0
        ),
        "worst_month_above_minus_5": bool(float(chosen["worst_month_gain"]) > -5.0),
        "minimum_group_above_minus_10": bool(float(chosen["minimum_group_gain"]) > -10.0),
    }
    summary = {
        "protocol": "V29_LOW_DOF_ANCHOR_ROUTE_ABOVE_V27_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "eligible_candidate_count": int(len(eligible)),
        "chosen": chosen,
        "details": chosen_detail,
        "gates": gates,
        "eligible_for_packaging": bool(all(gates.values())),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v29_anchor_route_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
