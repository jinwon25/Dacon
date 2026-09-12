"""Screen row-local transport gates for the high-upside v184 axis.

The v184 hierarchy was strongly positive in 2022 and late 2023 but negative
in 2024.  This audit asks whether a low-cardinality baseball context selected
on two complete OOF origins transports to the third.  It never fits or selects
with test rows, and every held-origin choice excludes that origin's labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v168_row_region_exact_contract_reaudit import metrics
from src.core.contract import _load_contract_axis


PROTOCOL = "V194_HIERARCHICAL_CONTEXT_TRANSPORT_V1"
AXES = ("full_2022", "late_2023", "full_2024")
WEIGHTS = (0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1)
GATE_COLUMNS = (
    "season", "game_month", "inning", "top_bottom", "balls_before",
    "strikes_before", "outs_before", "score_diff_pitcher_team",
    "num_runners_on", "li", "pitcher_hand", "batter_hand",
    "asof_pitcher_n", "asof_batter_n",
)


def _bucket(values: pd.Series, bins: list[float], labels: list[str]) -> np.ndarray:
    return (
        pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels)
        .astype(str)
        .to_numpy()
    )


def context_labels(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return predeclared, low-cardinality and row-local baseball contexts."""
    balls = pd.to_numeric(frame["balls_before"], errors="raise").to_numpy(np.int8)
    strikes = pd.to_numeric(frame["strikes_before"], errors="raise").to_numpy(np.int8)
    runners = pd.to_numeric(frame["num_runners_on"], errors="coerce").fillna(0).to_numpy(np.int8)
    li = pd.to_numeric(frame["li"], errors="coerce").fillna(1.0)
    inning = pd.to_numeric(frame["inning"], errors="coerce").fillna(5.0)
    score = pd.to_numeric(frame["score_diff_pitcher_team"], errors="coerce").fillna(0.0)
    count = np.char.add(np.char.add(balls.astype(str), "-"), strikes.astype(str))
    count_leverage = np.where(
        balls > strikes, "batter_ahead", np.where(balls < strikes, "pitcher_ahead", "even")
    )
    pitcher_n = _bucket(
        frame["asof_pitcher_n"], [-np.inf, 29, 199, 999, np.inf],
        ["p0_29", "p30_199", "p200_999", "p1000_plus"],
    )
    batter_n = _bucket(
        frame["asof_batter_n"], [-np.inf, 29, 199, 999, np.inf],
        ["b0_29", "b30_199", "b200_999", "b1000_plus"],
    )
    li_bucket = _bucket(li, [-np.inf, 0.7, 1.5, np.inf], ["li_low", "li_mid", "li_high"])
    inning_bucket = _bucket(
        inning, [-np.inf, 3, 6, np.inf], ["inning_early", "inning_middle", "inning_late"]
    )
    month_bucket = _bucket(
        frame["game_month"], [-np.inf, 5, 7, np.inf], ["month_early", "month_middle", "month_late"]
    )
    score_bucket = np.where(score > 1, "lead", np.where(score < -1, "trail", "close"))
    platoon = np.where(
        pd.to_numeric(frame["pitcher_hand"], errors="raise").to_numpy()
        == pd.to_numeric(frame["batter_hand"], errors="raise").to_numpy(),
        "same_hand", "opposite_hand",
    )
    labels = {
        "count": count,
        "count_leverage": count_leverage,
        "two_strike": np.where(strikes == 2, "two_strike", "under_two"),
        "three_ball": np.where(balls == 3, "three_ball", "under_three"),
        "runner": np.where(runners > 0, "runner_on", "bases_empty"),
        "runner_count": runners.astype(str),
        "li": li_bucket,
        "inning": inning_bucket,
        "month_phase": month_bucket,
        "score": score_bucket,
        "platoon": platoon,
        "pitcher_history": pitcher_n,
        "batter_history": batter_n,
        "top_bottom": frame["top_bottom"].fillna("NA").astype(str).to_numpy(),
        "outs": pd.to_numeric(frame["outs_before"], errors="coerce").fillna(-1).astype(int).astype(str).to_numpy(),
    }
    interaction_pairs = (
        ("count_leverage", "runner"),
        ("count_leverage", "li"),
        ("count", "runner"),
        ("count", "pitcher_history"),
        ("pitcher_history", "li"),
        ("pitcher_history", "runner"),
        ("inning", "li"),
        ("runner", "li"),
        ("platoon", "count_leverage"),
        ("score", "li"),
    )
    for left, right in interaction_pairs:
        labels[f"{left}__{right}"] = np.char.add(
            np.char.add(labels[left].astype(str), "|"), labels[right].astype(str)
        )
    return labels


def gate_library(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, np.ndarray]]:
    """Construct only gates whose named level exists in every origin."""
    label_maps = {axis: context_labels(frames[axis]) for axis in AXES}
    gates: dict[str, dict[str, np.ndarray]] = {
        "all": {axis: np.ones(len(frames[axis]), dtype=bool) for axis in AXES}
    }
    for family in label_maps[AXES[0]]:
        common = set(np.unique(label_maps[AXES[0]][family]).tolist())
        for axis in AXES[1:]:
            common &= set(np.unique(label_maps[axis][family]).tolist())
        for value in sorted(common):
            name = f"{family}={value}"
            masks = {axis: label_maps[axis][family] == value for axis in AXES}
            if min(float(mask.mean()) for mask in masks.values()) >= 0.02:
                gates[name] = masks
    return gates


def apply_local(
    base: np.ndarray,
    local: np.ndarray,
    exact: np.ndarray,
    gate: np.ndarray,
    weight: float,
) -> np.ndarray:
    output = np.asarray(base, dtype=np.float64).copy()
    active = np.asarray(exact, dtype=bool) & np.asarray(gate, dtype=bool)
    output[active] += float(weight) * (
        np.asarray(local, dtype=np.float64)[active] - output[active]
    )
    return np.clip(output, 0.001, 0.999)


def _select(rows: pd.DataFrame, fit_axes: tuple[str, ...]) -> pd.Series:
    eligible = rows.copy()
    for axis in fit_axes:
        eligible = eligible.loc[
            (eligible[f"{axis}_gain"] > 0.0)
            & (eligible[f"{axis}_month_fraction"] >= 0.5)
            & (eligible[f"{axis}_worst_month"] > -2.0)
        ]
    if eligible.empty:
        return rows.loc[rows["weight"].eq(0.0025) & rows["gate"].eq("all")].iloc[0]
    columns = [f"{axis}_gain" for axis in fit_axes]
    eligible = eligible.assign(
        fit_min=eligible[columns].min(axis=1),
        fit_mean=eligible[columns].mean(axis=1),
    )
    return eligible.sort_values(
        ["fit_min", "fit_mean", "minimum_active_fraction", "gate", "weight"],
        ascending=[False, False, False, True, True], kind="stable",
    ).iloc[0]


def run(
    train_csv: Path,
    contract_dir: Path,
    bridge_oof: Path,
    v192_axes: Path,
    v184_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = pd.read_csv(train_csv, usecols=list(GATE_COLUMNS), low_memory=False)
    season = context["season"].to_numpy(np.int16)
    raw = {
        year: context.loc[season == year].reset_index(drop=True)
        for year in (2022, 2023, 2024)
    }
    late23 = raw[2023]["game_month"].ge(8).to_numpy()
    frames = {
        "full_2022": raw[2022].reset_index(drop=True),
        "late_2023": raw[2023].loc[late23].reset_index(drop=True),
        "full_2024": raw[2024].reset_index(drop=True),
    }
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    with np.load(v192_axes, allow_pickle=False) as saved:
        base = {axis: saved[axis].astype(np.float64) for axis in AXES}
    local = {
        "full_2022": np.load(v184_dir / "hierarchical_residual_2022.npy", allow_pickle=False).astype(np.float64),
        "late_2023": np.load(v184_dir / "hierarchical_residual_2023.npy", allow_pickle=False).astype(np.float64)[late23],
        "full_2024": np.load(v184_dir / "hierarchical_residual_2024.npy", allow_pickle=False).astype(np.float64),
    }
    for axis in AXES:
        if not (len(frames[axis]) == len(base[axis]) == len(local[axis])):
            raise ValueError(f"alignment mismatch: {axis}")

    gates = gate_library(frames)
    rows: list[dict[str, Any]] = []
    detail: dict[str, dict[str, Any]] = {}
    for gate_name, masks in gates.items():
        for weight in WEIGHTS:
            key = f"{gate_name}__w={weight:g}"
            item: dict[str, Any] = {}
            row: dict[str, Any] = {
                "gate": gate_name,
                "weight": weight,
                "minimum_active_fraction": min(
                    float(np.mean(masks[axis] & np.asarray(axes[axis]["exact_mask"], dtype=bool)))
                    for axis in AXES
                ),
            }
            for axis in AXES:
                candidate = apply_local(
                    base[axis], local[axis], axes[axis]["exact_mask"], masks[axis], weight
                )
                score = metrics(axes[axis], base[axis], candidate)
                item[axis] = score
                row[f"{axis}_gain"] = score["gain"]
                row[f"{axis}_month_fraction"] = score["positive_month_fraction"]
                row[f"{axis}_worst_month"] = score["worst_month_gain"]
            detail[key] = item
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "gate_screen.csv", index=False, encoding="utf-8-sig")

    loo: dict[str, Any] = {}
    for held in AXES:
        fit_axes = tuple(axis for axis in AXES if axis != held)
        selected = _select(table, fit_axes)
        key = f"{selected['gate']}__w={float(selected['weight']):g}"
        loo[held] = {
            "fit_axes": list(fit_axes),
            "gate": str(selected["gate"]),
            "weight": float(selected["weight"]),
            "held_metrics": detail[key][held],
        }
    final = _select(table, AXES)
    final_key = f"{final['gate']}__w={float(final['weight']):g}"
    final_metrics = detail[final_key]
    loo_pass = all(loo[axis]["held_metrics"]["gain"] > 0.0 for axis in AXES)
    final_pass = all(
        final_metrics[axis]["gain"] > 0.0
        and final_metrics[axis]["positive_month_fraction"] >= 0.5
        for axis in AXES
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "transport_pass" if loo_pass and final_pass else "transport_reject",
        "candidate_count": int(len(table)),
        "gate_count": int(len(gates)),
        "leave_one_origin_out": loo,
        "leave_one_origin_out_passed": bool(loo_pass),
        "final_all_origin_selection": {
            "gate": str(final["gate"]),
            "weight": float(final["weight"]),
            "metrics": final_metrics,
        },
        "final_passed": bool(final_pass),
        "eligible_for_packaging": bool(loo_pass and final_pass),
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "held_origin_excluded_from_each_loo_selection": True,
        "low_cardinality_row_local_gates_only": True,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "other_test_rows_required": False,
        "public_score_used_for_selection": False,
        "row_local_inference": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--v192-axes", type=Path, required=True)
    parser.add_argument("--v184-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv, args.contract_dir, args.bridge_oof,
        args.v192_axes, args.v184_dir, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
