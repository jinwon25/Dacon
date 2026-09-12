"""Strict-forward audit of pitcher-by-batter-hand partial pooling above v290."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v298_recent_rate_denominator_signal import metrics
from src.core.contract import _load_contract_axis


PROTOCOL = "V311_PLATOON_PARTIAL_POOLING_STRICT_AUDIT_V1"
TARGET = "control_success"
PITCHER_SHRINK = 400.0
CORRECTION_CAP = 0.050


def logit(values: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def expit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))


def fit_platoon_lookup(history: pd.DataFrame) -> tuple[dict[tuple[int, int], float], dict[str, float]]:
    """Estimate only the interaction beyond pitcher and hand main effects."""

    frame = history.dropna(subset=["pitcher_id", "batter_hand", TARGET]).copy()
    global_rate = float(frame[TARGET].mean())
    hand = frame.groupby("batter_hand", observed=True)[TARGET].agg(["size", "sum"])
    hand_rate = (
        hand["sum"] + 1000.0 * global_rate
    ) / (hand["size"] + 1000.0)
    pitcher = frame.groupby("pitcher_id", observed=True)[TARGET].agg(["size", "sum"])
    pitcher_rate = (
        pitcher["sum"] + PITCHER_SHRINK * global_rate
    ) / (pitcher["size"] + PITCHER_SHRINK)
    cells = (
        frame.groupby(["pitcher_id", "batter_hand"], observed=True)[TARGET]
        .agg(["size", "sum"])
        .reset_index()
    )
    expected = (
        logit(cells["pitcher_id"].map(pitcher_rate).fillna(global_rate).to_numpy())
        + logit(cells["batter_hand"].map(hand_rate).fillna(global_rate).to_numpy())
        - float(logit(np.asarray([global_rate]))[0])
    )
    observed_rate = cells["sum"].to_numpy(np.float64) / cells["size"].to_numpy(np.float64)
    raw_delta = logit(observed_rate) - expected
    observed_var = 1.0 / np.clip(
        cells["size"].to_numpy(np.float64)
        * observed_rate * (1.0 - observed_rate),
        1e-6,
        None,
    )
    stable = cells["size"].to_numpy(np.float64) >= 30.0
    tau2 = max(
        float(np.var(raw_delta[stable], ddof=1) - np.mean(observed_var[stable])),
        1e-6,
    ) if stable.sum() >= 2 else 1e-6
    shrunk = raw_delta * tau2 / (tau2 + observed_var)
    lookup = {
        (int(pitcher_id), int(batter_hand)): float(delta)
        for pitcher_id, batter_hand, delta in zip(
            cells["pitcher_id"], cells["batter_hand"], shrunk
        )
    }
    diagnostics = {
        "global_rate": global_rate,
        "tau2": tau2,
        "cells": int(len(cells)),
        "stable_cells": int(stable.sum()),
    }
    return lookup, diagnostics


def map_platoon_direction(
    query: pd.DataFrame,
    parent: np.ndarray,
    lookup: dict[tuple[int, int], float],
) -> tuple[np.ndarray, np.ndarray]:
    keys = zip(query["pitcher_id"].astype(int), query["batter_hand"].astype(int))
    delta = np.asarray([lookup.get(key, 0.0) for key in keys], dtype=np.float64)
    active = delta != 0.0
    unit = expit(logit(parent) + np.clip(delta, -0.50, 0.50)) - parent
    unit = np.clip(unit, -CORRECTION_CAP, CORRECTION_CAP)
    unit[~active] = 0.0
    return unit, active


def pooled_dose(parts: list[tuple[np.ndarray, np.ndarray]]) -> float:
    numerator = float(sum(np.dot(residual, direction) for residual, direction in parts))
    denominator = float(sum(np.dot(direction, direction) for _, direction in parts))
    return float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator else 0.0


def restrictions() -> dict[str, bool]:
    return {
        "official_train_only": True,
        "evaluation_row_id_or_order_used": False,
        "other_evaluation_rows_required": False,
        "runtime_live_features_are_row_local": True,
        "strictly_prior_season_lookup": True,
        "source_dose_frozen_before_full_2024": True,
        "public_score_used_for_selection": False,
    }


def run(
    train_csv: Path,
    contract_dir: Path,
    v285_axes: Path,
    v290_axes: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=["season", "game_month", "pitcher_id", "batter_hand", TARGET],
        low_memory=False,
    )
    frames = {
        "full_2022": train.loc[train["season"].eq(2022)].reset_index(drop=True),
        "late_2023": train.loc[
            train["season"].eq(2023) & train["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": train.loc[train["season"].eq(2024)].reset_index(drop=True),
    }
    axes = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in frames
    }
    with np.load(v285_axes, allow_pickle=False) as saved:
        parent = {"full_2022": saved["candidate_full_2022"].astype(np.float64)}
    with np.load(v290_axes, allow_pickle=False) as saved:
        parent.update({
            "late_2023": saved["candidate_late_2023"].astype(np.float64),
            "full_2024": saved["candidate_full_2024"].astype(np.float64),
        })
    directions: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for year, name in ((2022, "full_2022"), (2023, "late_2023"), (2024, "full_2024")):
        lookup, diagnostics[name] = fit_platoon_lookup(
            train.loc[train["season"].lt(year)]
        )
        directions[name], covered = map_platoon_direction(frames[name], parent[name], lookup)
        active[name] = np.asarray(axes[name]["exact_mask"], dtype=bool) & covered
        if not np.array_equal(
            frames[name][TARGET].to_numpy(np.float64),
            np.asarray(axes[name]["target"], dtype=np.float64),
        ):
            raise ValueError(f"target alignment mismatch: {name}")

    frame22 = frames["full_2022"]
    early22 = frame22["game_month"].le(7).to_numpy() & active["full_2022"]
    late22 = frame22["game_month"].ge(8).to_numpy() & active["full_2022"]
    target = {name: frames[name][TARGET].to_numpy(np.float64) for name in frames}
    dose = pooled_dose([
        (
            target["full_2022"][late22] - parent["full_2022"][late22],
            directions["full_2022"][late22],
        ),
        (
            target["late_2023"][active["late_2023"]]
            - parent["late_2023"][active["late_2023"]],
            directions["late_2023"][active["late_2023"]],
        ),
    ])
    candidates = {
        name: np.clip(parent[name] + dose * directions[name], 0.001, 0.999)
        for name in frames
    }
    source = {
        "strict_prior_to_late_2022": metrics(
            frame22.loc[late22].reset_index(drop=True),
            target["full_2022"][late22], parent["full_2022"][late22],
            candidates["full_2022"][late22],
            np.ones(int(late22.sum()), dtype=bool),
        ),
        "strict_prior_to_late_2023": metrics(
            frames["late_2023"], target["late_2023"], parent["late_2023"],
            candidates["late_2023"], active["late_2023"],
        ),
    }
    locked = metrics(
        frames["full_2024"], target["full_2024"], parent["full_2024"],
        candidates["full_2024"], active["full_2024"],
    )
    source_pass = bool(
        dose > 0.0
        and all(item["gain"] > 0.0 for item in source.values())
        and all(item["positive_month_fraction"] >= 2.0 / 3.0 for item in source.values())
    )
    locked_pass = bool(
        locked["gain"] > 0.0
        and locked["positive_month_fraction"] >= 0.625
        and locked["worst_month_gain"] > -3.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{f"parent_{name}": parent[name] for name in frames},
        **{f"candidate_{name}": candidates[name] for name in frames},
        **{f"active_{name}": active[name] for name in frames},
        **{f"direction_{name}": directions[name] for name in frames},
    )
    result = {
        "protocol": PROTOCOL,
        "status": "robust_point_candidate" if source_pass and locked_pass else (
            "locked_reject" if source_pass else "source_reject"
        ),
        "source_selected_dose": dose,
        "lookup_diagnostics": diagnostics,
        "source": source,
        "locked_full_2024": locked,
        "source_gate_passed": source_pass,
        "locked_gate_passed": locked_pass,
        "eligible_for_packaging": bool(source_pass and locked_pass),
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v290-axes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.train_csv, args.contract_dir, args.v285_axes, args.v290_axes,
        args.output_dir,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
