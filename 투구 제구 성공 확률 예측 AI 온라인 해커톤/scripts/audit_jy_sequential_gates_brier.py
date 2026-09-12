"""Re-audit the frozen JY gate screen with the competition Brier metric.

The original 2026-08-28 screen is retained as historical evidence.  It used
summed log loss for candidate ordering, although the competition metric is a
Brier Skill Score.  This audit reproduces the exact same frozen proposals and
reports both metrics without opening any new model-selection dimensions.

The three v84 axes are historical proxy directions.  The v148 axis is only the
frozen bridge component, not a row-aligned reconstruction of the full deployed
JY formula.  Consequently this is a metric-correction audit, not a promotion
gate for a new submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

AXIS_FILES = {
    "full_2022": "artifacts/oof_champion_1161/v84_full_2022.npz",
    "late_2023": "artifacts/oof_champion_1161/v84_late_2023.npz",
    "full_2024_v84": "artifacts/oof_champion_1161/v84_full_2024.npz",
    "locked_bridge_2024": "artifacts/oof_champion_1170/v148_full_2024.npz",
}

GATES: dict[str, Callable[[dict[str, np.ndarray]], np.ndarray]] = {
    "runners": lambda masks: masks["runners_on"],
    "runners_or_high_li": lambda masks: masks["runners_on"] | masks["high_li"],
    "runners_or_pressure_count": lambda masks: (
        masks["runners_on"] | masks["pressure_count"]
    ),
    "runners_or_high_li_or_pressure_count": lambda masks: (
        masks["runners_on"] | masks["high_li"] | masks["pressure_count"]
    ),
}

SCALE_GATES = ("runners", "runners_or_high_li")
BRIDGE_SCALES = {"bridge022": 0.7, "bridge025": 1.0, "bridge027": 1.2}


def logloss_sum(target: np.ndarray, prediction: np.ndarray) -> float:
    probability = np.clip(np.asarray(prediction, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    truth = np.asarray(target, dtype=np.float64)
    return float(
        -(truth * np.log(probability) + (1.0 - truth) * np.log1p(-probability)).sum()
    )


def logloss_gain(
    target: np.ndarray, parent: np.ndarray, candidate: np.ndarray
) -> float:
    return logloss_sum(target, parent) - logloss_sum(target, candidate)


def brier_gain(
    target: np.ndarray, parent: np.ndarray, candidate: np.ndarray
) -> float:
    """Return the paired, unclipped competition-BSS improvement in points."""
    truth = np.asarray(target, dtype=np.float64)
    reference = float(truth.mean()) * (1.0 - float(truth.mean()))
    if reference <= 0.0:
        raise ValueError("Brier Skill Score is undefined for a constant target")
    parent_bs = float(np.mean(np.square(np.asarray(parent, dtype=np.float64) - truth)))
    candidate_bs = float(
        np.mean(np.square(np.asarray(candidate, dtype=np.float64) - truth))
    )
    return 100_000.0 * (parent_bs - candidate_bs) / reference


def row_brier_improvement(
    target: np.ndarray, parent: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    truth = np.asarray(target, dtype=np.float64)
    return np.square(np.asarray(parent, dtype=np.float64) - truth) - np.square(
        np.asarray(candidate, dtype=np.float64) - truth
    )


def context_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    pitcher_team = pd.to_numeric(frame["pitcher_team_id"], errors="coerce").fillna(-1)
    batter_team = pd.to_numeric(frame["batter_team_id"], errors="coerce").fillna(-1)
    r_core = frame["game_type"].astype(str).eq("R") & ~(
        pitcher_team.eq(13) | batter_team.eq(13)
    )
    return {
        "r_core": r_core.to_numpy(),
        "runners_on": (
            pd.to_numeric(frame["num_runners_on"], errors="coerce")
            .fillna(0)
            .to_numpy()
            > 0
        ),
        "high_li": (
            pd.to_numeric(frame["li"], errors="coerce").fillna(0.0).to_numpy()
            >= 1.5
        ),
        "pressure_count": (
            pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).to_numpy()
            >= 3
        )
        | (
            pd.to_numeric(frame["strikes_before"], errors="coerce")
            .fillna(0)
            .to_numpy()
            >= 2
        ),
    }


def frozen_proposal(
    axis_name: str, saved: np.lib.npyio.NpzFile, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    if axis_name == "locked_bridge_2024":
        parent = saved["parent"].astype(np.float64)
        bridge025 = saved["v142_full_2024"].astype(np.float64) + 0.25 * (
            saved["v138_full_2024"].astype(np.float64)
            - saved["v142_full_2024"].astype(np.float64)
        )
        return parent, parent + float(scale) * (bridge025 - parent)
    return (
        saved["common_parent"].astype(np.float64),
        saved["parent"].astype(np.float64),
    )


def axis_metrics(
    axis_name: str,
    path: Path,
    active: np.ndarray,
    scale: float,
) -> dict[str, float | int]:
    with np.load(path, allow_pickle=True) as saved:
        target = saved["target"].astype(np.float64)
        months = saved["game_month"].astype(np.int16)
        parent, proposal = frozen_proposal(axis_name, saved, scale)
    candidate = parent.copy()
    candidate[active] = proposal[active]
    month_gains = [
        brier_gain(target[months == month], parent[months == month], candidate[months == month])
        for month in sorted(np.unique(months))
    ]
    active_row_gain = row_brier_improvement(target, parent, candidate)[active]
    return {
        f"{axis_name}_bss_gain": brier_gain(target, parent, candidate),
        f"{axis_name}_logloss_sum_gain": logloss_gain(target, parent, candidate),
        f"{axis_name}_worst_month_bss_gain": float(min(month_gains)),
        f"{axis_name}_positive_month_fraction": float(
            np.mean(np.asarray(month_gains) > 0.0)
        ),
        f"{axis_name}_positive_active_row_fraction": float(
            np.mean(active_row_gain > 0.0)
        ),
        f"{axis_name}_rows": int(active.sum()),
        f"{axis_name}_fraction": float(active.mean()),
    }


def run(train_csv: Path, artifacts_root: Path, output_dir: Path) -> pd.DataFrame:
    paths = {name: artifacts_root / relative for name, relative in AXIS_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen OOF axes: {missing}")

    usecols = [
        "game_type",
        "pitcher_team_id",
        "batter_team_id",
        "num_runners_on",
        "li",
        "balls_before",
        "strikes_before",
    ]
    train = pd.read_csv(train_csv, usecols=usecols, encoding="utf-8-sig")
    masks_by_axis: dict[str, dict[str, np.ndarray]] = {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=True) as saved:
            raw_index = saved["raw_index"].astype(np.int64)
        masks_by_axis[name] = context_masks(train.iloc[raw_index].reset_index(drop=True))

    recipes: list[tuple[str, str, float]] = [
        (name, name, 1.0) for name in GATES
    ]
    recipes.extend(
        (f"{gate_name}_{scale_name}", gate_name, scale)
        for gate_name in SCALE_GATES
        for scale_name, scale in BRIDGE_SCALES.items()
    )

    rows: list[dict[str, object]] = []
    for candidate_name, gate_name, scale in recipes:
        row: dict[str, object] = {
            "candidate": candidate_name,
            "gate": gate_name,
            "bridge_scale": scale,
        }
        gate_fn = GATES[gate_name]
        for axis_name, path in paths.items():
            axis_masks = masks_by_axis[axis_name]
            active = axis_masks["r_core"] & gate_fn(axis_masks)
            row.update(axis_metrics(axis_name, path, active, scale))
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        [
            "locked_bridge_2024_bss_gain",
            "full_2024_v84_bss_gain",
            "late_2023_bss_gain",
            "full_2022_bss_gain",
        ],
        ascending=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "sequential_gate_brier_reaudit.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "protocol": "JY_SEQUENTIAL_GATE_BRIER_REAUDIT_V1",
        "selection_status": "diagnostic_only",
        "metric": "paired unclipped BSS-equivalent gain",
        "candidate_count": int(len(result)),
        "limitations": [
            "The v84 axes are historical proxy directions.",
            "The locked v148 axis isolates the bridge component and does not reconstruct the full deployed JY formula.",
            "No candidate may be promoted from this diagnostic alone.",
        ],
        "ranking": result.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--artifacts-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "jy_brier_metric_reaudit_20260828_01",
    )
    args = parser.parse_args()
    result = run(args.train_csv, args.artifacts_root, args.output_dir)
    columns = [
        "candidate",
        "full_2022_bss_gain",
        "late_2023_bss_gain",
        "full_2024_v84_bss_gain",
        "locked_bridge_2024_bss_gain",
        "locked_bridge_2024_worst_month_bss_gain",
    ]
    print(result[columns].to_string(index=False))


if __name__ == "__main__":
    main()
