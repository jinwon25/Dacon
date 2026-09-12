from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT.parent / "data" / "train.csv"
OUT_DIR = ROOT / "artifacts" / "row_region_variants_01"


AXES = {
    "full_2022": ROOT / "artifacts" / "oof_champion_1161" / "v84_full_2022.npz",
    "late_2023": ROOT / "artifacts" / "oof_champion_1161" / "v84_late_2023.npz",
    "full_2024_v84": ROOT / "artifacts" / "oof_champion_1161" / "v84_full_2024.npz",
    "locked_bridge_2024": ROOT / "artifacts" / "oof_champion_1170" / "v148_full_2024.npz",
}


GATES = {
    "runners": lambda f: f["runners_on"],
    "runners_or_high_li": lambda f: f["runners_on"] | f["high_li"],
    "runners_or_pressure_count": lambda f: f["runners_on"] | f["pressure_count"],
    "runners_or_high_li_or_pressure_count": lambda f: f["runners_on"] | f["high_li"] | f["pressure_count"],
}

BRIDGE_SCALES = {
    "bridge022": 0.7,
    "bridge025": 1.0,
    "bridge027": 1.2,
}

SCALE_GATES = {
    "runners": lambda f: f["runners_on"],
    "runners_or_high_li": lambda f: f["runners_on"] | f["high_li"],
}


def logloss_sum(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(-(y * np.log(p) + (1.0 - y) * np.log1p(-p)).sum())


def score_gain(y: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> float:
    return logloss_sum(y, base) - logloss_sum(y, candidate)


def masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    pitcher_team = pd.to_numeric(frame["pitcher_team_id"], errors="coerce").fillna(-1).astype("int64")
    batter_team = pd.to_numeric(frame["batter_team_id"], errors="coerce").fillna(-1).astype("int64")
    r_core = frame["game_type"].astype(str).eq("R") & ~(pitcher_team.eq(13) | batter_team.eq(13))
    return {
        "r_core": r_core.to_numpy(),
        "runners_on": (pd.to_numeric(frame["num_runners_on"], errors="coerce").fillna(0).to_numpy() > 0),
        "high_li": (pd.to_numeric(frame["li"], errors="coerce").fillna(0.0).to_numpy() >= 1.5),
        "pressure_count": (
            (pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).to_numpy() >= 3)
            | (pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).to_numpy() >= 2)
        ),
    }


def month_worst_gain(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, months: np.ndarray) -> float:
    gains = []
    for month in sorted(set(months.tolist())):
        m = months == month
        gains.append(score_gain(y[m], base[m], candidate[m]))
    return float(min(gains)) if gains else 0.0


def eval_axis(name: str, path: Path, train_cols: pd.DataFrame, gate_name: str, gate: np.ndarray, scale: float = 1.0) -> dict[str, float | int]:
    z = np.load(path, allow_pickle=True)
    y = z["target"].astype(float)
    months = z["game_month"]
    active = gate

    if name == "locked_bridge_2024":
        base = z["parent"].astype(float)
        bridge025 = z["v142_full_2024"].astype(float) + 0.25 * (
            z["v138_full_2024"].astype(float) - z["v142_full_2024"].astype(float)
        )
        proposal = base + scale * (bridge025 - base)
    else:
        base = z["common_parent"].astype(float)
        proposal = z["parent"].astype(float)

    candidate = base.copy()
    candidate[active] = proposal[active]
    return {
        f"{name}_gain": score_gain(y, base, candidate),
        f"{name}_worst_month_gain": month_worst_gain(y, base, candidate, months),
        f"{name}_rows": int(active.sum()),
        f"{name}_fraction": float(active.mean()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    usecols = [
        "game_type",
        "pitcher_team_id",
        "batter_team_id",
        "num_runners_on",
        "li",
        "balls_before",
        "strikes_before",
    ]
    train_cols = pd.read_csv(TRAIN, usecols=usecols, encoding="utf-8-sig")
    results = []

    loaded_masks: dict[str, dict[str, np.ndarray]] = {}
    for axis_name, path in AXES.items():
        z = np.load(path, allow_pickle=True)
        frame = train_cols.iloc[z["raw_index"]].reset_index(drop=True)
        loaded_masks[axis_name] = masks(frame)

    for gate_name, gate_fn in GATES.items():
        row: dict[str, object] = {"candidate": gate_name, "bridge_scale": 1.0}
        for axis_name, path in AXES.items():
            f = loaded_masks[axis_name]
            gate = f["r_core"] & gate_fn(f)
            row.update(eval_axis(axis_name, path, train_cols, gate_name, gate, scale=1.0))
        results.append(row)

    for scale_gate_name, scale_gate_fn in SCALE_GATES.items():
        for scale_name, scale in BRIDGE_SCALES.items():
            row = {"candidate": f"{scale_gate_name}_{scale_name}", "bridge_scale": scale}
            for axis_name, path in AXES.items():
                f = loaded_masks[axis_name]
                gate = f["r_core"] & scale_gate_fn(f)
                row.update(eval_axis(axis_name, path, train_cols, scale_gate_name, gate, scale=scale))
            results.append(row)

    df = pd.DataFrame(results)
    sort_cols = ["locked_bridge_2024_gain", "full_2024_v84_gain", "late_2023_gain", "full_2022_gain"]
    df = df.sort_values(sort_cols, ascending=False)
    df.to_csv(OUT_DIR / "sequential_gate_audit.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "sequential_gate_audit.json").write_text(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
