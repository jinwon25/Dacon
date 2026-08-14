from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import CAPACITY_KWH


KEY_COLUMNS = ["forecast_id", "forecast_kst_dtm"]


def parse_weights(value: str) -> dict[str, float]:
    if "=" not in value:
        return {"__all__": float(value)}
    weights: dict[str, float] = {}
    for part in value.split(","):
        key, raw = part.split("=", 1)
        weights[key.strip()] = float(raw)
    return weights


def blend_selectively(
    base_path: Path,
    member_path: Path,
    output_path: Path,
    weights: dict[str, float],
    max_disagreement: float,
    min_base_ratio: float,
) -> dict[str, dict[str, float]]:
    base = pd.read_csv(base_path, encoding="utf-8-sig")
    member = pd.read_csv(member_path, encoding="utf-8-sig")
    if list(base[KEY_COLUMNS].itertuples(index=False)) != list(member[KEY_COLUMNS].itertuples(index=False)):
        raise ValueError("Submission keys do not match.")

    out = base.copy()
    report: dict[str, dict[str, float]] = {}
    for target, capacity in CAPACITY_KWH.items():
        alpha = weights.get(target, weights.get("__all__", 0.0))
        base_pred = base[target].astype(float)
        member_pred = member[target].astype(float)
        eligible = base_pred >= min_base_ratio * capacity
        agreement = (member_pred - base_pred).abs() <= max_disagreement * capacity
        mask = eligible & agreement & (abs(alpha) > 0)
        blended = base_pred.copy()
        blended.loc[mask] = (1.0 - alpha) * base_pred.loc[mask] + alpha * member_pred.loc[mask]
        out[target] = np.clip(blended, 0, capacity)
        report[target] = {
            "alpha": float(alpha),
            "changed_rows": int(mask.sum()),
            "changed_ratio": float(mask.mean()),
            "mean_delta": float((out[target] - base[target]).mean()),
            "absmean_delta": float((out[target] - base[target]).abs().mean()),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="submissions/blend_over115_scada_stack5.csv")
    parser.add_argument("--member", default="submissions/power_curve_residual.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--max-disagreement", type=float, default=0.08)
    parser.add_argument("--min-base-ratio", type=float, default=0.10)
    args = parser.parse_args()

    report = blend_selectively(
        Path(args.base),
        Path(args.member),
        Path(args.output),
        parse_weights(args.weights),
        args.max_disagreement,
        args.min_base_ratio,
    )
    print(report)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
