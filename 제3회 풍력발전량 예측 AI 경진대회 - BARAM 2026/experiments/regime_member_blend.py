from __future__ import annotations

import argparse
import json
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


def parse_int_set(value: str) -> set[int] | None:
    if not value:
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def make_mask(
    base_pred: pd.Series,
    member_pred: pd.Series,
    capacity: float,
    timestamps: pd.Series,
    max_disagreement: float,
    min_base_ratio: float,
    max_base_ratio: float,
    months: set[int] | None,
    hours: set[int] | None,
    direction: str,
) -> pd.Series:
    base_ratio = base_pred / capacity
    normalized_delta = (member_pred - base_pred) / capacity
    mask = (
        normalized_delta.abs().le(max_disagreement)
        & base_ratio.ge(min_base_ratio)
        & base_ratio.le(max_base_ratio)
    )
    if months is not None:
        mask &= timestamps.dt.month.isin(months)
    if hours is not None:
        mask &= timestamps.dt.hour.isin(hours)
    if direction == "up":
        mask &= normalized_delta.gt(0)
    elif direction == "down":
        mask &= normalized_delta.lt(0)
    elif direction != "both":
        raise ValueError("direction must be one of: both, up, down")
    return mask


def blend_regime(
    base_path: Path,
    member_path: Path,
    output_path: Path,
    weights: dict[str, float],
    max_disagreement: float,
    min_base_ratio: float,
    max_base_ratio: float,
    months: set[int] | None,
    hours: set[int] | None,
    direction: str,
) -> dict[str, dict[str, float]]:
    base = pd.read_csv(base_path, encoding="utf-8-sig")
    member = pd.read_csv(member_path, encoding="utf-8-sig")
    if list(base[KEY_COLUMNS].itertuples(index=False)) != list(member[KEY_COLUMNS].itertuples(index=False)):
        raise ValueError("Submission keys do not match.")

    timestamps = pd.to_datetime(base["forecast_kst_dtm"])
    out = base.copy()
    report: dict[str, dict[str, float]] = {}

    for target, capacity in CAPACITY_KWH.items():
        alpha = weights.get(target, weights.get("__all__", 0.0))
        base_pred = base[target].astype(float)
        member_pred = member[target].astype(float)
        mask = make_mask(
            base_pred=base_pred,
            member_pred=member_pred,
            capacity=capacity,
            timestamps=timestamps,
            max_disagreement=max_disagreement,
            min_base_ratio=min_base_ratio,
            max_base_ratio=max_base_ratio,
            months=months,
            hours=hours,
            direction=direction,
        )
        blended = base_pred.copy()
        if abs(alpha) > 0:
            blended.loc[mask] = (1.0 - alpha) * base_pred.loc[mask] + alpha * member_pred.loc[mask]
        out[target] = np.clip(blended, 0, capacity)
        delta = out[target].astype(float) - base_pred
        changed = delta.abs() > 1e-12
        report[target] = {
            "alpha": float(alpha),
            "eligible_rows": int(mask.sum()),
            "changed_rows": int(changed.sum()),
            "changed_ratio": float(changed.mean()),
            "mean_delta": float(delta.mean()),
            "absmean_delta": float(delta.abs().mean()),
            "p95_abs_delta": float(delta.abs().quantile(0.95)),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="submissions/blend_over115_scada_stack5.csv")
    parser.add_argument("--member", default="submissions/scada_proxy_stack_hist.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--max-disagreement", type=float, default=0.06)
    parser.add_argument("--min-base-ratio", type=float, default=0.10)
    parser.add_argument("--max-base-ratio", type=float, default=1.00)
    parser.add_argument("--months", default="")
    parser.add_argument("--hours", default="")
    parser.add_argument("--direction", choices=["both", "up", "down"], default="both")
    args = parser.parse_args()

    report = blend_regime(
        base_path=Path(args.base),
        member_path=Path(args.member),
        output_path=Path(args.output),
        weights=parse_weights(args.weights),
        max_disagreement=args.max_disagreement,
        min_base_ratio=args.min_base_ratio,
        max_base_ratio=args.max_base_ratio,
        months=parse_int_set(args.months),
        hours=parse_int_set(args.hours),
        direction=args.direction,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
