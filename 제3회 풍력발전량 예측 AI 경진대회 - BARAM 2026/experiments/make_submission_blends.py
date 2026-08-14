from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.metrics import CAPACITY_KWH


def blend_submission(base_path: Path, member_path: Path, output_path: Path, weights: dict[str, float]) -> None:
    base = pd.read_csv(base_path, encoding="utf-8-sig")
    member = pd.read_csv(member_path, encoding="utf-8-sig")
    if list(base[["forecast_id", "forecast_kst_dtm"]].itertuples(index=False)) != list(
        member[["forecast_id", "forecast_kst_dtm"]].itertuples(index=False)
    ):
        raise ValueError("Submission keys do not match.")

    out = base.copy()
    for target, capacity in CAPACITY_KWH.items():
        weight = weights.get(target, weights.get("__all__", 0.0))
        out[target] = ((1.0 - weight) * base[target] + weight * member[target]).clip(0, capacity)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")


def parse_weights(value: str) -> dict[str, float]:
    if "=" not in value:
        return {"__all__": float(value)}
    weights: dict[str, float] = {}
    for part in value.split(","):
        key, raw = part.split("=", 1)
        weights[key.strip()] = float(raw)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="submissions/blend_v1_over115_cal125.csv")
    parser.add_argument("--member", default="submissions/scada_proxy_stack_hist.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--weights",
        required=True,
        help="Single alpha such as 0.05, or comma-separated target weights such as kpx_group_1=0.05,kpx_group_2=0.03",
    )
    args = parser.parse_args()

    blend_submission(Path(args.base), Path(args.member), Path(args.output), parse_weights(args.weights))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
