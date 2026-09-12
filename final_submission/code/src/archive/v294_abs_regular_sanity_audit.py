"""Reproduce the recent-F dose curve used to justify a 2024-R expert.

Official KBO material establishes that Futures ABS started in 2020, so the
2023 F break must not be labelled an ABS introduction.  It remains a useful
structural-break sanity analogue: a single recent post-break season predicts
the next F season better than mixed history.  This audit reproduces the fixed
five-seed F expert curve above v244 in both probability and logit space.  It
does not claim that an honest local validation exists for a model fitted on
the only post-ABS R season (2024) and deployed to 2025.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics


PROTOCOL = "V294_ABS_REGULAR_SANITY_AUDIT_V1"
WEIGHTS = (0.10, 0.20, 0.30)


def _logit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _expit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def blend(
    parent: np.ndarray,
    expert: np.ndarray,
    active: np.ndarray,
    weight: float,
    mode: str,
) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    if mode == "probability":
        output[active] += float(weight) * (expert - output[active])
    elif mode == "logit":
        output[active] = _expit(
            _logit(output[active])
            + float(weight) * (_logit(expert) - _logit(output[active]))
        )
    else:
        raise ValueError(f"unknown blend mode: {mode}")
    return np.clip(output, 0.001, 0.999)


def run(
    train_csv: Path,
    v285_axes: Path,
    v288_dir: Path,
    c332_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    full_2024 = train.loc[train["season"].eq(2024)].reset_index(drop=True)
    target = full_2024["control_success"].to_numpy(np.float64)
    active = full_2024["game_type"].astype(str).eq("F").to_numpy()
    with np.load(v285_axes, allow_pickle=False) as saved:
        parent = saved["baseline_full_2024"].astype(np.float64)
    expert = np.load(
        v288_dir / "expert_full_2024_multiseed.npy", allow_pickle=False
    ).astype(np.float64)
    if len(parent) != len(full_2024) or len(expert) != int(active.sum()):
        raise ValueError("F sanity axis alignment mismatch")

    axis = {
        "target": target,
        "exact_mask": np.ones(len(full_2024), dtype=bool),
        "game_month": full_2024["game_month"].to_numpy(),
    }
    curves: dict[str, list[dict[str, float]]] = {}
    for mode in ("probability", "logit"):
        rows = []
        for weight in WEIGHTS:
            candidate = blend(parent, expert, active, weight, mode)
            metrics = paired_metrics(axis, parent, candidate, active)
            shift = candidate - parent
            rows.append(
                {
                    "weight": float(weight),
                    "gain": float(metrics["gain"]),
                    "active_gain": float(metrics["active_gain"]),
                    "whole_row_rms_shift": float(np.sqrt(np.mean(np.square(shift)))),
                    "active_mean_abs_shift": float(np.mean(np.abs(shift[active]))),
                }
            )
        curves[mode] = rows

    c332_bytes = c332_summary.read_bytes()
    try:
        c332_text = c332_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # An earlier Windows run wrote this artifact with the active CP949 codepage.
        c332_text = c332_bytes.decode("cp949")
    c332 = json.loads(c332_text)
    probability_w010 = curves["probability"][0]["gain"]
    reproduced = bool(abs(probability_w010 - 3.8605805245) < 0.05)
    summary = {
        "protocol": PROTOCOL,
        "status": "sanity_reproduced" if reproduced else "sanity_failed",
        "official_fact_correction": {
            "futures_abs_start": 2020,
            "kbo_league_abs_start": 2024,
            "f_2023_break_is_abs_introduction": False,
            "interpretation": (
                "F-2023 is an empirical structural break, not the introduction "
                "of Futures ABS; analogy to R-2025 is therefore indirect."
            ),
        },
        "rows": {"full_2024": len(full_2024), "f_active": int(active.sum())},
        "curves": curves,
        "probability_w010_expected_gain": 3.8605805245,
        "probability_w010_reproduced": reproduced,
        "c332_regime_matched_oracle": c332["F_2024"],
        "deployment_contract": {
            "fit": "season=2024 and game_type=R",
            "predict": "2025 rows with game_type=R",
            "seeds": [2871, 2873, 2875, 2877, 2879],
            "weight": 0.10,
            "blend_mode": "logit",
            "honest_local_label_axis_available": False,
        },
        "restrictions": {
            "official_train_only": True,
            "test_csv_read": False,
            "public_score_used_to_tune_weight": False,
            "full_2024_not_claimed_as_honest_validation_for_final_r_model": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--v285-axes", type=Path, required=True)
    parser.add_argument("--v288-dir", type=Path, required=True)
    parser.add_argument("--c332-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.train_csv,
                args.v285_axes,
                args.v288_dir,
                args.c332_summary,
                args.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
