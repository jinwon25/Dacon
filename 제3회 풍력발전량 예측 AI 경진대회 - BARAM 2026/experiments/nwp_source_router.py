"""Forward-select sparse blends toward independent LDAPS/GFS experts.

Source experts are deliberately weak global replacements.  This diagnostic
tests whether weather-similar six-hour blocks can use only 5-15% of their
direction.  Q1 selects on Q2; a frozen policy opens H2 once.  It never writes a
submission.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.blocked_rolling_validation import load_issue_times
from experiments.dynamic_router_oracle import load_experts
from experiments.robust_weather_router import RobustPolicy, route_robust
from experiments.weather_similarity_router import (
    H2_START,
    Q2_START,
    ROOT,
    build_blocks,
    evaluate_period,
    period_blocks,
    utility_table,
)


SOURCE_CACHE = (
    ROOT
    / "artifacts_final"
    / "nwp_source_ablation"
    / "validation_predictions.npz"
)
CAT_SOURCE_CACHE = (
    ROOT
    / "artifacts_final"
    / "nwp_source_catboost"
    / "validation_predictions.npz"
)
OUTPUT = (
    ROOT / "artifacts_final" / "diagnostics" / "nwp_source_router_20260718.json"
)
END = pd.Timestamp("2025-01-01")
MIN_LOCKED_SCORE_DELTA = 0.002
MIN_LOCKED_FICR_DELTA = 0.003


def policies() -> tuple[RobustPolicy, ...]:
    return tuple(
        RobustPolicy(k, positive, risk, alpha, coverage, False)
        for k, positive, risk, alpha, coverage in itertools.product(
            (12, 24, 36),
            (0.60, 0.70),
            (0.25, 0.50),
            (0.05, 0.10, 0.15),
            (0.02, 0.05),
        )
    )


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _component_floor(evaluation: dict[str, Any]) -> float:
    return min(
        value[component]
        for value in evaluation["months"].values()
        for component in ("score", "one_minus_nmae", "ficr")
    )


def qualifies(evaluation: dict[str, Any]) -> bool:
    delta = evaluation["delta"]
    return bool(
        delta["score"] > 0.0
        and delta["one_minus_nmae"] > 0.0
        and delta["ficr"] > 0.0
        and _component_floor(evaluation) >= 0.0
    )


def main() -> None:
    index, truth, existing = load_experts()
    cache = np.load(SOURCE_CACHE, allow_pickle=False)
    cache_index = pd.DatetimeIndex(pd.to_datetime(cache["index_ns"]))
    if not cache_index.equals(index):
        raise ValueError("Source-expert and exact OOF indexes differ")
    source_names = ("ldaps_all", "ldaps_eligible", "gfs_all", "gfs_eligible")
    experts = {"incumbent_finesweep": existing["incumbent_finesweep"]}
    experts.update({name: cache[name].astype(float) for name in source_names})
    cat_cache = np.load(CAT_SOURCE_CACHE, allow_pickle=False)
    cat_index = pd.DatetimeIndex(pd.to_datetime(cat_cache["index_ns"]))
    if not cat_index.equals(index):
        raise ValueError("CatBoost source-expert and exact OOF indexes differ")
    cat_names = ("cat_ldaps_eligible", "cat_gfs_eligible")
    experts.update({name: cat_cache[name].astype(float) for name in cat_names})

    issue = load_issue_times(ROOT / "data" / "train" / "gfs_train.csv", index)
    frame, rows_by_block = build_blocks(index, issue, experts)
    q1 = period_blocks(frame, None, Q2_START)
    q2 = period_blocks(frame, Q2_START, H2_START)
    h1 = period_blocks(frame, None, H2_START)
    h2 = period_blocks(frame, H2_START, END)

    q1_utility = utility_table(truth, experts, rows_by_block, q1)
    records: list[dict[str, Any]] = []
    for policy in policies():
        candidate, routing = route_robust(
            frame,
            rows_by_block,
            experts,
            q1,
            q2,
            q1_utility,
            policy,
        )
        evaluation = evaluate_period(
            index,
            truth,
            experts["incumbent_finesweep"],
            candidate,
            rows_by_block,
            q2,
        )
        records.append(
            {
                "policy": policy,
                "routing": routing,
                "evaluation": evaluation,
                "qualifies": qualifies(evaluation),
            }
        )

    eligible = [record for record in records if record["qualifies"]]
    selected = max(
        eligible,
        key=lambda record: (
            _component_floor(record["evaluation"]),
            record["evaluation"]["delta"]["score"],
        ),
        default=None,
    )
    locked = None
    promoted = False
    if selected is not None:
        h1_utility = utility_table(truth, experts, rows_by_block, h1)
        candidate, routing = route_robust(
            frame,
            rows_by_block,
            experts,
            h1,
            h2,
            h1_utility,
            selected["policy"],
        )
        evaluation = evaluate_period(
            index,
            truth,
            experts["incumbent_finesweep"],
            candidate,
            rows_by_block,
            h2,
        )
        promoted = bool(
            qualifies(evaluation)
            and evaluation["delta"]["score"] >= MIN_LOCKED_SCORE_DELTA
            and evaluation["delta"]["ficr"] >= MIN_LOCKED_FICR_DELTA
        )
        locked = {
            "routing": routing,
            "evaluation": evaluation,
            "promoted": promoted,
        }

    report = {
        "method": "sparse weather-similarity routing toward source-isolated experts",
        "contract": {
            "source_model_selection": "2023-Q1-Q3 train / 2023-Q4 early stop",
            "development": "Q1 utility routes Q2",
            "locked": "frozen policy; H1 utility routes H2 once",
            "movement": "5-15% toward a source expert on at most 2-5% of blocks",
            "minimum_locked_effect": {
                "score": MIN_LOCKED_SCORE_DELTA,
                "ficr": MIN_LOCKED_FICR_DELTA,
            },
            "submission_created": False,
        },
        "experts": list(experts),
        "policy_count": len(records),
        "qualified_q2_policy_count": len(eligible),
        "selected_policy": None
        if selected is None
        else {"name": selected["policy"].name, **selected["policy"].__dict__},
        "selected_development": None
        if selected is None
        else {
            "routing": selected["routing"],
            "evaluation": selected["evaluation"],
        },
        "locked": locked,
        "decision": (
            "promotion gate passed; final source models may be trained for deployment"
            if promoted
            else "rejected; no submission created"
        ),
        "top_q2": [
            {
                "policy": {
                    "name": record["policy"].name,
                    **record["policy"].__dict__,
                },
                "routing": record["routing"],
                "evaluation": record["evaluation"],
                "qualifies": record["qualifies"],
            }
            for record in sorted(
                records,
                key=lambda record: record["evaluation"]["delta"]["score"],
                reverse=True,
            )[:10]
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(_to_builtin(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            _to_builtin(
                {
                    "qualified_q2_policy_count": len(eligible),
                    "selected_policy": report["selected_policy"],
                    "locked": locked,
                    "decision": report["decision"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
