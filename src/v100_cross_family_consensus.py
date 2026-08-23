"""Cross-family sign consensus between v87 FM and v98 conditional ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.v86_reliability_gated_r_fm import apply_reliability_offset
from src.v87_cross_season_consensus_r_fm import combine_corrections
from src.core.contract import _diagnostics, _load_contract_axis


PROTOCOL = "V100_CROSS_FAMILY_CONSENSUS_V1"


def cross_family_candidate(
    parent: np.ndarray,
    domain3: np.ndarray,
    fm_candidate: np.ndarray,
    conditional_correction: np.ndarray,
    conditional_eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.asarray(parent, dtype=np.float64)
    fm_shift = np.asarray(fm_candidate, dtype=np.float64) - parent
    conditional_shift = float(conditional_eta) * np.asarray(
        conditional_correction, dtype=np.float64
    )
    active = np.asarray(domain3).astype(str) == "R_CORE"
    agree = (
        (np.signbit(fm_shift) == np.signbit(conditional_shift))
        & (fm_shift != 0.0)
        & (conditional_shift != 0.0)
        & active
    )
    candidate = parent.copy()
    candidate[agree] = np.clip(
        parent[agree] + fm_shift[agree] + conditional_shift[agree],
        0.001,
        0.999,
    )
    return candidate, agree


def _axis_metrics(
    exact: dict[str, np.ndarray], candidate: np.ndarray, agree: np.ndarray
) -> dict[str, Any]:
    mask = exact["exact_mask"].astype(bool)
    parent = exact["parent"][mask].astype(np.float64)
    axis = {
        "target": exact["target"][mask].astype(np.float64),
        "parent": parent,
        "direct": parent + (candidate[mask] - parent),
        "domain3": exact["domain3"][mask],
        "game_month": exact["game_month"][mask],
    }
    metrics = _diagnostics(axis, ("R_CORE",), 1.0)
    core = axis["domain3"].astype(str) == "R_CORE"
    metrics["consensus_fraction_r_core"] = float(agree[mask][core].mean())
    return metrics


def _rank(
    policies: list[str], details: dict[str, dict[str, dict[str, Any]]], config: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    minimum_fraction = float(
        config["selection_gate"]["minimum_positive_month_fraction"]
    )
    worst_limit = float(config["selection_gate"]["worst_month_gain_strictly_above"])
    for policy in policies:
        axes = [details[policy][name] for name in ("full_2022", "late_2023")]
        passed = bool(
            all(item["gain"] > 0.0 for item in axes)
            and all(item["positive_month_fraction"] >= minimum_fraction for item in axes)
            and all(item["worst_month_gain"] > worst_limit for item in axes)
            and all(item["minimum_domain_gain"] >= 0.0 for item in axes)
        )
        rows.append(
            {
                "policy": policy,
                "source_gate_passed": passed,
                "minimum_gain": min(item["gain"] for item in axes),
                "minimum_month_fraction": min(
                    item["positive_month_fraction"] for item in axes
                ),
                "worst_month_gain": min(item["worst_month_gain"] for item in axes),
                "minimum_domain_gain": min(item["minimum_domain_gain"] for item in axes),
                "robust_score": min(
                    min(item["gain"] for item in axes),
                    min(item["worst_month_gain"] for item in axes),
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["source_gate_passed", "robust_score", "minimum_gain"], ascending=False
    ).reset_index(drop=True)


def run(
    train_csv: Path,
    contract_dir: Path,
    v87_dir: Path,
    v98_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_csv, low_memory=False)
    frame22 = raw.loc[raw["season"].eq(2022)].reset_index(drop=True)
    frame23 = raw.loc[
        raw["season"].eq(2023) & raw["game_month"].ge(8)
    ].reset_index(drop=True)
    frame24 = raw.loc[raw["season"].eq(2024)].reset_index(drop=True)
    exact22 = _load_contract_axis(contract_dir / "v84_full_2022.npz")
    exact23 = _load_contract_axis(contract_dir / "v84_late_2023.npz")
    exact24 = _load_contract_axis(contract_dir / "v84_full_2024.npz")
    with np.load(v87_dir / "source_corrections.npz") as saved:
        source = {name: saved[name].astype(np.float64) for name in saved.files}
    with np.load(v87_dir / "outer_full_2024.npz") as saved:
        correction24_old = saved["correction_old"].astype(np.float64)
        correction24_recent = saved["correction_recent"].astype(np.float64)
    conditional22 = np.load(v98_dir / "ablation_full_2022.npz")["correction"].astype(np.float64)
    conditional24 = np.load(v98_dir / "ablation_full_2024.npz")["correction"].astype(np.float64)
    common23 = _load_contract_axis(contract_dir / "common_full_2023.npz")
    conditional23_full = np.load(v98_dir / "ablation_full_2023.npz")["correction"].astype(np.float64)
    location23 = {
        int(value): index for index, value in enumerate(common23["raw_index"])
    }
    take23 = np.asarray(
        [location23[int(value)] for value in exact23["raw_index"]], dtype=np.int64
    )
    conditional23 = conditional23_full[take23]

    pairs = {
        "full_2022": (
            frame22, exact22, conditional22,
            source["correction22_old"], source["correction22_recent"],
        ),
        "late_2023": (
            frame23, exact23, conditional23,
            source["correction23_old"], source["correction23_recent"],
        ),
    }
    details: dict[str, dict[str, dict[str, Any]]] = {}
    source_candidates: dict[str, dict[str, np.ndarray]] = {}
    for policy in config["fm_policies"]:
        details[policy] = {}
        source_candidates[policy] = {}
        for name, (frame, exact, conditional, older, recent) in pairs.items():
            fm_correction, _ = combine_corrections(older, recent, policy)
            fm_candidate, _, _ = apply_reliability_offset(
                frame, exact["parent"], fm_correction, "none",
                eta=float(config["fm_eta"]),
            )
            candidate, agree = cross_family_candidate(
                exact["parent"], exact["domain3"], fm_candidate,
                conditional, float(config["conditional_eta"]),
            )
            source_candidates[policy][name] = candidate
            details[policy][name] = _axis_metrics(exact, candidate, agree)

    ranking = _rank(list(config["fm_policies"]), details, config)
    ranking.to_csv(output_dir / "source_policy_ranking.csv", index=False)
    selected = str(ranking.iloc[0]["policy"])
    source_passed = bool(ranking.iloc[0]["source_gate_passed"])

    # Open the locked 2024 axis once for the source-selected policy.
    fm24, _ = combine_corrections(
        correction24_old, correction24_recent, selected
    )
    fm_candidate24, _, _ = apply_reliability_offset(
        frame24, exact24["parent"], fm24, "none", eta=float(config["fm_eta"])
    )
    candidate24, agree24 = cross_family_candidate(
        exact24["parent"], exact24["domain3"], fm_candidate24,
        conditional24, float(config["conditional_eta"]),
    )
    locked = _axis_metrics(exact24, candidate24, agree24)
    locked_passed = bool(
        locked["gain"] > 0.0
        and locked["positive_month_fraction"]
        >= float(config["selection_gate"]["minimum_positive_month_fraction"])
        and locked["worst_month_gain"]
        > float(config["selection_gate"]["worst_month_gain_strictly_above"])
        and locked["minimum_domain_gain"] >= 0.0
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        candidate22=source_candidates[selected]["full_2022"],
        candidate23=source_candidates[selected]["late_2023"],
        candidate24=candidate24,
        agree24=agree24,
    )
    result = {
        "protocol": PROTOCOL,
        "selected_policy": selected,
        "source_gate_passed": source_passed,
        "source_ranking": ranking.to_dict(orient="records"),
        "source_details": details[selected],
        "locked_full_2024": locked,
        "locked_gate_passed": locked_passed,
        "point_gates_passed": bool(source_passed and locked_passed),
        "eligible_for_packaging": False,
        "packaging_reason": (
            "bootstrap and Reality Check pending"
            if source_passed and locked_passed
            else "source or locked point gate failed"
        ),
        "test_csv_read": False,
        "test_aggregate_used": False,
        "row_local_inference": True,
        "public_score_used_for_selection": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v87-dir", type=Path, required=True)
    parser.add_argument("--v98-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv, args.contract_dir, args.v87_dir,
        args.v98_dir, args.config, args.output_dir,
    )


if __name__ == "__main__":
    main()
