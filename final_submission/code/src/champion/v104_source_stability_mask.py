"""Source-only context stability mask for the frozen v103 correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.robust_local_evaluation import grouped_gain_table
from src.core.contract import _load_contract_axis
from src.core.axis_metrics import _axis_metrics, _robust_axis


PROTOCOL = "V104_SOURCE_STABILITY_MASK_V1"


def context_labels(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    balls = pd.to_numeric(frame["balls_before"], errors="raise").to_numpy(np.int8)
    strikes = pd.to_numeric(frame["strikes_before"], errors="raise").to_numpy(np.int8)
    pitcher_hand = frame["pitcher_hand"].astype(str).to_numpy()
    batter_hand = frame["batter_hand"].astype(str).to_numpy()
    history = pd.cut(
        pd.to_numeric(frame["asof_pitcher_n"], errors="raise"),
        bins=[-np.inf, 29, 199, 999, np.inf],
        labels=["0-29", "30-199", "200-999", "1000+"],
    ).astype(str).to_numpy()
    return {
        "count": np.char.add(np.char.add(balls.astype(str), "-"), strikes.astype(str)),
        "history": history,
        "platoon": np.char.add(np.char.add(pitcher_hand, "-"), batter_hand),
    }


def policy_mask(
    labels: dict[str, np.ndarray], safe: dict[str, list[str]], policy: str
) -> np.ndarray:
    flags = {
        key: np.isin(np.asarray(labels[key]).astype(str), np.asarray(safe[key]).astype(str))
        for key in ("count", "history", "platoon")
    }
    if policy == "count_only":
        return flags["count"]
    if policy == "history_only":
        return flags["history"]
    if policy == "platoon_only":
        return flags["platoon"]
    if policy == "count_history":
        return flags["count"] & flags["history"]
    if policy == "count_platoon":
        return flags["count"] & flags["platoon"]
    if policy == "history_platoon":
        return flags["history"] & flags["platoon"]
    if policy == "all_three":
        return flags["count"] & flags["history"] & flags["platoon"]
    if policy == "majority_two":
        return (
            flags["count"].astype(np.int8)
            + flags["history"].astype(np.int8)
            + flags["platoon"].astype(np.int8)
        ) >= 2
    raise ValueError(f"unknown policy: {policy}")


def _learn_safe_levels(
    frames: dict[str, pd.DataFrame],
    exact: dict[str, dict[str, np.ndarray]],
    raw_candidate: dict[str, np.ndarray],
    minimum_rows: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    tables: dict[str, dict[str, list[dict[str, Any]]]] = {}
    by_axis: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for axis_name in ("full_2022", "late_2023"):
        axis = exact[axis_name]
        mask = axis["exact_mask"].astype(bool)
        frame = frames[axis_name].loc[mask].reset_index(drop=True)
        labels = context_labels(frame)
        target = axis["target"][mask].astype(np.float64)
        parent = axis["parent"][mask].astype(np.float64)
        candidate = raw_candidate[axis_name][mask].astype(np.float64)
        tables[axis_name] = {}
        by_axis[axis_name] = {}
        for family, values in labels.items():
            table = grouped_gain_table(target, candidate, parent, values)
            records = table.to_dict(orient="records")
            tables[axis_name][family] = records
            by_axis[axis_name][family] = {
                str(row["group"]): {
                    "gain": float(row["gain"]), "n_rows": int(row["n_rows"])
                }
                for row in records
            }
    safe: dict[str, list[str]] = {}
    for family in ("count", "history", "platoon"):
        common = set(by_axis["full_2022"][family]) & set(by_axis["late_2023"][family])
        safe[family] = sorted(
            level
            for level in common
            if all(
                by_axis[axis_name][family][level]["gain"] > 0.0
                and by_axis[axis_name][family][level]["n_rows"] >= minimum_rows
                for axis_name in ("full_2022", "late_2023")
            )
        )
    return safe, tables


def _masked_candidate(
    parent: np.ndarray, raw_candidate: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    return np.where(np.asarray(mask, dtype=bool), raw_candidate, parent).astype(np.float64)


def _point_pass(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return bool(
        metrics["gain"] > 0.0
        and metrics["positive_month_fraction"]
        >= float(gate["minimum_positive_month_fraction"])
        and metrics["worst_month_gain"]
        > float(gate["worst_month_gain_strictly_above"])
        and metrics["minimum_domain_gain"] >= float(gate["minimum_domain_gain"])
    )


def run(
    train_csv: Path,
    contract_dir: Path,
    v103_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_csv, low_memory=False)
    frames = {
        "full_2022": raw.loc[raw["season"].eq(2022)].reset_index(drop=True),
        "late_2023": raw.loc[
            raw["season"].eq(2023) & raw["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": raw.loc[raw["season"].eq(2024)].reset_index(drop=True),
    }
    exact = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(contract_dir / "v84_full_2024.npz"),
    }
    late24_mask = exact["full_2024"]["game_month"].astype(np.int16) >= 8
    exact["late_2024"] = {
        key: np.asarray(value)[late24_mask]
        for key, value in exact["full_2024"].items()
    }
    frames["late_2024"] = frames["full_2024"].loc[late24_mask].reset_index(drop=True)
    with np.load(v103_dir / "fixed_union_axes.npz") as saved:
        raw_candidate = {
            "full_2022": saved["candidate_full_2022"].astype(np.float64),
            "late_2023": saved["candidate_late_2023"].astype(np.float64),
            "full_2024": saved["candidate_full_2024"].astype(np.float64),
            "late_2024": saved["candidate_late_2024"].astype(np.float64),
        }
    safe, source_tables = _learn_safe_levels(
        frames, exact, raw_candidate, int(config["minimum_group_rows_per_source"])
    )
    policies = [str(value) for value in config["policies"]]
    candidates: dict[str, dict[str, np.ndarray]] = {policy: {} for policy in policies}
    source_metrics: dict[str, dict[str, dict[str, Any]]] = {policy: {} for policy in policies}
    for axis_name in ("full_2022", "late_2023", "full_2024", "late_2024"):
        labels = context_labels(frames[axis_name])
        axis = exact[axis_name]
        for policy in policies:
            mask = policy_mask(labels, safe, policy)
            candidate = _masked_candidate(axis["parent"], raw_candidate[axis_name], mask)
            candidates[policy][axis_name] = candidate
            if axis_name in ("full_2022", "late_2023"):
                metrics = _axis_metrics(axis, candidate)
                metrics["mask_fraction"] = float(mask.mean())
                source_metrics[policy][axis_name] = metrics
    gate = config["selection_gate"]
    ranking_rows = []
    for policy in policies:
        items = [source_metrics[policy][name] for name in ("full_2022", "late_2023")]
        passed = all(_point_pass(item, gate) for item in items)
        ranking_rows.append(
            {
                "policy": policy,
                "source_gate_passed": passed,
                "minimum_gain": min(item["gain"] for item in items),
                "worst_month_gain": min(item["worst_month_gain"] for item in items),
                "minimum_month_fraction": min(
                    item["positive_month_fraction"] for item in items
                ),
                "mean_mask_fraction": float(np.mean([item["mask_fraction"] for item in items])),
            }
        )
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["source_gate_passed", "minimum_gain", "worst_month_gain"],
        ascending=False,
    ).reset_index(drop=True)
    ranking.to_csv(output_dir / "source_policy_ranking.csv", index=False)
    selected = str(ranking.iloc[0]["policy"])
    selected_source_pass = bool(ranking.iloc[0]["source_gate_passed"])
    locked_metrics = {
        name: _axis_metrics(exact[name], candidates[selected][name])
        for name in ("full_2024", "late_2024")
    }
    locked_point_pass = {
        name: _point_pass(item, gate) for name, item in locked_metrics.items()
    }
    family_by_axis = {
        name: [candidates[policy][name] for policy in policies]
        for name in ("full_2022", "late_2023", "full_2024", "late_2024")
    }
    robust = {
        name: _robust_axis(
            frames[name], exact[name], candidates[selected][name], family_by_axis[name],
            config, 10 * index,
        )
        for index, name in enumerate(
            ("full_2022", "late_2023", "full_2024", "late_2024")
        )
    }
    robust_pass = {
        name: bool(
            all(result[key]["p05"] > 0.0 for key in (
                "pitcher", "crossed_pitcher_batter", "chronological_block"
            ))
            and result["reality_check"]["p_value"]
            <= float(gate["reality_check_alpha"])
        )
        for name, result in robust.items()
    }
    eligible = bool(
        selected_source_pass
        and all(locked_point_pass.values())
        and all(robust_pass.values())
    )
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        **{name: candidates[selected][name] for name in candidates[selected]},
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote" if eligible else "reject",
        "safe_levels_learned_from_sources": safe,
        "source_group_tables": source_tables,
        "source_ranking": ranking.to_dict(orient="records"),
        "selected_policy": selected,
        "selected_source_metrics": source_metrics[selected],
        "selected_source_gate_passed": selected_source_pass,
        "locked_metrics": locked_metrics,
        "locked_point_gate_pass": locked_point_pass,
        "robust": robust,
        "robust_gate_pass": robust_pass,
        "eligible_for_packaging": eligible,
        "packaging_reason": "all gates passed" if eligible else "point or robustness gate failed",
        **config["restrictions"],
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
    parser.add_argument("--v103-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.train_csv, args.contract_dir, args.v103_dir, args.config, args.output_dir)


if __name__ == "__main__":
    main()
