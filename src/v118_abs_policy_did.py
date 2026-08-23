"""Policy-timed difference-in-differences residual audit above exact v104.

The model does not carry a league-wide calibration offset.  It transports
only shrunken context contrasts.  Hyperparameters are selected by asking
whether the 2019->2020 Futures transition contrast persisted in 2021 and
2022.  The frozen recipe is then used to estimate the 2023->early-2024
Regular transition above exact v104 and is audited once on late 2024.

All query features are row-local.  Test rows, Public scores, row order and
current-pitch measurements are never used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.core.diagnostics import diagnostics


PROTOCOL = "V118_ABS_POLICY_DID_V1"
TARGET = "control_success"


@dataclass(frozen=True)
class GroupSpec:
    name: str
    columns: tuple[str, ...]


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    output = _add_domain_and_pressure(frame.reset_index(drop=True).copy())
    balls = pd.to_numeric(output["balls_before"], errors="coerce").fillna(-1).astype(int)
    strikes = pd.to_numeric(output["strikes_before"], errors="coerce").fillna(-1).astype(int)
    output["count_state"] = balls.astype(str) + "-" + strikes.astype(str)
    output["hand_matchup"] = (
        output["pitcher_hand"].astype("string").fillna("__MISSING__").astype(str)
        + "-"
        + output["batter_hand"].astype("string").fillna("__MISSING__").astype(str)
    )
    return output


def group_key(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    values = frame.loc[:, list(columns)].astype("string").fillna("__MISSING__")
    key = values.iloc[:, 0].astype(str)
    for column in values.columns[1:]:
        key = key.str.cat(values[column].astype(str), sep="\x1f")
    return key.to_numpy(str)


def _centered_group_effect(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    columns: tuple[str, ...],
    domain_mask: np.ndarray,
    alpha: float,
    minimum_rows: int,
) -> dict[str, float]:
    outcome = np.asarray(outcome, dtype=np.float64)
    domain_mask = np.asarray(domain_mask, dtype=bool)
    if outcome.shape != (len(frame),) or domain_mask.shape != (len(frame),):
        raise ValueError("effect arrays are not aligned")
    if not domain_mask.any():
        return {}
    local = outcome[domain_mask]
    centre = float(local.mean())
    keys = group_key(frame, columns)[domain_mask]
    codes, names = pd.factorize(keys, sort=False)
    count = np.bincount(codes, minlength=len(names)).astype(np.float64)
    total = np.bincount(codes, weights=local - centre, minlength=len(names))
    effect = total / (count + float(alpha))
    return {
        str(name): float(value)
        for name, value, n_rows in zip(names, effect, count, strict=True)
        if n_rows >= int(minimum_rows)
    }


def difference_in_differences(
    pre: pd.DataFrame,
    pre_outcome: np.ndarray,
    post: pd.DataFrame,
    post_outcome: np.ndarray,
    columns: tuple[str, ...],
    *,
    treated_pre: np.ndarray,
    control_pre: np.ndarray,
    treated_post: np.ndarray,
    control_post: np.ndarray,
    alpha: float,
    minimum_rows: int,
) -> dict[str, float]:
    """Return (treated post-pre) - (control post-pre) context contrasts."""

    pre_t = _centered_group_effect(
        pre, pre_outcome, columns, treated_pre, alpha, minimum_rows
    )
    pre_c = _centered_group_effect(
        pre, pre_outcome, columns, control_pre, alpha, minimum_rows
    )
    post_t = _centered_group_effect(
        post, post_outcome, columns, treated_post, alpha, minimum_rows
    )
    post_c = _centered_group_effect(
        post, post_outcome, columns, control_post, alpha, minimum_rows
    )
    shared = sorted(set(pre_t) & set(pre_c) & set(post_t) & set(post_c))
    return {
        key: float((post_t[key] - pre_t[key]) - (post_c[key] - pre_c[key]))
        for key in shared
    }


def map_effect(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    effect: dict[str, float],
    correction_cap: float,
) -> np.ndarray:
    values = np.asarray(
        [effect.get(key, 0.0) for key in group_key(frame, columns)],
        dtype=np.float64,
    )
    return np.clip(values, -float(correction_cap), float(correction_cap))


def apply_effect(
    frame: pd.DataFrame,
    parent: np.ndarray,
    correction: np.ndarray,
    active: np.ndarray,
    eta: float,
) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if parent.shape != correction.shape or parent.shape != active.shape:
        raise ValueError("candidate arrays are not aligned")
    output = parent.copy()
    output[active] = np.clip(
        parent[active] + float(eta) * correction[active], 0.001, 0.999
    )
    return output


def compact(result: dict[str, Any], active_domain: str) -> dict[str, float]:
    return {
        "gain": float(result["gain"]),
        "positive_month_fraction": float(result["positive_month_fraction"]),
        "worst_month_gain": float(result["worst_month_gain"]),
        "active_domain_gain": float(result["domain_gains"].get(active_domain, 0.0)),
        "mean_abs_shift": float(result["mean_abs_shift"]),
    }


def _metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": frame[TARGET].to_numpy(np.float64),
            "game_month": frame["game_month"].to_numpy(np.int16),
            "domain3": frame["domain3"].astype(str).to_numpy(),
        }
    )


def _load_contract_axis(
    contract_dir: Path,
    raw: pd.DataFrame,
    name: str,
    parent_override: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    with np.load(contract_dir / f"{name}.npz", allow_pickle=False) as saved:
        index = saved["raw_index"].astype(np.int64)
        target = saved["target"].astype(np.float64)
        parent = saved["parent"].astype(np.float64)
        domain = saved["domain3"].astype(str)
    frame = add_context(raw.iloc[index].reset_index(drop=True))
    frame["domain3"] = domain
    if not np.array_equal(target, frame[TARGET].to_numpy(np.float64)):
        raise ValueError(f"target/order mismatch: {name}")
    if parent_override is not None:
        parent = np.asarray(parent_override, dtype=np.float64)
    if parent.shape != (len(frame),):
        raise ValueError(f"parent shape mismatch: {name}")
    return frame, parent


def _source_pass(row: dict[str, Any], gate: dict[str, Any]) -> bool:
    return bool(
        row["gain"] > 0.0
        and row["positive_month_fraction"] >= float(gate["positive_month_fraction_min"])
        and row["worst_month_gain"] > float(gate["worst_month_gain_min_exclusive"])
        and row["active_domain_gain"] > float(gate["active_domain_gain_min"])
    )


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_axes: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_csv.resolve(), low_memory=False)
    raw = add_context(raw)
    specs = [
        GroupSpec(name, tuple(columns))
        for name, columns in config["group_specs"].items()
    ]
    with np.load(v104_axes.resolve(), allow_pickle=False) as saved:
        v104 = {name: saved[name].astype(np.float64) for name in saved.files}

    common21, parent21 = _load_contract_axis(contract_dir, raw, "common_full_2021")
    common22, parent22_common = _load_contract_axis(contract_dir, raw, "common_full_2022")
    full22, parent22 = _load_contract_axis(
        contract_dir, raw, "v84_full_2022", v104["full_2022"]
    )
    late23, parent23 = _load_contract_axis(
        contract_dir, raw, "v84_late_2023", v104["late_2023"]
    )
    full24, parent24 = _load_contract_axis(
        contract_dir, raw, "v84_full_2024", v104["full_2024"]
    )
    pre19 = raw.loc[raw["season"].eq(2019)].reset_index(drop=True)
    post20 = raw.loc[raw["season"].eq(2020)].reset_index(drop=True)
    raw_pre = pre19[TARGET].to_numpy(np.float64)
    raw_post = post20[TARGET].to_numpy(np.float64)
    minimum_rows = int(config["minimum_group_rows"])
    cap = float(config["correction_cap"])

    rows: list[dict[str, Any]] = []
    source_effects: dict[tuple[str, float], dict[str, float]] = {}
    for spec in specs:
        for alpha in config["alpha_grid"]:
            effect = difference_in_differences(
                pre19,
                raw_pre,
                post20,
                raw_post,
                spec.columns,
                treated_pre=pre19["domain3"].eq("F").to_numpy(),
                control_pre=pre19["domain3"].ne("F").to_numpy(),
                treated_post=post20["domain3"].eq("F").to_numpy(),
                control_post=post20["domain3"].ne("F").to_numpy(),
                alpha=float(alpha),
                minimum_rows=minimum_rows,
            )
            source_effects[(spec.name, float(alpha))] = effect
            for axis_name, frame, parent in (
                ("persist_2021", common21, parent21),
                ("persist_2022", common22, parent22_common),
            ):
                correction = map_effect(frame, spec.columns, effect, cap)
                active = frame["domain3"].eq("F").to_numpy()
                for eta in config["eta_grid"]:
                    candidate = apply_effect(frame, parent, correction, active, float(eta))
                    result = diagnostics(_metric_frame(frame), parent, candidate, active)
                    rows.append(
                        {
                            "axis": axis_name,
                            "group": spec.name,
                            "alpha": float(alpha),
                            "eta": float(eta),
                            "effect_count": len(effect),
                            **compact(result, "F"),
                        }
                    )
    source_table = pd.DataFrame(rows)
    ranking_rows = []
    gate = config["source_gate"]
    for values, local in source_table.groupby(["group", "alpha", "eta"], observed=True):
        records = local.to_dict("records")
        passed = len(records) == 2 and all(_source_pass(row, gate) for row in records)
        ranking_rows.append(
            {
                "group": values[0],
                "alpha": float(values[1]),
                "eta": float(values[2]),
                "source_gate_passed": passed,
                "minimum_gain": float(local["gain"].min()),
                "minimum_month_fraction": float(local["positive_month_fraction"].min()),
                "worst_month_gain": float(local["worst_month_gain"].min()),
                "minimum_active_domain_gain": float(local["active_domain_gain"].min()),
            }
        )
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["source_gate_passed", "minimum_gain", "worst_month_gain"],
        ascending=False,
    ).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
    spec = next(item for item in specs if item.name == selected["group"])

    # Exact v104 2023->2024 policy-transition estimate.  Only March-July 2024
    # labels enter the fitted contrast; August-October are held out once.
    early24_mask = full24["game_month"].le(7).to_numpy()
    late24_mask = full24["game_month"].ge(8).to_numpy()
    early24 = full24.loc[early24_mask].reset_index(drop=True)
    early24_residual = early24[TARGET].to_numpy(np.float64) - parent24[early24_mask]
    pre23_residual = late23[TARGET].to_numpy(np.float64) - parent23
    event_effect = difference_in_differences(
        late23,
        pre23_residual,
        early24,
        early24_residual,
        spec.columns,
        treated_pre=late23["domain3"].eq("R_CORE").to_numpy(),
        control_pre=late23["domain3"].eq("F").to_numpy(),
        treated_post=early24["domain3"].eq("R_CORE").to_numpy(),
        control_post=early24["domain3"].eq("F").to_numpy(),
        alpha=float(selected["alpha"]),
        minimum_rows=minimum_rows,
    )
    late24 = full24.loc[late24_mask].reset_index(drop=True)
    correction24 = map_effect(late24, spec.columns, event_effect, cap)
    active24 = late24["domain3"].eq("R_CORE").to_numpy()
    candidate24 = apply_effect(
        late24,
        parent24[late24_mask],
        correction24,
        active24,
        float(selected["eta"]),
    )
    locked_result = diagnostics(
        _metric_frame(late24), parent24[late24_mask], candidate24, active24
    )
    locked = compact(locked_result, "R_CORE")
    locked_gate = config["locked_gate"]
    locked_pass = bool(
        locked["gain"] >= float(locked_gate["gain_min"])
        and locked["positive_month_fraction"] >= float(locked_gate["positive_month_fraction_min"])
        and locked["worst_month_gain"] > float(locked_gate["worst_month_gain_min_exclusive"])
        and locked["active_domain_gain"] > float(locked_gate["active_domain_gain_min"])
    )

    source_table.to_csv(output_dir / "source_metrics.csv", index=False)
    ranking.to_csv(output_dir / "source_ranking.csv", index=False)
    np.savez_compressed(
        output_dir / "locked_late_2024.npz",
        target=late24[TARGET].to_numpy(np.float64),
        parent=parent24[late24_mask],
        candidate=candidate24,
        correction=correction24,
        active=active24,
        pitcher_id=late24["pitcher_id"].to_numpy(),
        batter_id=late24["batter_id"].to_numpy(),
        game_month=late24["game_month"].to_numpy(np.int16),
        domain3=late24["domain3"].astype(str).to_numpy(),
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "robust_audit_required" if locked_pass else "reject",
        "selected_recipe": selected,
        "selected_source_metrics": source_table.loc[
            source_table["group"].eq(selected["group"])
            & source_table["alpha"].eq(selected["alpha"])
            & source_table["eta"].eq(selected["eta"])
        ].to_dict("records"),
        "source_gate_passed": bool(selected["source_gate_passed"]),
        "locked_late_2024": locked,
        "locked_point_gate_passed": locked_pass,
        "source_event_effect_count": len(
            source_effects[(spec.name, float(selected["alpha"]))]
        ),
        "locked_event_effect_count": len(event_effect),
        **config["restrictions"],
        "eligible_for_packaging": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-axes", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.train_csv, args.contract_dir, args.v104_axes, args.config, args.output_dir)


if __name__ == "__main__":
    main()
