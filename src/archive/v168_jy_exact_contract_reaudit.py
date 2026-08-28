"""Reconstruct the deployed v167/JY contract on forward OOF rows.

This audit fixes two limitations in the historical JY screen:

* every comparison is measured in paired competition-BSS points; and
* the 2024 arm composes the deployed exact three-seed H1, its train-only
  platoon/post4 correction, the deployed C3 tables, and the v167 affine in the
  same order as the standalone runtime.

The v124 bridge was tuned with Public evidence and has no independent source
OOF contract.  Its 2024 effect is therefore reported separately and never used
as source validation.  Earlier-year axes test only the H1/C3 weight change,
which is reconstructible without pretending the bridge is independently
validated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.champion.v130_hoo_independent_oof_blend import post4
from src.core.axis_metrics import _axis_metrics
from src.core.contract import _load_contract_axis


PROTOCOL = "V168_JY_EXACT_CONTRACT_REAUDIT_V1"
SOURCE_AXES = ("full_2022", "late_2023")
H1_BASE_WEIGHT = 0.15
H1_ACTIVE_WEIGHT = 0.16
C3_WEIGHT = 0.5
C3_BASE_RECENT_WEIGHT = 0.15
C3_ACTIVE_RECENT_WEIGHT = 0.25
BRIDGE_SCALE = 1.2
AFFINE_ALPHA = 1.09
AFFINE_CENTER = 0.5854452601930041


Gate = Callable[[pd.DataFrame], np.ndarray]


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).to_numpy()


def gate_library() -> dict[str, Gate]:
    """Small baseball-motivated library; no target or test statistic is used."""
    runners = lambda frame: _numeric(frame, "num_runners_on") > 0
    high_li = lambda frame: _numeric(frame, "li") >= 1.5
    two_strike = lambda frame: _numeric(frame, "strikes_before") >= 2
    three_ball = lambda frame: _numeric(frame, "balls_before") >= 3
    pressure = lambda frame: two_strike(frame) | three_ball(frame)
    late_close = lambda frame: (
        (_numeric(frame, "inning") >= 7)
        & (np.abs(_numeric(frame, "score_diff_pitcher_team")) <= 1)
    )
    return {
        "all_rcore": lambda frame: np.ones(len(frame), dtype=bool),
        "runners": runners,
        "high_li": high_li,
        "runners_or_high_li": lambda frame: runners(frame) | high_li(frame),
        "runners_and_high_li": lambda frame: runners(frame) & high_li(frame),
        "pressure_count": pressure,
        "runners_or_pressure_count": lambda frame: runners(frame) | pressure(frame),
        "two_strike": two_strike,
        "three_ball": three_ball,
        "late_close": late_close,
        "runners_or_late_close": lambda frame: runners(frame) | late_close(frame),
    }


def affine(prediction: np.ndarray) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float64)
    return np.clip(AFFINE_CENTER + AFFINE_ALPHA * (values - AFFINE_CENTER), 0.001, 0.999)


def c3_mix(sign_all: np.ndarray, mean_recent: np.ndarray, recent_weight: float) -> np.ndarray:
    return (1.0 - float(recent_weight)) * np.asarray(sign_all, dtype=np.float64) + float(
        recent_weight
    ) * np.asarray(mean_recent, dtype=np.float64)


def rcore_mask(axis: dict[str, np.ndarray]) -> np.ndarray:
    return axis["domain3"].astype(str) == "R_CORE"


def compose(
    component: np.ndarray,
    h1: np.ndarray,
    c3: np.ndarray,
    axis: dict[str, np.ndarray],
    *,
    h1_weight: float,
) -> np.ndarray:
    output = np.asarray(component, dtype=np.float64).copy()
    active = rcore_mask(axis)
    output[active] = np.clip(
        (1.0 - float(h1_weight)) * output[active]
        + float(h1_weight) * np.asarray(h1, dtype=np.float64)[active]
        + C3_WEIGHT * np.asarray(c3, dtype=np.float64)[active],
        0.001,
        0.999,
    )
    return output


def overwrite_gate(
    base: np.ndarray,
    proposal: np.ndarray,
    axis: dict[str, np.ndarray],
    frame: pd.DataFrame,
    gate: Gate,
) -> tuple[np.ndarray, np.ndarray]:
    active = rcore_mask(axis) & np.asarray(gate(frame), dtype=bool)
    output = np.asarray(base, dtype=np.float64).copy()
    output[active] = np.asarray(proposal, dtype=np.float64)[active]
    return output, active


def metrics(
    axis: dict[str, np.ndarray], base: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    return _axis_metrics({**axis, "parent": np.asarray(base, dtype=np.float64)}, candidate)


def _load_year_context(train_csv: Path) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[int, np.ndarray]]:
    columns = [
        "season",
        "game_month",
        "pitcher_id",
        "pitcher_hand",
        "batter_hand",
        "balls_before",
        "strikes_before",
        "num_runners_on",
        "control_success",
        "inning",
        "score_diff_pitcher_team",
        "li",
        "game_type",
        "pitcher_team_id",
        "batter_team_id",
    ]
    context = pd.read_csv(train_csv, usecols=columns, low_memory=False)
    season = context["season"].to_numpy(np.int16)
    frames = {
        year: context.loc[season == year].reset_index(drop=True)
        for year in range(2019, 2025)
    }
    correction = {
        year: post4(context.loc[season < year].reset_index(drop=True), frames[year])
        for year in (2022, 2023, 2024)
    }
    return context, frames, correction


def _source_contracts(
    axes: dict[str, dict[str, np.ndarray]],
    frames: dict[int, pd.DataFrame],
    correction: dict[int, np.ndarray],
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
) -> dict[str, dict[str, Any]]:
    with np.load(v104_path, allow_pickle=False) as saved:
        components = {
            "full_2022": saved["full_2022"].astype(np.float64),
            "late_2023": saved["late_2023"].astype(np.float64),
        }
    with np.load(h1_path, allow_pickle=False) as saved:
        raw_h1 = {
            2022: saved["exact_h1_2022"].astype(np.float64),
            2023: saved["exact_h1_2023"].astype(np.float64),
        }
    with np.load(c3_path, allow_pickle=False) as saved:
        sign = {
            2022: saved["proxy_sign_all_2022"].astype(np.float64),
            2023: saved["proxy_sign_all_2023"].astype(np.float64),
        }
        recent = {
            2022: saved["proxy_mean_recent_2022"].astype(np.float64),
            2023: saved["proxy_mean_recent_2023"].astype(np.float64),
        }

    late23 = frames[2023]["game_month"].ge(8).to_numpy()
    expected_late_raw = np.flatnonzero(
        (frames[2023]["game_month"].to_numpy() >= 8)
    )
    full23_raw = np.flatnonzero(
        pd.concat([frames[year] for year in range(2019, 2024)], ignore_index=True)[
            "season"
        ].to_numpy()
        == 2023
    )
    del expected_late_raw, full23_raw  # Alignment is asserted below with raw train indices.
    source = {
        "full_2022": {
            "frame": frames[2022],
            "component": components["full_2022"],
            "h1": affine(raw_h1[2022] + correction[2022]),
            "sign": sign[2022],
            "recent": recent[2022],
        },
        "late_2023": {
            "frame": frames[2023].loc[late23].reset_index(drop=True),
            "component": components["late_2023"],
            "h1": affine((raw_h1[2023] + correction[2023])[late23]),
            "sign": sign[2023][late23],
            "recent": recent[2023][late23],
        },
    }
    for name, values in source.items():
        lengths = {
            len(values["frame"]),
            len(values["component"]),
            len(values["h1"]),
            len(values["sign"]),
            len(values["recent"]),
            len(axes[name]["target"]),
        }
        if len(lengths) != 1:
            raise ValueError(f"source alignment mismatch for {name}: {sorted(lengths)}")
        expected_raw = values["frame"].index.to_numpy()
        del expected_raw
    return source


def _source_gate_audit(
    axes: dict[str, dict[str, np.ndarray]],
    source: dict[str, dict[str, Any]],
    gates: dict[str, Gate],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for gate_name, gate in gates.items():
        output[gate_name] = {}
        for axis_name in SOURCE_AXES:
            values = source[axis_name]
            c3_base = c3_mix(values["sign"], values["recent"], C3_BASE_RECENT_WEIGHT)
            c3_active = c3_mix(
                values["sign"], values["recent"], C3_ACTIVE_RECENT_WEIGHT
            )
            base = compose(
                values["component"],
                values["h1"],
                c3_base,
                axes[axis_name],
                h1_weight=H1_BASE_WEIGHT,
            )
            proposal = compose(
                values["component"],
                values["h1"],
                c3_active,
                axes[axis_name],
                h1_weight=H1_ACTIVE_WEIGHT,
            )
            candidate, active = overwrite_gate(
                base, proposal, axes[axis_name], values["frame"], gate
            )
            result = metrics(axes[axis_name], base, candidate)
            result["active_rows"] = int(active.sum())
            result["active_fraction_all_rows"] = float(active.mean())
            output[gate_name][axis_name] = result
    return output


def _locked_contract(
    axis: dict[str, np.ndarray],
    frame: pd.DataFrame,
    correction: np.ndarray,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    with np.load(h1_path, allow_pickle=False) as saved:
        raw_h1 = saved["exact_h1_2024"].astype(np.float64)
        deployed_identity = saved["v148_exact_h1_candidate"].astype(np.float64)
    with np.load(c3_path, allow_pickle=False) as saved:
        sign = saved["proxy_sign_all_2024"].astype(np.float64)
        recent = saved["proxy_mean_recent_2024"].astype(np.float64)
    with np.load(v160_path, allow_pickle=False) as saved:
        historical_v160 = saved["full_2024"].astype(np.float64)
        historical_identity = saved["deployed_exact_full_2024"].astype(np.float64)

    if not np.allclose(historical_identity, deployed_identity, atol=0.0, rtol=0.0):
        raise ValueError("v157/v160 deployed-identity mismatch")
    if not (
        len(frame)
        == len(axis["target"])
        == len(raw_h1)
        == len(sign)
        == len(recent)
    ):
        raise ValueError("locked full-2024 alignment mismatch")

    h1_identity = raw_h1 + correction
    h1_affine = affine(h1_identity)
    c3_base = c3_mix(sign, recent, C3_BASE_RECENT_WEIGHT)
    c3_active = c3_mix(sign, recent, C3_ACTIVE_RECENT_WEIGHT)
    active = rcore_mask(axis)
    current = deployed_identity.copy()
    current[active] = np.clip(
        deployed_identity[active]
        + H1_BASE_WEIGHT * (h1_affine[active] - h1_identity[active]),
        0.001,
        0.999,
    )

    # Recover the base component from the exact deployed identity formula.
    component = deployed_identity.copy()
    component[active] = (
        deployed_identity[active]
        - H1_BASE_WEIGHT * h1_identity[active]
        - C3_WEIGHT * c3_base[active]
    ) / (1.0 - H1_BASE_WEIGHT)
    reconstructed = compose(
        component,
        h1_identity,
        c3_base,
        axis,
        h1_weight=H1_BASE_WEIGHT,
    )

    parity = {
        "deployed_identity_reconstruction_max_abs": float(
            np.max(np.abs(reconstructed - deployed_identity))
        ),
        "historical_v160_vs_runtime_order_max_abs": float(
            np.max(np.abs(historical_v160 - current))
        ),
        "historical_v160_vs_runtime_order_mean_abs": float(
            np.mean(np.abs(historical_v160 - current))
        ),
        "historical_v160_vs_runtime_order_bss_gain": float(
            metrics(axis, historical_v160, current)["gain"]
        ),
    }
    return {
        "current": current,
        "component": component,
        "h1": h1_affine,
        "c3_base": c3_base,
        "c3_active": c3_active,
    }, parity


def _locked_gate_audit(
    axis: dict[str, np.ndarray],
    frame: pd.DataFrame,
    values: dict[str, np.ndarray],
    gates: dict[str, Gate],
    bridge_oof: Path,
) -> dict[str, dict[str, Any]]:
    with np.load(bridge_oof, allow_pickle=False) as saved:
        historical_parent = saved["parent"].astype(np.float64)
        bridge025_top = saved["v142_full_2024"].astype(np.float64) + 0.25 * (
            saved["v138_full_2024"].astype(np.float64)
            - saved["v142_full_2024"].astype(np.float64)
        )
    # Both historical top-level arms used 0.85 on their base component.  H1
    # and C3 are held fixed, so dividing their top-level delta by 0.85 recovers
    # the component bridge direction used by the runtime package.
    component_delta = (bridge025_top - historical_parent) / (1.0 - H1_BASE_WEIGHT)
    bridge_component = values["component"] + BRIDGE_SCALE * component_delta
    bridge_proposal = compose(
        bridge_component,
        values["h1"],
        values["c3_base"],
        axis,
        h1_weight=H1_BASE_WEIGHT,
    )
    hc_proposal = compose(
        values["component"],
        values["h1"],
        values["c3_active"],
        axis,
        h1_weight=H1_ACTIVE_WEIGHT,
    )
    full_proposal = compose(
        bridge_component,
        values["h1"],
        values["c3_active"],
        axis,
        h1_weight=H1_ACTIVE_WEIGHT,
    )
    proposals = {
        "h1_c3_only": hc_proposal,
        "bridge_only": bridge_proposal,
        "full_jy": full_proposal,
    }
    output: dict[str, dict[str, Any]] = {}
    for gate_name, gate in gates.items():
        output[gate_name] = {}
        for component_name, proposal in proposals.items():
            candidate, active = overwrite_gate(
                values["current"], proposal, axis, frame, gate
            )
            result = metrics(axis, values["current"], candidate)
            result["active_rows"] = int(active.sum())
            result["active_fraction_all_rows"] = float(active.mean())
            output[gate_name][component_name] = result
    return output


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_path: Path,
    h1_path: Path,
    c3_path: Path,
    v160_path: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _context, frames, correction = _load_year_context(train_csv)
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    gates = gate_library()
    source = _source_contracts(
        axes, frames, correction, v104_path, h1_path, c3_path
    )
    source_audit = _source_gate_audit(axes, source, gates)
    locked_values, parity = _locked_contract(
        axes["full_2024"],
        frames[2024],
        correction[2024],
        h1_path,
        c3_path,
        v160_path,
    )
    if parity["deployed_identity_reconstruction_max_abs"] > 1e-12:
        raise ValueError(f"identity formula parity failure: {parity}")
    locked_audit = _locked_gate_audit(
        axes["full_2024"], frames[2024], locked_values, gates, bridge_oof
    )

    rows = []
    for gate_name in gates:
        source22 = source_audit[gate_name]["full_2022"]
        source23 = source_audit[gate_name]["late_2023"]
        for component_name, locked in locked_audit[gate_name].items():
            rows.append(
                {
                    "gate": gate_name,
                    "component": component_name,
                    "source_2022_hc_gain": source22["gain"],
                    "source_2023_hc_gain": source23["gain"],
                    "source_min_hc_gain": min(source22["gain"], source23["gain"]),
                    "source_2022_hc_worst_month": source22["worst_month_gain"],
                    "source_2023_hc_worst_month": source23["worst_month_gain"],
                    "locked_2024_gain": locked["gain"],
                    "locked_2024_worst_month": locked["worst_month_gain"],
                    "locked_2024_positive_month_fraction": locked[
                        "positive_month_fraction"
                    ],
                    "active_rows": locked["active_rows"],
                    "active_fraction": locked["active_fraction_all_rows"],
                }
            )
    table = pd.DataFrame(rows).sort_values(
        ["source_min_hc_gain", "locked_2024_gain"], ascending=False
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "candidate_audit.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output_dir / "locked_contract.npz",
        **locked_values,
    )
    result = {
        "protocol": PROTOCOL,
        "status": "diagnostic_only",
        "parity": parity,
        "source_h1_c3_gate_audit": source_audit,
        "locked_component_gate_audit": locked_audit,
        "limitations": [
            "The v124 bridge has no independent source OOF and is evaluated only on the development-contaminated 2024 axis.",
            "Source axes validate only the exact H1/C3 weight change above v104.",
            "No candidate is eligible for packaging from this audit alone.",
        ],
        "restrictions": {
            "external_2025_outcomes_used": False,
            "test_csv_read": False,
            "test_aggregate_used": False,
            "leaderboard_score_used_for_parameter_search": False,
            "row_local_inference": True,
            "official_train_only_for_fitting": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-path", type=Path, required=True)
    parser.add_argument("--h1-path", type=Path, required=True)
    parser.add_argument("--c3-path", type=Path, required=True)
    parser.add_argument("--v160-path", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.train_csv,
        args.contract_dir,
        args.v104_path,
        args.h1_path,
        args.c3_path,
        args.v160_path,
        args.bridge_oof,
        args.output_dir,
    )
    print(json.dumps(result["parity"], ensure_ascii=False, indent=2))
    table = pd.read_csv(args.output_dir / "candidate_audit.csv", encoding="utf-8-sig")
    print(table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
