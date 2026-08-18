"""Three-stage, family-balanced multi-bank screen above frozen v27.

Stage 1 uses 2022 only to prefilter signal identities from four independently
generated families: exact-ASOF, recent exact-ASOF, multi-year state, and latent
failure mode.  Stage 2 uses late-2023 only to choose direction, deployment
domain, blend weight, and at most one disjoint-domain pair.  The frozen recipe
is then opened once on full-2024 and secondarily checked on late-2024.

The 2022 parent predates v27 and is therefore used only for identity
prefiltering.  No 2022 gain is presented as a v27 analogue.  Audit labels do
not choose identities, weights, routes, or pair members.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v30_diverse_covariance_screen import (
    _cached_v25_axes,
    diagnostics,
    v27_parent,
)


WEIGHTS = (0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20)
DOMAINS = ("ALL", "R_CORE", "R_ANCHOR", "F")
DIRECTIONS = ("toward_parent", "delta_v21")
FAMILY_QUOTA = 3
FORBIDDEN = ("oracle", "mode_label", "raw_by_mode")
EXACT_NAMES = (
    "exact_lgb",
    "exact_ridge",
    "trend_lgb",
    "binary_seed_ensemble",
    "l2_leaves7",
    "l2_leaves15",
)
RECENT_NAMES = ("prediction", "exact_lgb", "exact_ridge", "trend_lgb")


def signal_family(name: str) -> str:
    return str(name).split("::", 1)[0]


def _mode_dir(year: int) -> str:
    return (
        "latent_failure_mode_state_20260817_02"
        if year == 2022
        else "latent_failure_mode_state_20260816_01"
    )


def _load_bank(
    project: Path, year: int, names: set[str] | None = None
) -> dict[str, np.ndarray]:
    """Load only legal prediction columns shared by all three audit years."""

    def wanted(name: str) -> bool:
        return names is None or name in names

    output: dict[str, np.ndarray] = {}
    mode_path = (
        project
        / "artifacts"
        / _mode_dir(year)
        / f"latent_failure_mode_o{year}.npz"
    )
    with np.load(mode_path, allow_pickle=True) as saved:
        mode_names = [str(value) for value in saved["names"].tolist()]
        for index, raw_name in enumerate(mode_names):
            name = f"mode::{raw_name}"
            if wanted(name) and not any(token in raw_name.lower() for token in FORBIDDEN):
                output[name] = saved["raw"][:, index].astype(np.float64)

    variants_path = (
        project
        / "artifacts"
        / "multi_year_state_variants_20260816_01"
        / f"state_variants_o{year}.npz"
    )
    with np.load(variants_path, allow_pickle=True) as saved:
        for raw_name in saved.files:
            name = f"state::{raw_name}"
            if raw_name.startswith(("global_", "domain_")) and wanted(name):
                output[name] = saved[raw_name].astype(np.float64)
    selected_path = (
        project
        / "artifacts"
        / "multi_year_state_selected_20260816_01"
        / f"selected_state_o{year}.npz"
    )
    if wanted("state::selected"):
        with np.load(selected_path, allow_pickle=True) as saved:
            output["state::selected"] = saved["raw"].astype(np.float64)

    exact_path = (
        project
        / "artifacts"
        / "v14_exact_model_screen_20260815_01"
        / f"exact_model_screen_o{year}.npz"
    )
    with np.load(exact_path, allow_pickle=True) as saved:
        for raw_name in EXACT_NAMES:
            name = f"exact::{raw_name}"
            if wanted(name):
                output[name] = saved[raw_name].astype(np.float64)

    recent_path = (
        project
        / "artifacts"
        / "recent_shared_exact_asof_20260815_02"
        / f"recent_shared_o{year}.npz"
    )
    with np.load(recent_path, allow_pickle=True) as saved:
        for raw_name in RECENT_NAMES:
            name = f"recent::{raw_name}"
            if wanted(name):
                output[name] = saved[raw_name].astype(np.float64)
    if names is not None and set(output) != set(names):
        missing = sorted(set(names) - set(output))
        raise ValueError(f"bank signals missing for {year}: {missing}")
    return output


def _metadata(project: Path, year: int) -> dict[str, np.ndarray]:
    path = (
        project
        / "artifacts"
        / "multi_year_state_selected_20260816_01"
        / f"selected_state_o{year}.npz"
    )
    with np.load(path, allow_pickle=True) as saved:
        return {
            "target": saved["target"].astype(np.float64),
            "parent": saved["incumbent"].astype(np.float64),
            "month": saved["game_month"].astype(np.int16),
            "domain": saved["domain3"].astype(str),
        }


def _mask(domain: np.ndarray, route: str) -> np.ndarray:
    if route == "ALL":
        return np.ones(len(domain), dtype=bool)
    return np.asarray(domain, dtype=str) == route


def _gain_grid(
    target: np.ndarray,
    parent: np.ndarray,
    direction: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    local_target = np.asarray(target, dtype=np.float64)[mask]
    local_parent = np.asarray(parent, dtype=np.float64)[mask]
    local_direction = np.asarray(direction, dtype=np.float64)[mask]
    residual = local_target - local_parent
    reference = float(local_target.mean() * (1.0 - local_target.mean()))
    scale = 1_000_000.0 if reference <= 0.0 else 100_000.0 / reference
    linear = 2.0 * float(np.mean(residual * local_direction))
    quadratic = float(np.mean(np.square(local_direction)))
    return scale * (weights * linear - np.square(weights) * quadratic)


def grid_rows(
    frame: pd.DataFrame,
    parent: np.ndarray,
    v21: np.ndarray,
    raw: np.ndarray,
    *,
    signal: str,
    direction_mode: str,
    route: str,
) -> list[dict[str, object]]:
    if direction_mode == "toward_parent":
        direction = np.asarray(raw, dtype=np.float64) - parent
    elif direction_mode == "delta_v21":
        direction = np.asarray(raw, dtype=np.float64) - v21
    else:
        raise ValueError(f"unknown direction: {direction_mode}")
    route_mask = _mask(frame["domain3"].astype(str).to_numpy(), route)
    direction = np.where(route_mask, direction, 0.0)
    weights = np.asarray(WEIGHTS, dtype=np.float64)
    endpoints = parent[:, None] + direction[:, None] * weights[None, :]
    if float(endpoints.min()) < 0.001 or float(endpoints.max()) > 0.999:
        raise ValueError(f"candidate clips: {signal} {direction_mode} {route}")
    target = frame["target"].to_numpy(np.float64)
    all_mask = np.ones(len(frame), dtype=bool)
    gains = _gain_grid(target, parent, direction, all_mask, weights)
    month_values = []
    for month in sorted(frame["game_month"].unique()):
        month_mask = frame["game_month"].eq(month).to_numpy()
        if np.any(month_mask & route_mask):
            month_values.append(
                _gain_grid(target, parent, direction, month_mask, weights)
            )
    month_matrix = np.vstack(month_values)
    domain_values = {}
    for domain in ("R_CORE", "R_ANCHOR", "F"):
        domain_mask = frame["domain3"].astype(str).eq(domain).to_numpy()
        domain_values[domain] = _gain_grid(
            target, parent, direction, domain_mask, weights
        )
    domain_matrix = np.vstack(list(domain_values.values()))
    applied = (
        domain_matrix.min(axis=0)
        if route == "ALL"
        else domain_values[route]
    )
    output = []
    for index, weight in enumerate(weights):
        selection_score = min(
            float(gains[index]),
            float(month_matrix[:, index].min()),
            float(applied[index]),
        )
        output.append(
            {
                "signal": signal,
                "family": signal_family(signal),
                "direction": direction_mode,
                "domain": route,
                "weight": float(weight),
                "gain": float(gains[index]),
                "positive_month_fraction": float(
                    np.mean(month_matrix[:, index] > 0.0)
                ),
                "worst_month_gain": float(month_matrix[:, index].min()),
                "minimum_domain_gain": float(domain_matrix[:, index].min()),
                "applied_domain_gain": float(applied[index]),
                "selection_score": selection_score,
                "mean_abs_shift": float(
                    weight * np.mean(np.abs(direction))
                ),
            }
        )
    return output


def family_prefilter(stage1: pd.DataFrame, quota: int = FAMILY_QUOTA) -> list[str]:
    best_identity = (
        stage1.sort_values(["selection_score", "gain"], ascending=False)
        .drop_duplicates("signal")
    )
    selected = []
    for family in sorted(best_identity["family"].unique()):
        local = best_identity.loc[
            best_identity["family"].eq(family)
        ].head(quota)
        selected.extend(local["signal"].astype(str).tolist())
    return selected


def _recipe_key(members: list[dict[str, object]]) -> str:
    return ";;".join(
        "|".join(str(member[key]) for key in ("signal", "direction", "domain", "weight"))
        for member in members
    )


def _candidate(
    frame: pd.DataFrame,
    bank: dict[str, np.ndarray],
    members: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    parent = v27_parent(frame)
    v21 = frame["v21"].to_numpy(np.float64)
    candidate = parent.copy()
    active = np.zeros(len(frame), dtype=bool)
    domain = frame["domain3"].astype(str).to_numpy()
    for member in members:
        local_mask = _mask(domain, str(member["domain"]))
        raw = bank[str(member["signal"])]
        if str(member["direction"]) == "toward_parent":
            direction = raw - parent
        else:
            direction = raw - v21
        candidate[local_mask] += float(member["weight"]) * direction[local_mask]
        active |= local_mask
    return np.clip(candidate, 0.001, 0.999), active


def _selection_recipe_result(
    frame: pd.DataFrame,
    bank: dict[str, np.ndarray],
    members: list[dict[str, object]],
) -> dict[str, object]:
    parent = v27_parent(frame)
    candidate, active = _candidate(frame, bank, members)
    result = diagnostics(frame, parent, candidate, active)
    return {
        **{
            key: value
            for key, value in result.items()
            if key not in {"months", "domain_gains"}
        },
        "selection_score": min(
            float(result["gain"]),
            float(result["worst_month_gain"]),
            float(result["minimum_domain_gain"]),
        ),
    }


def _slice_bank(
    bank: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: value[mask] for name, value in bank.items()}


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta22 = _metadata(project, 2022)
    stage1_frame = pd.DataFrame(
        {
            "target": meta22["target"],
            "game_month": meta22["month"],
            "domain3": meta22["domain"],
        }
    )
    bank22 = _load_bank(project, 2022)
    stage1_rows = []
    for signal, raw_prediction in bank22.items():
        for route in DOMAINS:
            stage1_rows.extend(
                grid_rows(
                    stage1_frame,
                    meta22["parent"],
                    meta22["parent"],
                    raw_prediction,
                    signal=signal,
                    direction_mode="toward_parent",
                    route=route,
                )
            )
    stage1 = pd.DataFrame(stage1_rows).sort_values(
        ["selection_score", "gain"], ascending=False
    )
    stage1.to_csv(output_dir / "stage1_2022_metrics.csv", index=False)
    prefiltered = family_prefilter(stage1)
    del bank22

    raw_train = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    axes = _cached_v25_axes(project, raw_train)
    del raw_train
    selection = axes["selection_late_2023"]
    full23_month = _metadata(project, 2023)["month"]
    late23 = full23_month >= 8
    bank23 = _slice_bank(_load_bank(project, 2023, set(prefiltered)), late23)
    if any(len(value) != len(selection) for value in bank23.values()):
        raise ValueError("late-2023 bank row mismatch")

    selection_parent = v27_parent(selection)
    selection_v21 = selection["v21"].to_numpy(np.float64)
    stage2_rows = []
    for signal, raw_prediction in bank23.items():
        for direction_mode, route in itertools.product(DIRECTIONS, DOMAINS):
            stage2_rows.extend(
                grid_rows(
                    selection,
                    selection_parent,
                    selection_v21,
                    raw_prediction,
                    signal=signal,
                    direction_mode=direction_mode,
                    route=route,
                )
            )
    stage2 = pd.DataFrame(stage2_rows)
    stage2["passes_selection_gate"] = (
        stage2["gain"].gt(0.0)
        & stage2["positive_month_fraction"].eq(1.0)
        & stage2["worst_month_gain"].gt(0.0)
        & stage2["applied_domain_gain"].gt(0.0)
        & stage2["minimum_domain_gain"].gt(-5.0)
    )
    stage2 = stage2.sort_values(
        ["passes_selection_gate", "selection_score", "gain"], ascending=False
    ).reset_index(drop=True)
    stage2.to_csv(output_dir / "stage2_2023_metrics.csv", index=False)

    passing = stage2.loc[stage2["passes_selection_gate"]]
    shortlist = (
        passing.sort_values(["selection_score", "gain"], ascending=False)
        .drop_duplicates(["family", "domain"])
        .head(16)
    )
    recipes: list[dict[str, object]] = []
    for _, row in shortlist.iterrows():
        member = {
            key: row[key]
            for key in ("signal", "direction", "domain", "weight")
        }
        result = _selection_recipe_result(selection, bank23, [member])
        recipes.append({"members": [member], **result})

    specific = shortlist.loc[~shortlist["domain"].eq("ALL")]
    for (_, left), (_, right) in itertools.combinations(specific.iterrows(), 2):
        if left["domain"] == right["domain"] or left["family"] == right["family"]:
            continue
        members = [
            {
                key: row[key]
                for key in ("signal", "direction", "domain", "weight")
            }
            for row in (left, right)
        ]
        result = _selection_recipe_result(selection, bank23, members)
        if (
            result["gain"] > 0.0
            and result["positive_month_fraction"] == 1.0
            and result["worst_month_gain"] > 0.0
            and result["minimum_domain_gain"] > -5.0
        ):
            recipes.append({"members": members, **result})
    recipes.sort(
        key=lambda item: (item["selection_score"], item["gain"]), reverse=True
    )
    if recipes:
        chosen = recipes[0]
    else:
        fallback = stage2.iloc[0]
        member = {
            key: fallback[key]
            for key in ("signal", "direction", "domain", "weight")
        }
        chosen = {
            "members": [member],
            **_selection_recipe_result(selection, bank23, [member]),
        }
    chosen_members = chosen["members"]
    chosen_signals = {str(member["signal"]) for member in chosen_members}
    del bank23

    bank24_full = _load_bank(project, 2024, chosen_signals)
    audit_results: dict[str, dict[str, object]] = {}
    for axis_name in ("outer_full_2024", "replication_late_2024"):
        frame = axes[axis_name]
        if axis_name == "outer_full_2024":
            local_bank = bank24_full
        else:
            mask24 = _metadata(project, 2024)["month"] >= 8
            local_bank = _slice_bank(bank24_full, mask24)
        if any(len(value) != len(frame) for value in local_bank.values()):
            raise ValueError(f"bank row mismatch: {axis_name}")
        candidate, active = _candidate(frame, local_bank, chosen_members)
        result = diagnostics(frame, v27_parent(frame), candidate, active)
        audit_results[axis_name] = result
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=frame["target"].to_numpy(np.float64),
            v27=v27_parent(frame),
            candidate=candidate,
            active=active,
            domain3=frame["domain3"].astype(str).to_numpy(),
            game_month=frame["game_month"].to_numpy(np.int16),
        )

    gates = {
        "stage1_four_families": len({signal_family(name) for name in prefiltered}) == 4,
        "stage2_gate": bool(recipes),
        "outer_gain_at_least_5": audit_results["outer_full_2024"]["gain"] >= 5.0,
        "outer_month_fraction_at_least_075": audit_results["outer_full_2024"][
            "positive_month_fraction"
        ]
        >= 0.75,
        "outer_worst_month_above_minus_10": audit_results["outer_full_2024"][
            "worst_month_gain"
        ]
        > -10.0,
        "outer_minimum_domain_positive": audit_results["outer_full_2024"][
            "minimum_domain_gain"
        ]
        > 0.0,
        "replication_gain_positive": audit_results["replication_late_2024"][
            "gain"
        ]
        > 0.0,
        "replication_month_fraction_at_least_two_thirds": audit_results[
            "replication_late_2024"
        ]["positive_month_fraction"]
        >= 2.0 / 3.0,
        "replication_worst_month_above_minus_10": audit_results[
            "replication_late_2024"
        ]["worst_month_gain"]
        > -10.0,
    }
    summary = {
        "protocol": "V35_THREE_STAGE_FAMILY_BALANCED_MULTIBANK_ABOVE_V27_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "stage1": "2022 identity prefilter only; non-v27 parent",
        "stage2": "late-2023 direction/route/weight/pair selection above v27",
        "stage3": "frozen full-2024 audit; late-2024 secondary stability check",
        "signal_count": int(stage1["signal"].nunique()),
        "prefiltered_signals": prefiltered,
        "stage2_candidate_count": int(len(stage2)),
        "stage2_gate_count": int(stage2["passes_selection_gate"].sum()),
        "recipe_count": int(len(recipes)),
        "chosen": {
            "recipe": _recipe_key(chosen_members),
            "members": chosen_members,
            "selection_gain": chosen["gain"],
            "selection_worst_month_gain": chosen["worst_month_gain"],
            "selection_minimum_domain_gain": chosen["minimum_domain_gain"],
        },
        "audits": audit_results,
        "gates": gates,
        "eligible_for_packaging": bool(all(gates.values())),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "audit_labels_used_for_selection": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v35_three_stage_multibank_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
