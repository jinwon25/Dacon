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

from src.core.diagnostics import diagnostics, v27_parent
from src.core.axes import _cached_v25_axes
from src.core.banks import _load_bank, _mask, _metadata, grid_rows, signal_family


WEIGHTS = (0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20)
DOMAINS = ("ALL", "R_CORE", "R_ANCHOR", "F")
DIRECTIONS = ("toward_parent", "delta_v21")
FAMILY_QUOTA = 3
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
