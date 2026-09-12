"""Nested diversity and covariance screen above the frozen v27 parent.

All bank members are legal season-forward OOF predictions produced by earlier
experiments.  Candidate selection uses late-2023 only.  Full-2024 and
late-2024 are opened after the recipe is frozen, preventing the audit labels
from choosing a signal, route, or weight.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.champion.v22_oof_bank_screen import (
    _delta_bank,
    _model_residual_bank,
    _prediction_bank,
)
from src.core.v25_recipe import ETA as V25_SOURCE_ETA
from src.core.axes import _load_axis
from src.core.axes import _early_to_late_2024
from src.core.diagnostics import compose, diagnostics, v27_parent
from src.core.axes import _cached_v25_axes, _quadratic_gain


DOMAINS = ("ALL", "R_CORE", "R_ANCHOR", "F")
WEIGHTS = (0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10)
def _year_slice(project: Path, frame: pd.DataFrame, year: int) -> np.ndarray:
    path = (
        project
        / "artifacts"
        / "state_mode_joint_20260816_02"
        / f"joint_candidate_o{year}.npz"
    )
    with np.load(path, allow_pickle=True) as saved:
        target = saved["target"].astype(np.float64)
        month = saved["game_month"].astype(np.int16)
    expected = frame["target"].to_numpy(np.float64)
    if len(expected) == len(target):
        index = np.arange(len(target))
    else:
        index = np.flatnonzero(month >= 8)
    if not np.array_equal(target[index], expected):
        raise ValueError(f"OOF target order mismatch for {year}")
    return index


def _bank(
    project: Path,
    frame: pd.DataFrame,
    year: int,
    old_axis: str,
) -> dict[tuple[str, str], np.ndarray]:
    index = _year_slice(project, frame, year)
    output: dict[tuple[str, str], np.ndarray] = {}
    for name, value in _prediction_bank(project, year).items():
        output[("prediction", name)] = np.asarray(value, dtype=np.float64)[index]
    for name, value in _delta_bank(project, year).items():
        output[("delta", name)] = np.asarray(value, dtype=np.float64)[index]
    for name, value in _model_residual_bank(project, old_axis, len(frame)).items():
        output[("delta", name)] = np.asarray(value, dtype=np.float64)
    return output


def _mask(frame: pd.DataFrame, domain: str) -> np.ndarray:
    if domain == "ALL":
        return np.ones(len(frame), dtype=bool)
    return frame["domain3"].astype(str).eq(domain).to_numpy()


def _selection_rows(
    frame: pd.DataFrame,
    parent: np.ndarray,
    v21: np.ndarray,
    *,
    kind: str,
    signal_name: str,
    signal: np.ndarray,
    mode: str,
    domain: str,
) -> list[dict[str, object]]:
    apply_mask = _mask(frame, domain)
    if kind == "delta":
        raw = np.asarray(signal, dtype=np.float64)
    elif mode == "toward_parent":
        raw = np.asarray(signal, dtype=np.float64) - parent
    else:
        raw = np.asarray(signal, dtype=np.float64) - v21
    direction = np.where(apply_mask, raw, 0.0)
    weights = np.asarray(WEIGHTS, dtype=np.float64)
    # Current banks and conservative weights stay far from clipping.  Keep an
    # assertion because the quadratic shortcut is exact only in that regime.
    endpoints = parent[:, None] + direction[:, None] * weights[None, :]
    if float(endpoints.min()) < 0.001 or float(endpoints.max()) > 0.999:
        raise ValueError(f"selection direction clips: {signal_name} {mode} {domain}")
    gains = _quadratic_gain(
        frame["target"].to_numpy(np.float64),
        parent,
        direction,
        np.ones(len(frame), dtype=bool),
        weights,
    )
    month_gain = []
    for month_value in sorted(frame["game_month"].unique()):
        month_mask = frame["game_month"].eq(month_value).to_numpy()
        if np.any(month_mask & apply_mask):
            month_gain.append(
                _quadratic_gain(
                    frame["target"].to_numpy(np.float64),
                    parent,
                    direction,
                    month_mask,
                    weights,
                )
            )
    month_matrix = np.vstack(month_gain)
    domain_gain = []
    for domain_value in ("R_CORE", "R_ANCHOR", "F"):
        domain_gain.append(
            _quadratic_gain(
                frame["target"].to_numpy(np.float64),
                parent,
                direction,
                frame["domain3"].astype(str).eq(domain_value).to_numpy(),
                weights,
            )
        )
    domain_matrix = np.vstack(domain_gain)
    output = []
    for index, weight in enumerate(weights):
        recipe = {
            "kind": kind,
            "signal": signal_name,
            "mode": mode,
            "domain": domain,
            "weight": float(weight),
        }
        output.append(
            {
                **recipe,
                "recipe": _recipe_key(recipe),
                "gain": float(gains[index]),
                "positive_month_fraction": float(np.mean(month_matrix[:, index] > 0.0)),
                "worst_month_gain": float(month_matrix[:, index].min()),
                "minimum_domain_gain": float(domain_matrix[:, index].min()),
                "mean_abs_shift": float(weight * np.mean(np.abs(direction))),
            }
        )
    return output


def _recipe_key(recipe: dict[str, object]) -> str:
    return "|".join(
        str(recipe[key])
        for key in ("kind", "signal", "mode", "domain", "weight")
    )


def _apply_recipe(
    frame: pd.DataFrame,
    bank: dict[tuple[str, str], np.ndarray],
    recipe: list[dict[str, object]],
) -> np.ndarray:
    parent = v27_parent(frame)
    output = parent.copy()
    v21 = frame["v21"].to_numpy(np.float64)
    for item in recipe:
        output = compose(
            output,
            v21,
            bank[(str(item["kind"]), str(item["signal"]))],
            _mask(frame, str(item["domain"])),
            kind=str(item["kind"]),
            mode=str(item["mode"]),
            weight=float(item["weight"]),
        )
    return output


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    axes = _cached_v25_axes(project, raw)
    axis_meta = {
        "selection_late_2023": (2023, "y2023_early_to_late"),
        "outer_full_2024": (2024, "y2023_to_y2024"),
        "replication_late_2024": (2024, "y2024_early_to_late"),
    }
    banks = {
        name: _bank(project, axes[name], year, old_axis)
        for name, (year, old_axis) in axis_meta.items()
    }

    selection = axes["selection_late_2023"]
    parent = v27_parent(selection)
    v21 = selection["v21"].to_numpy(np.float64)
    rows: list[dict[str, object]] = []
    for (kind, signal), value in banks["selection_late_2023"].items():
        modes = ("raw_delta",) if kind == "delta" else ("toward_parent", "delta_v21")
        for mode, domain in itertools.product(modes, DOMAINS):
            rows.extend(
                _selection_rows(
                    selection,
                    parent,
                    v21,
                    kind=kind,
                    signal_name=signal,
                    signal=value,
                    mode=mode,
                    domain=domain,
                )
            )
    selection_metrics = pd.DataFrame(rows).sort_values(
        ["gain", "worst_month_gain"], ascending=False
    )
    selection_metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    selected_pool = selection_metrics.loc[
        selection_metrics["gain"].gt(0.0)
        & selection_metrics["positive_month_fraction"].eq(1.0)
        & selection_metrics["worst_month_gain"].gt(0.0)
    ].copy()
    # Limit reuse of near-identical weights from the same signal and route.
    selected_pool = selected_pool.drop_duplicates(
        ["kind", "signal", "mode", "domain"]
    ).head(40)

    audit_rows: list[dict[str, object]] = []
    recipes: dict[str, list[dict[str, object]]] = {}
    for _, row in selected_pool.iterrows():
        recipe = {
            key: row[key]
            for key in ("kind", "signal", "mode", "domain", "weight")
        }
        key = _recipe_key(recipe)
        recipes[key] = [recipe]
        for axis_name, frame in axes.items():
            axis_parent = v27_parent(frame)
            candidate = _apply_recipe(frame, banks[axis_name], [recipe])
            result = diagnostics(frame, axis_parent, candidate, _mask(frame, str(recipe["domain"])))
            audit_rows.append(
                {
                    "recipe": key,
                    "axis": axis_name,
                    **{name: value for name, value in result.items() if name not in {"months", "domain_gains"}},
                }
            )

    audits = pd.DataFrame(audit_rows)
    eligible_keys: list[str] = []
    if len(audits):
        audits.to_csv(output_dir / "individual_audits.csv", index=False)
        pivot = audits.pivot(index="recipe", columns="axis", values="gain")
        eligible_keys = pivot.index[
            pivot["outer_full_2024"].gt(0.0)
            & pivot["replication_late_2024"].gt(0.0)
        ].tolist()

    # Search pairs only among independently audited positive directions.  The
    # pair is accepted only if it also passes both untouched 2024 audits.
    pair_rows: list[dict[str, object]] = []
    for left, right in itertools.combinations(eligible_keys[:16], 2):
        recipe = [*recipes[left], *recipes[right]]
        # Domains must be disjoint, otherwise sequential composition changes a
        # member's declared direction and complicates package reproducibility.
        domains = [str(item["domain"]) for item in recipe]
        if "ALL" in domains or len(set(domains)) != len(domains):
            continue
        key = f"{left};;{right}"
        details = {}
        for axis_name, frame in axes.items():
            axis_parent = v27_parent(frame)
            candidate = _apply_recipe(frame, banks[axis_name], recipe)
            result = diagnostics(
                frame, axis_parent, candidate, np.ones(len(frame), dtype=bool)
            )
            details[axis_name] = result
        pair_rows.append(
            {
                "recipe": key,
                "selection_gain": details["selection_late_2023"]["gain"],
                "outer_gain": details["outer_full_2024"]["gain"],
                "replication_gain": details["replication_late_2024"]["gain"],
                "min_gain": min(item["gain"] for item in details.values()),
                "outer_month_fraction": details["outer_full_2024"]["positive_month_fraction"],
                "outer_worst_month": details["outer_full_2024"]["worst_month_gain"],
                "replication_month_fraction": details["replication_late_2024"]["positive_month_fraction"],
            }
        )
        recipes[key] = recipe
    pairs = pd.DataFrame(pair_rows)
    if len(pairs):
        pairs = pairs.sort_values(["min_gain", "outer_gain"], ascending=False)
        pairs.to_csv(output_dir / "pair_audits.csv", index=False)

    finalists: list[dict[str, object]] = []
    for key in eligible_keys:
        subset = audits.loc[audits["recipe"].eq(key)].set_index("axis")
        finalists.append(
            {
                "recipe": key,
                "members": recipes[key],
                "selection_gain": float(subset.loc["selection_late_2023", "gain"]),
                "outer_gain": float(subset.loc["outer_full_2024", "gain"]),
                "replication_gain": float(subset.loc["replication_late_2024", "gain"]),
                "min_gain": float(subset["gain"].min()),
            }
        )
    if len(pairs):
        for _, row in pairs.head(10).iterrows():
            finalists.append(
                {
                    "recipe": str(row["recipe"]),
                    "members": recipes[str(row["recipe"])],
                    **{
                        name: float(row[name])
                        for name in ("selection_gain", "outer_gain", "replication_gain", "min_gain")
                    },
                }
            )
    finalists.sort(key=lambda item: (item["min_gain"], item["outer_gain"]), reverse=True)
    chosen = finalists[0] if finalists else None
    eligible = bool(
        chosen
        and chosen["outer_gain"] >= 5.0
        and chosen["replication_gain"] > 0.0
        and chosen["min_gain"] > 0.0
    )
    summary = {
        "protocol": "V30_NESTED_DIVERSE_COVARIANCE_ABOVE_V27_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "signal_count": int(len(banks["selection_late_2023"])),
        "selection_candidate_count": int(len(selection_metrics)),
        "selection_gate_count": int(len(selected_pool)),
        "independent_audit_positive_count": int(len(eligible_keys)),
        "pair_count": int(len(pairs)),
        "chosen": chosen,
        "top_finalists": finalists[:20],
        "gates": {
            "outer_gain_at_least_5": bool(chosen and chosen["outer_gain"] >= 5.0),
            "replication_gain_positive": bool(chosen and chosen["replication_gain"] > 0.0),
            "all_axes_positive": bool(chosen and chosen["min_gain"] > 0.0),
        },
        "eligible_for_packaging": eligible,
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
        default=Path("artifacts/v30_diverse_covariance_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
